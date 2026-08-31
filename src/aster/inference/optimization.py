"""Native quantization and packed storage with explicit floating-point dequantized execution."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CalibrationData:
    inputs: torch.Tensor
    dataset_fingerprint: str

    def __post_init__(self):
        if self.inputs.ndim != 2 or min(self.inputs.shape) < 1 or not self.dataset_fingerprint:
            raise ValueError("Calibration needs nonempty aligned rows and a dataset fingerprint")
        if not torch.isfinite(self.inputs).all():
            raise ValueError("Calibration contains non-finite activations")


class PackedLinear(nn.Module):
    """Pack two 4-bit values per uint8, then explicitly dequantize for floating-point execution."""

    evidence_kind = "native_storage_reference"
    compute_provider = "torch_float_dequant_reference"

    def __init__(self, in_features, out_features, *, bits=4, group_size=128, bias=True):
        super().__init__()
        if bits not in {4, 8} or min(in_features, out_features, group_size) < 1:
            raise ValueError("PackedLinear supports positive shapes and 4/8 bits")
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        self.groups = math.ceil(in_features / group_size)
        self.padded_features = self.groups * group_size
        count = out_features * self.padded_features
        self.register_buffer(
            "packed_weight",
            torch.zeros(math.ceil(count / (2 if bits == 4 else 1)), dtype=torch.uint8),
        )
        self.register_buffer("scales", torch.ones(out_features, self.groups))
        self.register_buffer("input_scale", torch.ones(in_features))
        self.register_buffer("permutation", torch.arange(in_features))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)
        self.algorithm = "rtn_symmetric"
        self.calibration_fingerprint = None

    def configuration(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bits": self.bits,
            "group_size": self.group_size,
            "bias": self.bias is not None,
            "algorithm": self.algorithm,
            "calibration_fingerprint": self.calibration_fingerprint,
            "evidence_kind": self.evidence_kind,
            "compute_provider": self.compute_provider,
        }

    def _set_codes(self, codes, scales):
        offset = 2 ** (self.bits - 1)
        unsigned = (codes.reshape(-1).to(torch.int16) + offset).to(torch.uint8)
        if self.bits == 4:
            if unsigned.numel() % 2:
                unsigned = F.pad(unsigned, (0, 1))
            packed = unsigned[0::2] | (unsigned[1::2] << 4)
        else:
            packed = unsigned
        self.packed_weight.copy_(packed)
        self.scales.copy_(scales)

    def dequantized_weight(self, *, original_coordinates=False):
        if self.bits == 4:
            data = torch.stack((self.packed_weight & 15, self.packed_weight >> 4), dim=1).reshape(
                -1
            )
        else:
            data = self.packed_weight
        count = self.out_features * self.padded_features
        integers = data[:count].to(self.scales.dtype) - 2 ** (self.bits - 1)
        grouped = integers.reshape(self.out_features, self.groups, self.group_size)
        weight = (grouped * self.scales[..., None]).reshape(self.out_features, -1)[
            :, : self.in_features
        ]
        if original_coordinates:
            weight = weight[:, torch.argsort(self.permutation)] / self.input_scale[None, :]
        return weight

    def forward(self, inputs):
        scaled = (inputs / self.input_scale.to(inputs.dtype))[..., self.permutation]
        weight = self.dequantized_weight().to(dtype=inputs.dtype)
        return F.linear(
            scaled, weight, self.bias.to(inputs.dtype) if self.bias is not None else None
        )

    @property
    def stored_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self.buffers())


def _rtn(weight, bits, group_size, *, clipping=None):
    out_features, in_features = weight.shape
    groups = math.ceil(in_features / group_size)
    grouped = F.pad(weight, (0, groups * group_size - in_features)).reshape(
        out_features, groups, group_size
    )
    maximum = grouped.abs().amax(-1) if clipping is None else clipping
    qmax = 2 ** (bits - 1) - 1
    scales = maximum.clamp_min(1e-8) / qmax
    codes = (grouped / scales[..., None]).round().clamp(-qmax, qmax)
    return codes, scales


@torch.no_grad()
def quantize_linear(
    linear,
    *,
    bits=4,
    group_size=128,
    algorithm="rtn",
    calibration=None,
    smooth_alpha=0.5,
    damping=0.01,
    act_order=False,
    search_grid=20,
    clip_grid=10,
):
    """Apply the requested calibration/quantization algorithm rather than aliasing
    all methods to independent rounding."""
    if type(linear) is not nn.Linear or bits not in {4, 8} or group_size < 1:
        raise ValueError("A supported dense Linear and quantization layout are required")
    if algorithm not in {"rtn", "smoothquant", "gptq", "awq_linear"}:
        raise ValueError("Unknown quantization algorithm")
    weight = linear.weight.detach().float().cpu().clone()
    if not torch.isfinite(weight).all():
        raise ValueError("Cannot quantize non-finite weights")
    inputs = None
    if algorithm != "rtn":
        if calibration is None or calibration.inputs.shape[1] != linear.in_features:
            raise ValueError("This algorithm requires matching real calibration activations")
        inputs = calibration.inputs.detach().float().cpu()
    packed = PackedLinear(
        linear.in_features,
        linear.out_features,
        bits=bits,
        group_size=group_size,
        bias=linear.bias is not None,
    )
    if linear.bias is not None:
        packed.bias.copy_(linear.bias.detach().float().cpu())
    packed.algorithm = algorithm
    packed.calibration_fingerprint = (
        calibration.dataset_fingerprint if calibration is not None else None
    )
    input_scale = torch.ones(linear.in_features)
    if algorithm == "smoothquant":
        if not math.isfinite(smooth_alpha) or not 0 <= smooth_alpha <= 1:
            raise ValueError("SmoothQuant alpha must be within [0,1]")
        amax = inputs.abs().amax(0).clamp_min(1e-5)
        wmax = weight.abs().amax(0).clamp_min(1e-5)
        input_scale = (amax.pow(smooth_alpha) / wmax.pow(1 - smooth_alpha)).clamp_min(1e-5)
        weight *= input_scale[None, :]
    elif algorithm == "awq_linear":
        if search_grid < 2 or clip_grid < 1:
            raise ValueError("AWQ search requires a nontrivial grid")
        activation_scale = inputs.abs().mean(0).clamp_min(1e-8)
        reference = inputs @ weight.t()
        best_loss, best_scale = math.inf, None
        for index in range(search_grid):
            scale = activation_scale.pow(index / search_grid).clamp_min(1e-4)
            scale /= (scale.max() * scale.min()).sqrt()
            codes, factors = _rtn(weight * scale, bits, group_size)
            approximated = (codes * factors[..., None]).reshape(linear.out_features, -1)[
                :, : linear.in_features
            ]
            loss = float(((inputs / scale) @ approximated.t() - reference).square().mean())
            if loss < best_loss:
                best_loss, best_scale = loss, scale.clone()
        input_scale = best_scale
        weight *= input_scale[None, :]
        scaled_inputs = inputs / input_scale

        groups = math.ceil(linear.in_features / group_size)
        grouped = F.pad(weight, (0, groups * group_size - linear.in_features)).reshape(
            linear.out_features, groups, group_size
        )
        xs = F.pad(scaled_inputs, (0, groups * group_size - linear.in_features)).reshape(
            -1, groups, group_size
        )
        maximum = grouped.abs().amax(-1)
        best_max = maximum.clone()
        best_error = torch.full_like(maximum, torch.inf)
        for index in range(clip_grid):
            current_max = maximum * (1 - 0.5 * index / clip_grid)
            codes, factors = _rtn(weight, bits, group_size, clipping=current_max)
            error = grouped - codes * factors[..., None]

            errors = torch.einsum("ngd,ogd->nog", xs, error).square().mean(0)
            improved = errors < best_error
            best_max = torch.where(improved, current_max, best_max)
            best_error = torch.minimum(errors, best_error)
        codes, factors = _rtn(weight, bits, group_size, clipping=best_max)
        packed._set_codes(codes, factors)
    if algorithm == "gptq":
        if linear.in_features > 8192 or not math.isfinite(damping) or damping <= 0:
            raise ValueError("GPTQ reference needs <=8192 input channels and positive damping")
        hessian = inputs.t() @ inputs / max(1, len(inputs))
        dead = hessian.diag() == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0
        permutation = (
            hessian.diag().argsort(descending=True)
            if act_order
            else torch.arange(linear.in_features)
        )
        packed.permutation.copy_(permutation)
        weight = weight[:, permutation]
        hessian = hessian[permutation][:, permutation]
        hessian.diagonal().add_(damping * hessian.diag().mean())
        inverse_factor = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
        )
        groups = math.ceil(linear.in_features / group_size)
        quantized = torch.zeros(linear.out_features, groups * group_size)
        factors = torch.zeros(linear.out_features, groups)
        qmax = 2 ** (bits - 1) - 1
        for column in range(linear.in_features):
            group = column // group_size
            if column % group_size == 0:
                factors[:, group] = (
                    weight[:, column : column + group_size].abs().amax(1).clamp_min(1e-8) / qmax
                )
            scale = factors[:, group]
            values = weight[:, column].clone()
            quantized[:, column] = (values / scale).round().clamp(-qmax, qmax)
            error = (values - quantized[:, column] * scale) / inverse_factor[column, column]
            weight[:, column:] -= error[:, None] * inverse_factor[column, column:][None, :]
        packed._set_codes(quantized, factors)
    elif algorithm != "awq_linear":
        packed._set_codes(*_rtn(weight, bits, group_size))
    packed.input_scale.copy_(input_scale)
    return packed.to(linear.weight.device)


def collect_calibration(model, batches, *, targets, dataset_fingerprint, max_rows=2048):
    """Collect bounded real-forward inputs; a 2D mask excludes padded positions."""
    if max_rows < 1 or not dataset_fingerprint or not targets:
        raise ValueError("Calibration limits, targets and data identity are required")
    modules = dict(model.named_modules())
    if any(name not in modules or not isinstance(modules[name], nn.Linear) for name in targets):
        raise ValueError("Calibration targets must name native dense Linear layers")
    rows = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    mask = None

    def hook(name):
        def collect(module, args):
            values = args[0].detach()
            if mask is not None and values.ndim == 3 and values.shape[:2] == mask.shape:
                values = values[mask.bool()]
            else:
                values = values.reshape(-1, values.shape[-1])
            count = min(max_rows - counts[name], len(values))
            if count > 0:
                rows[name].append(values[:count].float().cpu())
                counts[name] += count

        return collect

    handles = [modules[name].register_forward_pre_hook(hook(name)) for name in targets]
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                mask = batch.get("attention_mask")
                if mask is not None and mask.ndim != 2:
                    raise ValueError(
                        "Calibration mask must explicitly identify valid token positions"
                    )
                model(
                    **{
                        key: value
                        for key, value in batch.items()
                        if key not in {"labels", "sample_ids"}
                    }
                )
                if all(count >= max_rows for count in counts.values()):
                    break
    finally:
        model.train(training)
        for handle in handles:
            handle.remove()
    if any(not values for values in rows.values()):
        raise ValueError("Calibration failed to observe every target")
    return {
        name: CalibrationData(torch.cat(values), dataset_fingerprint)
        for name, values in rows.items()
    }


def quantize_model(
    model, *, targets, bits=4, group_size=128, algorithm="rtn", calibration=None, **options
):
    """Return an independent transformed model; reject ambiguous shared Linear ownership."""
    modules = dict(model.named_modules())
    if not targets or any(
        name not in modules or type(modules[name]) is not nn.Linear for name in targets
    ):
        raise ValueError("Every quantization target must name a Linear")
    aliases = {}
    for name, module in model.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(name)
    if any(len(aliases[id(modules[name])]) > 1 for name in targets):
        raise ValueError("Shared module aliases need an explicit quantized sharing map")
    parameter_aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_aliases.setdefault(id(parameter), []).append(name)
    if any(len(parameter_aliases[id(modules[name].weight)]) > 1 for name in targets):
        raise ValueError("Tied weights need an explicit quantized sharing map")
    result = copy.deepcopy(model).eval()
    replacements = {}
    for name in targets:
        replacements[name] = quantize_linear(
            result.get_submodule(name),
            bits=bits,
            group_size=group_size,
            algorithm=algorithm,
            calibration=(calibration or {}).get(name),
            **options,
        )
    for name, module in replacements.items():
        parent, _, attribute = name.rpartition(".")
        setattr(result.get_submodule(parent) if parent else result, attribute, module)
    return result


def save_optimized_model(model, path, *, base_artifact_id, transformation_metadata=None):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "packed_model.pt").exists() or (path / "optimization.json").exists():
        raise ValueError("Refusing to overwrite an existing optimized export")
    if not base_artifact_id or not hasattr(model, "config"):
        raise ValueError("Optimized export needs a native config and base artifact identity")
    modules = {
        name: module.configuration()
        for name, module in model.named_modules()
        if isinstance(module, PackedLinear)
    }
    if not modules:
        raise ValueError("No packed modules to export")
    weights = path / "packed_model.pt"
    tensors = model.state_dict()
    torch.save(tensors, weights)
    manifest = {
        "schema_version": 1,
        "format": "aster_packed_linear_v1",
        "base_artifact_id": base_artifact_id,
        "config": model.config.to_dict(),
        "modules": modules,
        "transformations": transformation_metadata or {},
        "tensor_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "compute_provider": PackedLinear.compute_provider,
        "evidence_kind": PackedLinear.evidence_kind,
    }
    (path / "optimization.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return manifest


def load_optimized_model(path):

    from aster.models import build_model
    from aster.models.config import config_from_dict
    from aster.core import read_json

    path = Path(path)
    manifest = read_json(path / "optimization.json")
    if manifest["schema_version"] != 1 or manifest["format"] != "aster_packed_linear_v1":
        raise ValueError("Unsupported optimized artifact format")
    weights = path / "packed_model.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Packed weights checksum mismatch")
    model = build_model(config_from_dict(manifest["config"]))
    for name, specification in manifest["modules"].items():
        original = model.get_submodule(name)
        arguments = {
            key: specification[key]
            for key in ("in_features", "out_features", "bits", "group_size", "bias")
        }
        if type(original) is not nn.Linear or (original.in_features, original.out_features) != (
            arguments["in_features"],
            arguments["out_features"],
        ):
            raise ValueError("Packed module topology differs from the base architecture")
        module = PackedLinear(**arguments)
        module.algorithm = specification["algorithm"]
        module.calibration_fingerprint = specification["calibration_fingerprint"]
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute, module)
    tensors = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or any(
        not isinstance(value, torch.Tensor) for value in tensors.values()
    ):
        raise ValueError("Packed checkpoint must contain only named tensor state")
    if "tensor_dtypes" in manifest and manifest["tensor_dtypes"] != {
        name: str(value.dtype) for name, value in tensors.items()
    }:
        raise ValueError("Packed checkpoint tensor dtype manifest mismatch")

    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if len(names) > 1 and any(
            tensors[name].dtype != tensors[names[0]].dtype
            or not torch.equal(tensors[name], tensors[names[0]])
            for name in names[1:]
        ):
            raise ValueError("Packed checkpoint has inconsistent tied parameters")
    model.load_state_dict(tensors, strict=True, assign=True)
    for names in aliases.values():
        first = model.get_parameter(names[0])
        for name in names[1:]:
            parent, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, attribute, first)
    return model.eval()
