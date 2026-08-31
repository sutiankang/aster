"""Symmetric grouped weight-only QAT with a clipped straight-through estimator."""

from __future__ import annotations
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..inference.optimization import PackedLinear


def grouped_fake_quantize(weight, scales, *, bits, group_size):
    if bits not in {4, 8} or weight.ndim != 2 or group_size < 1:
        raise ValueError("QAT supports explicit 4/8bit grouped Linear weights")
    outputs, inputs = weight.shape
    groups = math.ceil(inputs / group_size)
    if scales.shape != (outputs, groups) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("QAT scales must cover every output/input group")
    grouped = F.pad(weight.float(), (0, groups * group_size - inputs)).reshape(
        outputs, groups, group_size
    )
    scale = scales.detach().float()[..., None]
    qmax = 2 ** (bits - 1) - 1

    codes = (grouped / scale).round()
    mask = (codes >= -qmax) & (codes <= qmax)
    rounded = codes.clamp(-qmax, qmax) * scale
    surrogate = grouped * mask.to(grouped.dtype)
    fake = surrogate + (rounded - surrogate).detach()
    return fake.reshape(outputs, -1)[:, :inputs].to(weight.dtype)


class QATLinear(nn.Module):
    def __init__(self, linear, *, bits=4, group_size=128):
        super().__init__()
        if (
            type(linear) is not nn.Linear
            or bits not in {4, 8}
            or type(group_size) is not int
            or group_size < 1
        ):
            raise ValueError("QAT target must be an unwrapped dense Linear")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bits, self.group_size = bits, group_size
        self.weight = nn.Parameter(
            linear.weight.detach().clone(), requires_grad=linear.weight.requires_grad
        )
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
            if linear.bias is not None
            else None
        )
        self.register_buffer(
            "scales",
            torch.ones(
                linear.out_features,
                math.ceil(linear.in_features / group_size),
                device=linear.weight.device,
            ),
        )
        self.register_buffer("observer_enabled", torch.tensor(True, device=linear.weight.device))
        self.register_buffer("fake_quant_enabled", torch.tensor(True, device=linear.weight.device))
        self.observe()

    @torch.no_grad()
    def observe(self):
        if not torch.isfinite(self.weight).all():
            raise ValueError("Non-finite QAT weight")
        groups = self.scales.shape[1]
        grouped = F.pad(
            self.weight.detach().float(), (0, groups * self.group_size - self.in_features)
        ).reshape(self.out_features, groups, self.group_size)
        self.scales.copy_(grouped.abs().amax(-1).clamp_min(1e-8) / (2 ** (self.bits - 1) - 1))

    def forward(self, inputs):
        if bool(self.observer_enabled):
            self.observe()
        weight = (
            grouped_fake_quantize(
                self.weight, self.scales, bits=self.bits, group_size=self.group_size
            )
            if bool(self.fake_quant_enabled)
            else self.weight
        )
        return F.linear(inputs, weight, self.bias)

    def precision_contract(self):

        return {
            "kind": "native_symmetric_weight_qat",
            "bits": self.bits,
            "group_size": self.group_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "scale_observer": "per_output_group_absolute_max",
            "gradient": "rounded_code_clipped_ste",
        }

    @torch.no_grad()
    def to_packed(self):
        if not bool(self.fake_quant_enabled):
            raise ValueError(
                "Cannot export disabled fake quantization as a numerically matching packed model"
            )
        if bool(self.observer_enabled):
            self.observe()
        if (
            not torch.isfinite(self.weight).all()
            or not torch.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("QAT export requires finite weights and positive finite scales")
        packed = PackedLinear(
            self.in_features,
            self.out_features,
            bits=self.bits,
            group_size=self.group_size,
            bias=self.bias is not None,
        ).to(self.weight.device)
        grouped = F.pad(
            self.weight.float(), (0, packed.padded_features - self.in_features)
        ).reshape(self.out_features, packed.groups, self.group_size)
        qmax = 2 ** (self.bits - 1) - 1
        codes = (grouped / self.scales[..., None]).round().clamp(-qmax, qmax)

        packed._set_codes(codes, self.scales)
        if self.bias is not None:
            packed.bias.copy_(self.bias)
        packed.algorithm = "qat_symmetric_weight_only"
        return packed


def prepare_qat(model, *, targets, bits=4, group_size=128):
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("QAT needs distinct explicit target paths")
    modules = dict(model.named_modules(remove_duplicate=False))
    if any(name not in modules or type(modules[name]) is not nn.Linear for name in targets):
        raise ValueError("QAT targets must identify native dense Linear modules")
    all_parameters = list(model.named_parameters(remove_duplicate=False))
    for name in targets:
        source = modules[name]
        if sum(parameter is source.weight for _, parameter in all_parameters) != 1:
            raise ValueError(
                "Tied/aliased QAT weights need an explicit shared-quantization transform"
            )
    result = copy.deepcopy(model)
    for name in targets:
        parent, _, child = name.rpartition(".")
        setattr(
            result.get_submodule(parent),
            child,
            QATLinear(result.get_submodule(name), bits=bits, group_size=group_size),
        )
    return result


def configure_qat(model, *, observe=None, fake_quant=None):
    found = False
    for module in model.modules():
        if isinstance(module, QATLinear):
            found = True
            if observe is not None:
                if type(observe) is not bool:
                    raise ValueError("Observer toggle must be bool")
                module.observer_enabled.fill_(observe)
            if fake_quant is not None:
                if type(fake_quant) is not bool:
                    raise ValueError("Fake-quant toggle must be bool")
                module.fake_quant_enabled.fill_(fake_quant)
    if not found:
        raise ValueError("No QAT modules in model")
    return model


def convert_qat(model):
    """Create a deployment copy while retaining training parameters and optimizer state
    in the original model."""
    result = copy.deepcopy(model).eval()
    names = [name for name, module in result.named_modules() if isinstance(module, QATLinear)]
    if not names:
        raise ValueError("No QAT modules to convert")
    for name in names:
        parent, _, child = name.rpartition(".")
        setattr(result.get_submodule(parent), child, result.get_submodule(name).to_packed())
    result.requires_grad_(False)
    return result
