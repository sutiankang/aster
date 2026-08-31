"""Strict local safetensors loading into native dense or tensor-parallel models."""

from __future__ import annotations
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import stat
import torch
from torch import nn

from ..models import build_model, LlamaConfig, Qwen2Config, Qwen3Config
from ..nn.position import RopeConfig, RotaryEmbedding
from .distributed import ParallelCausalPredictor


def _json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate checkpoint JSON key")
            result[key] = value
        return result

    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Checkpoint metadata exceeds bound")

    def invalid(value):
        raise ValueError("Non-finite checkpoint JSON")

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid
    )


def _local_file(root, name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or ":" in name
    ):
        raise ValueError("Checkpoint path is not a local artifact-relative file")
    path = root / name
    for part in (path, *path.parents):
        if part == root.parent:
            break
        attributes = part.lstat()
        if part.is_symlink() or getattr(attributes, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            raise ValueError("Checkpoint symlink/reparse points are not allowed")
    actual = path.resolve(strict=True)
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ValueError("Checkpoint file escapes the immutable artifact")
    return actual


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_config_from_hf(data):

    classes = {"llama": LlamaConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}
    if not isinstance(data, dict) or data.get("model_type") not in classes:
        raise ValueError("HF architecture is not supported by this native importer")
    if (
        data.get("auto_map")
        or data.get("quantization_config")
        or data.get("is_encoder_decoder", False)
    ):
        raise ValueError(
            "Remote-code, externally quantized and encoder-decoder imports are not enabled"
        )
    family = {field.name for field in fields(classes[data["model_type"]])} - {"rope"}
    metadata = {
        "model_type",
        "architectures",
        "transformers_version",
        "_name_or_path",
        "name_or_path",
        "torch_dtype",
        "dtype",
        "return_dict",
        "output_hidden_states",
        "output_attentions",
        "use_cache",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "id2label",
        "label2id",
        "problem_type",
        "num_labels",
        "tokenizer_class",
        "is_encoder_decoder",
        "auto_map",
        "pretraining_tp",
        "hidden_act",
        "mlp_bias",
        "attention_bias",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "chunk_size_feed_forward",
        "use_sliding_window",
        "layer_types",
        "max_window_layers",
    }
    if set(data) - family - metadata:
        raise ValueError(
            "Unrecognized HF config fields require an explicit semantic mapping: "
            + ", ".join(sorted(set(data) - family - metadata))
        )
    required = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rms_norm_eps",
        "initializer_range",
        "tie_word_embeddings",
    }
    if not required <= data.keys():
        raise ValueError("HF dimensions/normalization/tie fields must be explicit")
    if (
        data.get("hidden_act", "silu") != "silu"
        or data.get("mlp_bias", False)
        or data.get("chunk_size_feed_forward", 0) != 0
    ):
        raise ValueError("Unsupported HF activation/bias/feedforward semantics")
    expected_bias = data["model_type"] == "qwen2"
    if (
        data.get("attention_bias", expected_bias) != expected_bias
        or data.get("pretraining_tp", 1) != 1
    ):
        raise ValueError("HF attention bias or pretraining slicing semantics differ")
    if data.get("use_sliding_window", False) or (
        data.get("layer_types") is not None
        and data["layer_types"] != ["full_attention"] * data["num_hidden_layers"]
    ):
        raise ValueError("HF importer currently supports only explicit full-attention dense layers")
    rope = dict(data.get("rope_parameters") or data.get("rope_scaling") or {})
    if data.get("rope_parameters") and data.get("rope_scaling"):
        raise ValueError("Ambiguous old/new RoPE configuration")
    theta = rope.pop("rope_theta", data.get("rope_theta"))
    if theta is None:
        raise ValueError("RoPE theta must be present; importer never guesses family defaults")
    kind = rope.pop("rope_type", rope.pop("type", "default"))
    supported_rope = {field.name for field in fields(RopeConfig)} - {"kind", "theta", "interleaved"}
    if set(rope) - supported_rope:
        raise ValueError("Unsupported RoPE configuration fields")
    values = {name: data[name] for name in family if name in data}
    if data["model_type"] == "qwen2" and values.get("sliding_window") is None:
        values.pop("sliding_window", None)

    for name in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    ):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError("HF shape fields must be positive integers")
    if values["num_hidden_layers"] > 4096:
        raise ValueError("Layer count exceeds importer metadata bound")
    values["rope"] = RopeConfig(kind=kind, theta=theta, **rope)
    return classes[data["model_type"]](**values)


def load_hf_safetensors(
    path,
    *,
    device="cpu",
    dtype=None,
    parallel=None,
    expected_hashes=None,
    max_tensor_bytes=4 * 1024**3,
    max_total_bytes=256 * 1024**3,
):

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "HF safetensors import requires the separately provisioned safetensors package"
        ) from error
    root = Path(path).resolve(strict=True)
    if not root.is_dir() or min(max_tensor_bytes, max_total_bytes) < 1:
        raise ValueError("A local model directory and positive load bounds are required")
    config_file = _local_file(root, "config.json")
    config = native_config_from_hf(_json(config_file))
    index = root / "model.safetensors.index.json"
    files = {"config.json": config_file}
    if index.exists():
        index = _local_file(root, index.name)
        files[index.name] = index
        metadata = _json(index)
        weight_map = metadata.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("HF index must have a nonempty weight_map")
        for filename in set(weight_map.values()):
            files[filename] = _local_file(root, filename)
    else:
        files["model.safetensors"] = _local_file(root, "model.safetensors")
        with safe_open(str(files["model.safetensors"]), framework="pt", device="cpu") as source:
            weight_map = {name: "model.safetensors" for name in source.keys()}
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value.endswith(".safetensors")
        for key, value in weight_map.items()
    ):
        raise ValueError("HF index entries must be safetensors key/path pairs")
    hashes = {name: _sha(file) for name, file in files.items()}
    if expected_hashes is not None and hashes != dict(expected_hashes):
        raise ValueError("Checkpoint file set/hash differs from the pinned manifest")

    headers = {}
    element_bytes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
    for filename in set(weight_map.values()):
        with safe_open(str(files[filename]), framework="pt", device="cpu") as source:
            actual = set(source.keys())
            if actual != {name for name, file in weight_map.items() if file == filename}:
                raise ValueError("Shard contents and weight_map disagree")
            for name in actual:
                sliced = source.get_slice(name)
                shape, code = tuple(sliced.get_shape()), sliced.get_dtype()
                if code not in element_bytes:
                    raise ValueError("Only unquantized floating HF weights are supported")
                size = math.prod(shape) * element_bytes[code]
                if size > max_tensor_bytes:
                    raise ValueError("Source tensor exceeds configured load bound")
                headers[name] = shape, size
    total_bytes = sum(size for _, size in headers.values())
    if total_bytes > max_total_bytes:
        raise ValueError("Checkpoint parameter bytes exceed configured model bound")
    with torch.device("meta"):
        full = build_model(config)
        model = ParallelCausalPredictor(full, parallel) if parallel is not None else full
    full_shapes = {name: tuple(value.shape) for name, value in full.state_dict().items()}
    ties = {}
    for name, parameter in full.named_parameters(remove_duplicate=False):
        ties.setdefault(id(parameter), []).append(name)
    aliases = {}
    for names in ties.values():
        supplied = [name for name in names if name in headers]
        if supplied:
            aliases.update({name: supplied[0] for name in names if name not in headers})
    if set(headers) - set(full_shapes) or set(full_shapes) - set(headers) - set(aliases):
        raise ValueError("HF tensor keys differ from the exact native architecture")
    if any(headers[name][0] != full_shapes[name] for name in headers):
        raise ValueError("HF weight shape differs from the mapped native configuration")
    if dtype is not None and dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError("Import dtype must be a supported floating torch dtype")
    parameters = list(model.named_parameters(remove_duplicate=False))
    already, loaded_count, largest, loaded_bytes = {}, 0, 0, 0
    for local_name, original in parameters:
        if parallel is None:
            source_name = local_name
        elif local_name.startswith("layers."):
            _, position, tail = local_name.split(".", 2)
            source_name = f"model.layers.{model.stage_start + int(position)}.{tail}"
        elif local_name.startswith(("embed_tokens.", "norm.")):
            source_name = "model." + local_name
        else:
            source_name = local_name
        source_name = aliases.get(source_name, source_name)
        with safe_open(str(files[weight_map[source_name]]), framework="pt", device="cpu") as source:
            slicing = [slice(None)] * len(headers[source_name][0])
            dimension = getattr(original, "_aster_tp_dimension", None)
            if parallel is not None and dimension is not None:
                width = headers[source_name][0][dimension] // parallel.tp.size
                slicing[dimension] = slice(parallel.tp.rank * width, (parallel.tp.rank + 1) * width)
            value = source.get_slice(source_name)[tuple(slicing)]
            if value.shape != original.shape:
                raise ValueError("Native shard shape disagrees with checkpoint slice")
            materialized = value.numel() * value.element_size()
            largest, loaded_bytes = max(largest, materialized), loaded_bytes + materialized

            value = value.to(device=device, dtype=dtype or value.dtype, copy=True)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite checkpoint parameter")
            if id(original) in already:
                shared = already[id(original)]
                if shared.dtype != value.dtype or not torch.equal(shared, value):
                    raise ValueError("Tied checkpoint parameters disagree")
                parameter = shared
            else:
                parameter = nn.Parameter(value, requires_grad=False)
                already[id(original)] = parameter
                for attribute in ("_aster_tp_dimension", "_aster_tp_sharded"):
                    if hasattr(original, attribute):
                        setattr(parameter, attribute, getattr(original, attribute))
            owner, _, attribute = local_name.rpartition(".")
            setattr(model.get_submodule(owner), attribute, parameter)
            loaded_count += 1

    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = RotaryEmbedding(module.dim, module.config).inv_freq.to(device)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Unmaterialized model state remains; importer mapping is incomplete")
    if hashes != {name: _sha(file) for name, file in files.items()}:
        raise ValueError("Checkpoint changed while being imported")
    model.eval()
    model.load_report = {
        "format": "hf_safetensors_native_import",
        "source_hashes": hashes,
        "source_parameter_bytes": total_bytes,
        "loaded_tensor_count": loaded_count,
        "loaded_slice_bytes": loaded_bytes,
        "largest_materialized_tensor_bytes": largest,
        "construction": "meta_then_parameter_slices",
        "parallel": None if parallel is None else parallel.to_dict(),
        "native_architecture": config.architecture,
        "runtime": "aster_native",
        "measured_rss_peak_bytes": None,
    }
    return model


def load_hf_artifact(store, artifact_id, **kwargs):
    artifact = store.get(artifact_id, verify=True)
    model = load_hf_safetensors(artifact.path, **kwargs)
    model.load_report["artifact_id"] = artifact.id
    return model
