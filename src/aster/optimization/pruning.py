"""Structured SwiGLU pruning with coordinated gate/up rows and down columns."""

from __future__ import annotations
from dataclasses import dataclass, replace
import torch
from torch import nn

from ..models import build_model, LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig
from ..models.decoder import CausalLM, DecoderLayer
from ..inference.optimization import collect_calibration


@dataclass(frozen=True)
class PruningResult:
    model: nn.Module
    manifest: dict


def _validate(model):
    if (
        not isinstance(model, CausalLM)
        or type(model.config) not in {LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig}
        or any(type(layer) is not DecoderLayer for layer in model.model.layers)
    ):
        raise ValueError(
            "Structured MLP pruning currently supports explicit dense decoder families"
        )
    if any(
        type(getattr(layer.mlp, name)) is not nn.Linear
        for layer in model.model.layers
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise ValueError(
            "Prune before QAT/LoRA/packed transforms, or provide an explicit composite transform"
        )


def mlp_importance(model, *, batches=None, dataset_fingerprint=None, max_rows=2048):
    _validate(model)
    paths = [f"model.layers.{index}.mlp.down_proj" for index in range(len(model.model.layers))]
    calibration = (
        collect_calibration(
            model,
            batches,
            targets=paths,
            dataset_fingerprint=dataset_fingerprint,
            max_rows=max_rows,
        )
        if batches is not None
        else None
    )
    result = {}
    for index, layer in enumerate(model.model.layers):
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        if calibration is not None:
            activation_rms = calibration[paths[index]].inputs.square().mean(0).sqrt()
            result[index] = down.norm(dim=0) * activation_rms
        else:
            gate, up = (
                layer.mlp.gate_proj.weight.detach().float().cpu(),
                layer.mlp.up_proj.weight.detach().float().cpu(),
            )
            result[index] = down.norm(dim=0) * gate.norm(dim=1) * up.norm(dim=1)
    return result


def prune_mlp(
    model, *, intermediate_size, importance=None, parent_artifact_id, calibration_fingerprint=None
):
    _validate(model)
    original = model.config.intermediate_size
    if (
        type(intermediate_size) is not int
        or not 1 <= intermediate_size < original
        or not parent_artifact_id
    ):
        raise ValueError(
            "Pruning must reduce a positive intermediate width and name its parent artifact"
        )
    supplied = importance is not None
    importance = mlp_importance(model) if importance is None else importance
    if set(importance) != set(range(len(model.model.layers))):
        raise ValueError("Every MLP layer requires its own channel importance")
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    kept = {}
    for index, scores in importance.items():
        if (
            not isinstance(scores, torch.Tensor)
            or scores.shape != (original,)
            or not torch.isfinite(scores).all()
            or (scores < 0).any()
        ):
            raise ValueError("Channel salience must be finite nonnegative and match original width")

        selected = (
            torch.argsort(scores, descending=True, stable=True)[:intermediate_size].sort().values
        )
        kept[str(index)] = selected.tolist()
        for name, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
            key = f"model.layers.{index}.mlp.{name}.weight"
            state[key] = state[key].index_select(axis, selected.to(state[key].device))
    config = replace(model.config, intermediate_size=intermediate_size)
    with torch.device("meta"):
        result = build_model(config)
    result.load_state_dict(state, strict=True, assign=True)

    for new, old in zip(result.model.layers, model.model.layers):
        new.self_attn.rope.inv_freq = old.self_attn.rope.inv_freq.detach().clone()
    if config.tie_word_embeddings:
        result.lm_head.weight = result.model.embed_tokens.weight
    trainable = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in result.named_parameters():
        parameter.requires_grad_(trainable[name])
    result.train(model.training)
    before = sum(p.numel() for p in model.parameters())
    after = sum(p.numel() for p in result.parameters())
    manifest = {
        "kind": "structured_swiglu_intermediate",
        "parent_artifact_id": parent_artifact_id,
        "original_intermediate_size": original,
        "intermediate_size": intermediate_size,
        "kept_channels": kept,
        "importance": "explicit_scores" if supplied else "weight_norm_product",
        "calibration_fingerprint": calibration_fingerprint,
        "parameters_before": before,
        "parameters_after": after,
        "config": config.to_dict(),
        "recovery": "compatible_with_native_DistillationObjective",
    }
    return PruningResult(result, manifest)
