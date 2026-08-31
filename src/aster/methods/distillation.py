"""Logit and feature distillation, explicit gradient boundaries, and native linear LoRA."""

from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle
from .supervised import (
    model_inputs,
    token_targets,
    native_causal_config,
    preflight_causal_microbatches,
)


def distribution_divergence(
    student_logits, teacher_logits, *, kind="forward_kl", temperature=1.0, mixture=0.5
):
    if student_logits.shape != teacher_logits.shape or temperature <= 0:
        raise ValueError("Aligned distributions and positive temperature are required")
    if not 0 < mixture < 1:
        raise ValueError("mixture must be strictly between 0 and 1")
    log_s = F.log_softmax(student_logits.float() / temperature, -1)
    log_t = F.log_softmax(teacher_logits.float() / temperature, -1)
    forward = (log_t.exp() * (log_t - log_s)).sum(-1)
    reverse = (log_s.exp() * (log_s - log_t)).sum(-1)
    if kind == "forward_kl":
        result = forward
    elif kind == "reverse_kl":
        result = reverse
    elif kind == "mixed_kl":
        result = mixture * forward + (1 - mixture) * reverse
    elif kind == "js":
        weights = log_s.new_tensor([mixture, 1 - mixture]).log()
        log_m = torch.logsumexp(torch.stack((log_t + weights[0], log_s + weights[1])), dim=0)
        result = mixture * (log_t.exp() * (log_t - log_m)).sum(-1) + (1 - mixture) * (
            log_s.exp() * (log_s - log_m)
        ).sum(-1)
    else:
        raise ValueError("Unknown divergence; truncated top-k is not full KL")
    return result * temperature**2


def feature_distance(student, teacher, valid, *, kind="mse"):
    if student.shape[:2] != teacher.shape[:2] or valid.shape != student.shape[:2]:
        raise ValueError("Features must align in sample/token positions")
    student, teacher = student.float(), teacher.float()
    if kind == "relation":
        s = F.normalize(student, dim=-1)
        t = F.normalize(teacher, dim=-1)
        errors = (s @ s.transpose(-1, -2) - t @ t.transpose(-1, -2)).square()
        pair_mask = valid[:, :, None] & valid[:, None, :]
        return LossTerm(
            errors.masked_select(pair_mask).sum(),
            pair_mask.sum().to(errors.dtype),
            "token_pair",
            "feature",
        )
    if student.shape != teacher.shape:
        raise ValueError("Coordinate matching needs an explicit student-owned projection")
    if kind == "mse":
        errors = (student - teacher).square().mean(-1)
    elif kind == "cosine":
        errors = 1 - F.cosine_similarity(student, teacher, dim=-1)
    else:
        raise ValueError("Unknown feature objective")
    return LossTerm(
        errors.masked_select(valid).sum(), valid.sum().to(errors.dtype), "token", "feature"
    )


class DistillationObjective(nn.Module):
    def __init__(
        self,
        teacher,
        *,
        kind="forward_kl",
        temperature=1.0,
        kd_weight=0.5,
        causal=True,
        tokenizer_fingerprints=None,
        feature_weight=0.0,
        feature_kind="relation",
        layer_pairs=(),
    ):
        super().__init__()
        if not 0 <= kd_weight <= 1 or temperature <= 0 or feature_weight < 0:
            raise ValueError("Invalid distillation weights/temperature")
        if tokenizer_fingerprints is not None and (
            len(tokenizer_fingerprints) != 2
            or tokenizer_fingerprints[0] != tokenizer_fingerprints[1]
        ):
            raise ValueError(
                "Token KL requires identical vocab/template semantics; use sequence KD for unaligned vocabularies"
            )
        if feature_weight and not layer_pairs:
            raise ValueError("Feature distillation requires explicit student/teacher layer pairs")
        self.teacher = teacher.eval().requires_grad_(False)
        self.kind, self.temperature, self.kd_weight, self.causal = (
            kind,
            temperature,
            kd_weight,
            causal,
        )
        self.feature_weight, self.feature_kind, self.layer_pairs = (
            feature_weight,
            feature_kind,
            tuple(layer_pairs),
        )
        self.tokenizer_fingerprints = tokenizer_fingerprints

    def config_dict(self):
        return {
            "type": "distillation",
            "kind": self.kind,
            "temperature": self.temperature,
            "kd_weight": self.kd_weight,
            "causal": self.causal,
            "feature_weight": self.feature_weight,
            "feature_kind": self.feature_kind,
            "layer_pairs": self.layer_pairs,
            "tokenizer_fingerprints": self.tokenizer_fingerprints,
        }

    @torch.no_grad()
    def preflight_microbatches(self, model, batches):
        """Validate the complete accumulation window and all model roles used by this
        objective before forward or sharded parameter communication."""
        if {id(p) for p in model.parameters()} & {id(p) for p in self.teacher.parameters()}:
            raise ValueError("Student and frozen teacher cannot share parameter objects")
        if (
            self.kind not in {"forward_kl", "reverse_kl", "mixed_kl", "js"}
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
            or not math.isfinite(self.kd_weight)
            or not 0 <= self.kd_weight <= 1
            or not math.isfinite(self.feature_weight)
            or self.feature_weight < 0
        ):
            raise ValueError("Invalid declared distillation objective")
        batches = preflight_causal_microbatches(model, batches, causal=self.causal)
        preflight_causal_microbatches(self.teacher, batches, causal=self.causal)
        student_config, teacher_config = (
            native_causal_config(model),
            native_causal_config(self.teacher),
        )
        if self.causal and student_config is not None and teacher_config is not None:
            if student_config.vocab_size != teacher_config.vocab_size:
                raise ValueError("Token KL requires aligned student/teacher vocabulary dimensions")
            if self.feature_weight:
                if self.feature_kind not in {"relation", "mse", "cosine"} or not self.layer_pairs:
                    raise ValueError(
                        "Feature KD requires a supported distance and explicit layer pairs"
                    )
                if (
                    self.feature_kind != "relation"
                    and student_config.hidden_size != teacher_config.hidden_size
                ):
                    raise ValueError(
                        "Coordinate feature KD needs equal hidden sizes or an explicit projection"
                    )
                for pair in self.layer_pairs:
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                        raise ValueError("Feature KD layer pairs must contain two indices")
                    for index, config in zip(pair, (student_config, teacher_config)):
                        if (
                            type(index) is not int
                            or not -config.num_hidden_layers - 1
                            <= index
                            <= config.num_hidden_layers
                        ):
                            raise ValueError(
                                "Feature KD layer index exceeds that model's hidden-state sequence"
                            )
        return batches

    def forward(self, model, batch):
        if {id(p) for p in model.parameters()} & {id(p) for p in self.teacher.parameters()}:
            raise ValueError("Student and frozen teacher cannot share parameter objects")
        self.teacher.eval()
        inputs = model_inputs(batch)
        output = model(**inputs, use_cache=False, output_hidden_states=bool(self.feature_weight))

        with torch.no_grad():
            teacher = self.teacher(
                **inputs, use_cache=False, output_hidden_states=bool(self.feature_weight)
            )
        logits, labels, mask = token_targets(batch, output.logits, causal=self.causal)
        teacher_logits = teacher.logits[:, :-1] if self.causal else teacher.logits
        kd = distribution_divergence(
            logits, teacher_logits, kind=self.kind, temperature=self.temperature
        )
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.masked_fill(~mask, 0).reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        combined = (1 - self.kd_weight) * ce + self.kd_weight * kd
        terms = [
            LossTerm(
                combined.masked_select(mask).sum(),
                mask.sum().to(combined.dtype),
                "token",
                "distillation",
            )
        ]
        if self.feature_weight:
            if self.causal and native_causal_config(model) is not None:
                feature_padding = inputs.get("attention_mask")

                valid = (
                    torch.ones(output.logits.shape[:2], device=logits.device, dtype=torch.bool)
                    if feature_padding is None
                    else feature_padding.bool()
                )
            else:
                valid = batch.get(
                    "attention_mask", torch.ones(output.logits.shape[:2], device=logits.device)
                ).bool()
            if valid.ndim != 2:
                raise ValueError("Provide a 2D valid-position mask for feature distillation")
            for index, (student_layer, teacher_layer) in enumerate(self.layer_pairs):
                term = feature_distance(
                    output.hidden_states[student_layer],
                    teacher.hidden_states[teacher_layer],
                    valid,
                    kind=self.feature_kind,
                )
                terms.append(
                    LossTerm(
                        term.numerator,
                        term.denominator,
                        term.unit,
                        f"feature_{index}",
                        self.feature_weight / len(self.layer_pairs),
                    )
                )
        return terms[0] if len(terms) == 1 else LossBundle(tuple(terms))


class LoRALinear(nn.Module):
    """Apply W_eff = W + (alpha / rank) * B @ A with a frozen base projection.

    A uses Kaiming initialization; B starts at zero, preserving the initial base
    function. Dropout applies only to the adapter input. The merged evaluation
    projection is exact up to floating-point ordering; disable dropout before merging."""

    def __init__(self, base: nn.Linear, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        self.base, self.rank, self.scale = base.requires_grad_(False), rank, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, value):
        return (
            self.base(value) + F.linear(F.linear(self.dropout(value), self.a), self.b) * self.scale
        )

    @torch.no_grad()
    def merged(self):
        if self.training and self.dropout.p:
            raise ValueError("Merge requires eval mode when LoRA dropout is nonzero")
        result = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        result.weight.copy_(self.base.weight + self.scale * (self.b @ self.a))
        if result.bias is not None:
            result.bias.copy_(self.base.bias)
        return result


def inject_lora(model, *, targets, rank=4, alpha=8.0, dropout=0.0):
    matches = [(name, module) for name, module in model.named_modules() if name in set(targets)]
    if (
        len(matches) != len(set(targets))
        or not matches
        or any(not isinstance(m, nn.Linear) for _, m in matches)
    ):
        raise ValueError("Every explicit LoRA target must identify a distinct Linear module")
    model.requires_grad_(False)
    for name, module in matches:
        parent_name, _, child = name.rpartition(".")
        setattr(
            model.get_submodule(parent_name) if parent_name else model,
            child,
            LoRALinear(module, rank, alpha, dropout),
        )
    return model


def merge_lora(model):
    import copy

    result = copy.deepcopy(model).eval()
    for name, module in list(result.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child = name.rpartition(".")
            setattr(
                result.get_submodule(parent_name) if parent_name else result, child, module.merged()
            )
    return result
