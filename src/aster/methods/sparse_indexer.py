"""DSA attention-teacher and QSA microblock indexer distillation."""

import math
import torch
import torch.nn.functional as F
from aster.core import LossTerm, LossBundle


def indexer_distillation(scores, teacher_probabilities, visible, *, query_mask=None):
    """Compare scores [B,Q,K] against teacher attention [B,H,Q,K] or [B,Q,K]
    over the explicitly visible keys."""
    if scores.ndim != 3 or visible.shape != scores.shape or visible.dtype != torch.bool:
        raise ValueError("Indexer scores and boolean visibility must align [B,Q,K]")
    target = teacher_probabilities.detach().float()
    if target.ndim == 4:
        target = target.sum(1)
    if target.shape != scores.shape or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Teacher attention must be finite nonnegative probabilities")
    target = target.masked_fill(~visible, 0)
    valid = visible.any(-1) & (target.sum(-1) > 0)
    if query_mask is not None:
        if query_mask.shape != valid.shape:
            raise ValueError("Query mask must align with queries")
        valid &= query_mask.bool()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-20)
    safe_scores = scores.float().masked_fill(~visible, -torch.inf)
    safe_scores = torch.where(valid[..., None], safe_scores, torch.zeros_like(safe_scores))
    log_q = F.log_softmax(safe_scores, -1).masked_fill(~visible, 0)
    terms = torch.where(target > 0, target * (target.clamp_min(1e-20).log() - log_q), 0).sum(-1)
    return LossTerm(
        terms.masked_select(valid).sum(), valid.sum(dtype=torch.int64), "query", "indexer_kl"
    )


def prepare_dsa_stage(model, stage):
    """Declare the warmup or sparse-training stage before constructing a trainer.
    Changing parameter ownership requires a new trainer.

    References:
    https://arxiv.org/html/2512.02556v1"""
    from ..models.sparse import DeepSeekV32ForCausalLM

    if type(model) is not DeepSeekV32ForCausalLM or stage not in {
        "dense_warmup",
        "sparse_training",
    }:
        raise ValueError("DSA stage requires native DeepSeekV32 and a named training stage")
    if getattr(model, "_aster_training_owned", False):
        raise ValueError(
            "Cannot change DSA stage after Trainer ownership; export weights into a fresh model"
        )
    if model.config.attention_dropout != 0:
        raise ValueError("Audited DSA training profile requires zero attention dropout")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(stage == "sparse_training" or ".indexer." in name)
    for layer in model.model.layers:
        layer.self_attn.indexer_stage = stage
    return model


class DSAIndexerObjective(torch.nn.Module):
    """Compute token CE and per-layer attention-teacher KL in one forward.
    The shared trainer normalizes their different token/query counts independently."""

    def __init__(self, stage, *, indexer_weight=1.0):
        super().__init__()
        if stage not in {"dense_warmup", "sparse_training"}:
            raise ValueError("Unknown DSA training stage")
        if (
            type(indexer_weight) not in {int, float}
            or not math.isfinite(indexer_weight)
            or indexer_weight <= 0
        ):
            raise ValueError("DSA indexer weight must be finite and positive")
        self.stage, self.indexer_weight = stage, indexer_weight

    def config_dict(self):
        return dict(
            type="dsa_native_attention_kd",
            stage=self.stage,
            indexer_weight=self.indexer_weight,
            teacher="detached_pre_dropout_attention_head_sum",
            reduction="global_query_mean_layer_mean",
        )

    def _validate_model(self, model):
        from ..models.sparse import DeepSeekV32ForCausalLM
        from ..training.sharding import zero3_units

        if type(model) is not DeepSeekV32ForCausalLM or model.config.attention_dropout != 0:
            raise ValueError("DSA objective requires the audited native model with zero dropout")
        if any(layer.self_attn.indexer_stage != self.stage for layer in model.model.layers):
            raise ValueError("Call prepare_dsa_stage before constructing Trainer")

        placeholders = {id(p) for unit in zero3_units(model) for p in unit.module.parameters()}
        for name, parameter in model.named_parameters():
            expected = self.stage == "sparse_training" or ".indexer." in name
            if id(parameter) not in placeholders and parameter.requires_grad != expected:
                raise ValueError(
                    "DSA trainable parameter ownership differs from the declared stage"
                )

    def validate_training_context(self, model, parallel):
        self._validate_model(model)
        if any(
            size != 1 for name, size in vars(parallel.config).items() if name != "data_parallel"
        ):
            raise ValueError(
                "DSA two-stage profile currently admits pure DP with ZeRO0-3, not unverified model-parallel layouts"
            )

    def preflight_microbatches(self, model, batches):
        from .supervised import preflight_causal_microbatches, model_inputs

        self._validate_model(model)
        batches = preflight_causal_microbatches(model, batches)
        for batch in batches:
            if any(
                key in batch
                for key in ("teacher_attention", "dsa_teacher_attention", "teacher_probabilities")
            ):
                raise ValueError(
                    "DSA teachers must come from the actual main attention, not caller-supplied tensors"
                )
            mask = batch.get("indexer_query_mask")
            if mask is None:
                continue
            inputs = model_inputs(batch)
            tokens = inputs.get("input_ids", inputs.get("inputs_embeds"))
            if (
                not isinstance(mask, torch.Tensor)
                or mask.shape != tokens.shape[:2]
                or mask.device != tokens.device
                or mask.requires_grad
                or not ((mask == 0) | (mask == 1)).all()
            ):
                raise ValueError(
                    "indexer_query_mask must be fixed binary [B,Q] on the input device"
                )
        return batches

    def forward(self, model, batch):
        from .supervised import model_inputs, token_targets

        self.preflight_microbatches(model, [batch])
        inputs = model_inputs(batch)
        output = model(**inputs, use_cache=False)
        records = output.auxiliary["indexer"]
        query_mask = inputs.get("attention_mask")
        extra = batch.get("indexer_query_mask")
        if extra is not None:
            query_mask = extra.bool() if query_mask is None else query_mask.bool() & extra.bool()
        terms = []
        if self.stage == "sparse_training":
            prediction, labels, mask = token_targets(batch, output.logits)
            values = F.cross_entropy(
                prediction.flatten(0, 1), labels.masked_fill(~mask, 0).flatten(), reduction="none"
            ).reshape_as(labels)
            terms.append(
                LossTerm(
                    values.masked_select(mask).sum(), mask.sum(dtype=torch.int64), "token", "ce"
                )
            )
        for index, info in enumerate(records):
            term = indexer_distillation(
                info["scores"],
                info["teacher_probabilities"],
                info["training_visible"],
                query_mask=query_mask,
            )
            terms.append(
                LossTerm(
                    term.numerator,
                    term.denominator,
                    term.unit,
                    f"dsa_indexer_{index}",
                    self.indexer_weight / len(records),
                )
            )
        return LossBundle(tuple(terms))


def qsa_indexer_distillation(records, teacher_probabilities, *, query_mask=None):
    """Sum teacher token probabilities inside visible microblocks before normalizing
    over complete blocks."""
    target = teacher_probabilities.detach().float()
    if target.ndim == 4:
        target = target.sum(1)
    if target.ndim != 3 or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("QSA teacher attention must be finite nonnegative [B,Q,K] or [B,H,Q,K]")
    if query_mask is not None and query_mask.shape != target.shape[:2]:
        raise ValueError("Invalid QSA query mask")
    total, count, visited = None, 0, set()
    for row, query, blocks, scores in records:
        if (row, query) in visited:
            raise ValueError("QSA records contain duplicate query entries")
        visited.add((row, query))
        if (
            not 0 <= row < target.shape[0]
            or not 0 <= query < target.shape[1]
            or blocks.ndim != 2
            or blocks.dtype != torch.long
            or scores.shape != (len(blocks),)
        ):
            raise ValueError("QSA records do not align with teacher attention")
        if (
            not len(blocks)
            or (blocks < 0).any()
            or (blocks >= target.shape[2]).any()
            or len(blocks.flatten().unique()) != blocks.numel()
        ):
            raise ValueError("QSA microblocks must contain distinct valid token positions")

        if total is None:
            total = scores.sum() * 0
        mass = target[row, query].to(scores.device)[blocks].sum(-1)
        if (query_mask is not None and not bool(query_mask[row, query])) or not bool(
            mass.sum() > 0
        ):
            continue
        mass = mass / mass.sum()
        total = (
            total
            + torch.where(
                mass > 0, mass * (mass.clamp_min(1e-20).log() - scores.float().log_softmax(-1)), 0
            ).sum()
        )
        count += 1
    if total is None:
        total = teacher_probabilities.new_zeros((), dtype=torch.float32, requires_grad=True)
    return LossTerm(
        total,
        torch.tensor(count, dtype=torch.int64, device=total.device),
        "query",
        "qsa_indexer_kl",
    )


class QSAIndexerObjective(torch.nn.Module):
    """Compute token CE and explicitly supplied teacher microblock KL in one forward."""

    def __init__(self, layers, *, base_weight=1.0, indexer_weight=0.1):
        super().__init__()
        import math

        self.layers = tuple(layers)
        if (
            not self.layers
            or any(type(x) is not int or x < 0 for x in self.layers)
            or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("QSA objective needs distinct explicit layer indices")
        if (
            not all(math.isfinite(x) and x >= 0 for x in (base_weight, indexer_weight))
            or base_weight + indexer_weight <= 0
        ):
            raise ValueError(
                "QSA objective weights must be finite, nonnegative and nonzero together"
            )
        self.base_weight, self.indexer_weight = base_weight, indexer_weight

    def config_dict(self):
        return dict(
            type="qsa_microblock_ce_kd",
            layers=self.layers,
            base_weight=self.base_weight,
            indexer_weight=self.indexer_weight,
        )

    def forward(self, model, batch):
        from .supervised import model_inputs, token_targets

        inputs = model_inputs(batch)
        output = model(**inputs, use_cache=False)
        records = (output.auxiliary or {}).get("qsa_indexer")
        teachers = batch.get("qsa_teacher_attention")
        actual_layers = (output.auxiliary or {}).get("qsa_layer_indices", ())
        if (
            not isinstance(records, tuple)
            or not isinstance(teachers, dict)
            or set(teachers) != set(self.layers)
            or max(self.layers) >= len(records)
            or not set(self.layers).issubset(actual_layers)
        ):
            raise ValueError("QSA layer records and explicit teacher targets must align")
        prediction, labels, mask = token_targets(batch, output.logits)
        values = F.cross_entropy(
            prediction.flatten(0, 1), labels.masked_fill(~mask, 0).flatten(), reduction="none"
        ).reshape_as(labels)
        terms = [
            LossTerm(
                values.masked_select(mask).sum(),
                mask.sum(dtype=torch.int64),
                "token",
                "ce",
                self.base_weight,
            )
        ]
        query_mask = batch.get("indexer_query_mask", inputs.get("attention_mask"))
        for index in self.layers:
            value = qsa_indexer_distillation(records[index], teachers[index], query_mask=query_mask)
            terms.append(
                LossTerm(
                    value.numerator,
                    value.denominator,
                    value.unit,
                    f"qsa_indexer_{index}",
                    self.indexer_weight / len(self.layers),
                )
            )
        return LossBundle(tuple(terms))
