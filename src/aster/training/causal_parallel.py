"""Dense-decoder tensor-parallel training with shared trainer ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from aster.core import LossTerm, TokenOutput
from aster.models.config import LlamaConfig, Qwen2Config, Qwen3Config
from aster.models.decoder import CausalLM, DecoderLayer
from aster.methods.supervised import model_inputs, token_targets
from .parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    _Copy,
    _Reduce,
    _Gather,
    vocab_parallel_cross_entropy,
)


def _tensor_signature(value):

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return (str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest())


def _reject_attention_backend(source):

    if any(getattr(module, "attention_backend", None) is not None for module in source.modules()):
        raise ValueError(
            "Explicit attention_backend is not yet certified for native TP/PP training providers"
        )


def _symmetric(context, error, *, signatures=None, same_world=False):
    records = context.world.gather_objects((error, signatures))
    failures = [(rank, item[0]) for rank, item in enumerate(records) if item[0]]
    if failures:
        raise ValueError(f"Native TP preflight failed: {failures}")
    ranks = context.world.ranks if same_world else context.tp_pp.ranks
    if any(records[rank][1] != records[ranks[0]][1] for rank in ranks):
        mismatch = "Native TP replicas require identical declared inputs/state"
    else:
        mismatch = None
    errors = context.world.gather_objects(mismatch)
    if any(errors):
        raise ValueError(next(item for item in errors if item))


def _column(source, group):
    target = ColumnParallelLinear(
        source.in_features, source.out_features, group, bias=source.bias is not None
    ).to(source.weight)
    with torch.no_grad():
        target.weight.copy_(source.weight.chunk(group.size, 0)[group.rank])
        if source.bias is not None:
            target.bias.copy_(source.bias.chunk(group.size, 0)[group.rank])
    return target


def _row(source, group):
    target = RowParallelLinear(
        source.in_features, source.out_features, group, bias=source.bias is not None
    ).to(source.weight)
    with torch.no_grad():
        target.weight.copy_(source.weight.chunk(group.size, 1)[group.rank])
        if source.bias is not None:
            target.bias.copy_(source.bias)
    return target


class _ReplicatedKVProjection(nn.Module):
    def __init__(self, source, *, query_heads, kv_heads, head_dim, group):
        super().__init__()
        self.group, self.head_dim = group, head_dim
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        local_queries = query_heads // group.size
        self.head_indices = tuple(
            (group.rank * local_queries + i) // (query_heads // kv_heads)
            for i in range(local_queries)
        )
        for parameter in self.parameters():
            parameter._aster_extra_gradient_group = group

    def forward(self, hidden):
        indices = torch.tensor(self.head_indices, device=hidden.device)
        weight = (
            self.weight.reshape(-1, self.head_dim, self.weight.shape[-1])
            .index_select(0, indices)
            .flatten(0, 1)
        )
        bias = (
            self.bias.reshape(-1, self.head_dim).index_select(0, indices).flatten()
            if self.bias is not None
            else None
        )
        return F.linear(_Copy.apply(hidden, self.group), weight, bias)


class VocabParallelEmbedding(nn.Module):
    """Shard vocabulary rows; padded rows are removed from standard exported weights."""

    def __init__(self, source, group):
        super().__init__()
        self.group = group
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
        self.width = math.ceil(self.num_embeddings / group.size)
        self.start, self.end = (
            group.rank * self.width,
            min((group.rank + 1) * self.width, self.num_embeddings),
        )
        self.weight = nn.Parameter(source.weight.new_zeros(self.width, self.embedding_dim))
        with torch.no_grad():
            if self.end > self.start:
                self.weight[: self.end - self.start].copy_(source.weight[self.start : self.end])
        self.weight._aster_tp_sharded = True
        self.weight._aster_tp_dimension = 0
        self.weight._aster_tp_global_shape = (self.num_embeddings, self.embedding_dim)

    def forward(self, input_ids):
        own = (input_ids >= self.start) & (input_ids < self.end)
        local = (input_ids - self.start).clamp(0, self.width - 1)
        value = F.embedding(local, self.weight) * own.unsqueeze(-1)
        return _Reduce.apply(value, self.group)


class _VocabHead(nn.Module):
    def __init__(self, source, group, shared=None):
        super().__init__()
        self.group, self.vocab_size = group, source.out_features
        self.width = math.ceil(self.vocab_size / group.size)
        self.weight = (
            shared
            if shared is not None
            else nn.Parameter(source.weight.new_zeros(self.width, source.in_features))
        )
        if shared is None:
            start, end = (
                group.rank * self.width,
                min((group.rank + 1) * self.width, self.vocab_size),
            )
            with torch.no_grad():
                if end > start:
                    self.weight[: end - start].copy_(source.weight[start:end])
        self.weight._aster_tp_sharded = True
        self.weight._aster_tp_dimension = 0
        self.weight._aster_tp_global_shape = tuple(source.weight.shape)

    def forward(self, hidden):
        output = F.linear(_Copy.apply(hidden, self.group), self.weight)

        valid = (
            torch.arange(self.width, device=hidden.device) + self.group.rank * self.width
            < self.vocab_size
        )
        return output.masked_fill(~valid, -torch.inf)


@dataclass
class ShardedTokenOutput:
    logits: torch.Tensor
    vocab_start: int
    vocab_size: int
    hidden_states: tuple | None = None


class TensorParallelCausalLM(CausalLM):
    """A training layout exported to a standard dense model before deployment."""

    def __init__(self, source, context):
        nn.Module.__init__(self)
        _reject_attention_backend(source)
        source = deepcopy(source)
        self.config, self.model_key, self.context = source.config, source.model_key, context
        self.model, self.lm_head = source.model, source.lm_head
        self._aster_shared_runtime_handles = (context, context.tp, context.dp)
        group, config = context.tp, self.config
        for layer in self.model.layers:
            attention = layer.self_attn
            attention.q_proj = _column(attention.q_proj, group)
            if config.num_key_value_heads % group.size == 0:
                attention.k_proj = _column(attention.k_proj, group)
                attention.v_proj = _column(attention.v_proj, group)
                attention.num_kv_heads //= group.size
            else:
                for name in ("k_proj", "v_proj"):
                    setattr(
                        attention,
                        name,
                        _ReplicatedKVProjection(
                            getattr(attention, name),
                            query_heads=config.num_attention_heads,
                            kv_heads=config.num_key_value_heads,
                            head_dim=config.attention_head_dim,
                            group=group,
                        ),
                    )
                attention.num_kv_heads = config.num_attention_heads // group.size
            attention.num_heads //= group.size
            attention.o_proj = _row(attention.o_proj, group)

            for normalization in (attention.q_norm, attention.k_norm):
                for parameter in normalization.parameters():
                    parameter._aster_extra_gradient_group = group
            layer.mlp.gate_proj = _column(layer.mlp.gate_proj, group)
            layer.mlp.up_proj = _column(layer.mlp.up_proj, group)
            layer.mlp.down_proj = _row(layer.mlp.down_proj, group)
        embedding = VocabParallelEmbedding(self.model.embed_tokens, group)
        self.lm_head = _VocabHead(
            self.lm_head, group, embedding.weight if config.tie_word_embeddings else None
        )
        self.model.embed_tokens = embedding

    def forward_sharded(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if state is not None or use_cache:
            raise ValueError(
                "Training TP provider does not expose inference cache; export a deployment model"
            )
        output = super().forward(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        return ShardedTokenOutput(
            output.logits,
            self.context.tp.rank * math.ceil(self.config.vocab_size / self.context.tp.size),
            self.config.vocab_size,
            output.hidden_states,
        )

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        output = self.forward_sharded(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        full = _Gather.apply(output.logits, self.context.tp, -1)[..., : self.config.vocab_size]
        return TokenOutput(full, hidden_states=output.hidden_states)

    def save_pretrained(self, *args, **kwargs):
        raise RuntimeError(
            "Do not save local TP shards as a dense model; use collective Trainer.export_state_dict"
        )


def parallelize_causal_lm(model, context, *, pipeline_schedule="1f1b"):
    """Collectively validate, then construct independent shards without mutating the input model."""
    error, signature = None, None
    try:
        config = model.config
        _reject_attention_backend(model)
        if type(model) is not CausalLM or type(config) not in {
            LlamaConfig,
            Qwen2Config,
            Qwen3Config,
        }:
            raise TypeError(
                "Native training TP supports exact dense Llama/Qwen2/Qwen3 providers only"
            )
        if any(
            getattr(context.config, key) != 1
            for key in (
                "context_parallel",
                "gtp_remat",
                "expert_parallel",
                "expert_tensor_parallel",
            )
        ):
            raise ValueError(
                "Native causal training provider supports TP x PP x DP only; CP/GTP need explicit providers"
            )
        if pipeline_schedule not in {"gpipe", "serial", "1f1b"}:
            raise ValueError("Unsupported native causal pipeline schedule")
        if config.num_hidden_layers < context.pp.size:
            raise ValueError("Every pipeline stage must own at least one decoder layer")
        if context.pp.size > 2 and config.tie_word_embeddings:
            raise ValueError("Cross-stage tied embeddings currently support exactly two PP stages")
        if (
            config.num_attention_heads % context.tp.size
            or config.intermediate_size % context.tp.size
        ):
            raise ValueError(
                "Query heads and MLP intermediate width must divide tensor parallel size"
            )
        if any(type(layer) is not DecoderLayer for layer in model.model.layers):
            raise TypeError("Unknown decoder layer")
        if getattr(model, "_aster_training_owned", False):
            raise ValueError("Parallelize before creating the Trainer optimizer owner")
        if any(parameter.is_meta for parameter in model.parameters()):
            raise ValueError(
                "This provider requires materialized initial weights, not meta tensors"
            )
        signature = (
            config.to_dict(),
            pipeline_schedule,
            {key: _tensor_signature(value) for key, value in model.state_dict().items()},
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error, signatures=signature, same_world=True)
    result = None
    error = None
    try:
        result = TensorParallelCausalLM(model, context)
        if context.pp.size > 1:
            from .causal_pipeline import CausalPipelineStage

            result = CausalPipelineStage(result, context, schedule=pipeline_schedule)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error)
    return result


class TensorParallelCrossEntropyObjective(nn.Module):
    """Normalize replicated TP targets only over the DP sample group."""

    def __init__(self, context):
        super().__init__()
        self.context = context

    def config_dict(self):
        return {
            "type": "native_tensor_parallel_causal_ce",
            "version": 1,
            "parallel": self.context.to_dict(),
            "normalization_domain": "dp",
            "kv_layout": "shard_if_heads_divisible_else_replicate",
            "vocabulary": "ceil_padding_excluded_from_softmax_and_dense_export",
        }

    def _preflight(self, model, batch):
        error, signature, arguments = None, None, None
        try:
            from .causal_pipeline import CausalPipelineStage

            if (
                not isinstance(model, (TensorParallelCausalLM, CausalPipelineStage))
                or model.context is not self.context
            ):
                raise TypeError("TP objective requires its matching native model/context")
            if not isinstance(batch, dict) or set(batch) - {
                "input_ids",
                "inputs_embeds",
                "attention_mask",
                "position_ids",
                "labels",
                "loss_mask",
                "model_inputs",
            }:
                raise ValueError("Unknown TP language batch fields")
            arguments = model_inputs(batch)
            if "model_inputs" in batch and any(
                key in batch
                for key in ("input_ids", "inputs_embeds", "attention_mask", "position_ids")
            ):
                raise ValueError("Do not mix nested and top-level model inputs")
            if set(arguments) - {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}:
                raise ValueError("Unknown TP model input fields")
            ids, embeds = arguments.get("input_ids"), arguments.get("inputs_embeds")
            if (ids is None) == (embeds is None):
                raise ValueError("Provide input_ids or inputs_embeds, not both")
            if ids is not None:
                if (
                    ids.dtype != torch.long
                    or ids.ndim != 2
                    or ((ids < 0) | (ids >= model.config.vocab_size)).any()
                ):
                    raise ValueError("input_ids must be int64 [B,T] within vocabulary")
                shape, device = ids.shape, ids.device
            else:
                if (
                    embeds.ndim != 3
                    or not embeds.is_floating_point()
                    or embeds.shape[-1] != model.config.hidden_size
                    or not torch.isfinite(embeds).all()
                ):
                    raise ValueError("Invalid floating inputs_embeds")
                shape, device = embeds.shape[:2], embeds.device
            parameter = next(model.parameters())
            if device != getattr(parameter, "_aster_compute_device", parameter.device):
                raise ValueError("TP input device must match the declared model compute device")
            if shape[0] < 1:
                raise ValueError(
                    "Empty sample batches are not supported; use a fully masked padding sample for zero valid count"
                )
            if not 2 <= shape[1] <= model.config.max_position_embeddings:
                raise ValueError("Causal training needs 2..max_position_embeddings tokens")
            for name in ("attention_mask", "position_ids"):
                value = arguments.get(name)
                if value is not None:
                    if value.shape != shape or value.device != device:
                        raise ValueError(f"{name} must align with tokens/device")
                    if name == "position_ids" and (value.dtype != torch.long or (value < 0).any()):
                        raise ValueError("position_ids must be nonnegative int64")
                    if name == "attention_mask" and not ((value == 0) | (value == 1)).all():
                        raise ValueError("attention_mask must be binary")
            labels = batch.get("labels", ids)
            if (
                labels is None
                or labels.shape != shape
                or labels.dtype != torch.long
                or labels.device != device
            ):
                raise ValueError("TP labels must be aligned int64 on the input device")
            if ((labels != -100) & ((labels < 0) | (labels >= model.config.vocab_size))).any():
                raise ValueError("Label outside vocabulary")
            if "loss_mask" in batch:
                mask = batch["loss_mask"]
                if mask.shape != shape or mask.device != device or mask.dtype != torch.bool:
                    raise ValueError("loss_mask must be aligned boolean tensor")
            tensors = {**arguments, "labels": labels}
            if "loss_mask" in batch:
                tensors["loss_mask"] = batch["loss_mask"]
            signature = {key: _tensor_signature(value) for key, value in tensors.items()}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _symmetric(self.context, error, signatures=signature)
        modes = self.context.world.gather_objects("ids" if "input_ids" in arguments else "embeds")
        if len(set(modes)) != 1:
            raise ValueError(
                "All DP replicas must use the same embedding execution path before ZeRO collectives"
            )
        return arguments

    def forward(self, model, batch):
        arguments = self._preflight(model, batch)
        output = model.forward_sharded(**arguments)
        targets = {**batch, "labels": batch.get("labels", arguments.get("input_ids"))}
        logits, labels, mask = token_targets(targets, output.logits)
        labels = labels.masked_fill(~mask, -100)
        values = vocab_parallel_cross_entropy(logits, labels, self.context.tp)
        return LossTerm(values.masked_select(mask).sum(), mask.sum().to(torch.int64), "token", "ce")
