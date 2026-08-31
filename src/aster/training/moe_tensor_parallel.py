"""Mixtral expert, expert-tensor, and expert-data parallel layouts."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from aster.core import TokenOutput, LossTerm, LossBundle
from aster.models.decoder import CausalLM
from aster.models.moe import MixtralForCausalLM, MixtralLayer
from aster.models.config import MixtralConfig
from aster.methods.supervised import model_inputs, token_targets
from .causal_parallel import (
    _column,
    _row,
    _ReplicatedKVProjection,
    VocabParallelEmbedding,
    _VocabHead,
    _tensor_signature,
    _symmetric,
)
from .parallel import _Gather, _Scatter, vocab_parallel_cross_entropy
from .experts import _AllToAll
from .moe_parallel import ExpertParallelCausalLM, ExpertParallelCrossEntropyObjective


def _gather_variable(value, lengths, group):

    if group.size == 1:
        return value
    width = max(lengths)
    if not width:
        return value.new_empty((0, *value.shape[1:]))
    padded = value.new_zeros((width, *value.shape[1:]))
    padded[: len(value)] = value
    outputs = [torch.empty_like(padded) for _ in group.ranks]
    dist.all_gather(outputs, padded.contiguous(), group=group.handle)
    return torch.cat([piece[:length] for piece, length in zip(outputs, lengths)])


def _scatter_sum_variable(value, lengths, group):
    if group.size == 1:
        return value
    width = max(lengths)
    if not width:
        return value.new_empty((0, *value.shape[1:]))
    chunks = value.split(lengths, dim=0)
    padded = value.new_zeros((group.size, width, *value.shape[1:]))
    for index, chunk in enumerate(chunks):
        padded[index, : len(chunk)] = chunk
    result = value.new_empty((width, *value.shape[1:]))

    dist.reduce_scatter_tensor(result, padded.flatten(0, 1).contiguous(), group=group.handle)
    return result[: lengths[group.rank]].contiguous()


class _GatherVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, lengths, group):
        ctx.lengths, ctx.group = lengths, group
        return _gather_variable(value, lengths, group)

    @staticmethod
    def backward(ctx, gradient):
        return _scatter_sum_variable(gradient, ctx.lengths, ctx.group), None, None


class _ScatterSumVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, lengths, group):
        ctx.lengths, ctx.group = lengths, group
        return _scatter_sum_variable(value, lengths, group)

    @staticmethod
    def backward(ctx, gradient):
        return _gather_variable(gradient, ctx.lengths, ctx.group), None, None


class _TensorPackedExperts(nn.Module):
    def __init__(self, source, context):
        super().__init__()
        self.num_experts = source.num_experts // context.ep.size
        self.first = context.ep.rank * self.num_experts
        gate_up = source.gate_up_proj.detach().narrow(0, self.first, self.num_experts)
        gate_up = torch.cat(
            [part.chunk(context.etp.size, 1)[context.etp.rank] for part in gate_up.chunk(2, 1)], 1
        )
        down = (
            source.down_proj.detach()
            .narrow(0, self.first, self.num_experts)
            .chunk(context.etp.size, 2)[context.etp.rank]
        )
        for name, value, axis, stripes in (
            ("gate_up_proj", gate_up, 1, 2),
            ("down_proj", down, 2, 1),
        ):
            parameter = nn.Parameter(value.contiguous().clone())
            parameter._aster_gradient_group = context.edp
            parameter._aster_ep_dimension, parameter._aster_ep_group = 0, context.ep
            parameter._aster_tp_dimension, parameter._aster_tp_group = axis, context.etp
            parameter._aster_tp_stripes, parameter._aster_tp_sharded = stripes, True
            self.register_parameter(name, parameter)

    def forward(self, tokens, expert_ids):
        output = torch.zeros_like(tokens)
        for local in range(self.num_experts):
            positions = (expert_ids == self.first + local).nonzero(as_tuple=True)[0]
            gate, up = F.linear(tokens[positions], self.gate_up_proj[local]).chunk(2, -1)
            partial = F.linear(F.silu(gate) * up, self.down_proj[local])

            output = output.index_copy(0, positions, partial.to(output.dtype))
        return output


class _ExpertTensorBlock(nn.Module):
    def __init__(self, source, context):
        super().__init__()
        self.context, self.group = context, context.ep
        self.jitter, self.top_k = source.jitter, source.gate.top_k
        self.num_experts, self.experts_per_rank = (
            source.experts.num_experts,
            source.experts.num_experts // context.ep.size,
        )
        self.gate = nn.Linear(source.gate.weight.shape[1], self.num_experts, bias=False).to(
            source.gate.weight
        )
        with torch.no_grad():
            self.gate.weight.copy_(source.gate.weight)

        if context.tp.size > 1:
            self.gate.weight._aster_extra_gradient_group = context.tp
        self.experts = _TensorPackedExperts(source.experts, context)
        self.last_send_counts = self.last_receive_counts = self.last_etp_lengths = None

    def forward(self, hidden):
        shape, tp = hidden.shape, self.context.tp
        if self.training and self.jitter:
            hidden = hidden * torch.empty_like(hidden).uniform_(1 - self.jitter, 1 + self.jitter)
        full = hidden.reshape(-1, shape[-1])
        count = len(full)
        padding = (-count) % tp.size
        if padding:
            full = F.pad(full, (0, 0, 0, padding))

        tokens = _Scatter.apply(full, tp, 0)
        logits = self.gate(tokens)
        probabilities = logits.float().softmax(-1)
        indices = probabilities.topk(self.top_k, dim=-1).indices
        weights = probabilities.gather(-1, indices)
        weights = weights / weights.sum(-1, keepdim=True)
        destinations = indices.flatten() // self.experts_per_rank
        order = destinations.argsort(stable=True)
        counts = torch.bincount(destinations, minlength=self.group.size).to(torch.int64)
        receive = torch.empty_like(counts)
        if self.group.size > 1:
            dist.all_to_all_single(receive, counts, group=self.group.handle)
        else:
            receive.copy_(counts)
        send, recv = counts.tolist(), receive.tolist()
        self.last_send_counts, self.last_receive_counts = tuple(send), tuple(recv)
        repeated = tokens[:, None].expand(-1, self.top_k, -1).reshape(-1, shape[-1])[order]
        dispatched = _AllToAll.apply(repeated, send, recv, self.group)
        selected = indices.flatten()[order]
        if self.group.size > 1:
            expert_ids = selected.new_empty(sum(recv))
            dist.all_to_all_single(
                expert_ids,
                selected.contiguous(),
                output_split_sizes=recv,
                input_split_sizes=send,
                group=self.group.handle,
            )
        else:
            expert_ids = selected
        lengths = tuple(self.context.etp.gather_objects(len(dispatched)))
        self.last_etp_lengths = lengths

        complete = _GatherVariable.apply(dispatched, lengths, self.context.etp)
        all_ids = _gather_variable(expert_ids, lengths, self.context.etp)
        partial = self.experts(complete, all_ids)
        local = _ScatterSumVariable.apply(partial, lengths, self.context.etp)
        returned = _AllToAll.apply(local, recv, send, self.group)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        values = returned[inverse].reshape(len(tokens), self.top_k, shape[-1])
        combined = (values * weights.to(values.dtype).unsqueeze(-1)).sum(1)
        output = _Gather.apply(combined, tp, 0)[:count].reshape(shape)

        routing = {
            "logits": _Gather.apply(logits, tp, 0)[:count],
            "weights": _Gather.apply(weights, tp, 0)[:count],
            "indices": _Gather.apply(indices, tp, 0)[:count],
        }
        return output, routing


@dataclass
class _ShardedMoEOutput:
    logits: torch.Tensor
    auxiliary: dict
    hidden_states: tuple | None


class ExpertTensorParallelCausalLM(ExpertParallelCausalLM):
    def __init__(self, source, context):
        nn.Module.__init__(self)
        copied = deepcopy(source)
        self.config, self.model_key, self.context = copied.config, copied.model_key, context
        if context.tp.size > 1:
            self._aster_replicated_rng_group = context.tp
        self.model, self.lm_head = copied.model, copied.lm_head
        for layer in self.model.layers:
            attention = layer.self_attn
            if context.tp.size > 1:
                attention.q_proj = _column(attention.q_proj, context.tp)
                if self.config.num_key_value_heads % context.tp.size == 0:
                    attention.k_proj = _column(attention.k_proj, context.tp)
                    attention.v_proj = _column(attention.v_proj, context.tp)
                    attention.num_kv_heads //= context.tp.size
                else:
                    for name in ("k_proj", "v_proj"):
                        setattr(
                            attention,
                            name,
                            _ReplicatedKVProjection(
                                getattr(attention, name),
                                query_heads=self.config.num_attention_heads,
                                kv_heads=self.config.num_key_value_heads,
                                head_dim=self.config.attention_head_dim,
                                group=context.tp,
                            ),
                        )
                    attention.num_kv_heads = self.config.num_attention_heads // context.tp.size
                attention.num_heads //= context.tp.size
                attention.o_proj = _row(attention.o_proj, context.tp)
                for normalization in (attention.q_norm, attention.k_norm):
                    for parameter in normalization.parameters():
                        parameter._aster_extra_gradient_group = context.tp
            layer.mlp = _ExpertTensorBlock(layer.mlp, context)
        embedding = VocabParallelEmbedding(self.model.embed_tokens, context.tp)
        self.lm_head = _VocabHead(
            self.lm_head, context.tp, embedding.weight if self.config.tie_word_embeddings else None
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
                "Expert training provider has no inference KV cache; export the dense model"
            )
        result = CausalLM.forward(
            self,
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        return _ShardedMoEOutput(result.logits, result.auxiliary, result.hidden_states)

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
        result = self.forward_sharded(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = _Gather.apply(result.logits, self.context.tp, -1)[..., : self.config.vocab_size]
        return TokenOutput(logits, hidden_states=result.hidden_states, auxiliary=result.auxiliary)


def parallelize_mixtral_tensor(model, context):
    error, signature = None, None
    try:
        if type(model) is not MixtralForCausalLM or type(model.config) is not MixtralConfig:
            raise TypeError("Expert tensor training currently supports exact native Mixtral only")
        if any(
            getattr(context.config, key) != 1
            for key in ("pipeline_parallel", "context_parallel", "gtp_remat")
        ):
            raise ValueError("Expert tensor training does not support PP/CP/GTP folding")
        if context.tp.size not in {1, context.etp.size}:
            raise ValueError("Attention TP must equal ETP or one")
        if (
            model.config.intermediate_size % context.etp.size
            or model.config.num_local_experts % context.ep.size
        ):
            raise ValueError("Expert count must divide EP and intermediate_size must divide ETP")
        if model.config.num_attention_heads % context.tp.size:
            raise ValueError("Query heads must divide attention TP")
        if context.tp.size > 1 and model.config.attention_dropout:
            raise ValueError(
                "Attention TP dropout needs independent head RNG streams; this provider currently requires attention_dropout=0"
            )
        if any(type(layer) is not MixtralLayer for layer in model.model.layers):
            raise TypeError("Unknown Mixtral layer")
        if getattr(model, "_aster_training_owned", False) or any(
            parameter.is_meta for parameter in model.parameters()
        ):
            raise ValueError("Parallelize a materialized model before assigning Trainer ownership")
        signature = (
            model.config.to_dict(),
            {name: _tensor_signature(value) for name, value in model.state_dict().items()},
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error, signatures=signature, same_world=True)
    result, error = None, None
    try:
        result = ExpertTensorParallelCausalLM(model, context)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error)
    return result


class ExpertTensorParallelCrossEntropyObjective(ExpertParallelCrossEntropyObjective):
    def config_dict(self):
        return {
            **super().config_dict(),
            "type": "native_mixtral_ep_etp_ce",
            "version": 1,
            "attention_token_layout": "scatter_before_router_gather_after_return",
            "vocabulary": "distributed_softmax_excludes_ceil_padding",
            "expert_fc1_stripes": 2,
        }

    def preflight_microbatches(self, model, batches):
        super().preflight_microbatches(model, batches)
        error = (
            None
            if isinstance(model, ExpertTensorParallelCausalLM)
            else "ETP objective requires its matching full-model provider"
        )
        _symmetric(self.context, error)
        error, signature = None, None
        try:
            signature = [
                {
                    key: None if value is None else _tensor_signature(value)
                    for key, value in {
                        **model_inputs(batch),
                        "labels": batch.get("labels", model_inputs(batch).get("input_ids")),
                        **({"loss_mask": batch["loss_mask"]} if "loss_mask" in batch else {}),
                    }.items()
                }
                for batch in batches
            ]
            if self.context.tp.size > 1 and model.config.router_jitter_noise:
                signature.append({"cpu_rng": _tensor_signature(torch.get_rng_state())})
                device = next(model.parameters()).device
                device = getattr(next(model.parameters()), "_aster_compute_device", device)
                if device.type == "cuda":
                    signature.append(
                        {"cuda_rng": _tensor_signature(torch.cuda.get_rng_state(device))}
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _symmetric(self.context, error, signatures=signature)
        return batches

    def forward(self, model, batch):
        inputs = model_inputs(batch)
        output = model.forward_sharded(**inputs, use_cache=False)
        logits, labels, mask = token_targets(
            {**batch, "labels": batch.get("labels", inputs.get("input_ids"))}, output.logits
        )
        values = vocab_parallel_cross_entropy(
            logits, labels.masked_fill(~mask, -100), self.context.tp
        )
        ce = LossTerm(values.masked_select(mask).sum(), mask.sum(dtype=torch.int64), "token", "ce")
        if not self.coefficient:
            return ce
        batch_size, length = output.logits.shape[:2]
        valid = inputs.get(
            "attention_mask", torch.ones(batch_size, length, dtype=torch.bool, device=logits.device)
        ).bool()
        counts = valid.sum(-1)

        numerator = output.auxiliary["router"][0]["logits"].sum() * 0.0
        for routing in output.auxiliary["router"]:
            experts, top_k = routing["logits"].shape[-1], routing["indices"].shape[-1]
            probabilities = (
                routing["logits"].float().softmax(-1).reshape(batch_size, length, experts)
            )
            selected = (
                F.one_hot(routing["indices"], experts)
                .float()
                .reshape(batch_size, length, top_k, experts)
            )
            frequency = (selected * valid[:, :, None, None]).sum((1, 2))
            scores = (probabilities * valid[:, :, None]).sum(1)
            losses = (frequency * scores).sum(-1) * experts / (top_k * counts.clamp_min(1).square())
            numerator = numerator + losses.masked_select(counts > 0).sum()
        return LossBundle(
            (
                ce,
                LossTerm(
                    numerator,
                    (counts > 0).sum(dtype=torch.int64),
                    "sequence",
                    "router_seq_aux",
                    self.coefficient,
                ),
            )
        )
