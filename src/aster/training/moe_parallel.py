"""Native Mixtral expert/data-parallel training."""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from aster.core import LossTerm, LossBundle
from aster.models.config import MixtralConfig
from aster.models.decoder import CausalLM
from aster.models.moe import MixtralForCausalLM, MixtralLayer
from aster.methods.supervised import model_inputs, token_targets
from .causal_parallel import _tensor_signature, _symmetric
from .experts import _AllToAll


class _LocalPackedExperts(nn.Module):
    def __init__(self, source, context):
        super().__init__()
        self.num_experts = source.num_experts // context.ep.size
        self.first = context.ep.rank * self.num_experts
        for name in ("gate_up_proj", "down_proj"):
            value = getattr(source, name).detach().narrow(0, self.first, self.num_experts).clone()
            parameter = nn.Parameter(value)
            parameter._aster_gradient_group = context.edp
            parameter._aster_ep_dimension = 0
            parameter._aster_ep_group = context.ep
            self.register_parameter(name, parameter)

    def forward(self, hidden, expert_ids):
        result = torch.zeros_like(hidden)
        for local in range(self.num_experts):
            positions = (expert_ids == self.first + local).nonzero(as_tuple=True)[0]

            gate, up = F.linear(hidden[positions], self.gate_up_proj[local]).chunk(2, -1)
            values = F.linear(F.silu(gate) * up, self.down_proj[local])

            result = result.index_copy(0, positions, values.to(result.dtype))
        return result


class _MixtralParallelBlock(nn.Module):
    def __init__(self, source, context):
        super().__init__()
        self.group, self.jitter = context.ep, source.jitter
        self.num_experts = source.experts.num_experts
        self.top_k = source.gate.top_k
        self.experts_per_rank = self.num_experts // self.group.size

        self.gate = nn.Linear(source.gate.weight.shape[1], self.num_experts, bias=False).to(
            source.gate.weight
        )
        with torch.no_grad():
            self.gate.weight.copy_(source.gate.weight)
        self.experts = _LocalPackedExperts(source.experts, context)
        self.last_send_counts = self.last_receive_counts = None

    def forward(self, hidden):
        if self.training and self.jitter:
            hidden = hidden * torch.empty_like(hidden).uniform_(1 - self.jitter, 1 + self.jitter)
        shape = hidden.shape
        tokens = hidden.reshape(-1, shape[-1])
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
        local = self.experts(dispatched, expert_ids)
        returned = _AllToAll.apply(local, recv, send, self.group)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        values = returned[inverse].reshape(len(tokens), self.top_k, shape[-1])
        output = (values * weights.to(values.dtype).unsqueeze(-1)).sum(1).reshape(shape)
        return output, {"logits": logits, "weights": weights, "indices": indices}


class ExpertParallelCausalLM(CausalLM):
    def __init__(self, source, context):
        nn.Module.__init__(self)

        copied = deepcopy(source)
        self.config, self.model_key = copied.config, copied.model_key
        self.context, self.model, self.lm_head = context, copied.model, copied.lm_head
        for layer in self.model.layers:
            layer.mlp = _MixtralParallelBlock(layer.mlp, context)

    def save_pretrained(self, *args, **kwargs):
        raise RuntimeError(
            "Local EP storage is not a dense artifact; use collective Trainer.export_state_dict"
        )


def parallelize_mixtral(model, context):
    """Collectively validate before partitioning expert parameters and retaining dense/router replicas."""
    error, signature = None, None
    try:
        if type(model) is not MixtralForCausalLM or type(model.config) is not MixtralConfig:
            raise TypeError("Native MoE provider currently supports exact Mixtral only")
        if any(
            getattr(context.config, key) != 1
            for key in (
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
                "gtp_remat",
                "expert_tensor_parallel",
            )
        ):
            raise ValueError(
                "Mixtral native provider supports EP x EDP only; ETP/TP/PP/CP/GTP are not implemented"
            )
        if model.config.num_local_experts % context.ep.size:
            raise ValueError("num_local_experts must divide EP")
        if any(type(layer) is not MixtralLayer for layer in model.model.layers):
            raise TypeError("Unknown Mixtral layer")
        if getattr(model, "_aster_training_owned", False):
            raise ValueError("Parallelize before assigning Trainer ownership")
        if any(parameter.is_meta for parameter in model.parameters()):
            raise ValueError("Mixtral provider requires materialized initial parameters")
        signature = (
            model.config.to_dict(),
            {key: _tensor_signature(value) for key, value in model.state_dict().items()},
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error, signatures=signature, same_world=True)
    error, result = None, None
    try:
        result = ExpertParallelCausalLM(model, context)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _symmetric(context, error)
    return result


class ExpertParallelCrossEntropyObjective(nn.Module):
    """Normalize CE by valid target tokens and optional sequence auxiliaries separately."""

    def __init__(self, context, *, router_aux_coefficient=0.0):
        super().__init__()
        if (
            isinstance(router_aux_coefficient, bool)
            or not math.isfinite(router_aux_coefficient)
            or router_aux_coefficient < 0
        ):
            raise ValueError("router_aux_coefficient must be finite and nonnegative")
        self.context, self.coefficient = context, float(router_aux_coefficient)

    def config_dict(self):
        return {
            "type": "native_mixtral_ep_ce",
            "version": 1,
            "parallel": self.context.to_dict(),
            "ce_denominator": "global_supervised_tokens",
            "router_aux_coefficient": self.coefficient,
            "router_aux_scope": "sequence",
            "router_aux_layers": "sum",
            "router_frequency": "selected_count/(K*T)",
        }

    def preflight_microbatches(self, model, batches):
        error, paths = None, []
        try:
            if not isinstance(model, ExpertParallelCausalLM) or model.context is not self.context:
                raise TypeError("MoE objective needs its matching native provider/context")
            for batch in batches:
                if not isinstance(batch, dict) or set(batch) - {
                    "input_ids",
                    "inputs_embeds",
                    "attention_mask",
                    "position_ids",
                    "labels",
                    "loss_mask",
                    "model_inputs",
                }:
                    raise ValueError("Unknown MoE batch fields")
                inputs = model_inputs(batch)
                if "model_inputs" in batch and any(
                    key in batch
                    for key in ("input_ids", "inputs_embeds", "attention_mask", "position_ids")
                ):
                    raise ValueError("Do not mix nested and top-level model inputs")
                if set(inputs) - {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}:
                    raise ValueError("Unknown MoE model inputs")
                ids, embeds = inputs.get("input_ids"), inputs.get("inputs_embeds")
                if (ids is None) == (embeds is None):
                    raise ValueError("Provide exactly one of input_ids/inputs_embeds")
                paths.append("ids" if ids is not None else "embeds")
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
                if shape[0] < 1 or shape[1] < 2 or shape[1] > model.config.max_position_embeddings:
                    raise ValueError(
                        "Mixtral requires B>=1 and valid sequence length; empty-loss ranks must supply an explicitly fully masked row"
                    )
                expected_device = next(model.parameters()).device

                expected_device = getattr(
                    next(model.parameters()), "_aster_compute_device", expected_device
                )
                if device != expected_device:
                    raise ValueError("Inputs must be on the declared compute device")
                labels = batch.get("labels", ids)
                if (
                    labels is None
                    or labels.dtype != torch.long
                    or labels.shape != shape
                    or labels.device != device
                    or (
                        (labels != -100) & ((labels < 0) | (labels >= model.config.vocab_size))
                    ).any()
                ):
                    raise ValueError("Labels must be aligned int64 vocabulary IDs or -100")
                for key, value in (
                    ("attention_mask", inputs.get("attention_mask")),
                    ("loss_mask", batch.get("loss_mask")),
                ):
                    if value is not None and (
                        value.shape != shape
                        or value.device != device
                        or not ((value == 0) | (value == 1)).all()
                    ):
                        raise ValueError(f"{key} must be an aligned binary mask")
                positions = inputs.get("position_ids")
                if positions is not None and (
                    positions.shape != shape
                    or positions.dtype != torch.long
                    or positions.device != device
                    or (positions < 0).any()
                ):
                    raise ValueError("Invalid position_ids")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        _symmetric(self.context, error, signatures=paths, same_world=True)
        return batches

    def forward(self, model, batch):
        inputs = model_inputs(batch)
        output = model(**inputs, use_cache=False)
        logits, targets, mask = token_targets(batch, output.logits)
        values = F.cross_entropy(
            logits.flatten(0, 1), targets.masked_fill(~mask, 0).flatten(), reduction="none"
        ).reshape_as(targets)
        ce = LossTerm(values.masked_select(mask).sum(), mask.sum(dtype=torch.int64), "token", "ce")
        if not self.coefficient:
            return ce
        batch_size, length = output.logits.shape[:2]
        valid = inputs.get(
            "attention_mask", torch.ones(batch_size, length, dtype=torch.bool, device=logits.device)
        ).bool()
        counts = valid.sum(-1)
        numerator = logits.sum() * 0.0
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
            losses = (
                (frequency * scores).sum(-1)
                * probabilities.shape[-1]
                / (routing["indices"].shape[-1] * counts.clamp_min(1).square())
            )
            numerator = numerator + losses.masked_select(counts > 0).sum()
        auxiliary = LossTerm(
            numerator,
            (counts > 0).sum(dtype=torch.int64),
            "sequence",
            "router_seq_aux",
            self.coefficient,
        )
        return LossBundle((ce, auxiliary))
