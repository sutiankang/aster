"""Sparse token-to-expert dispatch and explicit router auxiliary losses."""

import torch
from torch import nn
import torch.nn.functional as F
from .parameter_codec import register_parameter_codec


class PackedExperts(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        *,
        std=0.02,
        swiglu_limit=None,
        activation="silu",
    ):
        super().__init__()
        self.num_experts = num_experts
        if swiglu_limit is not None and swiglu_limit <= 0:
            raise ValueError("SwiGLU limit must be positive")
        self.swiglu_limit = swiglu_limit
        if (
            activation not in {"silu", "gelu", "gelu_pytorch_tanh"}
            or swiglu_limit is not None
            and activation != "silu"
        ):
            raise ValueError("Unsupported routed-expert activation/clipping combination")
        self.activation = activation
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.gate_up_proj, std=std)
        nn.init.normal_(self.down_proj, std=std)

    def forward(self, hidden, indices, weights):
        if hidden.ndim != 2 or indices.shape != weights.shape or indices.shape[0] != len(hidden):
            raise ValueError("Expert routing shapes mismatch")
        result = torch.zeros_like(hidden)
        for expert in range(self.num_experts):
            tokens, slots = torch.where(indices == expert)
            if not tokens.numel():
                continue
            gate, up = F.linear(hidden[tokens], self.gate_up_proj[expert]).chunk(2, -1)
            if self.swiglu_limit is not None:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            activated = (
                F.silu(gate)
                if self.activation == "silu"
                else F.gelu(
                    gate, approximate="tanh" if self.activation == "gelu_pytorch_tanh" else "none"
                )
            )
            value = F.linear(activated * up, self.down_proj[expert])

            result.index_add_(0, tokens, (value * weights[tokens, slots, None]).to(result.dtype))
        return result


class RouterProjection(nn.Module):
    def __init__(self, hidden_size, num_experts, *, sigmoid, std):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.weight, std=std)
        self.sigmoid = sigmoid

    def forward(self, hidden):
        return (
            F.linear(hidden.float(), self.weight.float())
            if self.sigmoid
            else F.linear(hidden, self.weight)
        )


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_experts,
        top_k,
        *,
        groups=1,
        topk_groups=1,
        sigmoid=False,
        normalize=True,
        scale=1.0,
        std=0.02,
    ):
        super().__init__()
        self.projection = RouterProjection(hidden_size, num_experts, sigmoid=sigmoid, std=std)

        register_parameter_codec(self, {"projection.weight": "weight"})
        self.top_k, self.groups, self.topk_groups = top_k, groups, topk_groups
        self.sigmoid, self.normalize, self.scale = sigmoid, normalize, scale
        if sigmoid:
            self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, hidden):
        logits = self.projection(hidden)
        probabilities = logits.sigmoid() if self.sigmoid else logits.float().softmax(-1)
        choice = probabilities
        if self.sigmoid:
            choice = probabilities + self.e_score_correction_bias
            group_score = choice.view(len(hidden), self.groups, -1).topk(2, dim=-1).values.sum(-1)
            selected = group_score.topk(self.topk_groups, sorted=False).indices
            group_mask = torch.zeros_like(group_score, dtype=torch.bool).scatter_(1, selected, True)
            choice = choice.masked_fill(
                ~group_mask[..., None]
                .expand(-1, -1, choice.shape[-1] // self.groups)
                .reshape_as(choice),
                -torch.inf,
            )
        indices = choice.topk(self.top_k, sorted=not self.sigmoid).indices
        weights = probabilities.gather(-1, indices)
        if self.normalize:
            weights = weights / (weights.sum(-1, keepdim=True) + (1e-20 if self.sigmoid else 0))
        return logits, weights * self.scale, indices

    @property
    def weight(self):

        return self.projection.weight


def router_balance_loss(logits, top_k, valid=None):
    """Mixtral auxiliary: E * sum(selection_frequency * mean_router_probability),
    summing the top-k selection dimension."""
    probabilities = logits.float().softmax(-1)
    selection = F.one_hot(probabilities.topk(top_k, -1).indices, logits.shape[-1]).float()
    valid = (
        torch.ones(len(logits), dtype=torch.bool, device=logits.device)
        if valid is None
        else valid.reshape(-1).bool()
    )
    if len(valid) != len(logits):
        raise ValueError("Router valid mask does not align with tokens")
    count = valid.sum().clamp_min(1)
    fraction = (selection * valid[:, None, None]).sum(0) / count
    mean = (probabilities * valid[:, None]).sum(0) / count
    return logits.shape[-1] * (fraction * mean[None]).sum()
