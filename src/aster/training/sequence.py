"""Differentiable sequence- and context-parallel communication."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from .parallel import Group


class _GatherReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dimension):
        ctx.group, ctx.dimension = group, dimension
        if group.size == 1:
            return value
        outputs = [torch.empty_like(value) for _ in group.ranks]
        dist.all_gather(outputs, value.contiguous(), group=group.handle)
        return torch.cat(outputs, dim=dimension)

    @staticmethod
    def backward(ctx, gradient):
        group, dimension = ctx.group, ctx.dimension
        if group.size == 1:
            return gradient, None, None
        packed = torch.cat(
            [chunk.contiguous().flatten() for chunk in gradient.chunk(group.size, dim=dimension)]
        )
        shape = list(gradient.shape)
        shape[dimension] //= group.size
        result = torch.empty(
            packed.numel() // group.size, device=gradient.device, dtype=gradient.dtype
        )
        dist.reduce_scatter_tensor(result, packed, group=group.handle)
        return result.reshape(shape), None, None


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dimension):
        ctx.group, ctx.dimension = group, dimension
        if value.shape[dimension] % group.size:
            raise ValueError("SP 序列必须整除 TP；padding 需附有效 mask")
        if group.size == 1:
            return value
        packed = torch.cat(
            [chunk.contiguous().flatten() for chunk in value.chunk(group.size, dim=dimension)]
        )
        shape = list(value.shape)
        shape[dimension] //= group.size
        result = torch.empty(packed.numel() // group.size, device=value.device, dtype=value.dtype)
        dist.reduce_scatter_tensor(result, packed, group=group.handle)
        return result.reshape(shape)

    @staticmethod
    def backward(ctx, gradient):
        group = ctx.group
        if group.size == 1:
            return gradient, None, None
        outputs = [torch.empty_like(gradient) for _ in group.ranks]
        dist.all_gather(outputs, gradient.contiguous(), group=group.handle)
        return torch.cat(outputs, dim=ctx.dimension), None, None


def gather_sequence(value, group: Group, *, dimension=-2):
    """All-gather forward with reduce-scatter backward; gradients from different ranks add."""
    return _GatherReduce.apply(value, group, dimension)


def reduce_scatter_sequence(value, group: Group, *, dimension=-2):
    return _ReduceScatter.apply(value, group, dimension)


class _SequenceBias(nn.Module):
    def __init__(self, width, group):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(width))
        self.bias._aster_extra_gradient_group = group

    def forward(self, value):
        return value + self.bias


class SequenceParallelMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, group: Group):
        super().__init__()
        if intermediate_size % group.size:
            raise ValueError("MLP intermediate_size 必须整除 TP")
        self.group = group
        self.up = nn.Linear(hidden_size, intermediate_size // group.size)
        self.down = nn.Linear(intermediate_size // group.size, hidden_size, bias=False)
        self.bias_layer = _SequenceBias(hidden_size, group)
        for parameter in self.up.parameters():
            parameter._aster_tp_sharded = True
            parameter._aster_tp_dimension = 0
        self.down.weight._aster_tp_sharded = True
        self.down.weight._aster_tp_dimension = 1

    @property
    def bias(self):
        return self.bias_layer.bias

    def forward(self, local_sequence):
        complete = gather_sequence(local_sequence, self.group)
        partial = self.down(F.gelu(self.up(complete)))
        return self.bias_layer(reduce_scatter_sequence(partial, self.group))


def context_parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group: Group,
    *,
    causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Attend from a local [B,H,L,D] sequence chunk to globally addressed boolean key masks."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError("CP attention 需要 [B,H,L,D] 且 K/V 同形")
    if query.shape[-2] != key.shape[-2] or query.shape[1] % key.shape[1]:
        raise ValueError("CP 分块长度/注意力头数不兼容")
    full_key, full_value = gather_sequence(key, group), gather_sequence(value, group)
    if query.shape[1] != key.shape[1]:
        repeats = query.shape[1] // key.shape[1]
        full_key = full_key.repeat_interleave(repeats, dim=1)
        full_value = full_value.repeat_interleave(repeats, dim=1)
    local_length = query.shape[-2]
    allowed = torch.ones((local_length, full_key.shape[-2]), dtype=torch.bool, device=query.device)
    if causal:
        positions = torch.arange(local_length, device=query.device) + group.rank * local_length
        allowed &= (
            torch.arange(full_key.shape[-2], device=query.device)[None, :] <= positions[:, None]
        )
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("CP mask 必须为 bool，True 表示允许注意力")
        allowed = allowed & attention_mask
    return F.scaled_dot_product_attention(
        query, full_key, full_value, attn_mask=allowed, dropout_p=0.0, scale=scale
    )
