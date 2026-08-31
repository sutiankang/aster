"""Native ring attention with online softmax and explicit backward communication."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from .parallel import Group


def _exchange_start(value, group, tag):
    if group.size == 1:
        return value, []
    received = torch.empty_like(value)
    operations = [
        dist.P2POp(
            dist.isend,
            value.contiguous(),
            group.ranks[(group.rank + 1) % group.size],
            group.handle,
            tag,
        ),
        dist.P2POp(
            dist.irecv, received, group.ranks[(group.rank - 1) % group.size], group.handle, tag
        ),
    ]
    return received, dist.batch_isend_irecv(operations)


def _wait(requests):
    for request in requests:
        request.wait()


def _mask(query, local_length, owner, group, causal, attention_mask):
    length = query.shape[-2]
    allowed = torch.ones((length, local_length), dtype=torch.bool, device=query.device)
    if causal:
        q_positions = torch.arange(length, device=query.device) + group.rank * length
        k_positions = torch.arange(local_length, device=query.device) + owner * local_length
        allowed &= k_positions[None, :] <= q_positions[:, None]
    if attention_mask is not None:
        allowed = allowed & attention_mask[..., owner * local_length : (owner + 1) * local_length]
    return allowed


def _heads(tensor, query_heads):
    return (
        tensor.repeat_interleave(query_heads // tensor.shape[1], dim=1)
        if tensor.shape[1] != query_heads
        else tensor
    )


def _unrepeat(tensor, original_heads):
    if tensor.shape[1] == original_heads:
        return tensor
    return tensor.reshape(
        tensor.shape[0], original_heads, tensor.shape[1] // original_heads, *tensor.shape[2:]
    ).sum(2)


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, group, causal, attention_mask, scale):
        dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        with torch.autocast(query.device.type, enabled=False):
            q = query.to(dtype)
            current = torch.stack((key, value)).contiguous()
            maximum = torch.full((*q.shape[:-1], 1), -torch.inf, device=q.device, dtype=dtype)
            normalizer = torch.zeros_like(maximum)
            accumulator = torch.zeros_like(q)
            for step in range(group.size):
                owner = (group.rank - step) % group.size
                if step + 1 < group.size:
                    received, requests = _exchange_start(current, group, 8100)
                k, v = [_heads(block.to(dtype), q.shape[1]) for block in current]
                allowed = _mask(q, k.shape[-2], owner, group, causal, attention_mask)
                scores = (q @ k.transpose(-1, -2)) * scale
                scores = scores.masked_fill(~allowed, -torch.inf)
                new_maximum = torch.maximum(maximum, scores.max(-1, keepdim=True).values)

                correction = torch.where(
                    torch.isfinite(maximum), (maximum - new_maximum).exp(), 0.0
                )
                probabilities = torch.where(allowed, (scores - new_maximum).exp(), 0.0)
                accumulator = accumulator * correction + probabilities @ v
                normalizer = normalizer * correction + probabilities.sum(-1, keepdim=True)
                maximum = new_maximum
                if step + 1 < group.size:
                    _wait(requests)
                    current = received
            output = torch.where(
                normalizer > 0, accumulator / normalizer.clamp_min(torch.finfo(dtype).tiny), 0.0
            )
            lse = maximum + normalizer.log()
        ctx.group, ctx.causal, ctx.scale = group, causal, scale
        ctx.mask = attention_mask
        ctx.save_for_backward(query, key, value, output, lse)
        return output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        query, key, value, output, lse = ctx.saved_tensors
        group = ctx.group
        dtype = output.dtype
        with torch.autocast(query.device.type, enabled=False):
            q, do = query.to(dtype), output_gradient.to(dtype)
            dq = torch.zeros_like(q)
            current = torch.stack((key, value)).contiguous()
            gradient_bucket = torch.zeros_like(current, dtype=dtype)
            row_dot = (do * output).sum(-1, keepdim=True)
            for step in range(group.size):
                owner = (group.rank - step) % group.size
                k, v = [_heads(block.to(dtype), q.shape[1]) for block in current]
                allowed = _mask(q, k.shape[-2], owner, group, ctx.causal, ctx.mask)
                scores = (q @ k.transpose(-1, -2)) * ctx.scale
                probabilities = torch.where(
                    allowed & torch.isfinite(lse), (scores - lse).exp(), 0.0
                )
                ds = probabilities * ((do @ v.transpose(-1, -2)) - row_dot)
                dq += (ds @ k) * ctx.scale
                gradient_bucket[0] += _unrepeat(
                    (ds.transpose(-1, -2) @ q) * ctx.scale, key.shape[1]
                )
                gradient_bucket[1] += _unrepeat(
                    probabilities.transpose(-1, -2) @ do, value.shape[1]
                )

                received, kv_requests = _exchange_start(current, group, 8200)
                received_gradient, grad_requests = _exchange_start(gradient_bucket, group, 8201)
                _wait(kv_requests)
                _wait(grad_requests)
                current, gradient_bucket = received, received_gradient
        return (
            dq.to(query.dtype),
            gradient_bucket[0].to(key.dtype),
            gradient_bucket[1].to(value.dtype),
            None,
            None,
            None,
            None,
        )


def ring_context_parallel_attention(
    query, key, value, group: Group, *, causal=True, attention_mask=None, scale=None
):
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("ring attention 需要 [B,H,L,D] 且 K/V 同形")
    if (
        query.shape[0] != key.shape[0]
        or query.shape[-2:] != key.shape[-2:]
        or query.shape[1] % key.shape[1]
    ):
        raise ValueError("ring attention 要求等长序列块、匹配 head_dim 和可整除 GQA 头数")
    if attention_mask is not None and (
        attention_mask.dtype != torch.bool or attention_mask.shape[-1] != key.shape[-2] * group.size
    ):
        raise ValueError("ring mask 必须为 bool，并包含完整全局 key 维")
    return _RingAttention.apply(
        query,
        key,
        value,
        group,
        causal,
        attention_mask,
        query.shape[-1] ** -0.5 if scale is None else scale,
    )
