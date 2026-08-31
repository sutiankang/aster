"""Stable blockwise online softmax without materializing full attention scores."""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .kv_quantization import QuantizedKV, finite_kv, kv_tile


@dataclass(frozen=True)
class AttentionBlock:
    key: torch.Tensor | QuantizedKV
    value: torch.Tensor | QuantizedKV
    offset: int
    padding: torch.Tensor | None = None


@dataclass
class AttentionWork:
    """Record actual tile dimensions, not GPU allocator usage or measured throughput."""

    query_tiles: int = 0
    key_tiles: int = 0
    max_score_elements: int = 0
    max_query_tile: int = 0
    max_key_tile: int = 0


@dataclass(frozen=True)
class SoftmaxState:
    maximum: torch.Tensor  # [B,Hkv,G,Q]
    mass: torch.Tensor  # Σ exp(score-maximum)
    weighted: torch.Tensor

    def output(self, *, dtype=None):

        value = self.weighted / self.mass.clamp_min(torch.finfo(self.mass.dtype).tiny)[..., None]
        batch, heads, groups, query, width = value.shape
        return value.reshape(batch, heads * groups, query, width).to(dtype or value.dtype)


@torch.no_grad()
def merge_softmax(left, right):
    """Merge (maximum, exponential sum, weighted numerator) statistics over disjoint
    KV partitions; averaging normalized attention outputs is incorrect."""
    if (
        left.maximum.shape != right.maximum.shape
        or left.weighted.shape != right.weighted.shape
        or left.maximum.dtype != right.maximum.dtype
        or left.maximum.device != right.maximum.device
    ):
        raise ValueError(
            "Attention partitions must describe the same queries and accumulator layout"
        )
    maximum = torch.maximum(left.maximum, right.maximum)
    safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    a = torch.where(torch.isfinite(left.maximum), (left.maximum - safe).exp(), 0.0)
    b = torch.where(torch.isfinite(right.maximum), (right.maximum - safe).exp(), 0.0)
    return SoftmaxState(
        maximum,
        left.mass * a + right.mass * b,
        left.weighted * a[..., None] + right.weighted * b[..., None],
    )


def _binary_mask(value, shape, device, name):
    if value is None:
        return None
    if value.shape != shape or value.device != device or not ((value == 0) | (value == 1)).all():
        raise ValueError(name + " must be a colocated aligned binary mask")
    return value.bool()


@torch.no_grad()
def online_attention_statistics(
    query,
    blocks,
    *,
    query_positions,
    causal=True,
    window=None,
    query_padding=None,
    scale=None,
    query_block_size=32,
    key_block_size=64,
    work=None,
):
    """Scan pages using at most [B,Hq,Qblock,Kblock] additional score storage."""
    blocks = tuple(blocks)
    if (
        query.ndim != 4
        or min(query.shape) < 1
        or not query.is_floating_point()
        or type(query_block_size) is not int
        or type(key_block_size) is not int
        or min(query_block_size, key_block_size) < 1
        or not blocks
    ):
        raise ValueError("Need nonempty query/pages and positive integer tile sizes")
    if not torch.isfinite(query).all():
        raise ValueError("Query must be finite")
    if type(causal) is not bool or (window is not None and (type(window) is not int or window < 1)):
        raise ValueError("Invalid causal/window semantics")
    batch, query_heads, queries, key_width = query.shape
    if (
        query_positions.shape != (batch, queries)
        or query_positions.device != query.device
        or query_positions.dtype not in {torch.int32, torch.int64}
        or (query_positions < 0).any()
    ):
        raise ValueError("Query positions must explicitly align the absolute token axis")
    query_padding = _binary_mask(query_padding, (batch, queries), query.device, "query padding")
    if (
        not isinstance(blocks[0], AttentionBlock)
        or blocks[0].key.ndim != 4
        or blocks[0].value.ndim != 4
    ):
        raise ValueError("Expected explicit rank-four KV blocks")
    heads, value_width = blocks[0].key.shape[1], blocks[0].value.shape[-1]
    if heads < 1 or query_heads % heads:
        raise ValueError("Query heads must divide into KV groups")
    intervals = []
    for block in blocks:
        if (
            not isinstance(block, AttentionBlock)
            or block.key.ndim != 4
            or block.value.ndim != 4
            or block.key.shape[:2] != (batch, heads)
            or block.key.shape[-1] != key_width
            or block.key.shape[:3] != block.value.shape[:3]
            or block.value.shape[-1] != value_width
            or block.key.shape[-2] < 1
            or type(block.offset) is not int
            or block.offset < 0
            or block.key.device != query.device
            or block.value.device != query.device
            or block.key.dtype != query.dtype
            or block.value.dtype != query.dtype
        ):
            raise ValueError("KV page layout/dtype/device differs from query")
        _binary_mask(block.padding, (batch, block.key.shape[-2]), query.device, "key padding")
        if not finite_kv(block.key) or not finite_kv(block.value):
            raise ValueError("KV blocks must contain finite values")
        intervals.append((block.offset, block.offset + block.key.shape[-2]))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping KV blocks would count the same key twice")
    scale = key_width**-0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be finite and positive")
    work = work if work is not None else AttentionWork()
    accumulator_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    groups = query_heads // heads
    maxima, masses, numerators = [], [], []
    for begin in range(0, queries, query_block_size):
        end = min(queries, begin + query_block_size)
        q = (
            query[..., begin:end, :]
            .to(accumulator_dtype)
            .reshape(batch, heads, groups, end - begin, key_width)
        )
        positions = query_positions[:, begin:end]
        shape = (batch, heads, groups, end - begin)
        state = SoftmaxState(
            torch.full(shape, -torch.inf, device=query.device, dtype=accumulator_dtype),
            torch.zeros(shape, device=query.device, dtype=accumulator_dtype),
            torch.zeros((*shape, value_width), device=query.device, dtype=accumulator_dtype),
        )
        work.query_tiles += 1
        for block in blocks:
            for start in range(0, block.key.shape[-2], key_block_size):
                stop = min(block.key.shape[-2], start + key_block_size)
                keys = kv_tile(block.key, start, stop - start, accumulator_dtype)
                values = kv_tile(block.value, start, stop - start, accumulator_dtype)
                key_positions = torch.arange(
                    block.offset + start, block.offset + stop, device=query.device
                )
                visible = torch.ones(
                    (batch, end - begin, stop - start), dtype=torch.bool, device=query.device
                )
                if causal:
                    visible &= key_positions[None, None, :] <= positions[:, :, None]
                if window is not None:
                    visible &= key_positions[None, None, :] > positions[:, :, None] - window
                if block.padding is not None:
                    visible &= block.padding[:, None, start:stop].bool()
                if query_padding is not None:
                    visible &= query_padding[:, begin:end, None]
                if not visible.any():
                    continue

                scores = torch.einsum("bhgqd,bhkd->bhgqk", q, keys) * scale
                if not torch.isfinite(scores).all():
                    raise ValueError(
                        "Attention score overflow; cannot repair an infinite dot product with softmax"
                    )
                scores = scores.masked_fill(~visible[:, None, None], -torch.inf)
                maximum = scores.amax(-1)
                safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
                probability = (scores - safe[..., None]).exp()
                local = SoftmaxState(
                    maximum,
                    probability.sum(-1),
                    torch.einsum("bhgqk,bhkd->bhgqd", probability, values),
                )
                state = merge_softmax(state, local)
                work.key_tiles += 1
                work.max_query_tile = max(work.max_query_tile, end - begin)
                work.max_key_tile = max(work.max_key_tile, stop - start)
                work.max_score_elements = max(work.max_score_elements, scores.numel())
        maxima.append(state.maximum)
        masses.append(state.mass)
        numerators.append(state.weighted)
    return SoftmaxState(torch.cat(maxima, -1), torch.cat(masses, -1), torch.cat(numerators, -2))


@torch.no_grad()
def online_attention(query, blocks, **kwargs):
    return online_attention_statistics(query, blocks, **kwargs).output(dtype=query.dtype)
