"""Tiled attention with recomputed backward and direct page-table execution."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from importlib import metadata
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .online_attention import (
    AttentionBlock,
    AttentionWork,
    SoftmaxState,
    merge_softmax,
    online_attention_statistics,
)


ATTENTION_SOURCES = (
    {
        "repository": "https://github.com/triton-lang/triton",
        "revision": "c817b9b63d40ead1ed023b7663f5ea14f676f4bc",
        "tag": "v3.4.0",
        "path": "python/tutorials/06-fused-attention.py",
        "sha256": "5f312a051cf0f1b55d0aa64d04e76c74d7aa8622096ad77b75f5d444fd91b6a7",
    },
    {
        "repository": "https://github.com/Dao-AILab/flash-attention",
        "revision": "060c9188beec3a8b62b33a3bfa6d5d2d44975fab",
        "tag": "v2.8.3",
        "path": "csrc/flash_attn/src/flash_fwd_kernel.h",
        "sha256": "765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3",
    },
)


_IMPLEMENTATION = {
    name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    for name in ("fused_attention.py", "_triton_attention.py", "online_attention.py")
}
_IMPLEMENTATION["nn/attention.py"] = hashlib.sha256(
    (Path(__file__).parent.parent / "nn/attention.py").read_bytes()
).hexdigest()


def _compiler_version():
    try:
        return metadata.version("triton")
    except metadata.PackageNotFoundError:
        return None


class UnsupportedAttentionBackend(ValueError):
    """Select an explicit caller-approved fallback only for unsupported configurations;
    do not catch numerical or execution failures as compatibility errors."""


@dataclass
class KernelWork(AttentionWork):
    backend: str | None = None
    fallback_reason: str | None = None
    backward_tiles: int = 0
    max_backward_score_elements: int = 0
    saved_tensor_elements: int = 0

    page_launches: int = 0
    compiler_version: str | None = None


def _mask(value, shape, device, name):
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if (
        not isinstance(value, torch.Tensor)
        or value.device != device
        or value.shape != shape
        or value.requires_grad
        or not ((value == 0) | (value == 1)).all()
    ):
        raise ValueError(name + " must be fixed, aligned, colocated binary data")
    return value.bool().contiguous()


@torch.no_grad()
def _validate(
    q, k, v, positions, offset, key_padding, query_padding, causal, window, scale, qb, kb
):
    if (
        any(not isinstance(x, torch.Tensor) or x.ndim != 4 or min(x.shape) < 1 for x in (q, k, v))
        or q.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
        or q.shape[0] != k.shape[0]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("Expected compatible floating [B,H,Q/K,D] GQA inputs")
    if any(not torch.isfinite(x).all() for x in (q, k, v)):
        raise ValueError("Attention inputs must be finite")
    if (
        not isinstance(positions, torch.Tensor)
        or positions.shape != (q.shape[0], q.shape[2])
        or positions.device != q.device
        or positions.dtype not in {torch.int32, torch.int64}
        or (positions < 0).any()
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError(
            "Attention requires explicit nonnegative absolute positions and key offset"
        )
    if type(causal) is not bool or (window is not None and (type(window) is not int or window < 1)):
        raise ValueError("Invalid causal/window profile")
    if any(type(x) is not int or x < 1 for x in (qb, kb)):
        raise ValueError("Tile sizes must be positive integers")
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if type(scale) not in {int, float} or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    km = _mask(key_padding, (q.shape[0], k.shape[2]), q.device, "Key padding")
    qm = _mask(query_padding, (q.shape[0], q.shape[2]), q.device, "Query padding")
    return positions.contiguous(), km, qm, float(scale)


def _select_backend(requested, fallback, q, v, qb, kb):
    if requested not in {"torch_tiled", "triton_fused"} or fallback not in {None, "torch_tiled"}:
        raise ValueError(
            "Select torch_tiled or triton_fused, with explicit optional torch_tiled fallback"
        )
    if requested == "torch_tiled":
        if fallback is not None:
            raise ValueError("A reference backend does not itself need a fallback")
        return requested, None
    reason = None
    if q.device.type != "cuda" or torch.version.hip is not None:
        reason = "triton_fused profile requires NVIDIA CUDA"
    elif (
        q.dtype not in {torch.float16, torch.bfloat16}
        or q.shape[-1] not in {32, 64, 128}
        or v.shape[-1] != q.shape[-1]
    ):
        reason = "triton_fused requires FP16/BF16 and equal head widths 32/64/128"
    elif qb != 32 or kb != 32:
        reason = "triton_fused initial profile requires 32 by 32 tiles"
    elif torch.cuda.get_device_capability(q.device)[0] < 8:
        reason = "triton_fused initial profile requires compute capability at least 8.0"
    elif importlib.util.find_spec("triton") is None:
        reason = "Optional Triton compiler is not installed"
    if reason and fallback is None:
        raise UnsupportedAttentionBackend(reason)
    return (fallback, reason) if reason else (requested, None)


def _visible(positions, offset, start, stop, km, qm, causal, window):
    kp = torch.arange(offset + start, offset + stop, device=positions.device)
    visible = km[:, None, start:stop] & qm[:, :, None]
    if causal:
        visible = visible & (kp[None, None] <= positions[:, :, None])
    if window is not None:
        visible = visible & (kp[None, None] > positions[:, :, None] - window)
    return visible[:, None, None]


class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, km, qm, offset, causal, window, scale, qb, kb, work):

        with torch.autocast(device_type=q.device.type, enabled=False):
            stats = online_attention_statistics(
                q,
                [AttentionBlock(k, v, offset, km)],
                query_positions=positions,
                query_padding=qm,
                causal=causal,
                window=window,
                scale=scale,
                query_block_size=qb,
                key_block_size=kb,
                work=work,
            )
        output = stats.output()
        lse = torch.where(stats.mass > 0, stats.maximum + stats.mass.log(), -torch.inf)
        lse = lse.reshape(q.shape[:3])
        ctx.save_for_backward(q, k, v, positions, km, qm, output, lse)
        ctx.settings = offset, causal, window, scale, qb, kb
        ctx.work = work
        work.saved_tensor_elements = sum(
            x.numel() for x in (q, k, v, positions, km, qm, output, lse)
        )
        return output.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            return _TiledAttention.backward_math(ctx, grad_output)

    @staticmethod
    def backward_math(ctx, grad_output):
        q, k, v, positions, km, qm, output, lse = ctx.saved_tensors
        offset, causal, window, scale, qb, kb = ctx.settings
        batch, heads, count, width = q.shape
        kvheads, keys, value_width = k.shape[1], k.shape[2], v.shape[-1]
        groups = heads // kvheads
        dtype = output.dtype
        dq, dk, dv = (torch.zeros_like(x, dtype=dtype) for x in (q, k, v))

        for begin in range(0, count, qb):
            end = min(begin + qb, count)
            shape = (batch, kvheads, groups, end - begin)
            qt = q[..., begin:end, :].to(dtype).reshape(*shape, width)
            do = grad_output[..., begin:end, :].to(dtype).reshape(*shape, value_width)
            ot = output[..., begin:end, :].reshape(*shape, value_width)
            delta = (do * ot).sum(-1)
            logs = lse[..., begin:end].reshape(shape)
            safe_logs = torch.where(torch.isfinite(logs), logs, 0.0)
            dqt = torch.zeros_like(qt)
            for start in range(0, keys, kb):
                stop = min(start + kb, keys)
                kt, vt = k[..., start:stop, :].to(dtype), v[..., start:stop, :].to(dtype)
                visible = _visible(
                    positions[:, begin:end],
                    offset,
                    start,
                    stop,
                    km,
                    qm[:, begin:end],
                    causal,
                    window,
                )
                scores = torch.einsum("bhgqd,bhkd->bhgqk", qt, kt) * scale
                p = (scores.masked_fill(~visible, -torch.inf) - safe_logs[..., None]).exp()
                dp = torch.einsum("bhgqd,bhkd->bhgqk", do, vt)
                ds = p * (dp - delta[..., None])
                dqt += torch.einsum("bhgqk,bhkd->bhgqd", ds, kt) * scale
                dk[..., start:stop, :] += torch.einsum("bhgqk,bhgqd->bhkd", ds, qt) * scale
                dv[..., start:stop, :] += torch.einsum("bhgqk,bhgqd->bhkd", p, do)
                ctx.work.backward_tiles += 1
                ctx.work.max_backward_score_elements = max(
                    ctx.work.max_backward_score_elements, scores.numel()
                )
            dq[..., begin:end, :] = dqt.reshape(batch, heads, end - begin, width)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), *(None for _ in range(10))


def fused_attention(
    q,
    k,
    v,
    *,
    query_positions,
    key_offset=0,
    key_padding=None,
    query_padding=None,
    causal=True,
    window=None,
    scale=None,
    backend="torch_tiled",
    fallback=None,
    query_block_size=32,
    key_block_size=32,
    dropout=0.0,
    additive_bias=None,
    work=None,
):
    """Tiled forward/backward with an explicit supported domain. No second derivatives,
    dropout, arbitrary additive bias, or packed-cu-seqlens support."""
    if dropout != 0 or additive_bias is not None:
        raise UnsupportedAttentionBackend("This profile has no dropout or additive attention bias")
    positions, km, qm, scale = _validate(
        q,
        k,
        v,
        query_positions,
        key_offset,
        key_padding,
        query_padding,
        causal,
        window,
        scale,
        query_block_size,
        key_block_size,
    )
    effective, reason = _select_backend(backend, fallback, q, v, query_block_size, key_block_size)
    work = work if work is not None else KernelWork()
    work.backend, work.fallback_reason = effective, reason
    args = (
        q,
        k,
        v,
        positions,
        km,
        qm,
        key_offset,
        causal,
        window,
        scale,
        query_block_size,
        key_block_size,
        work,
    )
    if effective == "torch_tiled":
        return _TiledAttention.apply(*args)
    from ._triton_attention import TritonAttention

    work.compiler_version = _compiler_version()
    return TritonAttention.apply(*args)


@torch.no_grad()
def paged_fused_attention(
    q,
    blocks,
    *,
    query_positions,
    causal=True,
    window=None,
    query_padding=None,
    scale=None,
    backend="triton_fused",
    fallback=None,
    query_block_size=32,
    key_block_size=32,
    work=None,
):
    """Read actual pages and merge partial results by logsumexp without concatenating KV."""
    blocks = tuple(blocks)
    if not blocks or any(not isinstance(x, AttentionBlock) for x in blocks):
        raise ValueError("Paged attention needs explicit nonempty AttentionBlocks")

    validated = [
        _validate(
            q,
            x.key,
            x.value,
            query_positions,
            x.offset,
            x.padding,
            query_padding,
            causal,
            window,
            scale,
            query_block_size,
            key_block_size,
        )
        for x in blocks
    ]
    intervals = sorted((x.offset, x.offset + x.key.shape[-2]) for x in blocks)
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping pages would duplicate keys")
    if any(
        x.key.shape[1] != blocks[0].key.shape[1] or x.value.shape[-1] != blocks[0].value.shape[-1]
        for x in blocks
    ):
        raise ValueError("All pages must share the same GQA/value layout")
    effective, reason = _select_backend(
        backend, fallback, q, blocks[0].value, query_block_size, key_block_size
    )
    work = work if work is not None else KernelWork()
    work.backend, work.fallback_reason = effective, reason
    if effective == "torch_tiled":
        with torch.autocast(device_type=q.device.type, enabled=False):
            return online_attention_statistics(
                q,
                blocks,
                query_positions=query_positions,
                causal=causal,
                window=window,
                query_padding=query_padding,
                scale=scale,
                query_block_size=query_block_size,
                key_block_size=key_block_size,
                work=work,
            ).output(dtype=q.dtype)
    from ._triton_attention import forward_statistics

    work.compiler_version = _compiler_version()
    state = None
    for block, (positions, km, qm, actual_scale) in zip(blocks, validated):
        output, lse = forward_statistics(
            q,
            block.key,
            block.value,
            positions,
            km,
            qm,
            block.offset,
            causal,
            window,
            actual_scale,
            query_block_size,
            key_block_size,
        )

        shape = (q.shape[0], block.key.shape[1], q.shape[1] // block.key.shape[1], q.shape[2])
        finite = torch.isfinite(lse).reshape(shape)
        local = SoftmaxState(
            lse.reshape(shape), finite.to(output.dtype), output.reshape(*shape, output.shape[-1])
        )
        state = local if state is None else merge_softmax(state, local)
        work.page_launches += 1
    return state.output(dtype=q.dtype)


def assert_dense_attention_layout(model, *, allow_zero3_units=False):

    from aster.nn.attention import GroupedQueryAttention

    context = getattr(model, "context", None)
    if context is not None and any(
        getattr(getattr(context, axis, None), "size", 1) != 1
        for axis in ("tp", "pp", "cp", "ep", "etp", "gtp_remat")
    ):
        raise UnsupportedAttentionBackend(
            "Fused attention initial model provider is not certified for model parallelism"
        )
    for parameter in model.parameters():
        if (
            hasattr(parameter, "_aster_tp_dimension")
            or getattr(parameter, "_aster_tp_sharded", False)
            or hasattr(parameter, "_aster_extra_gradient_group")
        ):
            raise UnsupportedAttentionBackend(
                "Parallel parameter ownership requires a separate certified provider"
            )
    for layer in model.modules():
        if isinstance(layer, GroupedQueryAttention):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projection = getattr(layer, name)
                if type(projection) is not nn.Linear and allow_zero3_units:
                    from aster.training.sharding import Zero3Unit

                    if type(projection) is Zero3Unit:
                        projection = projection.module
                if type(projection) is not nn.Linear:
                    raise UnsupportedAttentionBackend(
                        "Only unwrapped dense attention projections are supported"
                    )


class AttentionBackend(nn.Module):
    """A parameter-free execution provider; weight keys remain stable while the trainer
    records its precision contract."""

    def __init__(
        self, *, backend="torch_tiled", fallback=None, query_block_size=32, key_block_size=32
    ):
        super().__init__()
        if backend not in {"torch_tiled", "triton_fused"} or fallback not in {None, "torch_tiled"}:
            raise ValueError("Invalid explicit backend/fallback")
        if backend == "torch_tiled" and fallback is not None:
            raise ValueError("Reference backend must not declare fallback")
        if any(type(x) is not int or x < 1 for x in (query_block_size, key_block_size)):
            raise ValueError("Invalid tile size")
        self.backend, self.fallback = backend, fallback
        self.query_block_size, self.key_block_size = query_block_size, key_block_size
        self.work = KernelWork()

    def precision_contract(self):
        return {
            "kind": "native_tiled_attention_v1",
            "backend": self.backend,
            "fallback": self.fallback,
            "query_block_size": self.query_block_size,
            "key_block_size": self.key_block_size,
            "accumulator": "fp64_for_double_else_fp32",
            "dropout": 0.0,
            "higher_order_gradients": False,
            "sources": [dict(x) for x in ATTENTION_SOURCES],
            "implementation_sha256": dict(_IMPLEMENTATION),
            "triton_version": _compiler_version(),
        }

    def validate_attention(self, attention):

        assert_dense_attention_layout(attention, allow_zero3_units=True)

    def forward(self, q, k, v, **kwargs):
        return fused_attention(
            q,
            k,
            v,
            backend=self.backend,
            fallback=self.fallback,
            query_block_size=self.query_block_size,
            key_block_size=self.key_block_size,
            work=self.work,
            **kwargs,
        )


def set_attention_backend(
    model, *, backend="torch_tiled", fallback=None, query_block_size=32, key_block_size=32
):
    """Configure supported native dense Llama/Qwen2/Qwen3 before constructing the trainer."""
    from aster.models.decoder import CausalLM
    from aster.models.config import LlamaConfig, Qwen2Config, Qwen3Config

    if any(getattr(module, "_aster_training_owned", False) for module in model.modules()):
        raise UnsupportedAttentionBackend(
            "Select attention backend before creating the training-owned optimizer role"
        )
    if type(model) is not CausalLM or type(model.config) not in {
        LlamaConfig,
        Qwen2Config,
        Qwen3Config,
    }:
        raise UnsupportedAttentionBackend(
            "Native fused provider currently supports dense Llama/Qwen2/Qwen3 only"
        )
    assert_dense_attention_layout(model)
    layers = [x.self_attn for x in model.model.layers]
    if any(x.dropout != 0 for x in layers):
        raise UnsupportedAttentionBackend("Nonzero attention dropout is unsupported")

    providers = [
        AttentionBackend(
            backend=backend,
            fallback=fallback,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
        )
        for _ in layers
    ]
    for layer, provider in zip(layers, providers):
        layer.attention_backend = provider
    return model
