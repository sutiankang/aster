"""Native Triton online-softmax forward/backward kernels, imported by an explicit provider."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _visibility(
    POS,
    KM,
    QM,
    batch,
    qi,
    ki,
    Q: tl.constexpr,
    K: tl.constexpr,
    OFFSET: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    qpos = tl.load(POS + batch * Q + qi, qi < Q, other=0)
    qok = tl.load(QM + batch * Q + qi, qi < Q, other=0)
    kok = tl.load(KM + batch * K + ki, ki < K, other=0)
    visible = (qi[:, None] < Q) & (ki[None, :] < K) & qok[:, None] & kok[None, :]
    if CAUSAL:
        visible = visible & (OFFSET + ki[None, :] <= qpos[:, None])
    if WINDOW > 0:
        visible = visible & (OFFSET + ki[None, :] > qpos[:, None] - WINDOW)
    return visible


@triton.jit
def _forward(
    QP,
    KP,
    VP,
    POS,
    KM,
    QM,
    OUT,
    LSE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    OFFSET: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    kvhead = head // (HQ // HK)
    qi, di = row * BM + tl.arange(0, BM), tl.arange(0, D)
    q = tl.load(
        QP + ((batch * HQ + head) * Q + qi[:, None]) * D + di[None, :], qi[:, None] < Q, other=0
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    mass = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    for start in range(0, K, BN):
        ki = start + tl.arange(0, BN)
        key = tl.load(
            KP + ((batch * HK + kvhead) * K + ki[:, None]) * D + di[None, :],
            ki[:, None] < K,
            other=0,
        )
        value = tl.load(
            VP + ((batch * HK + kvhead) * K + ki[:, None]) * D + di[None, :],
            ki[:, None] < K,
            other=0,
        )
        visible = _visibility(POS, KM, QM, batch, qi, ki, Q, K, OFFSET, CAUSAL, WINDOW)
        scores = tl.where(visible, tl.dot(q, tl.trans(key)) * SCALE, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))

        safe = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe)
        p = tl.exp(scores - safe[:, None])
        mass = mass * alpha + tl.sum(p, 1)
        numerator = numerator * alpha[:, None] + tl.dot(p.to(value.dtype), value)
        maximum = next_max
    result = numerator / tl.where(mass > 0.0, mass, 1.0)[:, None]
    logsum = tl.where(mass > 0.0, maximum + tl.log(mass), -float("inf"))
    tl.store(
        OUT + ((batch * HQ + head) * Q + qi[:, None]) * D + di[None, :], result, qi[:, None] < Q
    )
    tl.store(LSE + (batch * HQ + head) * Q + qi, logsum, qi < Q)


@triton.jit
def _backward_q(
    QP,
    KP,
    VP,
    POS,
    KM,
    QM,
    OUT,
    LSE,
    DO,
    DQ,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    OFFSET: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    kvhead = head // (HQ // HK)
    qi, di = row * BM + tl.arange(0, BM), tl.arange(0, D)
    qoff = ((batch * HQ + head) * Q + qi[:, None]) * D + di[None, :]
    q = tl.load(QP + qoff, qi[:, None] < Q, other=0)
    grad = tl.load(DO + qoff, qi[:, None] < Q, other=0)
    out = tl.load(OUT + qoff, qi[:, None] < Q, other=0)
    delta = tl.sum(out * grad.to(tl.float32), 1)
    logs = tl.load(LSE + bh * Q + qi, qi < Q, other=-float("inf"))
    logs = tl.where(logs == -float("inf"), 0.0, logs)
    dq = tl.zeros((BM, D), tl.float32)
    for start in range(0, K, BN):
        ki = start + tl.arange(0, BN)
        koff = ((batch * HK + kvhead) * K + ki[:, None]) * D + di[None, :]
        key = tl.load(KP + koff, ki[:, None] < K, other=0)
        value = tl.load(VP + koff, ki[:, None] < K, other=0)
        visible = _visibility(POS, KM, QM, batch, qi, ki, Q, K, OFFSET, CAUSAL, WINDOW)
        scores = tl.where(visible, tl.dot(q, tl.trans(key)) * SCALE, -float("inf"))
        p = tl.exp(scores - logs[:, None])
        dp = tl.dot(grad, tl.trans(value))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(key.dtype), key) * SCALE
    tl.store(DQ + qoff, dq, qi[:, None] < Q)


@triton.jit
def _backward_kv(
    QP,
    KP,
    VP,
    POS,
    KM,
    QM,
    OUT,
    LSE,
    DO,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    OFFSET: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    column, bh = tl.program_id(0), tl.program_id(1)
    batch, kvhead = bh // HK, bh % HK
    ki, di = column * BN + tl.arange(0, BN), tl.arange(0, D)
    koff = ((batch * HK + kvhead) * K + ki[:, None]) * D + di[None, :]
    key = tl.load(KP + koff, ki[:, None] < K, other=0)
    value = tl.load(VP + koff, ki[:, None] < K, other=0)
    dk, dv = tl.zeros((BN, D), tl.float32), tl.zeros((BN, D), tl.float32)

    for group in range(0, HQ // HK):
        head = kvhead * (HQ // HK) + group
        for start in range(0, Q, BM):
            qi = start + tl.arange(0, BM)
            qoff = ((batch * HQ + head) * Q + qi[:, None]) * D + di[None, :]
            q = tl.load(QP + qoff, qi[:, None] < Q, other=0)
            grad = tl.load(DO + qoff, qi[:, None] < Q, other=0)
            out = tl.load(OUT + qoff, qi[:, None] < Q, other=0)
            delta = tl.sum(out * grad.to(tl.float32), 1)
            logs = tl.load(LSE + (batch * HQ + head) * Q + qi, qi < Q, other=-float("inf"))
            logs = tl.where(logs == -float("inf"), 0.0, logs)
            visible = _visibility(POS, KM, QM, batch, qi, ki, Q, K, OFFSET, CAUSAL, WINDOW)
            scores = tl.where(visible, tl.dot(q, tl.trans(key)) * SCALE, -float("inf"))
            p = tl.exp(scores - logs[:, None])
            dp = tl.dot(grad, tl.trans(value))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * SCALE
            dv += tl.dot(tl.trans(p.to(grad.dtype)), grad)
    tl.store(DK + koff, dk, ki[:, None] < K)
    tl.store(DV + koff, dv, ki[:, None] < K)


def _arguments(q, k, offset, causal, window, scale, qb, kb):
    return dict(
        HQ=q.shape[1],
        HK=k.shape[1],
        Q=q.shape[2],
        K=k.shape[2],
        D=q.shape[3],
        SCALE=scale,
        OFFSET=offset,
        CAUSAL=causal,
        WINDOW=0 if window is None else window,
        BM=qb,
        BN=kb,
        num_warps=4,
        num_stages=2,
    )


def forward_statistics(q, k, v, positions, km, qm, offset, causal, window, scale, qb, kb):

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    with torch.cuda.device(q.device):
        _forward[(triton.cdiv(q.shape[2], qb), q.shape[0] * q.shape[1])](
            q,
            k,
            v,
            positions,
            km,
            qm,
            output,
            lse,
            **_arguments(q, k, offset, causal, window, scale, qb, kb),
        )

    if not torch.isfinite(output).all() or torch.isnan(lse).any() or torch.isposinf(lse).any():
        raise ValueError("Native Triton attention produced non-finite output/statistics")
    return output, lse


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, positions, km, qm, offset, causal, window, scale, qb, kb, work):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output, lse = forward_statistics(
            q, k, v, positions, km, qm, offset, causal, window, scale, qb, kb
        )
        ctx.save_for_backward(q, k, v, positions, km, qm, output, lse)
        ctx.arguments = _arguments(q, k, offset, causal, window, scale, qb, kb)
        ctx.work, ctx.tiles = work, (qb, kb)
        work.saved_tensor_elements = sum(
            x.numel() for x in (q, k, v, positions, km, qm, output, lse)
        )
        work.query_tiles += triton.cdiv(q.shape[2], qb)
        work.key_tiles += triton.cdiv(q.shape[2], qb) * triton.cdiv(k.shape[2], kb)
        work.max_score_elements = max(
            work.max_score_elements, min(qb, q.shape[2]) * min(kb, k.shape[2])
        )
        return output.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, positions, km, qm, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        qb, kb = ctx.tiles
        with torch.cuda.device(q.device):
            _backward_q[(triton.cdiv(q.shape[2], qb), q.shape[0] * q.shape[1])](
                q, k, v, positions, km, qm, output, lse, grad_output, dq, **ctx.arguments
            )
            _backward_kv[(triton.cdiv(k.shape[2], kb), k.shape[0] * k.shape[1])](
                q, k, v, positions, km, qm, output, lse, grad_output, dk, dv, **ctx.arguments
            )
        ctx.work.backward_tiles += 2 * triton.cdiv(q.shape[2], qb) * triton.cdiv(k.shape[2], kb)
        ctx.work.max_backward_score_elements = max(
            ctx.work.max_backward_score_elements, min(qb, q.shape[2]) * min(kb, k.shape[2])
        )
        return dq, dk, dv, *(None for _ in range(10))
