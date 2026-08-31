from copy import deepcopy
import ast
from pathlib import Path
import sys

import pytest
import torch

from aster.optimization.fused_attention import (
    fused_attention,
    paged_fused_attention,
    KernelWork,
    UnsupportedAttentionBackend,
    set_attention_backend,
    assert_dense_attention_layout,
)
from aster.optimization.online_attention import AttentionBlock


def oracle(q, k, v, positions, *, offset=0, km=None, qm=None, causal=True, window=None, scale=None):
    repeat = q.shape[1] // k.shape[1]
    keys = offset + torch.arange(k.shape[2], device=q.device)
    mask = torch.ones((q.shape[0], q.shape[2], k.shape[2]), dtype=torch.bool, device=q.device)
    if causal:
        mask &= keys[None, None] <= positions[..., None]
    if window is not None:
        mask &= keys[None, None] > positions[..., None] - window
    if km is not None:
        mask &= km[:, None].bool()
    if qm is not None:
        mask &= qm[:, :, None].bool()
    mask = mask[:, None]
    scores = (q @ k.repeat_interleave(repeat, 1).transpose(-1, -2)) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    scores = scores.masked_fill(~mask, -torch.inf)
    scores = torch.where(mask.any(-1, keepdim=True), scores, 0.0)
    return scores.softmax(-1).masked_fill(~mask, 0) @ v.repeat_interleave(repeat, 1)


@pytest.mark.parametrize(
    "causal,window", [(True, None), (True, 1), (True, 6), (False, None), (False, 4)]
)
def test_tiled_forward_backward_gqa_absolute_padding_and_nonaligned_tiles(causal, window):
    torch.manual_seed(120)
    torch.set_num_threads(1)
    q = torch.randn(2, 6, 11, 7, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 19, 7, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 19, 5, dtype=torch.float64, requires_grad=True)
    pos = torch.arange(21, 32)[None].expand(2, -1)
    km = torch.ones(2, 19, dtype=torch.bool)
    km[0, 2::3] = False
    km[1] = False
    qm = torch.ones(2, 11, dtype=torch.bool)
    qm[0, 4] = False
    work = KernelWork()
    expected = oracle(q, k, v, pos, offset=13, km=km, qm=qm, causal=causal, window=window)
    actual = fused_attention(
        q,
        k,
        v,
        query_positions=pos,
        key_offset=13,
        key_padding=km,
        query_padding=qm,
        causal=causal,
        window=window,
        query_block_size=3,
        key_block_size=4,
        work=work,
    )
    gradient = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, (q, k, v), gradient)
    actual_grad = torch.autograd.grad(actual, (q, k, v), gradient)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    assert actual[1].count_nonzero() == actual[0, :, 4].count_nonzero() == 0
    assert all(torch.isfinite(x).all() for x in actual_grad)
    assert (
        work.max_score_elements <= 2 * 6 * 3 * 4
        and work.max_backward_score_elements <= 2 * 6 * 3 * 4
    )
    assert work.backward_tiles > 1 and work.backend == "torch_tiled"


def test_gradcheck_linear_saved_tensors_and_no_second_order_claim():
    torch.set_num_threads(1)
    torch.manual_seed(30)
    inputs = (
        torch.randn(1, 2, 3, 2, dtype=torch.double, requires_grad=True),
        torch.randn(1, 1, 5, 2, dtype=torch.double, requires_grad=True),
        torch.randn(1, 1, 5, 3, dtype=torch.double, requires_grad=True),
    )
    function = lambda *xs: fused_attention(
        *xs, query_positions=torch.arange(2, 5)[None], query_block_size=2, key_block_size=2
    )
    assert torch.autograd.gradcheck(function, inputs, eps=1e-6, atol=2e-5, rtol=1e-4)
    grad = torch.autograd.grad(function(*inputs).sum(), inputs, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(sum(x.sum() for x in grad), inputs)
    q = torch.randn(1, 4, 71, 8, requires_grad=True)
    k, v = (torch.randn(1, 2, 109, 8, requires_grad=True) for _ in range(2))
    saved = []

    def pack(x):
        saved.append((x.shape, x.numel()))
        return x

    work = KernelWork()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        out = fused_attention(
            q,
            k,
            v,
            query_positions=torch.arange(38, 109)[None],
            query_block_size=5,
            key_block_size=7,
            work=work,
        )
        out.square().sum().backward()

    assert len(saved) == 9 and all(tuple(shape[-2:]) != (71, 109) for shape, _ in saved)
    assert sum(n for _, n in saved) == work.saved_tensor_elements + out.numel()
    assert work.max_backward_score_elements <= 4 * 5 * 7


def test_no_full_score_intermediate_in_forward_or_backward():
    from torch.utils._python_dispatch import TorchDispatchMode

    q = torch.randn(1, 4, 17, 8, requires_grad=True)
    k, v = (torch.randn(1, 2, 31, 8, requires_grad=True) for _ in range(2))

    class NoQuadratic(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            values = result if isinstance(result, (tuple, list)) else (result,)
            assert all(
                not isinstance(x, torch.Tensor) or x.ndim < 2 or tuple(x.shape[-2:]) != (17, 31)
                for x in values
            )
            return result

    with NoQuadratic():
        fused_attention(
            q,
            k,
            v,
            query_positions=torch.arange(14, 31)[None],
            query_block_size=3,
            key_block_size=5,
        ).square().sum().backward()


def test_explicit_backend_masks_overflow_and_lazy_import():
    q = torch.ones(1, 2, 3, 8)
    k = torch.ones(1, 1, 5, 8)
    v = k.clone()
    args = dict(query_positions=torch.arange(2, 5)[None])
    with pytest.raises(UnsupportedAttentionBackend, match="CUDA"):
        fused_attention(q, k, v, backend="triton_fused", **args)
    work = KernelWork()
    got = fused_attention(
        q, k, v, backend="triton_fused", fallback="torch_tiled", work=work, **args
    )
    assert torch.equal(got, q) and work.backend == "torch_tiled" and "CUDA" in work.fallback_reason
    assert "aster.optimization._triton_attention" not in sys.modules
    for options in ({"dropout": 0.1}, {"additive_bias": torch.zeros(3, 5)}):
        with pytest.raises(UnsupportedAttentionBackend):
            fused_attention(q, k, v, **options, **args)
    for options in (
        {"key_padding": torch.ones(1, 4)},
        {"key_padding": torch.full((1, 5), 2)},
        {"query_padding": torch.ones(1, 3, requires_grad=True)},
        {"key_offset": -1},
        {"window": 0},
        {"scale": float("nan")},
        {"key_block_size": True},
    ):
        with pytest.raises(ValueError):
            fused_attention(q, k, v, **options, **args)
    with pytest.raises(ValueError, match="overflow"):
        fused_attention(q * 1e30, k * 1e30, v, **args)


def test_paged_fallback_partition_merge_reordered_pages_and_masks():
    torch.manual_seed(904)
    q = torch.randn(2, 4, 7, 8, dtype=torch.double)
    k, v = (torch.randn(2, 2, 21, 8, dtype=torch.double) for _ in range(2))
    pos = torch.arange(20, 27)[None].expand(2, -1)
    km = torch.ones(2, 21, dtype=torch.bool)
    km[0, 1::3] = False
    km[1] = False
    blocks = [
        AttentionBlock(k[..., a:b, :], v[..., a:b, :], 6 + a, km[:, a:b])
        for a, b in ((0, 2), (2, 9), (9, 13), (13, 21))
    ]
    work = KernelWork()
    actual = paged_fused_attention(
        q,
        list(reversed(blocks)),
        query_positions=pos,
        window=9,
        backend="triton_fused",
        fallback="torch_tiled",
        query_block_size=3,
        key_block_size=4,
        work=work,
    )
    expected = oracle(q, k, v, pos, offset=6, km=km, window=9)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert work.backend == "torch_tiled" and work.page_launches == 0
    with pytest.raises(ValueError, match="Overlapping"):
        paged_fused_attention(q, [blocks[0], blocks[0]], query_positions=pos)


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
def test_real_model_logits_gradients_and_default_weight_keys_unchanged(family, monkeypatch):
    from aster.models import build_model, LlamaConfig, Qwen2Config, Qwen3Config
    import aster.nn.attention as attention

    torch.manual_seed(212)
    cfg = {"llama": LlamaConfig, "qwen2": Qwen2Config, "qwen3": Qwen3Config}[family]
    options = (
        {"use_sliding_window": True, "sliding_window": 4, "max_window_layers": 1}
        if family == "qwen2"
        else {}
    )
    native = build_model(
        cfg(
            vocab_size=24,
            hidden_size=24,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            **options,
        )
    )
    test = set_attention_backend(deepcopy(native), query_block_size=3, key_block_size=2)
    assert native.state_dict().keys() == test.state_dict().keys()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [2, 4, 6, 8, 10, 12, 14]])
    padding = torch.ones_like(ids)
    padding[0, :2] = 0
    expected = native(ids, attention_mask=padding).logits
    expected.square().sum().backward()

    def forbid(*args, **kwargs):
        raise AssertionError("provider must not build a full mask or call default attention")

    monkeypatch.setattr(attention, "attention_mask", forbid)
    monkeypatch.setattr(attention, "scaled_attention", forbid)
    actual = test(ids, attention_mask=padding).logits
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(test.parameters(), native.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-4)
    prefix = test(ids[:, :5], attention_mask=padding[:, :5], use_cache=True)
    suffix = test(ids[:, 5:], attention_mask=padding, state=prefix.state, use_cache=True).logits
    torch.testing.assert_close(suffix, actual[:, 5:], atol=2e-6, rtol=2e-5)
    assert all(x.self_attn.attention_backend.work.backward_tiles > 0 for x in test.model.layers)


def test_autocast_keeps_fp32_accumulation_and_source_is_native():
    torch.manual_seed(541)
    q = torch.randn(1, 4, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    k, v = (torch.randn(1, 2, 13, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(2))
    args = dict(query_positions=torch.arange(8, 13)[None], query_block_size=2, key_block_size=3)
    expected = fused_attention(q, k, v, **args)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fused_attention(q, k, v, **args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual.float().square().sum()).backward()
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
    path = Path(__file__).resolve().parents[2] / "src/aster/optimization/_triton_attention.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [x.module for x in ast.walk(tree) if isinstance(x, ast.ImportFrom)]
    assert not any(x and ("flash_attn" in x or "xformers" in x) for x in imports)
    functions = {x.name for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)}
    assert {"_forward", "_backward_q", "_backward_kv"} <= functions


def test_parallel_parameter_and_projection_are_rejected():
    from aster.models import build_model, LlamaConfig

    model = build_model(LlamaConfig())
    model.model.layers[0].self_attn.k_proj.weight._aster_tp_dimension = 0
    with pytest.raises(UnsupportedAttentionBackend, match="Parallel parameter"):
        set_attention_backend(model)
    del model.model.layers[0].self_attn.k_proj.weight._aster_tp_dimension
    model.model.layers[0].self_attn.k_proj = torch.nn.Sequential(
        model.model.layers[0].self_attn.k_proj
    )
    with pytest.raises(UnsupportedAttentionBackend, match="unwrapped"):
        assert_dense_attention_layout(model)
    late = set_attention_backend(build_model(LlamaConfig()))
    late.model.layers[0].self_attn.q_proj.weight._aster_tp_dimension = 0
    with pytest.raises(UnsupportedAttentionBackend, match="Parallel parameter"):
        late(torch.tensor([[1, 2, 3]]))


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA: own Triton F/B compiler and numerical profile not hardware validated",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("causal,window", [(True, None), (True, 19), (False, None)])
def test_actual_triton_forward_gqa_backward_padding_offset(dtype, head_dim, causal, window):
    pytest.importorskip("triton")
    torch.manual_seed(691)
    q = (torch.randn(2, 6, 37, head_dim, device="cuda", dtype=dtype) * 0.4).requires_grad_()
    k, v = (
        (torch.randn(2, 2, 69, head_dim, device="cuda", dtype=dtype) * 0.4).requires_grad_()
        for _ in range(2)
    )
    pos = torch.arange(53, 90, device="cuda")[None].expand(2, -1)
    km = torch.ones(2, 69, device="cuda", dtype=torch.bool)
    km[0, ::4] = False
    km[1] = False
    qm = torch.ones(2, 37, device="cuda", dtype=torch.bool)
    qm[0, -1] = False
    actual = fused_attention(
        q,
        k,
        v,
        query_positions=pos,
        key_offset=21,
        key_padding=km,
        query_padding=qm,
        backend="triton_fused",
        causal=causal,
        window=window,
    )
    refs = [x.detach().float().requires_grad_() for x in (q, k, v)]
    expected = oracle(*refs, pos, offset=21, km=km, qm=qm, causal=causal, window=window)
    grad = torch.randn_like(actual) * 0.4
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, refs, grad.float())
    torch.testing.assert_close(actual.float(), expected, atol=0.006, rtol=0.02)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a.float(), b, atol=0.008, rtol=0.03)
    assert torch.count_nonzero(actual[1]) == 0 and all(
        torch.isfinite(x).all() for x in actual_grads
    )
