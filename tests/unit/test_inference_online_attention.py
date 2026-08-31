import pytest
import torch

from aster.optimization.online_attention import (
    AttentionBlock,
    AttentionWork,
    online_attention,
    online_attention_statistics,
    merge_softmax,
)
from aster.inference.state import PagedStatePool, CacheCapacityError, StateError


def oracle(q, key, value, positions, padding, *, causal, window, query_padding=None):
    heads = q.shape[1] // key.shape[1]
    scores = (q @ key.repeat_interleave(heads, 1).transpose(-1, -2)) * q.shape[-1] ** -0.5
    keys = torch.arange(key.shape[-2], device=q.device)
    mask = torch.ones((q.shape[0], q.shape[-2], key.shape[-2]), dtype=torch.bool, device=q.device)
    if causal:
        mask &= keys[None, None] <= positions[..., None]
    if window is not None:
        mask &= keys[None, None] > positions[..., None] - window
    mask &= padding[:, None]
    if query_padding is not None:
        mask &= query_padding[:, :, None]
    mask = mask[:, None]
    scores = scores.masked_fill(~mask, -torch.inf)
    scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
    return scores.softmax(-1).masked_fill(~mask, 0) @ value.repeat_interleave(heads, 1)


@pytest.mark.parametrize("window", [None, 1, 5, 17])
@pytest.mark.parametrize("causal", [True, False])
def test_page_scan_gqa_padding_window_query_chunks_and_partition_merge(window, causal):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    q = torch.randn(2, 6, 13, 8, dtype=torch.float64)
    key = torch.randn(2, 2, 31, 8, dtype=torch.float64)
    value = torch.randn(2, 2, 31, 5, dtype=torch.float64)
    positions = torch.arange(18, 31).expand(2, -1)
    padding = torch.ones(2, 31, dtype=torch.bool)
    padding[0, 2::3] = False
    padding[1] = False
    query_padding = torch.ones(2, 13, dtype=torch.bool)
    query_padding[0, -1] = False
    boundaries = [0, 3, 10, 14, 21, 31]
    blocks = [
        AttentionBlock(key[..., a:b, :], value[..., a:b, :], a, padding[:, a:b])
        for a, b in zip(boundaries, boundaries[1:])
    ]
    work = AttentionWork()
    kwargs = dict(
        query_positions=positions,
        causal=causal,
        window=window,
        query_padding=query_padding,
        query_block_size=4,
        key_block_size=5,
    )
    actual = online_attention(q, blocks, work=work, **kwargs)
    expected = oracle(
        q, key, value, positions, padding, causal=causal, window=window, query_padding=query_padding
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    left = online_attention_statistics(q, blocks[:2], **kwargs)
    right = online_attention_statistics(q, blocks[2:], **kwargs)
    torch.testing.assert_close(
        merge_softmax(left, right).output(), expected, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        online_attention(q, list(reversed(blocks)), **kwargs), expected, rtol=1e-12, atol=1e-12
    )
    assert work.max_score_elements <= 2 * 6 * 4 * 5
    assert work.max_score_elements < 2 * 6 * 13 * 31
    assert work.max_key_tile == 5 and work.max_query_tile == 4
    assert actual[1].count_nonzero() == 0 and actual[0, :, -1].count_nonzero() == 0


def test_extreme_finite_scores_and_no_autograd_history():
    torch.manual_seed(10)
    q = (torch.randn(1, 4, 9, 8) * 100).requires_grad_()
    key = torch.randn(1, 2, 40, 8) * 100
    value = torch.randn(1, 2, 40, 7)
    positions = torch.arange(31, 40)[None]
    pad = torch.ones(1, 40, dtype=torch.bool)
    blocks = [
        AttentionBlock(key[..., i : i + 3, :], value[..., i : i + 3, :], i) for i in range(0, 40, 3)
    ]
    got = online_attention(
        q, blocks, query_positions=positions, query_block_size=2, key_block_size=3
    )
    expected = oracle(
        q.double(), key.double(), value.double(), positions, pad, causal=True, window=None
    )
    torch.testing.assert_close(got.double(), expected, rtol=3e-5, atol=3e-5)
    assert torch.isfinite(got).all() and not got.requires_grad
    with pytest.raises(ValueError, match="Overlapping"):
        online_attention(q, [blocks[0], blocks[0]], query_positions=positions)


def state(length, start=0.0):
    keys = torch.arange(length * 2).reshape(1, 1, length, 2).float() + start
    return ((keys, keys + 1),)


def test_delta_append_real_zero_copy_views_cow_and_stale_leases():
    pool = PagedStatePool(block_size=4, max_blocks=4)
    original = pool.create("bound-model")
    pool.append_delta(original, state(3))
    child = pool.fork(original)
    old_ref = original.pages[0]
    with pool.read_pages(original) as views:
        assert views[0].payload[0].data_ptr() == pool._page(old_ref).payload[0].data_ptr()
        pool.append_delta(original, state(2, 100.0))
        assert original.pages[0] != old_ref and child.pages[0] == old_ref
        torch.testing.assert_close(views[0].payload[0], state(3)[0][0])
        pool.release(child)
        assert pool._page(old_ref).readers == 1
    with pytest.raises(StateError, match="Stale"):
        pool._page(old_ref)
    value = pool.materialize(original)[0][0]
    torch.testing.assert_close(value, torch.cat([state(3)[0][0], state(2, 100.0)[0][0]], -2))
    pool.release(original)
    assert pool.used_blocks == 0


def test_delta_allocation_failure_preserves_every_original_page():
    pool = PagedStatePool(block_size=2, max_blocks=1)
    seq = pool.create("domain")
    pool.append_delta(seq, state(2))
    refs = list(seq.pages)
    with pytest.raises(CacheCapacityError):
        pool.append_delta(seq, state(1))
    assert seq.pages == refs and seq.length == 2
    torch.testing.assert_close(pool.materialize(seq)[0][0], state(2)[0][0])


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA device; GPU numerical/performance validation is separate",
)
def test_cuda_online_attention_numerics_not_a_throughput_claim():
    q = torch.randn(1, 4, 17, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 51, 16, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    positions = torch.arange(34, 51, device="cuda")[None]
    blocks = [
        AttentionBlock(k[..., i : i + 8, :], v[..., i : i + 8, :], i) for i in range(0, 51, 8)
    ]
    got = online_attention(
        q, blocks, query_positions=positions, query_block_size=4, key_block_size=8
    )
    expected = oracle(
        q.float(),
        k.float(),
        v.float(),
        positions,
        torch.ones(1, 51, device="cuda", dtype=torch.bool),
        causal=True,
        window=None,
    )
    torch.testing.assert_close(got.float(), expected, atol=0.002, rtol=0.002)
