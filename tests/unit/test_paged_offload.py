import asyncio
import threading
import pytest
import torch

from aster.inference import (
    PagedStateArchive,
    PagedStatePool,
    KVQuantization,
    PagedAttentionRunner,
    InferenceEngine,
    SamplingConfig,
)
from aster.inference.state import StateError, CacheCapacityError
from aster.optimization.kv_quantization import QuantizedKV
from aster.models import build_model, LlamaConfig


def make_state(format=None, max_blocks=4):
    pool = PagedStatePool(
        block_size=3, max_blocks=max_blocks, quantization=KVQuantization(format) if format else None
    )
    state = pool.create("tenant-policy")
    torch.manual_seed(817)
    key, value = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)
    mask = torch.tensor([True, False, True, True, True]).reshape(1, 1, 5, 1)
    pool.append_delta(state, ((key, value), (mask,)))
    return pool, state


@pytest.mark.parametrize("format", [None, "int8", "fp8_e4m3fn", "fp8_e5m2"])
def test_raw_code_roundtrip_without_requantization_or_float_history(format):
    pool, state = make_state(format)
    archive = PagedStateArchive(pool)

    def codes(sequence):
        with pool.read_pages(sequence) as pages:
            return [
                [
                    (x.values.float().clone(), x.scales.clone())
                    if isinstance(x, QuantizedKV)
                    else (x.clone(),)
                    for x in page.payload
                ]
                for page in pages
            ]

    before = codes(state)
    pool.materialize = lambda *args: pytest.fail("Archive must preserve physical page codes")
    handle = archive.put(state)
    pool.release(state)
    assert pool.used_blocks == 0 and archive.stored_bytes > 0
    restored = archive.restore(handle, identity="tenant-policy")
    assert restored.length == 5
    for expected, actual in zip(before, codes(restored)):
        for xs, ys in zip(expected, actual):
            for x, y in zip(xs, ys):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
    archive.release(handle, identity="tenant-policy")
    pool.release(restored)
    assert archive.metrics() == dict(
        host_bytes=0, snapshots=0, offloaded_tokens=5, restored_tokens=5, pinned_memory=False
    )
    assert pool.used_blocks == 0


def test_budget_identity_failed_restore_and_copy_failure_are_transactional(monkeypatch):
    pool, state = make_state("int8", max_blocks=2)
    archive = PagedStateArchive(pool)
    handle = archive.put(state)
    with pytest.raises(StateError):
        archive.restore(handle, identity="other-tenant")
    with pytest.raises(StateError):
        archive.release(handle, identity="other-tenant")
    with pytest.raises(CacheCapacityError):
        archive.restore(handle, identity="tenant-policy")
    assert pool.used_blocks == 2 and archive.metrics()["snapshots"] == 1
    full = PagedStateArchive(pool, max_bytes=archive.stored_bytes - 1)
    with pytest.raises(CacheCapacityError):
        full.put(state)
    assert full.stored_bytes == 0
    pool.release(state)
    import aster.inference.state as module

    original = module.copy_kv

    def broken(*args):
        raise RuntimeError("injected transfer failure")

    monkeypatch.setattr(module, "copy_kv", broken)
    with pytest.raises(RuntimeError, match="injected"):
        archive.restore(handle, identity="tenant-policy")
    assert pool.used_blocks == 0 and archive.metrics()["snapshots"] == 1
    monkeypatch.setattr(module, "copy_kv", original)
    recovered = archive.restore(handle, identity="tenant-policy")
    archive.release(handle, identity="tenant-policy")
    pool.release(recovered)


@pytest.mark.parametrize("operation", ["put", "restore"])
def test_cancel_waits_for_transfer_then_reclaims_result(operation, monkeypatch):
    pool, state = make_state("fp8_e4m3fn")
    archive = PagedStateArchive(pool)
    handle = archive.put(state) if operation == "restore" else None
    if handle:
        pool.release(state)
    entered, finish = threading.Event(), threading.Event()
    original = getattr(archive, operation)

    def slow(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(archive, operation, slow)

    async def run():
        future = asyncio.create_task(
            archive.put_async(state)
            if operation == "put"
            else archive.restore_async(handle, identity="tenant-policy")
        )
        assert await asyncio.to_thread(entered.wait, 5)
        future.cancel()
        await asyncio.sleep(0)
        assert not future.done()
        future.cancel()
        await asyncio.sleep(0)
        assert not future.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await future

    try:
        asyncio.run(run())
    finally:
        finish.set()
    if handle:
        archive.release(handle, identity="tenant-policy")
    else:
        pool.release(state)
    assert pool.used_blocks == archive.stored_bytes == 0


@pytest.mark.parametrize("format", [None, "int8", "fp8_e4m3fn"])
@pytest.mark.parametrize("host_budget", [1, 1000000])
def test_continuous_scheduler_preemption_swaps_or_explicitly_recomputes(format, host_budget):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = build_model(
        LlamaConfig(
            vocab_size=24,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).eval()

    async def run(capacity, use_archive):
        runner = PagedAttentionRunner(
            model,
            policy_artifact_id="packed",
            backend="torch_online_paged",
            block_size=3,
            max_blocks=capacity,
            key_block_size=3,
            kv_quantization=KVQuantization(format) if format else None,
        )
        archive = PagedStateArchive(runner.pool, max_bytes=host_budget) if use_archive else None
        engine = InferenceEngine(
            runner, max_active=2, max_batch_tokens=8, prefill_chunk_size=3, offload_archive=archive
        )
        try:
            handles = [
                await engine.submit(
                    prompt, SamplingConfig(max_new_tokens=5, temperature=0.7, seed=17)
                )
                for prompt in ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
            ]
            results = await asyncio.gather(*(h.collect() for h in handles))
            stats = engine.observation()
        finally:
            await engine.close()
        assert runner.pool.used_blocks == 0
        if archive:
            assert archive.stored_bytes == 0
        return results, stats

    reference, _ = asyncio.run(run(40, False))
    actual, stats = asyncio.run(run(4, True))
    assert [r.token_ids for r in actual] == [r.token_ids for r in reference]
    assert all(r.stop_reason == "length" for r in actual)
    assert sum(r.preemption_count for r in actual) > 0
    for a, b in zip(actual, reference):
        torch.testing.assert_close(
            torch.tensor(a.raw_model_logprobs),
            torch.tensor(b.raw_model_logprobs),
            atol=1e-6,
            rtol=1e-6,
        )
    if host_budget > 1:
        assert stats["offload"]["restored_tokens"] > 0
        assert stats["offload_capacity_fallbacks"] == 0
    else:
        assert stats["offload_capacity_fallbacks"] > 0
        assert stats["offload"]["restored_tokens"] == 0
