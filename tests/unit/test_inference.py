import asyncio
from dataclasses import replace
import threading
import time

import pytest
import torch

from aster.inference import (
    KVStateCodec,
    PagedStatePool,
    PrefixCache,
    PrefixIdentity,
    CacheCapacityError,
    StateError,
    SamplingConfig,
    distributions,
    sample_token,
    speculative_accept,
    ModelRunner,
    InferenceEngine,
    OverloadedError,
)
from aster.models import build_model, LlamaConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(length, offset=0.0, widths=(3, 5)):
    return (
        (
            torch.arange(length * widths[0]).float().reshape(1, 1, length, widths[0]) + offset,
            torch.arange(length * widths[1]).float().reshape(1, 1, length, widths[1]) + offset,
        ),
    )


def model():
    torch.manual_seed(4)
    return build_model(
        LlamaConfig(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )


def test_pages_hold_real_tensor_and_cow_branches():
    pool = PagedStatePool(block_size=4, max_blocks=4)
    original = pool.create("policy")
    pool.append(original, state(3))
    shared = pool.fork(original)
    first_page = original.pages[0]
    pool.append(shared, state(5, offset=100))
    assert original.pages[0] == first_page != shared.pages[0]
    assert pool.used_blocks == 3
    torch.testing.assert_close(pool.materialize(original)[0][0], state(3)[0][0])
    extended = pool.materialize(shared)[0][0]
    torch.testing.assert_close(extended[:, :, :3], state(3)[0][0])
    torch.testing.assert_close(extended[:, :, 3:], state(5, 100)[0][0][:, :, 3:])
    pool.release(original)
    pool.release(shared)
    assert pool.used_blocks == 0


def test_atomic_oom_stale_reference_and_deferred_release():
    pool = PagedStatePool(block_size=2, max_blocks=1)
    sequence = pool.create("policy")
    pool.append(sequence, state(2))
    refs = list(sequence.pages)
    with pytest.raises(CacheCapacityError):
        pool.append(sequence, state(3))
    assert sequence.pages == refs and sequence.length == 2
    with pool.borrow(sequence):
        pool.release(sequence)
        assert pool.used_blocks == 1
        other = pool.create("policy")
        with pytest.raises(CacheCapacityError):
            pool.append(other, state(1))
    assert pool.used_blocks == 0
    pool.append(other, state(1))
    with pytest.raises(StateError, match="Stale"):
        pool._page(refs[0])


def test_prefix_complete_pages_tenant_and_policy_isolation():
    pool = PagedStatePool(block_size=2, max_blocks=12)
    cache = PrefixCache(pool)
    identity = PrefixIdentity("artifact-a", processor="processor-a", tenant="alice")
    sequence = pool.create(identity.fingerprint())
    pool.append(sequence, state(5))
    cache.publish(identity, [1, 2, 3, 4, 5], sequence)
    hit = cache.lookup(identity, [1, 2, 3, 4, 9])
    assert hit.length == 4
    assert cache.lookup(replace(identity, tenant="bob"), [1, 2, 3, 4, 9]).length == 0
    assert (
        cache.lookup(replace(identity, policy_artifact_id="artifact-b"), [1, 2, 3, 4, 9]).length
        == 0
    )
    assert (
        cache.lookup(replace(identity, multimodal_digest="new-image"), [1, 2, 3, 4, 9]).length == 0
    )
    assert cache.lookup(identity, [1, 2, 3, 4]).length == 2
    pool.release(hit)
    pool.release(sequence)
    cache.clear()


def test_codec_rejects_recurrent_and_mismatched_shapes():
    with pytest.raises(ValueError):
        KVStateCodec(kind="recurrent")
    codec = KVStateCodec(kind="mla_latent")
    leaves, tree, dims, length = codec.flatten(state(4))
    assert [leaf.shape[-1] for leaf in leaves] == [3, 5]
    assert codec.unflatten(leaves, tree)[0][1].shape == (1, 1, 4, 5)
    with pytest.raises(StateError):
        codec.flatten(((torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 3, 5)),))


def test_sampling_raw_and_behavior_probabilities():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    config = SamplingConfig(temperature=0.5, top_k=2, top_p=0.8)
    raw, behavior = distributions(logits, config)
    torch.testing.assert_close(raw, logits.log_softmax(-1))
    assert behavior.exp().sum() == 1
    assert not torch.allclose(raw, behavior)
    greedy = sample_token(logits, SamplingConfig(temperature=0), torch.Generator().manual_seed(0))
    assert greedy.token_id == 3 and greedy.behavior_logprob == 0 and greedy.raw_model_logprob < 0
    with pytest.raises(ValueError):
        distributions(torch.tensor([torch.nan, 1.0]), config)
    with pytest.raises(ValueError):
        distributions(torch.tensor([-torch.inf, -torch.inf]), config)


def test_speculative_rejection_matches_target_distribution():
    p, q = (
        torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64),
        torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64),
    )
    generator = torch.Generator().manual_seed(71)
    counts = torch.zeros(3)
    for _ in range(5000):
        proposed = int(torch.multinomial(q, 1, generator=generator))
        actual, _ = speculative_accept(p, q, proposed, generator)
        counts[actual] += 1
    torch.testing.assert_close(counts / 5000, p.float(), atol=0.025, rtol=0)


def test_native_runner_cache_matches_full_forward_and_saves_prefill():
    original = model().eval()
    runner = ModelRunner(original, policy_artifact_id="test-native", block_size=2, max_blocks=32)
    sequence = runner.pool.create("domain")
    prompt = [1, 4, 8, 2]
    first = runner.forward_batch([sequence], [prompt])[0]
    with torch.no_grad():
        expected = original(torch.tensor([prompt])).logits[0, -1]
    torch.testing.assert_close(first, expected, atol=1e-6, rtol=1e-5)
    second = runner.forward_batch([sequence], [[6]])[0]
    with torch.no_grad():
        expected = original(torch.tensor([prompt + [6]])).logits[0, -1]
    torch.testing.assert_close(second, expected, atol=1e-6, rtol=1e-5)
    assert runner.input_tokens_computed == 5
    assert sequence.length == 5

    with torch.no_grad():
        next(original.parameters()).add_(100.0)
    assert not torch.equal(next(original.parameters()), next(runner.model.parameters()))


def test_native_dynamic_batch_and_chunked_prefill_parity():
    async def exercise():
        runner = ModelRunner(model(), policy_artifact_id="native", block_size=2, max_blocks=64)
        engine = InferenceEngine(runner, max_active=4, max_batch_tokens=8, prefill_chunk_size=2)
        first = await engine.submit(
            [1, 2, 3, 4, 5], SamplingConfig(max_new_tokens=4, temperature=0)
        )
        second = await engine.submit(
            [1, 2, 3, 4, 5], SamplingConfig(max_new_tokens=4, temperature=0)
        )
        results = await asyncio.gather(first.collect(), second.collect())
        assert results[0].token_ids == results[1].token_ids
        assert results[0].stop_reason == "length"
        assert len(results[0].metrics()["itl_seconds"]) == 3
        fresh = await engine.submit(
            [1, 2, 3, 4, 5], SamplingConfig(max_new_tokens=4, temperature=0)
        )
        result = await fresh.collect()
        assert result.prefix_hit_tokens == 4
        assert result.token_ids == results[0].token_ids
        await engine.close()
        assert runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_admission_timeout_backpressure_and_shutdown():
    async def exercise():
        engine = InferenceEngine(
            ModelRunner(model(), policy_artifact_id="native"),
            max_active=1,
            max_queued=1,
            max_output_events=1,
        )
        slow = await engine.submit([1], SamplingConfig(max_new_tokens=8))
        timed = await engine.submit([2], SamplingConfig(max_new_tokens=4), timeout_s=0.000001)
        with pytest.raises(OverloadedError):
            await engine.submit([3])
        assert (await timed.result()).stop_reason == "timeout"
        result = await slow.result()
        assert result.stop_reason == "backpressure" and len(result.token_ids) == 1
        assert result.metrics()["tpot_seconds"] is None
        await engine.close()
        with pytest.raises(RuntimeError):
            await engine.submit([1])

    asyncio.run(exercise())


def test_cancel_waits_for_native_worker_acknowledgment():
    class BlockingRunner(ModelRunner):
        def forward_batch(self, *args):
            entered.set()
            released.wait(timeout=3)
            return super().forward_batch(*args)

    entered, released = threading.Event(), threading.Event()

    async def exercise():
        runner = BlockingRunner(model(), policy_artifact_id="native")
        engine = InferenceEngine(runner)
        handle = await engine.submit([1, 2], SamplingConfig(max_new_tokens=5))
        assert await asyncio.to_thread(entered.wait, 2)
        cancel = asyncio.create_task(handle.cancel())
        await asyncio.sleep(0.01)
        assert not cancel.done()
        released.set()
        result = await cancel
        assert result.stop_reason == "cancelled" and not result.token_ids
        assert result.metrics()["ttft_seconds"] is None
        await engine.close()
        assert runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_model_failure_is_structured_and_does_not_poison_worker():
    class OnceBrokenRunner(ModelRunner):
        failed = False

        def forward_batch(self, *args):
            if not self.failed:
                self.failed = True
                raise RuntimeError("secret should never enter response")
            return super().forward_batch(*args)

    async def exercise():
        engine = InferenceEngine(OnceBrokenRunner(model(), policy_artifact_id="native"))
        handle = await engine.submit([1])
        result = await handle.collect()
        assert result.error_code == "model_execution_failed"
        next_handle = await engine.submit([1], SamplingConfig(max_new_tokens=2))
        assert (await next_handle.collect()).stop_reason == "length"
        await engine.close()

    asyncio.run(exercise())
