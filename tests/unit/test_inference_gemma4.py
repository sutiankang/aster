import asyncio
from dataclasses import replace
import pytest
import torch

from aster.models import Gemma4TextConfig, Gemma4Config, build_model, pack_gemma4_images
from aster.inference import (
    Gemma4SnapshotRunner,
    KVStateCodec,
    InferenceEngine,
    PrefixIdentity,
    SamplingConfig,
    StateError,
    CacheCapacityError,
    StateArchive,
    StatefulTokenRunner,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def text_model():
    torch.manual_seed(52)
    return build_model(Gemma4TextConfig()).eval()


def visual_model(video=False):
    torch.manual_seed(22)
    config = Gemma4Config(
        text_config=replace(Gemma4TextConfig(), use_bidirectional_attention="vision")
    )
    model = build_model(config).eval()
    pixels = torch.rand(2 if video else 1, 3, 8, 8)
    packed = pack_gemma4_images(pixels, config.vision_config)
    count = 8 if video else 4
    tokens = [1] + [config.video_token_id if video else config.image_token_id] * count + [4, 5, 6]
    if video:
        inputs = {
            "pixel_values_videos": packed["pixel_values"][None],
            "video_position_ids": packed["pixel_position_ids"][None],
        }
    else:
        inputs = {
            "pixel_values": packed["pixel_values"],
            "image_position_ids": packed["pixel_position_ids"],
        }
    inputs["mm_token_type_ids"] = torch.tensor([[0] + [2 if video else 1] * count + [0, 0, 0]])
    return model, tokens, inputs


def dense_greedy(model, prompt, count, modality_inputs=None):
    context, output = list(prompt), []
    with torch.no_grad():
        for _ in range(count):
            inputs = dict(modality_inputs or {})
            if "mm_token_type_ids" in inputs:
                inputs["mm_token_type_ids"] = torch.nn.functional.pad(
                    inputs["mm_token_type_ids"], (0, len(context) - len(prompt))
                )
            logits = model(torch.tensor([context]), **inputs).logits[0, -1]
            token = int(logits.argmax())
            output.append(token)
            context.append(token)
    return tuple(output)


def test_gemma4_shared_owners_real_snapshot_cow_and_replay_rollback():
    model = text_model()
    runner = Gemma4SnapshotRunner(model, policy_artifact_id="gemma4:fixture")
    tokens = [1, 2, 3, 4, 5, 6]
    source, identity = runner.create_sequence(tokens)
    runner.forward_batch([source], [tokens])
    cached = runner.pool.materialize(source)
    assert cached.kind == "gemma4_shared_kv" and len(cached.layers) == 2
    assert cached.layers[0][0].shape == (1, 2, 3, 8)
    assert cached.layers[1][0].shape == (1, 1, 6, 16)
    with pytest.raises(ValueError):
        KVStateCodec(kind=cached.kind)
    with pytest.raises(ValueError):
        cached.truncate(3)
    branch = runner.pool.fork(source)
    assert branch.snapshot == source.snapshot and runner.pool.used_blocks == 1
    actual = runner.forward_batch([branch], [[7, 8]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            actual, model(torch.tensor([tokens + [7, 8]])).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    assert branch.snapshot != source.snapshot and runner.pool.cow_commits == 1
    original = runner.pool.materialize(source)
    torch.testing.assert_close(original.layers[0][0], cached.layers[0][0], rtol=0, atol=0)
    before = runner.input_tokens_computed
    runner.pool.truncate(branch, 2)
    assert runner.input_tokens_computed - before == 2 and runner.pool.replay_rollbacks == 1
    assert runner.pool.materialize(branch).seen_tokens == 2
    actual = runner.forward_batch([branch], [[9]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            actual, model(torch.tensor([[1, 2, 9]])).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    modified = runner.pool.materialize(source)
    modified.layers[0][0].zero_()
    assert runner.pool.materialize(source).layers[0][0].abs().sum() > 0
    runner.pool.release(branch)
    runner.pool.release(source)
    assert runner.pool.used_bytes == 0 and runner.modality_bytes == 0 and not runner._bindings


def test_gemma4_snapshot_transaction_and_inflight_owner_capacity():
    model = text_model()
    runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4", max_cache_bytes=1100)
    one, _ = runner.create_sequence([1, 2])
    two, _ = runner.create_sequence([3, 4])
    runner.forward_batch([one, two], [[1, 2], [3, 4]])
    assert runner.pool.used_bytes == 1024
    with pytest.raises(CacheCapacityError):
        runner.forward_batch([one, two], [[5], [6]])
    assert one.length == two.length == 2 and runner.pool.tokens(one) == (1, 2)
    runner.pool.release(two)
    with runner.pool.borrow(one):
        runner.pool.release(one)
        assert runner.pool.used_bytes > 0
    assert runner.pool.used_bytes == 0
    with pytest.raises(StateError):
        runner.pool.materialize(one)


def test_gemma4_engine_prefix_preemption_and_greedy_match_dense():
    async def execute():
        model = text_model()
        prompt = [1, 2, 3, 4, 5, 6]
        config = SamplingConfig(max_new_tokens=4, temperature=0)
        runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4", max_cache_bytes=1750)
        engine = InferenceEngine(runner, max_active=2, max_batch_tokens=4, prefill_chunk_size=2)
        handles = [await engine.submit(prompt, config) for _ in range(2)]
        results = await asyncio.gather(*(handle.collect() for handle in handles))
        expected = dense_greedy(model, prompt, 4)
        assert all(result.token_ids == expected and result.error_code is None for result in results)
        assert sum(result.preemption_count for result in results) > 0
        await engine.close()
        assert runner.pool.used_bytes == 0 and not runner._bindings
        spacious = Gemma4SnapshotRunner(model, policy_artifact_id="g4", max_cache_bytes=100000)
        engine = InferenceEngine(spacious, max_batch_tokens=4, prefill_chunk_size=2)
        first = await (await engine.submit(prompt, config)).collect()
        previous = spacious.input_tokens_computed
        second = await (await engine.submit(prompt, config)).collect()
        assert first.token_ids == second.token_ids == expected and second.prefix_hit_tokens == 5
        assert spacious.input_tokens_computed - previous == 4
        assert second.behavior_logprobs == (0.0,) * 4
        await engine.close()
        assert not spacious._bindings

    asyncio.run(execute())


@pytest.mark.parametrize("video", [False, True])
def test_gemma4_visual_real_prefix_rollback_inputs_frozen_and_layout_isolation(video):
    model, prompt, media = visual_model(video)
    runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4vl", processor_id="patches-fixed")
    original = {name: tensor.clone() for name, tensor in media.items()}
    sequence, identity = runner.create_sequence(prompt, modality_inputs=media)
    for name, value in media.items():
        if value.is_floating_point():
            value.zero_()

    actual = runner.forward_batch([sequence], [prompt])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            actual, model(torch.tensor([prompt]), **original).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    prefix = runner.create_prefix_cache(max_entries=4)
    prefix.publish(identity, prompt, sequence)
    owned = prefix.lookup(identity, prompt)
    assert owned.length == len(prompt) - 1
    actual = runner.forward_batch([owned], [[prompt[-1]]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            actual, model(torch.tensor([prompt]), **original).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    with pytest.raises(StateError, match="part"):
        runner.pool.truncate(owned, 2)
    assert owned.length == len(prompt)
    end = len(prompt) - 2
    runner.pool.truncate(owned, end)
    actual = runner.forward_batch([owned], [[11]])[0]
    shortened = {**original, "mm_token_type_ids": original["mm_token_type_ids"][:, : end + 1]}
    with torch.no_grad():
        torch.testing.assert_close(
            actual,
            model(torch.tensor([prompt[:end] + [11]]), **shortened).logits[0, -1],
            atol=2e-6,
            rtol=2e-5,
        )
    changed = {name: value.clone() for name, value in original.items()}
    key = "pixel_values_videos" if video else "pixel_values"
    changed[key] = 1 - changed[key]
    other, other_identity = runner.create_sequence(prompt, modality_inputs=changed)
    assert other_identity.multimodal_digest != identity.multimodal_digest
    miss = prefix.lookup(other_identity, prompt)
    assert miss.length == 0
    tenant, tenant_identity = runner.create_sequence(
        prompt, modality_inputs=original, identity=PrefixIdentity("g4vl", tenant="other")
    )
    tenant_miss = prefix.lookup(tenant_identity, prompt)
    assert tenant_miss.length == 0

    moved_prompt = [2] + prompt
    moved_media = {
        **original,
        "mm_token_type_ids": torch.nn.functional.pad(original["mm_token_type_ids"], (1, 0)),
    }
    moved, moved_identity = runner.create_sequence(moved_prompt, modality_inputs=moved_media)
    assert moved_identity.multimodal_digest != identity.multimodal_digest
    for item in (owned, sequence, other, miss, tenant, tenant_miss, moved):
        runner.pool.release(item)
    prefix.clear()
    assert runner.pool.used_bytes == runner.modality_bytes == 0 and not runner._bindings


@pytest.mark.parametrize("video", [False, True])
def test_gemma4_visual_scheduler_complete_prefill_and_greedy(video):
    async def execute():
        model, prompt, inputs = visual_model(video)
        runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4vl", processor_id="patches")
        engine = InferenceEngine(runner, max_batch_tokens=32, prefill_chunk_size=2)
        config = SamplingConfig(max_new_tokens=3, temperature=0)
        result = await (await engine.submit(prompt, config, modality_inputs=inputs)).collect()
        again = await (await engine.submit(prompt, config, modality_inputs=inputs)).collect()
        assert result.token_ids == again.token_ids == dense_greedy(model, prompt, 3, inputs)
        assert again.prefix_hit_tokens == len(prompt) - 1
        await engine.close()
        assert not runner._bindings
        constrained = InferenceEngine(
            Gemma4SnapshotRunner(model, policy_artifact_id="g4vl", processor_id="patches"),
            max_batch_tokens=2,
        )
        with pytest.raises(ValueError, match="visual block"):
            await constrained.submit(prompt, config, modality_inputs=inputs)
        await constrained.close()

    asyncio.run(execute())


def test_gemma4_opaque_archive_and_stateful_path_support_but_no_lossy_state():
    model = text_model()
    runner = StatefulTokenRunner(model, policy_artifact_id="g4")
    prefix = runner.forward(torch.tensor([[1, 2, 3, 4, 5]]))
    archive = StateArchive(max_bytes=10000)
    key = archive.put(prefix.state.native_state, identity="g4/processor/tenant")
    restored = archive.get(key, identity="g4/processor/tenant")
    with torch.no_grad():
        torch.testing.assert_close(
            model(torch.tensor([[6]]), state=restored).logits,
            model(torch.tensor([[1, 2, 3, 4, 5, 6]])).logits[:, -1:],
            atol=2e-6,
            rtol=2e-5,
        )
    with pytest.raises(StateError):
        archive.put(restored, identity="g4", quantize=True)


@pytest.mark.parametrize("video", [False, True])
def test_visual_preemption_replays_complete_media_and_leaves_no_bound_inputs(video):
    async def execute():
        model, prompt, inputs = visual_model(video)

        capacity = 384 + (len(prompt) + 2) * 128 + 64
        runner = Gemma4SnapshotRunner(
            model, policy_artifact_id="g4", processor_id="p", max_cache_bytes=capacity
        )
        engine = InferenceEngine(runner, max_active=2, max_batch_tokens=32, prefill_chunk_size=2)
        config = SamplingConfig(max_new_tokens=3, temperature=0)
        handles = [await engine.submit(prompt, config, modality_inputs=inputs) for _ in range(2)]
        results = await asyncio.gather(*(handle.collect() for handle in handles))
        expected = dense_greedy(model, prompt, 3, inputs)
        assert all(result.token_ids == expected and result.error_code is None for result in results)
        assert sum(result.preemption_count for result in results) > 0
        await engine.close()
        assert not runner._bindings and runner.modality_bytes == runner.pool.used_bytes == 0

    asyncio.run(execute())


def test_visual_binding_rejects_forged_identity_unknown_tokens_and_input_budget():
    model, prompt, inputs = visual_model()
    runner = Gemma4SnapshotRunner(
        model, policy_artifact_id="g4", processor_id="p", max_modality_bytes=1
    )
    with pytest.raises(CacheCapacityError):
        runner.create_sequence(prompt, modality_inputs=inputs)
    assert not runner._bindings
    runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4", processor_id="p")
    with pytest.raises(StateError):
        runner.create_sequence(
            prompt,
            modality_inputs=inputs,
            identity=PrefixIdentity("g4", multimodal_digest="fabricated"),
        )
    with pytest.raises(StateError):
        runner.create_sequence(
            prompt, modality_inputs=inputs, identity=PrefixIdentity("g4", processor="different")
        )
    with pytest.raises(ValueError):
        runner.sampling_context_ids([1, 90])
    assert runner.sampling_context_ids([1, 60, 61, 2]) == (1, 2)


def test_snapshot_metadata_cannot_relabel_cached_history_and_generic_vlm_generate():
    model = text_model()
    runner = Gemma4SnapshotRunner(model, policy_artifact_id="g4")
    state, _ = runner.create_sequence([1, 2, 3])
    runner.forward_batch([state], [[1, 2, 3]])
    original_identity = state.identity
    state.identity = "forged"
    with pytest.raises(StateError, match="metadata"):
        runner.pool.materialize(state)
    state.identity = original_identity
    state.length = 2
    with pytest.raises(StateError, match="metadata"):
        runner.pool.materialize(state)
    state.length = 3
    runner.pool.release(state)
    model, prompt, media = visual_model()
    typed = StatefulTokenRunner(model, policy_artifact_id="g4vl", processor_id="patches")
    result = typed.generate(
        prompt, SamplingConfig(max_new_tokens=2, temperature=0), modality_inputs=media
    )
    assert result.token_ids == dense_greedy(model, prompt, 2, media)
    assert result.sampling_transform_order[0].startswith("exclude_declared_out_of_vocab")
