import asyncio
from dataclasses import replace
import json

import pytest
import torch

from aster.inference import (
    PagedAttentionRunner,
    InferenceEngine,
    SamplingConfig,
    HTTPServer,
    PrefixIdentity,
)
from aster.models import build_model, LlamaConfig, Qwen2Config, Qwen3Config, Gemma4TextConfig


def model(configuration=LlamaConfig, **options):
    torch.manual_seed(9)
    torch.set_num_threads(1)
    return build_model(
        configuration(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            **options,
        )
    ).eval()


def runner(native, **options):
    return PagedAttentionRunner(
        native,
        policy_artifact_id="native-paged-test",
        backend="torch_online_paged",
        block_size=3,
        max_blocks=80,
        query_block_size=2,
        key_block_size=3,
        **options,
    )


def forbid(*args, **kwargs):
    raise AssertionError("Paged runner must not materialize KV or call complete attention forward")


@pytest.mark.parametrize(
    "configuration,options",
    [
        (LlamaConfig, {}),
        (Qwen2Config, {}),
        (Qwen3Config, {}),
        (Qwen2Config, {"use_sliding_window": True, "sliding_window": 4, "max_window_layers": 1}),
        (
            Qwen3Config,
            {"sliding_window": 4, "layer_types": ("sliding_attention", "full_attention")},
        ),
    ],
)
def test_native_logits_page_boundaries_long_prefix_chunk_and_padding(configuration, options):
    native = model(configuration, **options)
    paged = runner(native)
    paged.pool.materialize = forbid
    for layer in paged.model.model.layers:
        layer.self_attn.forward = forbid
    sequences = [paged.pool.create("same-domain") for _ in range(2)]
    inputs = torch.tensor(
        [[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 4, 6, 8], [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 3, 5, 7]]
    )
    padding = torch.ones_like(inputs)
    padding[0, :2] = 0
    padding[1, 3] = 0
    with torch.no_grad():
        expected = native(inputs, attention_mask=padding).logits
    outputs = []
    for a, b in [(0, 5), (5, 6), (6, 10), (10, 13)]:
        output = paged.forward_batch(
            sequences,
            inputs[:, a:b].tolist(),
            return_all_logits=True,
            padding_masks=padding[:, a:b],
        )
        outputs.append(torch.stack(output))
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=2e-5, atol=2e-6)
    assert paged.input_tokens_computed == inputs.numel() and sequences[0].length == 13
    assert paged.attention_work.max_score_elements <= 4 * 2 * 3
    assert paged.attention_work.key_tiles > paged.forward_calls


def test_actual_model_shared_partial_page_cow_rollback_and_branch_logits():
    native = model(Qwen3Config)
    paged = runner(native)
    root = paged.pool.create("bound-domain")
    prefix = [1, 4, 7, 10]
    paged.forward_batch([root], [prefix])
    branch = paged.pool.fork(root)
    original_tail = root.pages[-1]
    left = paged.forward_batch([root], [[2, 3]])[0]
    right = paged.forward_batch([branch], [[8, 9]])[0]
    assert root.pages[-1] != original_tail == branch.pages[-1]
    with torch.no_grad():
        torch.testing.assert_close(
            left, native(torch.tensor([prefix + [2, 3]])).logits[0, -1], atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            right, native(torch.tensor([prefix + [8, 9]])).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    paged.pool.truncate(root, 3)
    rewound = paged.forward_batch([root], [[6, 12]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            rewound,
            native(torch.tensor([prefix[:3] + [6, 12]])).logits[0, -1],
            atol=2e-6,
            rtol=2e-5,
        )
    paged.pool.release(root)
    paged.pool.release(branch)
    assert paged.pool.used_blocks == 0


def greedy(native, prompt, count):
    tokens, generated = list(prompt), []
    with torch.no_grad():
        for _ in range(count):
            token = int(native(torch.tensor([tokens])).logits[0, -1].argmax())
            generated.append(token)
            tokens.append(token)
    return tuple(generated)


def test_engine_streaming_prefix_cache_and_concurrent_greedy_without_dense_kv():
    native = model(Qwen2Config, use_sliding_window=True, sliding_window=5)
    paged = runner(native)
    paged.pool.materialize = forbid

    async def exercise():
        engine = InferenceEngine(paged, max_active=2, max_batch_tokens=6, prefill_chunk_size=4)
        prompt = [1, 3, 5, 7, 9, 11, 13]
        try:
            first = await (
                await engine.submit(prompt, SamplingConfig(temperature=0, max_new_tokens=5))
            ).collect()
            assert first.token_ids == greedy(native, prompt, 5)
            handles = [
                await engine.submit(prompt, SamplingConfig(temperature=0, max_new_tokens=5))
                for _ in range(2)
            ]
            results = await asyncio.gather(*(handle.collect() for handle in handles))
            assert all(
                result.token_ids == first.token_ids and result.prefix_hit_tokens >= 3
                for result in results
            )

            async with HTTPServer(engine) as server:
                reader, writer = await asyncio.open_connection(*server.address)
                body = json.dumps(
                    {"prompt_token_ids": prompt, "max_tokens": 3, "temperature": 0, "stream": True}
                ).encode()
                writer.write(
                    f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 10)
                writer.close()
                await writer.wait_closed()
                assert b"data: [DONE]" in response
                values = [
                    json.loads(line[6:])
                    for line in response.splitlines()
                    if line.startswith(b"data: {")
                ]
                assert (
                    tuple(event["aster"]["token_id"] for event in values[:-1])
                    == first.token_ids[:3]
                )
        finally:
            await engine.close()
        assert paged.pool.used_blocks == 0

    asyncio.run(exercise())


def test_backend_and_family_selection_are_explicit():
    native = model()
    with pytest.raises(ValueError, match="explicit backend"):
        PagedAttentionRunner(native, policy_artifact_id="x", backend="cuda_paged")
    with pytest.raises(ValueError, match="Llama/Qwen2/Qwen3"):
        PagedAttentionRunner(
            build_model(Gemma4TextConfig()), policy_artifact_id="x", backend="torch_online_paged"
        )


def test_long_prefix_real_dispatch_has_no_kv_cat_and_scores_are_bounded():
    from torch.utils._python_dispatch import TorchDispatchMode

    native = model()
    paged = runner(native)
    sequence = paged.pool.create("long-prefix-domain")
    prompt = [1] + [2 + i % 20 for i in range(89)]
    continuation = [3 + i % 17 for i in range(41)]
    with torch.no_grad():
        expected = native(torch.tensor([prompt + continuation])).logits[0, -len(continuation) :]
    paged.pool.materialize = forbid
    paged.forward_batch([sequence], [prompt])

    class NoHistoryConcat(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.cat.default:
                tensors, dimension = args[0], (args[1] if len(args) > 1 else 0)

                assert not (tensors and tensors[0].ndim == 4 and dimension in {2, -2})
            return func(*args, **(kwargs or {}))

    with NoHistoryConcat():
        actual = paged.forward_batch([sequence], [continuation], return_all_logits=True)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert paged.attention_work.max_score_elements <= 4 * 2 * 3
    assert paged.attention_work.max_score_elements < 4 * 41 * 131


def test_paged_scheduler_oom_preemption_recomputes_and_terminates():
    native = model()

    async def exercise():
        paged = PagedAttentionRunner(
            native,
            policy_artifact_id="paged-capacity",
            backend="torch_online_paged",
            block_size=2,
            max_blocks=4,
            query_block_size=2,
            key_block_size=2,
        )
        paged.pool.materialize = forbid
        engine = InferenceEngine(paged, max_active=2, max_batch_tokens=16, prefill_chunk_size=8)
        prompt, sampling = [1, 2, 3, 4], SamplingConfig(max_new_tokens=4, temperature=0)
        try:
            handles = [await engine.submit(prompt, sampling) for _ in range(2)]
            results = await asyncio.wait_for(
                asyncio.gather(*(handle.collect() for handle in handles)), 10
            )
            assert all(
                result.token_ids == greedy(native, prompt, 4) and result.stop_reason == "length"
                for result in results
            )
            assert sum(result.preemption_count for result in results) > 0
            assert paged.input_tokens_computed > 14
        finally:
            await engine.close()
        assert paged.pool.used_blocks == 0
        undersized = PagedAttentionRunner(
            native,
            policy_artifact_id="too-small",
            backend="torch_online_paged",
            block_size=2,
            max_blocks=1,
        )
        failed = InferenceEngine(undersized)
        try:
            result = await asyncio.wait_for(
                (await failed.submit([1, 2, 3], SamplingConfig(max_new_tokens=1))).collect(), 5
            )
            assert result.error_code == "cache_capacity" and result.stop_reason == "error"
        finally:
            await failed.close()
        assert undersized.pool.used_blocks == 0

    asyncio.run(exercise())


def test_paged_speculative_target_verification_uses_chunk_and_rolls_back():
    from aster.inference import SpeculativeDecoder, ModelRunner

    native = model()
    target = runner(native)
    target.pool.materialize = forbid

    draft_model = model()
    with torch.no_grad():
        draft_model.lm_head.weight.neg_()
    draft = ModelRunner(draft_model, policy_artifact_id="different-draft", block_size=3)
    decoder = SpeculativeDecoder(
        target, draft, num_draft_tokens=3, vocabulary_fingerprint="native-24-token-v1"
    )
    result = decoder.generate([1, 3, 5], SamplingConfig(max_new_tokens=7, temperature=0))
    assert result.token_ids == greedy(native, [1, 3, 5], 7)
    assert result.draft_token_count > len(result.accepted_draft_tokens)
    assert target.pool.used_blocks == draft.pool.used_blocks == 0


def test_verified_local_model_artifact_loads_same_paged_execution(tmp_path):
    from aster.core import ArtifactStore
    from aster.models import load_model

    native = model()
    native.save_pretrained(tmp_path / "model")
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.publish(
        tmp_path / "model", kind="native_model", metadata={"test": "paged-attention"}
    )
    paged = PagedAttentionRunner.from_artifact(
        store, artifact.id, loader=load_model, backend="torch_online_paged", block_size=2
    )
    paged.pool.materialize = forbid
    sequence = paged.pool.create("verified-artifact-domain")
    logits = paged.forward_batch([sequence], [[1, 2, 3]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            logits, native(torch.tensor([[1, 2, 3]])).logits[0, -1], atol=2e-6, rtol=2e-5
        )
    assert paged.policy_artifact_id == artifact.id
