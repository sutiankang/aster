import asyncio
import pytest
import torch
from torch import nn

from aster.core import TokenOutput
from aster.models import build_model, LlamaConfig, DeepSeekV3Config
from aster.inference import (
    ModelRunner,
    InferenceEngine,
    SamplingConfig,
    KVStateCodec,
    SpeculativeDecoder,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    torch.manual_seed(44)
    return build_model(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )


def greedy_baseline(model, prompt, count):
    result, context = [], list(prompt)
    with torch.no_grad():
        for _ in range(count):
            token = int(model(torch.tensor([context])).logits[0, -1].argmax())
            context.append(token)
            result.append(token)
    return tuple(result)


def test_speculative_native_all_accept_and_fewer_target_calls():
    model = tiny().eval()
    target = ModelRunner(model, policy_artifact_id="target", block_size=4)
    draft = ModelRunner(model, policy_artifact_id="draft", block_size=4)
    decoder = SpeculativeDecoder(
        target, draft, num_draft_tokens=4, vocabulary_fingerprint="same-native-vocab"
    )
    events = []
    result = decoder.generate(
        [1, 2, 3], SamplingConfig(max_new_tokens=9, temperature=0), on_token=events.append
    )
    assert result.token_ids == greedy_baseline(model, [1, 2, 3], 9)
    assert result.accepted_draft_tokens == result.token_ids
    assert result.draft_token_count == 9 and target.forward_calls == 4
    assert tuple(event.token_id for event in events) == result.token_ids
    assert target.pool.used_blocks == draft.pool.used_blocks == 0


class ConstantModel(nn.Module):
    def __init__(self, favored):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.favored = favored

    def forward(
        self, input_ids, *, state=None, use_cache=False, attention_mask=None, position_ids=None
    ):
        previous = state[0][0] if state is not None else torch.zeros(input_ids.shape[0], 1, 0, 1)
        current = torch.cat((previous, input_ids[:, None, :, None].float()), dim=2)
        scores = torch.zeros((*input_ids.shape, 4))
        scores[:, :, self.favored] = 4.0
        return TokenOutput(scores, ((current, current.clone()),))


def test_speculative_rejection_discards_draft_and_rolls_back_real_pages():
    target = ModelRunner(ConstantModel(1), policy_artifact_id="target", block_size=2)
    draft = ModelRunner(ConstantModel(3), policy_artifact_id="draft", block_size=2)
    decoder = SpeculativeDecoder(
        target, draft, num_draft_tokens=3, vocabulary_fingerprint="four-token"
    )
    result = decoder.generate([0], SamplingConfig(max_new_tokens=5, temperature=0))
    assert result.token_ids == (1,) * 5 and not result.accepted_draft_tokens
    assert all(value == 0 for value in result.behavior_logprobs)
    assert result.draft_token_count > len(result.token_ids)
    assert target.pool.used_blocks == draft.pool.used_blocks == 0


def test_capacity_preemption_recomputes_without_changing_committed_tokens():
    async def exercise():
        model = tiny().eval()
        runner = ModelRunner(model, policy_artifact_id="native", block_size=2, max_blocks=4)
        engine = InferenceEngine(runner, max_active=2, max_batch_tokens=16, prefill_chunk_size=8)
        config = SamplingConfig(max_new_tokens=4, temperature=0)
        first = await engine.submit([1, 2, 3, 4], config)
        second = await engine.submit([1, 2, 3, 4], config)
        results = await asyncio.wait_for(asyncio.gather(first.collect(), second.collect()), 10)
        expected = greedy_baseline(model, [1, 2, 3, 4], 4)
        assert all(
            result.token_ids == expected and result.stop_reason == "length" for result in results
        )
        assert sum(result.preemption_count for result in results) >= 1
        assert runner.input_tokens_computed > 14
        await engine.close()
        assert runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_single_request_larger_than_cache_fails_without_infinite_preempt():
    async def exercise():
        engine = InferenceEngine(
            ModelRunner(tiny(), policy_artifact_id="native", block_size=2, max_blocks=1)
        )
        request = await engine.submit([1, 2, 3], SamplingConfig(max_new_tokens=1))
        result = await asyncio.wait_for(request.collect(), 3)
        assert result.error_code == "cache_capacity"
        await engine.close()

    asyncio.run(exercise())
