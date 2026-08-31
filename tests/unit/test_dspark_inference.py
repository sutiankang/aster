from copy import deepcopy

import pytest
import torch

from aster.inference import ModelRunner, SamplingConfig
from aster.inference.dspark import DSparkDecoder, confident_prefix_length
from aster.models import CausalLM, Qwen3Config
from aster.models.dspark import DSparkConfig, DSparkDraft, block_attention_mask


def models(*, uniform=False):
    torch.set_num_threads(1)
    torch.manual_seed(171)
    target = CausalLM(
        Qwen3Config(
            vocab_size=13,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    if uniform:
        with torch.no_grad():
            target.lm_head.weight.zero_()
    draft = (
        DSparkDraft(
            DSparkConfig(
                target.config,
                target_layer_ids=(-1, 1),
                num_anchors=1,
                block_size=3,
                markov_rank=4,
                markov_head_type="rnn",
            )
        )
        .initialize_from_target(target)
        .eval()
    )
    if uniform:
        with torch.no_grad():
            draft.markov_head.markov_w2.weight.zero_()
    return target, draft


def decoder(target, draft, **kwargs):
    runner = ModelRunner(target, policy_artifact_id="target_test13_v1", block_size=2, max_blocks=64)
    return DSparkDecoder(
        runner,
        draft,
        draft_policy_artifact_id="draft_test13_v1",
        vocabulary_fingerprint="vocab13_v1",
        block_size=2,
        max_blocks=64,
        **kwargs,
    )


def test_dspark_incremental_context_projection_matches_full_backbone_and_has_no_query_kv():
    target, draft = models()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        states = target(ids, output_hidden_states=True).hidden_states
    features = torch.cat((states[0], states[2]), -1)
    cache = None
    previous = 0
    for anchor in (0, 2, 5):
        noise = torch.tensor([[int(ids[0, anchor]), 0, 0]])
        actual, cache = draft.backbone_cached(noise, features[:, previous:anchor], state=cache)
        positions = torch.arange(anchor + 3)[None]
        expected = draft.backbone(
            noise,
            features[:, :anchor],
            positions,
            torch.ones(1, 1, 3, anchor + 3, dtype=torch.bool),
        )
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
        assert all(k.shape[2] == v.shape[2] == anchor for k, v in cache)
        previous = anchor


@pytest.mark.parametrize("threshold", [0.0, 0.6, 1.0])
def test_dspark_greedy_exact_target_tokens_logprob_and_all_page_release(threshold):
    target, draft = models()
    runtime = decoder(target, draft, confidence_threshold=threshold)
    prompt = [1, 4, 2]
    expected, logprobs = [], []
    with torch.no_grad():
        for _ in range(11):
            logits = target(torch.tensor([prompt + expected])).logits[0, -1]
            token = int(logits.argmax())
            expected.append(token)
            logprobs.append(float(logits.log_softmax(-1)[token]))
    events = []
    result = runtime.generate(
        prompt, SamplingConfig(max_new_tokens=11, temperature=0), on_token=events.append
    )
    assert result.token_ids == tuple(expected) and len(events) == 11
    torch.testing.assert_close(
        torch.tensor(result.raw_model_logprobs), torch.tensor(logprobs), atol=4e-7, rtol=2e-6
    )
    assert all(x == 0 for x in result.behavior_logprobs)
    assert runtime.pool.used_blocks == runtime.target.pool.used_blocks == 0
    if threshold == 1:
        assert result.draft_token_count == 0 and result.dspark_stats["bonus_tokens"] == 11
    else:
        assert result.dspark_stats["target_verification_calls"] <= 11


def test_dspark_accept_all_uses_bonus_and_projects_only_new_context():
    target, draft = models(uniform=True)
    runtime = decoder(target, draft)
    result = runtime.generate([2, 3], SamplingConfig(max_new_tokens=9, temperature=0))
    assert result.token_ids == (0,) * 9 and result.accepted_draft_tokens == (0,) * 7
    assert (
        result.dspark_stats["target_verification_calls"] == 3
        and result.dspark_stats["bonus_tokens"] == 2
    )
    assert result.dspark_stats["projected_context_tokens"] == 9
    assert runtime.target.forward_calls == 4
    assert runtime.pool.used_blocks == runtime.target.pool.used_blocks == 0


def test_dspark_confidence_schedule_is_per_step_and_never_waives_target_verification():
    assert confident_prefix_length(torch.logit(torch.tensor([0.8, 0.8, 0.8])), 0.7) == 3
    assert confident_prefix_length(torch.logit(torch.tensor([0.8, 0.6, 0.9])), 0.7) == 1
    assert confident_prefix_length(torch.tensor([-100.0, 100.0]), 0.0) == 2
    target, draft = models()
    with torch.no_grad():
        draft.confidence_head.bias.fill_(100.0)
    runtime = decoder(target, draft, confidence_threshold=0.9)
    result = runtime.generate([1], SamplingConfig(max_new_tokens=8, temperature=0))
    assert result.dspark_stats["target_verification_calls"] > 0
    assert result.dspark_stats["rejected_blocks"] > 0


def test_dspark_callback_failure_cancellation_and_eos_release_both_pools():
    target, draft = models(uniform=True)
    runtime = decoder(target, draft)
    cancelled = runtime.generate([1, 2], cancelled=lambda: True)
    assert cancelled.stop_reason == "cancelled" and not cancelled.token_ids
    eos = runtime.generate([1], SamplingConfig(max_new_tokens=4, temperature=0, eos_token_ids=(0,)))
    assert eos.stop_reason == "eos" and eos.token_ids == (0,)

    def broken(event):
        raise RuntimeError("consumer disconnected")

    with pytest.raises(RuntimeError, match="disconnected"):
        runtime.generate([1], on_token=broken)
    assert runtime.pool.used_blocks == runtime.target.pool.used_blocks == 0


def test_dspark_wrong_target_identity_rejected_even_with_same_config():
    target, draft = models()
    other = deepcopy(target)
    with torch.no_grad():
        other.lm_head.weight.add_(0.1)
    with pytest.raises(ValueError, match="another target"):
        decoder(other, draft)


def test_dspark_serving_precision_does_not_depend_on_callers_autocast():
    target, draft = models()
    runtime = decoder(target, draft)
    settings = SamplingConfig(max_new_tokens=6, temperature=0.7, seed=99)
    expected = runtime.generate([1, 2], settings)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = runtime.generate([1, 2], settings)
    assert (
        actual.token_ids == expected.token_ids
        and actual.raw_model_logprobs == expected.raw_model_logprobs
    )


def test_dspark_last_anchor_at_exact_target_capacity_needs_no_extra_verification_token():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    target = CausalLM(
        Qwen3Config(
            vocab_size=13,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4,
        )
    ).eval()
    draft = DSparkDraft(
        DSparkConfig(target.config, block_size=1, num_anchors=1, markov_rank=2)
    ).initialize_from_target(target)
    runtime = decoder(target, draft)
    with torch.no_grad():
        expected = int(target(torch.tensor([[1, 2, 3, 4]])).logits[0, -1].argmax())
    result = runtime.generate([1, 2, 3, 4], SamplingConfig(max_new_tokens=1, temperature=0))
    assert result.token_ids == (expected,) and result.draft_token_count == 0
    assert result.dspark_stats["bonus_tokens"] == 1 and not runtime.target.pool.used_blocks
