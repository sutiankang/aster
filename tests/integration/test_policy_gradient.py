import asyncio
import copy

import pytest
import torch

from aster.core import digest_json
from aster.data import ByteTokenizer
from aster.inference import SamplingConfig
from aster.models import LlamaConfig, build_model
from aster.training import Trainer
from aster.methods.policy_gradient import OnPolicyRLMethod, RLOOObjective, leave_one_out_advantages
from aster.methods.supervised import sequence_logprobs


def config():
    return LlamaConfig(
        vocab_size=259,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )


def test_rloo_baseline_and_sequence_ratio_match_independent_equations():
    torch.manual_seed(9)
    torch.set_num_threads(1)
    rewards = torch.tensor([1.0, 7.0, 4.0, -1.0, 3.0])
    groups = torch.tensor([2, 0, 2, 0, 2])
    expected = torch.tensor([1.0 - 3.5, 8.0, 4.0 - 2.0, -8.0, 3.0 - 2.5])
    torch.testing.assert_close(leave_one_out_advantages(rewards, groups), expected)
    model = build_model(config())
    batch = {
        "input_ids": torch.tensor([[1, 4, 5, 6], [1, 8, 9, 0]]),
        "labels": torch.tensor([[-100, -100, 5, 6], [-100, 8, 9, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }
    logp, valid = sequence_logprobs(model, batch)
    batch["old_behavior_log_probs"] = logp.detach() - 0.03 * valid
    batch["advantages"] = torch.tensor([1.2, -0.7])
    term = RLOOObjective()(model, batch)
    ratio = torch.exp((logp * valid).sum(-1) - (batch["old_behavior_log_probs"] * valid).sum(-1))
    oracle = -torch.minimum(
        ratio * batch["advantages"], ratio.clamp(0.8, 1.2) * batch["advantages"]
    ).mean()
    torch.testing.assert_close(term.numerator / term.denominator, oracle)
    got = torch.autograd.grad(
        term.numerator / term.denominator, tuple(model.parameters()), retain_graph=True
    )
    want = torch.autograd.grad(oracle, tuple(model.parameters()))
    for left, right in zip(got, want):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("algorithm", ["rloo", "grpo"])
def test_online_native_generation_reward_update_and_exact_resume(tmp_path, algorithm):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    tokenizer = ByteTokenizer()
    reference = build_model(config())
    teacher_before = copy.deepcopy(reference.state_dict())
    model = copy.deepcopy(reference)
    engine = Trainer(model, lr=0.0003, accumulation_steps=2)
    method = OnPolicyRLMethod(
        engine,
        reference,
        tokenizer,
        reward=lambda row, group: float(sum(row.completion_ids) % 7),
        reward_id="test-token-sum-mod7-v1",
        reference_tokenizer_fingerprint=digest_json(tokenizer.to_dict()),
        algorithm=algorithm,
        group_size=3,
        max_prompt_tokens=12,
    )
    prompts = [tokenizer.encode("Q"), tokenizer.encode("R")]
    sampling = SamplingConfig(max_new_tokens=3, seed=27)
    initial = copy.deepcopy(model.state_dict())
    result = asyncio.run(method.update(prompts, sampling=sampling))
    assert result.updated and method.updates == 1 and len(method.last_records) == 6
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    assert all(record["reward"] is not None for record in method.last_records)
    for key, value in teacher_before.items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    engine.save_checkpoint(tmp_path / "checkpoint")
    asyncio.run(method.update(prompts, sampling=sampling))
    expected = copy.deepcopy(model.state_dict())
    trajectories = copy.deepcopy(method.last_records)
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    asyncio.run(method.update(prompts, sampling=sampling))
    assert method.last_records == trajectories
    for key, value in expected.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="untruncated"):
        asyncio.run(method.update(prompts, sampling=SamplingConfig(temperature=0.0)))
