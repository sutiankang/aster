import torch
import pytest

from aster.agents.agent_rl import AgentPolicyObjective, collate_agent_trajectories
from aster.models import LlamaConfig, build_model
from aster.methods.supervised import sequence_logprobs


def fixture():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = build_model(
        LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )

    def trace(prompt, action):
        return {
            "prompt_token_ids": prompt,
            "action_token_ids": action,
            "loss_mask": [0] * len(prompt) + [1] * len(action),
            "raw_model_logprobs": [-1.0] * len(action),
            "behavior_logprobs": [-1.0] * len(action),
        }

    records = [
        {"traces": [trace([1, 2], [3, 4]), trace([1, 6, 4], [5])]},
        {"traces": [trace([1], [4, 5, 6])]},
    ]
    batch = collate_agent_trajectories(records, pad_token_id=0, device="cpu")
    current, mask = sequence_logprobs(model, batch)
    batch["old_behavior_log_probs"] = current.detach() - 0.07 * mask
    batch["reference_log_probs"] = current.detach() + 0.04 * mask
    batch["advantages"] = torch.tensor([1.3, -0.7])
    return model, batch


@pytest.mark.parametrize("algorithm", ["rloo", "grpo"])
def test_multiturn_objective_is_trajectory_not_decision_mean(algorithm):
    model, batch = fixture()
    logp, valid = sequence_logprobs(model, batch)
    difference = (logp - batch["old_behavior_log_probs"]) * valid
    advantage = batch["advantages"]
    if algorithm == "rloo":
        ratios = torch.stack([difference[:2].sum().exp(), difference[2:].sum().exp()])
        oracle = -torch.minimum(ratios * advantage, ratios.clamp(0.8, 1.2) * advantage).mean()
    else:
        row_advantage = advantage[torch.tensor([0, 0, 1])][:, None]
        ratio = difference.exp()
        delta = batch["reference_log_probs"] - logp
        values = (
            -torch.minimum(ratio * row_advantage, ratio.clamp(0.8, 1.2) * row_advantage)
            + 0.03 * (delta.exp() - 1 - delta)
        ) * valid
        oracle = (values[:2].sum() / valid[:2].sum() + values[2:].sum() / valid[2:].sum()) / 2
    got = AgentPolicyObjective(algorithm, kl_weight=0.03)(model, batch).mean
    torch.testing.assert_close(got, oracle)
    actual = torch.autograd.grad(got, tuple(model.parameters()), retain_graph=True)
    expected = torch.autograd.grad(oracle, tuple(model.parameters()))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    assert batch["trajectory_index"].tolist() == [0, 0, 1]
    assert not valid[1, :2].any()


def test_collator_rejects_tool_observation_as_action():
    row = {
        "traces": [
            {
                "prompt_token_ids": [1, 2],
                "action_token_ids": [3],
                "loss_mask": [0, 1, 1],
                "raw_model_logprobs": [-1.0],
                "behavior_logprobs": [-1.0],
            }
        ]
    }
    with pytest.raises(ValueError, match="actual action"):
        collate_agent_trajectories([row], pad_token_id=0, device="cpu")
