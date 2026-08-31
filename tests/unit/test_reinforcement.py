import copy
import torch
from aster.data.replay import ReplayBuffer, NStepAccumulator
from aster.methods.reinforcement import (
    generalized_advantage,
    lambda_returns,
    CategoricalActorCritic,
    PPOObjective,
    GaussianActor,
    TwinQ,
    SACMethod,
    DQNObjective,
    mlp,
    group_relative_advantages,
)
from aster.training import Trainer


def test_gae_terminal_and_time_limit_boundary():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values, next_values = torch.zeros_like(rewards), torch.tensor([[10.0, 20.0, 30.0]])
    terminated = torch.tensor([[False, False, True]])
    truncated = torch.tensor([[False, True, False]])
    advantages, _ = generalized_advantage(
        rewards, values, next_values, terminated, truncated, gamma=0.5, lam=1.0
    )
    torch.testing.assert_close(advantages, torch.tensor([[12.0, 12.0, 3.0]]))
    returns = lambda_returns(torch.ones(1, 2), torch.zeros(1, 3), torch.ones(1, 2) * 0.5, lam=1.0)
    torch.testing.assert_close(returns, torch.tensor([[1.5, 1.0]]))


def test_replay_restores_exact_next_sample_and_stale_priority():
    buffer = ReplayBuffer(3, seed=8, priority_alpha=0.6)
    for i in range(4):
        buffer.add({"x": torch.tensor([float(i)]), "done": torch.tensor(False)}, priority=i + 1)
    state = buffer.state_dict()
    first = buffer.sample(5)
    buffer.load_state_dict(state)
    second = buffer.sample(5)
    torch.testing.assert_close(first["x"], second["x"])
    before = buffer.priorities.copy()
    buffer.update_priorities([0], [-100], [99.0])
    assert (before == buffer.priorities).all()


def test_n_step_truncation_bootstrap():
    accumulator = NStepAccumulator(3, 0.5)
    first = {
        "observation": [0.0],
        "action": 1,
        "reward": 2.0,
        "next_observation": [1.0],
        "terminated": False,
        "truncated": False,
    }
    assert accumulator.add(first) == []
    second = dict(first, reward=4.0, next_observation=[2.0], truncated=True)
    outputs = accumulator.add(second)
    assert outputs[0]["reward"] == 4.0 and outputs[0]["discount"] == 0.25
    assert outputs[1]["discount"] == 0.5 and outputs[0]["next_observation"] == [2.0]


def test_ppo_dqn_shared_updates():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = CategoricalActorCritic(4, 2, hidden=16)
    observations = torch.randn(8, 4)
    with torch.no_grad():
        actions, old, values = model.act(observations)
    batch = {
        "observations": observations,
        "actions": actions,
        "old_log_probs": old,
        "old_values": values,
        "advantages": torch.randn(8),
        "returns": torch.randn(8),
    }
    assert Trainer(model, PPOObjective()).step([batch]).updated
    network = mlp(4, 2, hidden=16)
    objective = DQNObjective(copy.deepcopy(network))
    batch.update(
        next_observations=torch.randn(8, 4),
        terminated=torch.zeros(8, dtype=torch.bool),
        rewards=torch.randn(8),
    )
    assert Trainer(network, objective).step([batch]).updated


def test_sac_multirole_checkpoint_next_update(tmp_path):
    torch.manual_seed(7)
    actor = GaussianActor(3, 2, hidden=16)
    engine = Trainer(actor, lr=0.001)
    method = SACMethod(engine, TwinQ(3, 2, hidden=16))
    batch = {
        "observations": torch.randn(6, 3),
        "next_observations": torch.randn(6, 3),
        "actions": torch.rand(6, 2) * 2 - 1,
        "rewards": torch.randn(6),
        "terminated": torch.zeros(6, dtype=torch.bool),
    }
    results = method.update([batch])
    assert all(result.updated for result in results.values())
    engine.save_checkpoint(tmp_path / "sac")
    method.update([batch])
    expected = copy.deepcopy(actor.state_dict())
    engine.load_checkpoint(tmp_path / "sac", trusted=True)
    method.update([batch])
    for key, value in expected.items():
        torch.testing.assert_close(value, actor.state_dict()[key])
    assert all(p.grad is None for p in method.target.parameters())


def test_grpo_group_centering():
    rewards = torch.tensor([1.0, 3.0, 8.0, 8.0])
    groups = torch.tensor([0, 0, 1, 1])
    advantage = group_relative_advantages(rewards, groups)
    assert advantage[0] < 0 < advantage[1]
    torch.testing.assert_close(advantage[2:], torch.zeros(2))
