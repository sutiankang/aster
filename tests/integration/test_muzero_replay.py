from copy import deepcopy
from dataclasses import replace
import pytest
import torch

from aster.models.muzero import MuZeroConfig, MuZeroModel
from aster.methods.muzero import MuZeroMethod, MuZeroObjective
from aster.methods.muzero_replay import MuZeroEpisode, MuZeroReplay, MuZeroSearch
from aster.training import Trainer


def _config():
    return MuZeroConfig(
        observation_dim=3, num_actions=2, latent_dim=8, hidden_size=16, support_size=5, discount=0.0
    )


def _episode(action=1, *, truncated=False):
    return MuZeroEpisode(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([action]),
        torch.tensor([float(action)]),
        torch.tensor([not truncated]),
        torch.tensor([truncated]),
    )


def test_replay_coordinates_keep_terminal_reward_and_truncation_bootstrap():
    torch.set_num_threads(1)
    torch.manual_seed(331)
    engine = Trainer(MuZeroModel(replace(_config(), discount=0.9)))
    search = MuZeroSearch.from_trainer(engine, num_simulations=4, search_options={"max_depth": 1})
    replay = MuZeroReplay(engine.model.config, unroll_steps=3)
    for truncation in (False, True):
        episode = _episode(truncated=truncation)
        analysis = search.reanalyze(episode)
        replay.add_episode(episode, analysis)
    store = replay.buffer.storage
    assert store["valid"][:2].tolist() == [[True, False, False, False], [True, True, False, False]]
    assert store["reward_valid"][:2].tolist() == [[True, False, False], [True, False, False]]
    assert store["reward_targets"][0, 0] == 1
    assert store["value_targets"][0, 0] == 1

    torch.testing.assert_close(store["value_targets"][1, 0], 1 + 0.9 * analysis.search_values[-1])
    MuZeroObjective().validate(engine.model, replay.sample(4))
    changed = replace(episode, rewards=torch.tensor([9.0]))
    with pytest.raises(ValueError, match="belong"):
        replay.add_episode(changed, analysis)


@pytest.mark.parametrize("stage,algorithm", [(0, "muzero"), (3, "gumbel_muzero")])
def test_search_replay_rng_and_optimizer_restore_as_one_checkpoint(stage, algorithm, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(332)
    engine = Trainer(MuZeroModel(_config()), lr=0.005, zero_stage=stage)
    method = MuZeroMethod(engine)
    search = MuZeroSearch.from_trainer(
        engine, seed=91, num_simulations=8, algorithm=algorithm, search_options={"max_depth": 2}
    )
    replay = MuZeroReplay(engine.model.config, capacity=20, unroll_steps=2, seed=99)
    engine.register_state("planner", search)
    engine.register_state("episodes", replay)
    for action in (0, 1):
        episode = _episode(action)
        replay.add_episode(episode, search.reanalyze(episode))
    method.update([replay.sample(4)])
    engine.save_checkpoint(tmp_path / "whole")
    expected_search = search.plan(_episode().observations)
    expected_batch = replay.sample(4)
    expected = method.update([expected_batch])
    weights = deepcopy(engine.export_state_dict())
    search.refresh(engine)
    engine.load_checkpoint(tmp_path / "whole", trusted=True)
    actual_search = search.plan(_episode().observations)
    actual_batch = replay.sample(4)
    actual = method.update([actual_batch])
    torch.testing.assert_close(
        actual_search.action_weights, expected_search.action_weights, rtol=0, atol=0
    )
    torch.testing.assert_close(actual_search.action, expected_search.action, rtol=0, atol=0)
    for key in expected_batch:
        torch.testing.assert_close(expected_batch[key], actual_batch[key], rtol=0, atol=0)
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_reanalyzed_native_muzero_learns_one_step_bandit_planning():
    torch.set_num_threads(1)
    torch.manual_seed(333)
    engine = Trainer(MuZeroModel(_config()), lr=0.01)
    method = MuZeroMethod(engine)
    search = MuZeroSearch.from_trainer(
        engine,
        seed=92,
        num_simulations=16,
        search_options={"max_depth": 1, "dirichlet_fraction": 0.0, "temperature": 0.0},
    )
    replay = MuZeroReplay(
        engine.model.config, capacity=8, unroll_steps=1, seed=98, priority_alpha=0.0
    )
    initial_id = search.model_id

    for cycle in range(4):
        for action in (0, 1):
            episode = _episode(action)
            replay.add_episode(episode, search.reanalyze(episode))
        for _ in range(25):
            method.update([replay.sample(8)])
        search.refresh(engine)
    result = search.plan(_episode().observations[:1])
    assert result.action.item() == 1
    assert result.action_weights[0, 1] > 0.7
    assert search.model_id != initial_id
    assert result.search_tree.qvalues()[0, 1] > result.search_tree.qvalues()[0, 0] + 0.5


def test_search_rejects_mutation_and_episode_rejects_cross_reset():
    engine = Trainer(MuZeroModel(_config()))
    search = MuZeroSearch.from_trainer(engine, num_simulations=4)
    with torch.no_grad():
        next(search.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="snapshot changed"):
        search.plan(_episode().observations)
    with pytest.raises(ValueError, match="final reset"):
        replace(_episode(), terminated=torch.tensor([False])).validate(_config())
    invalid = torch.ones(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="invalid action"):
        replace(_episode(), invalid_actions=invalid).validate(_config())
