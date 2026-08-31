from copy import deepcopy
import math

import pytest
import torch

from aster.models.planet import PlaNetConfig, PlaNetWorldModel
from aster.methods.planet import PlaNetObjective
from aster.methods.planet_loop import PlaNetLoop, PlaNetReplay, planet_chunk_offsets
from aster.training import Trainer


class VectorSimulator:
    def __init__(self, *, horizon=6, pixel=False):
        self.horizon, self.pixel = horizon, pixel
        self.position = self.steps = 0

    def config_dict(self):
        return dict(name="toy_scalar_control", horizon=self.horizon, pixel=self.pixel)

    def observation(self):
        if self.pixel:
            return torch.full((1, 64, 64), int((self.position + 2) / 4 * 255), dtype=torch.uint8)
        return torch.tensor([self.position, 1.0], dtype=torch.float32)

    def reset(self):
        self.position, self.steps = 0.0, 0
        return self.observation()

    def step(self, action):
        self.position = max(-2.0, min(2.0, 0.8 * self.position + float(action[0])))
        self.steps += 1
        return (
            self.observation(),
            -((self.position - 0.6) ** 2),
            False,
            self.steps >= self.horizon,
            {},
        )

    def state_dict(self):
        return dict(position=self.position, steps=self.steps)

    def load_state_dict(self, state):
        self.position, self.steps = state["position"], state["steps"]


def configuration(pixel=False):
    return PlaNetConfig(
        observation_dim=0 if pixel else 2,
        action_dim=1,
        state_size=3,
        belief_size=6,
        hidden_size=8,
        reward_layers=1,
        reward_hidden_size=8,
        image_channels=1,
        conv_channels=2,
    )


def episode(c, steps=12):
    obs = (
        torch.arange(steps + 1, dtype=torch.float32)[:, None]
        .expand(steps + 1, c.observation_dim)
        .clone()
    )
    end = torch.zeros(steps + 1, dtype=torch.bool)
    end[-1] = True
    first = torch.zeros_like(end)
    first[0] = True
    return dict(
        observations=obs,
        previous_actions=torch.arange(steps + 1, dtype=torch.float32)[:, None] / (steps + 1),
        rewards=torch.arange(steps + 1, dtype=torch.float32),
        is_first=first,
        terminated=torch.zeros_like(end),
        truncated=end,
    )


def test_planet_official_random_chunk_formula_and_reload_replay_restore():
    c = configuration()
    for length in (4, 7, 8, 19):
        rng = torch.Generator().manual_seed(31)
        clone = torch.Generator().manual_seed(31)
        count = max(1, length // 4 - 1)
        offset = int(torch.randint(length - count * 4 + 1, (), generator=clone))
        assert planet_chunk_offsets(length, 4, generator=rng) == tuple(
            offset + i * 4 for i in range(count)
        )
    replay = PlaNetReplay(c, sequence_length=4, seed=231, max_episodes=2)
    for length in (12, 9, 7):
        replay.add(episode(c, length))
    assert len(replay.episodes) == 2 and replay.insertions == 3
    replay.sample(1)
    state = deepcopy(replay.state_dict())
    expected = replay.sample(4)
    restored = PlaNetReplay(c, sequence_length=4, seed=999, max_episodes=2)
    restored.load_state_dict(state)
    actual = restored.sample(4)
    for key in actual:
        assert torch.equal(actual[key], expected[key])

    assert torch.equal(actual["observations"][..., 0], actual["rewards"])
    for row in actual["observations"][..., 0]:
        assert torch.equal(row[1:] - row[:-1], torch.ones(3))
    bad = episode(c)
    bad["previous_actions"][0] = 0.5
    with pytest.raises(ValueError, match="reset row"):
        replay.add(bad)
    assert replay.insertions == 3


@pytest.mark.parametrize("zero,precision", [(0, "fp32"), (3, "fp32"), (3, "bf16")])
def test_planet_real_collection_training_mid_episode_checkpoint_and_next_update(
    zero, precision, tmp_path
):
    torch.set_num_threads(1)
    torch.manual_seed(404)
    c = configuration()
    engine = Trainer(
        PlaNetWorldModel(c),
        PlaNetObjective(sequence_length=4, free_nats=0),
        zero_stage=zero,
        precision=precision,
        accumulation_steps=2,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.002),
    )
    replay = PlaNetReplay(c, sequence_length=4, seed=19)
    env = VectorSimulator()
    loop = PlaNetLoop(
        engine,
        env,
        replay,
        batch_size=2,
        seed=12,
        action_noise=0.1,
        planner=dict(horizon=2, population=6, elites=2, iterations=2),
    )
    rng = torch.get_rng_state().clone()
    result = loop.collect_steps(12, random=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert result["episodes"] == 2 and len(result["episode_returns"]) == 2
    for data in replay.episodes.values():
        assert (
            len(data["observations"]) == 7
            and data["previous_actions"][0] == 0
            and data["rewards"][0] == 0
        )
        assert data["truncated"][-1] and not data["terminated"].any()
    assert loop.train_step().updated
    loop.refresh_world()
    loop.collect_steps(2)
    checkpoint = engine.save_checkpoint(tmp_path / "mid_episode")
    expected_collection = loop.collect_steps(4)
    expected_update = loop.train_step()
    expected_weights = deepcopy(engine.export_state_dict())
    expected_environment = deepcopy(env.state_dict())
    engine.load_checkpoint(checkpoint, trusted=True)
    actual_collection = loop.collect_steps(4)
    actual_update = loop.train_step()
    assert actual_collection == expected_collection and actual_update.loss == expected_update.loss
    assert env.state_dict() == expected_environment and loop.episodes == 3 and loop.updates == 2
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)


def test_planet_raw_pixel_collection_preprocess_and_failed_simulator_requires_restore(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(405)
    c = configuration(pixel=True)
    engine = Trainer(
        PlaNetWorldModel(c), PlaNetObjective(sequence_length=3), zero_stage=3, lr=0.0001
    )
    env = VectorSimulator(horizon=3, pixel=True)
    replay = PlaNetReplay(c, sequence_length=3, seed=21)
    loop = PlaNetLoop(engine, env, replay, batch_size=1, seed=13)
    loop.collect_steps(3, random=True)
    stored = replay.episodes[0]["observations"]
    assert stored.dtype == torch.uint8
    state = deepcopy(replay.state_dict())
    first = replay.sample(1)["observations"]
    replay.load_state_dict(state)
    assert torch.equal(first, replay.sample(1)["observations"])
    assert first.is_floating_point() and first.min() >= -0.5 and first.max() < 0.5
    assert loop.train_step().updated
    checkpoint = engine.save_checkpoint(tmp_path / "valid")
    original = env.step

    def failed(action):
        original(action)
        raise RuntimeError("simulator advanced then failed")

    env.step = failed
    with pytest.raises(RuntimeError, match="advanced"):
        loop.collect_steps(1, random=True)
    assert loop.failed
    with pytest.raises(RuntimeError, match="partial simulator"):
        loop.state_dict()
    env.step = original
    engine.load_checkpoint(checkpoint, trusted=True)
    assert not loop.failed and loop.active is None and env.steps == 3


def test_planet_insufficient_replay_failure_preserves_sampling_rng():
    torch.set_num_threads(1)
    c = configuration()
    engine = Trainer(PlaNetWorldModel(c), PlaNetObjective(sequence_length=8))
    replay = PlaNetReplay(c, sequence_length=8)
    loop = PlaNetLoop(engine, VectorSimulator(horizon=3), replay, batch_size=2)
    loop.collect_steps(3, random=True)
    before = replay.rng.get_state().clone()
    with pytest.raises(ValueError, match="long enough"):
        loop.train_step()
    assert torch.equal(before, replay.rng.get_state()) and loop.updates == replay.samples == 0


def test_planet_bad_planner_rejected_before_environment_or_target_mutation():
    c = configuration()
    engine = Trainer(PlaNetWorldModel(c), PlaNetObjective(sequence_length=4))
    env = VectorSimulator()
    with pytest.raises(ValueError, match="planner"):
        PlaNetLoop(
            engine, env, PlaNetReplay(c, sequence_length=4), planner=dict(population=2, elites=5)
        )
    assert env.steps == 0 and set(engine.roles) == {"model"}
