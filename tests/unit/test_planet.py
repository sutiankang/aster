from copy import deepcopy
import math

import pytest
import torch

from aster.models import build_model
from aster.models.config import config_from_dict
from aster.models.planet import PlaNetConfig, PlaNetWorldModel
from aster.methods.planet import (
    PlaNetObjective,
    gaussian_state_kl,
    preprocess_planet_images,
    postprocess_planet_images,
)
from aster.planning.planet import planet_cem_plan
from aster.training import Trainer


def config(**kwargs):
    return PlaNetConfig(
        **dict(
            observation_dim=4,
            action_dim=2,
            state_size=3,
            belief_size=8,
            hidden_size=8,
            reward_hidden_size=8,
            reward_layers=1,
            **kwargs,
        )
    )


def batch(b=2, t=4):
    torch.manual_seed(13)
    return dict(
        observations=torch.randn(b, t, 4),
        previous_actions=torch.randn(b, t, 2),
        is_first=torch.zeros(b, t, dtype=torch.bool),
        rewards=torch.randn(b, t),
        prior_noise=torch.randn(b, t, 3),
        posterior_noise=torch.randn(b, t, 3),
    )


def test_planet_config_save_load_and_real_pixels(tmp_path):
    torch.set_num_threads(1)
    c = PlaNetConfig(
        action_dim=2,
        state_size=3,
        belief_size=8,
        hidden_size=8,
        reward_hidden_size=8,
        reward_layers=1,
        conv_channels=2,
    )
    model = build_model(config_from_dict(c.to_dict()))
    pixels = torch.randint(256, (2, 3, 3, 64, 64), dtype=torch.uint8)
    image = preprocess_planet_images(pixels, generator=torch.Generator().manual_seed(8))
    assert image.min() >= -0.5 and image.max() < 0.5
    torch.testing.assert_close(postprocess_planet_images(image), (pixels // 8) * 8, atol=0, rtol=0)
    output = model(
        image,
        torch.zeros(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
        prior_noise=torch.zeros(2, 3, 3),
        posterior_noise=torch.zeros(2, 3, 3),
    )
    assert output["reconstruction"].shape == image.shape
    objective = PlaNetObjective(sequence_length=3, free_nats=0.0)
    data = dict(
        observations=image,
        previous_actions=torch.zeros(2, 3, 2),
        is_first=torch.zeros(2, 3, dtype=torch.bool),
        rewards=torch.zeros(2, 3),
    )
    loss = objective(model, data)
    sum(term.mean * term.weight for term in loss.terms).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.save_pretrained(tmp_path / "model")
    restored = PlaNetWorldModel.from_pretrained(tmp_path / "model")
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    assert restored.config == c


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_planet_reset_state_replay_and_wrong_family_rejected(seed):
    torch.manual_seed(seed)
    model, data = PlaNetWorldModel(config()), batch()
    data["is_first"][:, 2] = True
    output = model(**{k: v for k, v in data.items() if k != "rewards"})
    changed = {k: v.clone() for k, v in data.items() if k != "rewards"}
    for key in ("observations", "previous_actions", "prior_noise", "posterior_noise"):
        changed[key][:, :2] += 5
    changed["previous_actions"][:, 2] += 5
    replay = model(**changed)
    # Identical encoder batch shapes isolate the reset contract from GEMM rounding:
    # neither pre-reset history nor the boundary action may affect any state field.
    for key in ("mean", "stddev", "sample", "belief"):
        torch.testing.assert_close(
            getattr(output["state"], key)[:, 2:],
            getattr(replay["state"], key)[:, 2:],
            atol=0,
            rtol=0,
        )
    changed["is_first"].zero_()
    without_reset = model(**changed)
    assert not torch.allclose(output["state"].belief[:, 2:], without_reset["state"].belief[:, 2:])

    tail = model(**{k: v[:, 2:] for k, v in data.items() if k != "rewards"})
    # The shorter sequence changes the encoder's GEMM batch size. PyTorch does not
    # guarantee bitwise equality for sliced versus batched floating-point math.
    # Keep this numerical check tight; the history-isolation check above stays exact.
    tolerance = 4 * torch.finfo(output["state"].mean.dtype).eps
    for key in ("mean", "stddev", "sample", "belief"):
        torch.testing.assert_close(
            getattr(output["state"], key)[:, 2:],
            getattr(tail["state"], key),
            atol=tolerance,
            rtol=tolerance,
        )
    state = output["state"].map(lambda v: v[:, 1])
    cloned = state.fork()
    cloned.sample.add_(10)
    assert not torch.equal(state.sample, cloned.sample)
    assert state.reorder(torch.tensor([1, 0])).sample.shape == (2, 3)
    wrong = PlaNetWorldModel(config(mean_only=True))
    with pytest.raises(ValueError, match="configuration"):
        wrong.transition(state, torch.zeros(2, 2))


def test_planet_unit_gaussian_loss_kl_and_valid_pair_normalization():
    model, data = PlaNetWorldModel(config(mean_only=True)), batch()
    data["valid"] = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    data["is_first"][0, 2] = True
    objective = PlaNetObjective(
        sequence_length=4, free_nats=0.01, overshooting_distance=3, overshooting_weight=0.7
    )
    result = model(**{k: v for k, v in data.items() if k not in {"rewards", "valid"}})
    terms = {v.name: v for v in objective(model, data).terms}
    assert all(term.denominator.dtype == torch.int64 for term in terms.values())
    assert terms["image"].denominator == 6

    assert terms["overshooting"].denominator == 3
    expected = torch.distributions.Independent(
        torch.distributions.Normal(result["reconstruction"], 1), 1
    )
    torch.testing.assert_close(
        terms["image"].numerator, -expected.log_prob(data["observations"])[data["valid"]].sum()
    )
    q = torch.distributions.Independent(
        torch.distributions.Normal(result["state"].mean, result["state"].stddev), 1
    )
    p = torch.distributions.Independent(
        torch.distributions.Normal(result["prior"].mean, result["prior"].stddev), 1
    )
    reference = torch.distributions.kl_divergence(q, p)
    torch.testing.assert_close(gaussian_state_kl(result["state"], result["prior"]), reference)
    torch.testing.assert_close(
        terms["divergence"].numerator, (reference - 0.01).clamp_min(0)[data["valid"]].sum()
    )
    damaged = deepcopy(data)
    damaged["observations"][1, 2:] = 50
    damaged["previous_actions"][1, 2:] = 50
    damaged["rewards"][1, 2:] = 50
    second = {v.name: v for v in objective(model, damaged).terms}
    for name, term in terms.items():
        torch.testing.assert_close(term.numerator, second[name].numerator, atol=0, rtol=0)


@pytest.mark.parametrize("zero,precision", [(0, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")])
def test_planet_shared_training_ema_exact_resume(zero, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(22)
    model = PlaNetWorldModel(config())
    objective = PlaNetObjective(
        sequence_length=4, free_nats=0.0, overshooting_distance=3, overshooting_weight=0.3
    )
    engine = Trainer(
        model,
        objective,
        zero_stage=zero,
        precision=precision,
        accumulation_steps=2,
        ema_decay=0.9,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001, eps=1e-4),
    )
    data = batch()
    data.pop("prior_noise")
    data.pop("posterior_noise")
    for _ in range(2):
        assert engine.step([data, data]).updated
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    expected = engine.step([data, data])
    weights = deepcopy(engine.export_state_dict())
    ema = deepcopy(engine.export_state_dict(ema=True))
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = engine.step([data, data])
    assert expected.loss == actual.loss
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    for name, value in engine.export_state_dict(ema=True).items():
        torch.testing.assert_close(value, ema[name], atol=0, rtol=0)
    invalid = {**data, "observations": data["observations"][:, :3]}
    with pytest.raises(ValueError, match="sequence length"):
        engine.step([data, invalid])


def test_planet_cem_batched_actions_rng_and_reward_only():
    torch.set_num_threads(1)
    model = PlaNetWorldModel(config(mean_only=True))
    initial = model.initial(2)
    initial.sample[1] = 1
    decoded = []
    hook = model.decoder.register_forward_hook(lambda *args: decoded.append(True))
    action, stats = planet_cem_plan(
        model,
        initial,
        horizon=3,
        population=16,
        elites=4,
        iterations=3,
        generator=torch.Generator().manual_seed(11),
    )
    second, other = planet_cem_plan(
        model,
        initial,
        horizon=3,
        population=16,
        elites=4,
        iterations=3,
        generator=torch.Generator().manual_seed(11),
    )
    hook.remove()
    assert not decoded and model.training
    torch.testing.assert_close(action, second, atol=0, rtol=0)
    assert action.shape == (2, 2) and (action.abs() <= 1).all()
    assert stats["trajectories"] == 96 and stats["model_rollout_steps"] == 9
    assert (stats["stddev"] >= 0.001).all()
    zeros, _ = planet_cem_plan(model, initial, iterations=0)
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_planet_conditional_dynamics_learns():
    torch.set_num_threads(1)
    torch.manual_seed(124)
    model = PlaNetWorldModel(config(mean_only=True))
    data = batch(b=5)
    data["observations"] = torch.cat(
        (data["previous_actions"], data["previous_actions"].square()), -1
    )
    data["rewards"] = data["previous_actions"].sum(-1)
    objective = PlaNetObjective(
        sequence_length=4,
        free_nats=0.0,
        divergence_weight=0.01,
        overshooting_distance=3,
        overshooting_weight=0.01,
    )
    engine = Trainer(
        model, objective, optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.01, eps=1e-4)
    )

    def error():
        with torch.no_grad():
            output = model(data["observations"], data["previous_actions"], data["is_first"])
            return (
                (output["reconstruction"] - data["observations"]).square().mean()
                + (output["reward"] - data["rewards"]).square().mean()
            ).item()

    initial = error()
    for _ in range(80):
        assert engine.step([data]).updated
    assert error() < initial * 0.15


def test_planet_validation_rejects_gaps_bad_noise_and_integer_pixels():
    model, data = PlaNetWorldModel(config()), batch()
    objective = PlaNetObjective(sequence_length=4)
    with pytest.raises(ValueError, match="contiguous prefix"):
        objective(
            model, {**data, "valid": torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)}
        )
    with pytest.raises(ValueError, match="noise"):
        objective(model, {**data, "posterior_noise": torch.full((2, 4, 3), math.nan)})
    with pytest.raises(ValueError, match="floating-point"):
        objective(model, {**data, "observations": data["observations"].to(torch.uint8)})
