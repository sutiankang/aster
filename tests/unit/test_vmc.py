from copy import deepcopy

import pytest
import torch

from aster.models import build_model
from aster.models.config import config_from_dict
from aster.models.vmc import (
    VMCVAEConfig,
    VMCVAE,
    MDNRNNConfig,
    MDNRNN,
    VMCControllerConfig,
    VMCController,
    MDNOutput,
    sample_mdn,
)
from aster.methods.vmc import VMCVAEObjective, MDNRNNObjective, encode_vmc_episodes
from aster.planning.vmc import VMCControllerSearch, vmc_population_returns
from aster.training import Trainer


def _config(**kwargs):
    return MDNRNNConfig(**dict(latent_size=3, action_dim=1, hidden_size=8, mixtures=2, **kwargs))


def _batch():
    torch.manual_seed(142)
    return dict(
        mean=torch.randn(2, 4, 3),
        logvar=torch.randn(2, 4, 3) * 0.1,
        actions=torch.rand(2, 4, 1) * 2 - 1,
        restart=torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.bool),
    )


def test_vmc_vae_training_encoding_and_model_persistence(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(69)
    model = VMCVAE(VMCVAEConfig(latent_size=3, conv_channels=2))
    images = torch.rand(2, 2, 3, 64, 64)
    objective = VMCVAEObjective(kl_tolerance=0.1)
    engine = Trainer(
        model, objective, zero_stage=3, optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001)
    )
    result = engine.step([dict(images=images.flatten(0, 1))])
    assert result.updated and result.terms["kl"]["denominator"] == 4
    rng = torch.get_rng_state().clone()
    encoded = encode_vmc_episodes(model, images, chunk_size=1)
    assert torch.equal(torch.get_rng_state(), rng)
    assert encoded["mean"].shape == (2, 2, 3) and not encoded["mean"].requires_grad
    materialized = VMCVAE(model.config)
    materialized.load_state_dict(engine.export_state_dict())
    materialized.save_pretrained(tmp_path / "vae")
    restored = VMCVAE.from_pretrained(tmp_path / "vae")
    assert restored.config == model.config
    torch.testing.assert_close(
        restored.encode(images[:, 0])[0], encoded["mean"][:, 0], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize(
    "zero,precision,norm",
    [(0, "fp32", False), (3, "fp32", True), (0, "bf16", True), (3, "bf16", False)],
)
def test_mdn_native_shared_trainer_dropout_and_exact_resume(zero, precision, norm, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(253)
    model = build_model(
        config_from_dict(
            _config(
                layer_norm=norm, input_dropout=0.1, output_dropout=0.1, recurrent_dropout=0.1
            ).to_dict()
        )
    )
    objective = MDNRNNObjective(sequence_length=4)
    engine = Trainer(
        model,
        objective,
        precision=precision,
        zero_stage=zero,
        accumulation_steps=2,
        ema_decay=0.9,
        max_grad_norm=None,
        max_grad_value=1.0,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001, eps=1e-4),
    )
    data = _batch()
    for _ in range(2):
        assert engine.step([data, data]).updated
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    expected = engine.step([data, data])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = engine.step([data, data])
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="sequence length"):
        engine.step([data, {**data, "mean": data["mean"][:, :3]}])


def test_mdn_restart_and_independent_coordinate_distribution_formula():
    torch.manual_seed(146)
    model = MDNRNN(_config())
    data = _batch()
    latent = data["mean"]
    output = model(latent, data["actions"], data["restart"])
    tail = model(latent[1:2, 2:], data["actions"][1:2, 2:], data["restart"][1:2, 2:])
    torch.testing.assert_close(output.mean[1:2, 2:], tail.mean, atol=1e-7, rtol=1e-6)
    objective = MDNRNNObjective(sequence_length=4, restart_factor=7)
    batch = dict(latents=latent, actions=data["actions"], restart=data["restart"])
    terms = objective(model, batch).terms
    prediction = model(latent[:, :-1], data["actions"][:, :-1], data["restart"][:, :-1])
    reference = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=prediction.logmix),
        torch.distributions.Normal(prediction.mean, prediction.logstd.exp()),
    )
    torch.testing.assert_close(terms[0].mean, -reference.log_prob(latent[:, 1:]).mean())
    assert terms[0].denominator.dtype == torch.int64 and terms[0].denominator == 18
    assert terms[1].denominator == 6
    changed = {**batch, "valid": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)}
    assert objective(model, changed).terms[0].denominator == 12


def test_mdn_temperature_gaussian_scale_and_deterministic_restart():
    model = MDNRNN(_config())

    out = MDNOutput(
        torch.zeros(1, 1, 3, 1),
        torch.ones(1, 1, 3, 1),
        torch.zeros(1, 1, 3, 1),
        torch.tensor([[0.1]]),
        model.initial(1),
    )
    first, restart = sample_mdn(out, temperature=1, generator=torch.Generator().manual_seed(8))
    second, _ = sample_mdn(out, temperature=4, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(second - 1, (first - 1) * 2)
    assert restart.item()
    with pytest.raises(ValueError, match="temperature"):
        sample_mdn(out, temperature=0)


def test_vmc_dream_termination_and_controller_search_restore(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(623)
    model = MDNRNN(_config())
    engine = Trainer(model, MDNRNNObjective(sequence_length=4))
    assert engine.step([_batch()]).updated
    means, logvars = torch.ones(4, 3), torch.zeros(4, 3)
    search = VMCControllerSearch(
        engine, means, logvars, population=6, episodes=2, horizon=4, seed=3
    )
    before = torch.get_rng_state().clone()
    first = search.step()
    assert torch.equal(before, torch.get_rng_state())
    assert first["diagnostics"]["episode_returns"].shape == (6, 2)
    assert (first["diagnostics"]["terminated"] ^ first["diagnostics"]["truncated"]).all()
    state = deepcopy(search.state_dict())
    expected = search.step()
    restored = VMCControllerSearch(
        engine, means, logvars, population=6, episodes=2, horizon=4, seed=3
    )
    restored.load_state_dict(state)
    actual = restored.step()
    assert actual["mean_return"] == expected["mean_return"]
    torch.testing.assert_close(restored.evolution.mean, search.evolution.mean, atol=0, rtol=0)
    controller = restored.controller()
    controller.save_pretrained(tmp_path / "controller")
    reloaded = VMCController.from_pretrained(tmp_path / "controller")
    torch.testing.assert_close(reloaded.weight, controller.weight, atol=0, rtol=0)

    snapshot = deepcopy(search.world.state_dict())
    assert engine.step([_batch()]).updated
    for key, value in search.world.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)
    incompatible = VMCControllerSearch(
        engine, means, logvars, population=6, episodes=2, horizon=4, seed=3
    )
    with pytest.raises(ValueError, match="identity"):
        incompatible.load_state_dict(state)


def test_vmc_learned_dynamics_to_evolved_controller_improves_dream_return():

    torch.set_num_threads(1)
    torch.manual_seed(338)
    c = MDNRNNConfig(latent_size=1, action_dim=1, hidden_size=8, mixtures=1)
    model = MDNRNN(c)

    actions = torch.rand(16, 13, 1) * 2 - 1
    actions[:8] = 0.6 + 0.4 * torch.rand(8, 13, 1)
    restart = torch.ones(16, 13, dtype=torch.bool)
    restart[:, 1:] = actions[:, :-1, 0] < 0.35
    data = dict(latents=1 + 0.08 * torch.randn(16, 13, 1), actions=actions, restart=restart)
    engine = Trainer(
        model,
        MDNRNNObjective(sequence_length=13, restart_factor=2),
        max_grad_norm=None,
        max_grad_value=1.0,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.01, eps=1e-4),
    )
    for _ in range(160):
        assert engine.step([data]).updated
    means, logvars = torch.ones(4, 1), torch.full((4, 1), -5.0)
    search = VMCControllerSearch(
        engine,
        means,
        logvars,
        population=12,
        episodes=3,
        horizon=12,
        temperature=1.0,
        sigma=0.5,
        seed=17,
    )
    zero = torch.zeros(1, search.controller_config.feature_size, 1)
    before, _ = vmc_population_returns(
        search.world,
        search.controller_config,
        zero,
        means,
        logvars,
        episodes=12,
        horizon=12,
        temperature=1.0,
        seed=23,
    )
    for _ in range(8):
        search.step()
    after, _ = vmc_population_returns(
        search.world,
        search.controller_config,
        search.controller().weight.detach()[None],
        means,
        logvars,
        episodes=12,
        horizon=12,
        temperature=1.0,
        seed=23,
    )
    assert after.item() >= before.item() + 5 and after.item() >= 10
