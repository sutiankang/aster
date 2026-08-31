from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from aster.methods.adversarial_autoencoder import AdversarialAutoencoderMethod
from aster.methods.perceptual_autoencoder import PerceptualAutoencoderObjective
from aster.models.adversarial import PatchDiscriminator, PatchDiscriminatorConfig
from aster.models.generative import AutoencoderKL, AutoencoderConfig
from aster.models.perceptual import LPIPS, LPIPSConfig
from aster.training import Trainer


def objects():
    torch.set_num_threads(1)
    torch.manual_seed(744)
    generator = AutoencoderKL(
        AutoencoderConfig(base_channels=4, latent_channels=2, channel_mult=(1, 2), num_res_blocks=1)
    )
    discriminator = PatchDiscriminator(PatchDiscriminatorConfig(base_channels=4, num_layers=1))
    metric = LPIPS(LPIPSConfig(channels=(2, 3, 4, 4, 4), allow_untrained=True))
    batches = [
        dict(sample=torch.rand(n, 3, 16, 16) * 2 - 1, posterior_noise=torch.randn(n, 2, 8, 8))
        for n in (1, 2)
    ]
    discriminator.initialize(torch.cat([b["sample"] for b in batches]))
    return generator, discriminator, metric, batches


def independent_step(
    generator, discriminator, metric, full, generator_optimizer, discriminator_optimizer, *, active
):

    generator.requires_grad_(True)
    discriminator.requires_grad_(False)
    posterior = generator.encode(full["sample"])
    fake = generator.decode(
        posterior.mean + (0.5 * posterior.logvar).exp() * full["posterior_noise"]
    )
    nll = ((full["sample"] - fake).abs() + metric(full["sample"], fake)).mean()
    kl = (
        0.5
        * (posterior.mean.square() + posterior.logvar.exp() - 1 - posterior.logvar)
        .flatten(1)
        .sum(1)
        .mean()
    )
    adversarial = -discriminator(fake).mean()
    nll_grad = torch.autograd.grad(nll, generator.decoder[-1].weight, retain_graph=True)[0]
    gan_grad = torch.autograd.grad(adversarial, generator.decoder[-1].weight, retain_graph=True)[0]
    coefficient = (
        (nll_grad.norm() / (gan_grad.norm() + 1e-4)).clamp(0, 1e4).detach() * 0.4
        if active
        else torch.tensor(0.0)
    )
    generator_loss = nll + 1e-6 * kl + 0.7 * coefficient * adversarial
    generator_optimizer.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()
    generator.requires_grad_(False)
    discriminator.requires_grad_(True)
    with torch.no_grad():
        posterior = generator.encode(full["sample"])
        fake = generator.decode(
            posterior.mean + (0.5 * posterior.logvar).exp() * full["posterior_noise"]
        )
    logits_real, logits_fake = discriminator(full["sample"]), discriminator(fake)
    discriminator_loss = (
        0.5
        * (F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean())
        * (0.7 if active else 0.0)
    )
    discriminator_optimizer.zero_grad()
    discriminator_loss.backward()
    discriminator_optimizer.step()
    generator.requires_grad_(True)
    return generator_loss.detach(), discriminator_loss.detach(), coefficient * 0.7


@pytest.mark.parametrize("stage", [0, 3])
def test_adversarial_method_full_batch_independent_update_ratio_and_role_ownership(stage, tmp_path):
    generator, discriminator, metric, batches = objects()
    dense_g, dense_d, dense_p = deepcopy(generator), deepcopy(discriminator), deepcopy(metric)
    full = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}

    optimizer_factory = lambda parameters: torch.optim.SGD(parameters, lr=0.001, momentum=0.9)
    dense_go, dense_do = (
        optimizer_factory(dense_g.parameters()),
        optimizer_factory(dense_d.parameters()),
    )
    engine = Trainer(
        generator,
        optimizer_factory=optimizer_factory,
        accumulation_steps=2,
        zero_stage=stage,
        max_grad_norm=None,
    )
    method = AdversarialAutoencoderMethod(
        engine,
        metric,
        discriminator,
        discriminator_optimizer_factory=optimizer_factory,
        pixel_reduction="mean",
        disc_start=1,
        disc_factor=0.7,
        disc_weight=0.4,
    )
    frozen = metric.weight_identity()
    for step in range(2):
        expected_g, expected_d, expected_coefficient = independent_step(
            dense_g, dense_d, dense_p, full, dense_go, dense_do, active=step >= 1
        )
        result = method.update(batches)
        assert result.updated and method.updates == step + 1
        assert abs(result.generator.loss - float(expected_g)) < 4e-6
        assert abs(result.discriminator.loss - float(expected_d)) < 4e-6
        record = engine.last_gradient_ratio(method.policy_name)
        assert record["active"] == (step >= 1)
        assert abs(record["effective_weight"] - float(expected_coefficient)) < 2e-6
        for role, dense in (("model", dense_g), (method.discriminator_role, dense_d)):
            exported = engine.export_state_dict(role=role, only_rank_zero=False)
            for name, value in exported.items():
                torch.testing.assert_close(value, dense.state_dict()[name], atol=3e-6, rtol=3e-5)
        assert metric.weight_identity() == frozen
    checkpoint = engine.save_checkpoint(tmp_path / f"zero{stage}")
    expected = method.update(batches)
    expected_g = deepcopy(engine.export_state_dict(only_rank_zero=False))
    expected_d = deepcopy(
        engine.export_state_dict(role=method.discriminator_role, only_rank_zero=False)
    )
    expected_ratio = engine.last_gradient_ratio(method.policy_name)
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = method.update(batches)
    assert (
        actual.generator.loss == expected.generator.loss
        and actual.discriminator.loss == expected.discriminator.loss
    )
    assert engine.last_gradient_ratio(method.policy_name) == expected_ratio
    for role, state in (("model", expected_g), (method.discriminator_role, expected_d)):
        for name, value in engine.export_state_dict(role=role, only_rank_zero=False).items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)


def test_adversarial_bf16_zero3_stochastic_fresh_model_checkpoint(tmp_path):
    generator, discriminator, metric, batches = objects()

    batches = [dict(sample=b["sample"]) for b in batches]
    engine = Trainer(generator, zero_stage=3, precision="bf16", accumulation_steps=2, ema_decay=0.9)
    method = AdversarialAutoencoderMethod(engine, metric, discriminator, pixel_reduction="mean")
    assert method.update(batches).updated
    checkpoint = engine.save_checkpoint(tmp_path / "gan_bf16")
    expected = method.update(batches)
    states = {
        role: deepcopy(engine.export_state_dict(role=role, only_rank_zero=False))
        for role in ("model", method.discriminator_role)
    }
    other_g, other_d, other_p, _ = objects()
    fresh = Trainer(other_g, zero_stage=3, precision="bf16", accumulation_steps=2, ema_decay=0.9)
    restored = AdversarialAutoencoderMethod(fresh, other_p, other_d, pixel_reduction="mean")
    fresh.load_checkpoint(checkpoint, trusted=True)
    actual = restored.update(batches)
    assert (
        actual.generator.loss == expected.generator.loss
        and actual.discriminator.loss == expected.discriminator.loss
    )
    for role, weights in states.items():
        for name, value in fresh.export_state_dict(role=role, only_rank_zero=False).items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)


def test_adversarial_adam_matches_independent_optimizer_given_actual_combined_gradients():
    generator, discriminator, metric, batches = objects()
    dense_g, dense_d, dense_p = deepcopy(generator), deepcopy(discriminator), deepcopy(metric)
    full = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
    factory = lambda parameters: torch.optim.Adam(parameters, lr=0.0001, betas=(0.5, 0.9))
    go, do = factory(dense_g.parameters()), factory(dense_d.parameters())

    terms = PerceptualAutoencoderObjective(dense_p, pixel_reduction="mean")(dense_g, full).terms
    sum(term.mean * term.weight for term in terms).backward()
    full_gradients = {name: p.grad.clone() for name, p in dense_g.named_parameters()}
    engine = Trainer(generator, optimizer_factory=factory, accumulation_steps=2, max_grad_norm=None)
    method = AdversarialAutoencoderMethod(
        engine,
        metric,
        discriminator,
        discriminator_optimizer_factory=factory,
        pixel_reduction="mean",
        disc_start=1,
    )
    assert method.update(batches).updated
    for role, live, dense, optimizer in (
        ("model", generator, dense_g, go),
        (method.discriminator_role, discriminator, dense_d, do),
    ):
        for (name, actual), (_, expected) in zip(live.named_parameters(), dense.named_parameters()):
            if role == "model":
                torch.testing.assert_close(actual.grad, full_gradients[name], atol=2e-6, rtol=3e-5)
            expected.grad = actual.grad.detach().clone()
        optimizer.step()
        for name, value in engine.export_state_dict(role=role, only_rank_zero=False).items():
            torch.testing.assert_close(value, dense.state_dict()[name], atol=0, rtol=0)


def test_adversarial_invalid_later_batch_rejects_before_any_generator_or_discriminator_work():
    generator, discriminator, metric, batches = objects()
    engine = Trainer(generator, zero_stage=3, accumulation_steps=2)
    method = AdversarialAutoencoderMethod(engine, metric, discriminator)
    bad = deepcopy(batches)
    bad[1]["posterior_noise"] = torch.zeros(1)
    calls = []
    hooks = [
        engine.model.encoder.register_forward_pre_hook(lambda *_: calls.append("g")),
        discriminator.register_forward_pre_hook(lambda *_: calls.append("d")),
    ]
    try:
        with pytest.raises(ValueError, match="posterior noise"):
            method.update(bad)
    finally:
        for hook in hooks:
            hook.remove()
    assert not calls and not engine._failed and method.updates == 0


def test_adversarial_half_committed_phase_blocks_checkpoint_until_complete_restore(
    tmp_path, monkeypatch
):
    generator, discriminator, metric, batches = objects()
    engine = Trainer(generator, zero_stage=3, accumulation_steps=2)
    method = AdversarialAutoencoderMethod(engine, metric, discriminator, pixel_reduction="mean")
    checkpoint = engine.save_checkpoint(tmp_path / "initial")
    phase = engine.phase

    def fail_discriminator(name, **kwargs):
        if kwargs.get("role") == method.discriminator_role:
            raise RuntimeError("Injected discriminator failure")
        return phase(name, **kwargs)

    monkeypatch.setattr(engine, "phase", fail_discriminator)
    with pytest.raises(RuntimeError, match="Injected"):
        method.update(batches)
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    with pytest.raises(ValueError, match="Incomplete"):
        method.update(batches)
    monkeypatch.setattr(engine, "phase", phase)
    engine.load_checkpoint(checkpoint, trusted=True)
    assert method.updates == 0 and not method._incomplete and method.update(batches).updated


def test_adversarial_uninitialized_or_batchnorm_setup_is_rejected_without_extra_roles():
    generator, discriminator, metric, _ = objects()
    engine = Trainer(generator)
    uninitialized = PatchDiscriminator(discriminator.config)
    with pytest.raises(ValueError, match="Initialize discriminator"):
        AdversarialAutoencoderMethod(engine, metric, uninitialized)
    assert list(engine.roles) == ["model"] and not engine.states
