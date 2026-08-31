from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

from aster.methods.adversarial_autoencoder import (
    AdversarialGeneratorObjective,
    AdversarialDiscriminatorObjective,
)
from aster.methods.perceptual_autoencoder import PerceptualAutoencoderObjective
from aster.models.adversarial import PatchDiscriminator, PatchDiscriminatorConfig
from aster.models.generative import AutoencoderKL, AutoencoderConfig
from aster.models.perceptual import LPIPS, LPIPSConfig


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(512)
    generator = AutoencoderKL(
        AutoencoderConfig(base_channels=4, latent_channels=2, channel_mult=(1, 2), num_res_blocks=1)
    )
    metric = LPIPS(LPIPSConfig(channels=(2, 3, 4, 4, 4), allow_untrained=True))
    reconstruction = PerceptualAutoencoderObjective(
        metric, pixel_reduction="mean", logvar=0.2, perceptual_weight=0.3
    )
    discriminator = PatchDiscriminator(PatchDiscriminatorConfig(base_channels=4, num_layers=1))
    batch = dict(sample=torch.rand(2, 3, 16, 16) * 2 - 1, posterior_noise=torch.randn(2, 2, 8, 8))
    discriminator.initialize(batch["sample"])
    return generator, discriminator, metric, reconstruction, batch


@pytest.mark.parametrize("active", [True, False])
def test_generator_reuses_same_reconstruction_with_nll_kl_and_gan_gradients(active):
    generator, discriminator, metric, reconstruction, batch = setup()
    discriminator.requires_grad_(False)
    objective = AdversarialGeneratorObjective(
        reconstruction, discriminator, disc_factor=0.7, active=active
    )
    calls = []
    hook = generator.encoder.register_forward_pre_hook(lambda *_: calls.append(True))
    terms = objective(generator, batch).terms
    hook.remove()
    assert len(calls) == 1
    clean = batch["sample"]
    posterior = generator.encode(clean)
    fake = generator.decode(
        posterior.mean + (0.5 * posterior.logvar).exp() * batch["posterior_noise"]
    )
    nll = (((clean - fake).abs() + 0.3 * metric(clean, fake)) * math.exp(-0.2) + 0.2).mean()
    kl = (
        0.5
        * (posterior.mean.square() + posterior.logvar.exp() - 1 - posterior.logvar)
        .flatten(1)
        .sum(1)
        .mean()
    )
    g_loss = -discriminator(fake).mean()
    expected = nll + reconstruction.kl_weight * kl + (g_loss * 0.7 if active else 0.0)
    actual = sum(t.mean * t.weight for t in terms)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    a = torch.autograd.grad(actual, tuple(generator.parameters()), retain_graph=True)
    b = torch.autograd.grad(expected, tuple(generator.parameters()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert all(p.grad is None for p in discriminator.parameters())
    assert all(not p.requires_grad and p.grad is None for p in metric.parameters())


@pytest.mark.parametrize("loss", ["hinge", "vanilla"])
def test_discriminator_formula_fresh_fake_and_warmup_zero_gradient_optimizer_clock(loss):
    generator, discriminator, _, reconstruction, batch = setup()
    generator.requires_grad_(False)
    objective = AdversarialDiscriminatorObjective(
        generator, reconstruction, loss=loss, disc_factor=0.7
    )
    actual = objective(discriminator, batch).mean
    with torch.no_grad():
        fake, _ = reconstruction.reconstruct(generator, batch)
    real_logits, fake_logits = discriminator(batch["sample"]), discriminator(fake)
    expected = (
        0.35 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())
        if loss == "hinge"
        else 0.35 * (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean())
    )
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, tuple(discriminator.parameters()), retain_graph=True)
    b = torch.autograd.grad(expected, tuple(discriminator.parameters()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)

    with torch.no_grad():
        generator.decoder[-1].bias.add_(0.5)
    assert not torch.equal(objective(discriminator, batch).mean, actual)
    objective.active = False
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.9))
    before = deepcopy(discriminator.state_dict())
    warm = objective(discriminator, batch)
    assert warm.weight == 1 and warm.mean == 0
    warm.mean.backward()
    optimizer.step()
    assert all(
        p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in discriminator.parameters()
    )
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    assert all(torch.equal(v, discriminator.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in generator.parameters())
