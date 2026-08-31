import pytest
import torch
from torch import nn
from aster.core import FieldOutput
from aster.models.generative import (
    UNet2D,
    UNetConfig,
    DiT,
    DiTConfig,
    AutoencoderKL,
    AutoencoderConfig,
    DiagonalGaussian,
)
from aster.methods.generation import (
    DiffusionSchedule,
    DiffusionObjective,
    FlowPath,
    FlowObjective,
    sample_flow,
    sample_diffusion,
    EDMObjective,
    AutoencoderObjective,
)
from aster.methods.generative_distillation import (
    consistency_prediction,
    dmd_surrogate,
    drifting_loss,
)
from aster.training import Trainer


def test_unet_dit_native_shared_training():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    sample = torch.randn(2, 2, 8, 8)
    unet = UNet2D(
        UNetConfig(
            in_channels=2,
            model_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_levels=(1,),
            num_heads=2,
        )
    )
    schedule = DiffusionSchedule.create(8)
    trainer = Trainer(unet, DiffusionObjective(schedule))
    result = trainer.step(
        [{"sample": sample, "time": torch.tensor([0, 7]), "noise": torch.randn_like(sample)}]
    )
    assert result.updated
    assert sample_diffusion(unet, sample, schedule, method="ddim").shape == sample.shape
    dit = DiT(DiTConfig(in_channels=2, hidden_size=16, num_layers=2, num_heads=2))
    assert Trainer(dit, FlowObjective()).step([{"sample": sample}]).updated
    assert sample_flow(dit, sample, steps=3).isfinite().all()


def test_diffusion_parameterizations_respacing_and_perfect_ddim():
    torch.manual_seed(1)
    schedule = DiffusionSchedule.create(20)

    clean, noise = (
        torch.randn(2, 1, 3, 3, dtype=torch.float64),
        torch.randn(2, 1, 3, 3, dtype=torch.float64),
    )
    time = torch.tensor([2, 19])
    a = schedule.at("alpha_bar", time, clean)
    noisy = schedule.noise(clean, time, noise)
    predictions = {
        "epsilon": noise,
        "x0": clean,
        "v": a.sqrt() * noise - (1 - a).sqrt() * clean,
        "score": -noise / (1 - a).sqrt(),
    }
    for kind, value in predictions.items():
        estimated, epsilon = schedule.clean_and_noise(FieldOutput(value, kind), noisy, time)
        torch.testing.assert_close(estimated, clean, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(epsilon, noise, atol=5e-5, rtol=5e-5)
    short = schedule.respaced([0, 5, 12, 19])
    torch.testing.assert_close(short.alpha_bar, schedule.alpha_bar[[0, 5, 12, 19]])

    class Oracle(nn.Module):
        def forward(self, x, t, condition=None):
            return FieldOutput(clean, "x0")

    torch.testing.assert_close(sample_diffusion(Oracle(), noise, short), clean)


def test_learned_variance_and_edm_gradients():
    model = UNet2D(
        UNetConfig(
            in_channels=1,
            out_channels=2,
            model_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=2,
        )
    )
    objective = DiffusionObjective(DiffusionSchedule.create(8), learned_variance=True)
    bundle = objective(
        model, {"sample": torch.rand(2, 1, 4, 4) * 2 - 1, "time": torch.tensor([0, 4])}
    )
    assert len(bundle.terms) == 2 and all(t.mean.isfinite() for t in bundle.terms)
    sum(t.mean for t in bundle.terms).backward()
    assert model.output[-1].weight.grad.isfinite().all()
    residual = UNet2D(
        UNetConfig(
            in_channels=1,
            model_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=2,
            prediction_type="edm_residual",
        )
    )
    assert Trainer(residual, EDMObjective()).step([{"sample": torch.randn(2, 1, 4, 4)}]).updated


def test_flow_time_conventions_and_integrators():
    data, noise = torch.ones(2, 3), torch.zeros(2, 3)

    class Constant(nn.Module):
        def __init__(self, sign):
            super().__init__()
            self.sign = sign

        def forward(self, x, t, condition=None):
            return FieldOutput(torch.full_like(x, self.sign), "velocity")

    for direction, sign in [("noise_to_data", 1.0), ("data_to_noise", -1.0)]:
        path = FlowPath(direction=direction)
        _, target = path.sample(data, noise, torch.tensor([0.2, 0.5]))
        torch.testing.assert_close(target, torch.full_like(data, sign))
        for solver in ("euler", "heun", "rk4"):
            torch.testing.assert_close(
                sample_flow(Constant(sign), noise, direction=direction, solver=solver, steps=5),
                data,
            )


def test_vae_reconstruction_latent_contract():
    vae = AutoencoderKL(
        AutoencoderConfig(
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            latent_channels=2,
            scaling_factor=0.5,
            shift_factor=0.2,
        )
    )
    images = torch.randn(2, 3, 8, 8)
    posterior = vae.encode(images)
    assert posterior.mean.shape == (2, 2, 4, 4)
    assert (posterior.kl() >= 0).all()
    torch.testing.assert_close(
        vae.decode(vae.latent(images, sample=False), scaled=True), vae.decode(posterior.mean)
    )
    assert Trainer(vae, AutoencoderObjective()).step([{"sample": images}]).updated


@pytest.mark.parametrize(
    "dtype,value",
    [(torch.bfloat16, 0.01), (torch.float16, 12.0), (torch.float32, 0.001), (torch.float64, 0.001)],
)
def test_vae_kl_stable_statistics_and_actual_encoder_gradient(dtype, value):
    mean = torch.zeros(2, 2, 3, 3, dtype=dtype, requires_grad=True)
    logvar = torch.full_like(mean, value, requires_grad=True)
    kl = DiagonalGaussian(mean, logvar).kl()
    expected = 0.5 * (
        mean.double().square() + torch.expm1(logvar.double()) - logvar.double()
    ).flatten(1).sum(1)
    assert kl.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(kl).all() and (kl > 0).all()
    torch.testing.assert_close(kl.double(), expected, rtol=2e-4, atol=1e-10)

    (kl.sum() * 1e-6).backward()
    reference_gradient = (0.5 * torch.expm1(logvar.detach().double()) * 1e-6).to(dtype)
    assert torch.isfinite(logvar.grad).all() and torch.count_nonzero(logvar.grad) == logvar.numel()
    torch.testing.assert_close(logvar.grad, reference_gradient, rtol=2e-4, atol=1e-11)
    assert torch.equal(mean.grad, torch.zeros_like(mean))


def test_distillation_boundaries_and_driving_force():
    class Zero(nn.Module):
        def forward(self, x, t, condition=None):
            return FieldOutput(torch.zeros_like(x), "consistency_residual")

    x = torch.randn(2, 3)
    torch.testing.assert_close(consistency_prediction(Zero(), x, torch.full((2,), 0.002)), x)
    generated = torch.tensor([[1.0, 2.0]], requires_grad=True)
    real, fake = torch.zeros_like(generated), torch.ones_like(generated)
    term = dmd_surrogate(generated, real, fake, normalize=False)
    term.mean.backward()
    torch.testing.assert_close(generated.grad, torch.full_like(generated, 0.5))
    points = torch.randn(2, 3, 4, requires_grad=True)
    positives = torch.randn(2, 5, 4, requires_grad=True)
    term, info = drifting_loss(points, positives)
    term.mean.backward()
    assert points.grad.isfinite().all() and points.grad.abs().sum() > 0
    assert positives.grad is None and info["scale"] > 0
