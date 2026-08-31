import torch
from torch import nn
from aster.core import FieldOutput
from aster.methods.generation import DiffusionSchedule, karras_sigmas
from aster.methods.solvers import (
    SDE,
    ScoreSDEObjective,
    sample_score_sde,
    sample_dpmpp_2m,
    ProgressiveDistillationObjective,
)


def test_sde_marginal_and_perfect_score_loss():
    clean, noise = torch.randn(2, 3), torch.randn(2, 3)
    time = torch.tensor([0.2, 0.8])
    for kind in ("vp", "ve", "subvp"):
        sde = SDE(kind=kind)
        _, std = sde.marginal(time)

        class Oracle(nn.Module):
            def forward(self, x, t, condition=None):
                return FieldOutput(-noise / std[:, None], "score")

        term = ScoreSDEObjective(sde)(Oracle(), {"sample": clean, "noise": noise, "time": time})
        assert term.mean < 1e-10


def test_dpmpp_exact_constant_denoiser():
    target = torch.randn(2, 4)

    def denoiser(x, sigma, condition):
        return target

    torch.testing.assert_close(
        sample_dpmpp_2m(denoiser, torch.randn_like(target), karras_sigmas(6)), target
    )


def test_probability_flow_gaussian_stationarity():
    class GaussianScore(nn.Module):
        def forward(self, x, t, condition=None):
            return FieldOutput(-x, "score")

    noise = torch.randn(2, 4)

    torch.testing.assert_close(
        sample_score_sde(GaussianScore(), noise, steps=10, probability_flow=True), noise
    )


def test_progressive_teacher_transport_target():
    target = torch.randn(2, 1, 3, 3, dtype=torch.float64)

    class Oracle(nn.Module):
        def forward(self, x, t, condition=None):
            return FieldOutput(target, "x0")

    objective = ProgressiveDistillationObjective(Oracle(), DiffusionSchedule.create(20))
    result = objective(
        Oracle(),
        {
            "sample": target,
            "time_high": torch.tensor([19, 15]),
            "time_middle": torch.tensor([10, 8]),
            "time_low": torch.tensor([-1, 0]),
        },
    )
    assert result.mean < 1e-20
