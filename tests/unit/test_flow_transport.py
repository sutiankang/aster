import itertools
import math
import pytest
import torch
from torch import nn

from aster.core import FieldOutput
from aster.methods.flow_transport import (
    exact_assignment,
    sinkhorn_plan,
    transport_pairing,
    integrate_flow,
    flow_log_likelihood,
)
from aster.methods.generation import FlowObjective
from aster.models import DiTConfig, build_model
from aster.training import Trainer


class LinearVelocity(nn.Module):
    def __init__(self, rate=0.4):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))

    def forward(self, sample, time, condition=None):
        return FieldOutput(sample * self.rate, "velocity")


def test_exact_assignment_matches_bruteforce_and_sinkhorn_marginals():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    cost = torch.rand(5, 5, dtype=torch.float64)
    columns = exact_assignment(cost)
    actual = cost[torch.arange(5), columns].sum()
    oracle = min(
        cost[torch.arange(5), list(permutation)].sum()
        for permutation in itertools.permutations(range(5))
    )
    torch.testing.assert_close(actual, oracle, rtol=0, atol=1e-14)
    rectangular = torch.rand(3, 5, dtype=torch.float64)
    plan = sinkhorn_plan(rectangular, regularization=0.2)
    torch.testing.assert_close(
        plan.sum(1), torch.full((3,), 1 / 3, dtype=torch.float64), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        plan.sum(0), torch.full((5,), 0.2, dtype=torch.float64), atol=1e-7, rtol=0
    )
    with pytest.raises(RuntimeError, match="converge"):
        sinkhorn_plan(rectangular, regularization=0.01, iterations=1, tolerance=1e-14)


@pytest.mark.parametrize("trace", ["exact", "hutchinson"])
def test_flow_inverse_and_likelihood_match_analytic_gaussian(trace):
    torch.set_num_threads(1)
    model = LinearVelocity().eval()
    noise = torch.tensor([[0.1, -0.3, 0.7], [0.2, -0.5, 0.8]], dtype=torch.float64)
    sample = integrate_flow(model, noise, torch.linspace(0, 1, 33))
    torch.testing.assert_close(sample, noise * math.exp(0.4), rtol=1e-9, atol=1e-10)
    inverse = integrate_flow(model, sample, torch.linspace(1, 0, 33))
    torch.testing.assert_close(inverse, noise, rtol=1e-9, atol=1e-10)
    result = flow_log_likelihood(
        model,
        sample,
        torch.linspace(1, 0, 33),
        trace=trace,
        generator=torch.Generator().manual_seed(17),
    )

    expected = -0.5 * (noise.square() + math.log(2 * math.pi)).sum(-1) - 3 * 0.4
    torch.testing.assert_close(result.log_prob, expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        result.divergence_integral, torch.full((2,), -1.2, dtype=torch.float64)
    )
    assert result.function_evaluations == 128 and model.rate.grad is None


def test_ot_pairing_enters_native_field_training():
    torch.manual_seed(33)
    torch.set_num_threads(1)
    target = torch.randn(4, 1, 4, 4)
    noise = torch.randn_like(target)
    pairs = transport_pairing(noise, target)
    assert sorted(pairs["target_indices"].tolist()) == list(range(4))
    assert pairs["mean_cost"] <= (noise - target).square().flatten(1).sum(-1).mean() + 1e-7
    model = build_model(
        DiTConfig(
            in_channels=1,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            patch_size=2,
            prediction_type="velocity",
        )
    )
    engine = Trainer(model, FlowObjective(), lr=0.001)
    assert engine.step(
        [{"sample": pairs["target"], "noise": pairs["source"], "time": torch.full((4,), 0.5)}]
    ).updated


def test_continuous_transport_rejects_nonfinite_hyperparameters_and_invalid_density_probe():
    cost = torch.zeros(2, 2)
    for arguments in (
        {"regularization": float("nan")},
        {"tolerance": float("inf")},
        {"iterations": 1.5},
    ):
        with pytest.raises(ValueError):
            sinkhorn_plan(cost, **arguments)
    model = LinearVelocity().eval()
    sample = torch.ones(2, 3)
    with pytest.raises(ValueError, match="Rademacher"):
        flow_log_likelihood(model, sample, [1.0, 0.0], probe=torch.zeros_like(sample))
    with pytest.raises(ValueError, match="floating"):
        integrate_flow(model, sample.long(), [0.0, 1.0])
