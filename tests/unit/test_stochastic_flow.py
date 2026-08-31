from copy import deepcopy
import pytest
import torch
from torch import nn

from aster.core import FieldOutput
from aster.methods import GaussianFlowPath, GaussianFlowObjective
from aster.methods.factory import build_objective
from aster.methods.flow_transport import integrate_flow
from aster.models import DiTConfig, build_model
from aster.training import Trainer


@pytest.mark.parametrize(
    "kind,sigma", [("conditional", 0.3), ("target", 0.2), ("schrodinger", 0.7)]
)
def test_gaussian_path_matches_author_formula_and_input_gradients(kind, sigma):
    torch.manual_seed(198)
    values = [torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    source, target, noise = values
    time = torch.tensor([0.11, 0.48, 0.93], dtype=torch.float64)
    actual = GaussianFlowPath(kind, sigma).sample(source, target, time, noise)
    t = time[:, None, None]
    mean = t * target if kind == "target" else (1 - t) * source + t * target
    std = (
        1 - (1 - sigma) * t
        if kind == "target"
        else (sigma * (t * (1 - t)).sqrt() if kind == "schrodinger" else torch.full_like(t, sigma))
    )
    expected = mean + std * noise
    velocity = (
        (target - (1 - sigma) * expected) / (1 - (1 - sigma) * t)
        if kind == "target"
        else target
        - source
        + (
            (1 - 2 * t) / (2 * t * (1 - t) + 1e-8) * (expected - mean)
            if kind == "schrodinger"
            else 0
        )
    )
    torch.testing.assert_close(actual.sample, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual.velocity, velocity, atol=1e-14, rtol=1e-14)
    left = torch.autograd.grad(
        (actual.sample.square() + actual.velocity.square()).sum(), values, allow_unused=True
    )
    right = torch.autograd.grad(
        (expected.square() + velocity.square()).sum(), values, allow_unused=True
    )
    for a, b in zip(left, right):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_target_path_endpoint_is_smoothed_data_and_ode_recovers_it():
    class TargetVelocity(nn.Module):
        def forward(self, sample, time, condition=None):
            return FieldOutput((condition - 0.8 * sample) / (1 - 0.8 * time[:, None]), "velocity")

    noise = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    target = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    actual = integrate_flow(TargetVelocity(), noise, torch.linspace(0, 1, 33), condition=target)
    torch.testing.assert_close(actual, target + 0.2 * noise, atol=1e-10, rtol=1e-10)
    endpoint = GaussianFlowPath("target", 0.0).sample(noise, target, torch.ones(2), noise)
    torch.testing.assert_close(endpoint.sample, target)
    torch.testing.assert_close(endpoint.velocity, target - noise)


def test_ot_condition_indices_reach_real_objective_and_count_is_integer():
    class Capture(nn.Module):
        def forward(self, sample, time, condition=None):
            self.condition, self.sample = condition, sample
            return FieldOutput(torch.zeros_like(sample), "velocity")

    data = torch.tensor([[2.0], [0.0], [1.0]])
    noise = torch.tensor([[0.0], [1.0], [2.0]])
    condition = {
        "classes": torch.tensor([20, 0, 10]),
        "nested": (torch.tensor([[2.0], [0.0], [1.0]]),),
    }
    model = Capture()
    objective = GaussianFlowObjective(coupling="exact")
    term = objective(
        model, dict(sample=data, noise=noise, condition=condition, time=torch.full((3,), 0.4))
    )
    torch.testing.assert_close(model.sample, noise)
    assert model.condition["classes"].tolist() == [0, 10, 20]
    assert model.condition["nested"][0].flatten().tolist() == [0.0, 1.0, 2.0]
    assert term.numerator == 0 and term.denominator.dtype == torch.int64


@pytest.mark.parametrize("stage,coupling", [(0, "sinkhorn"), (3, "exact")])
def test_native_dit_sb_training_and_rng_exact_checkpoint(stage, coupling, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(199)
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
    objective = build_objective(
        dict(name="gaussian_flow", path=dict(kind="schrodinger", sigma=1.0), coupling=coupling)
    )
    engine = Trainer(model, objective, lr=0.002, zero_stage=stage)
    data = {"sample": torch.randn(3, 1, 4, 4) * 0.1}
    assert engine.step([data]).updated
    engine.save_checkpoint(tmp_path / "flow")
    expected = engine.step([data])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(tmp_path / "flow", trusted=True)
    actual = engine.step([data])
    assert actual.loss == expected.loss and actual.updated
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_gaussian_paths_reject_ambiguous_or_singular_inputs():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="open interval"):
        GaussianFlowPath("schrodinger", 1.0).sample(x, x, torch.tensor([0.0, 1.0]), x)
    with pytest.raises(ValueError, match="finite"):
        GaussianFlowPath().sample(x, x, torch.tensor([float("nan"), 0.5]), x)
    with pytest.raises(ValueError, match="sigma"):
        GaussianFlowPath("schrodinger", 0.0)
    with pytest.raises(ValueError, match="source endpoint"):
        GaussianFlowObjective(GaussianFlowPath("target"), coupling="exact")
    with pytest.raises(ValueError, match="ambiguous"):
        GaussianFlowObjective(GaussianFlowPath("target"))(
            None, dict(sample=x, noise=x, perturbation=x)
        )
    with pytest.raises(ValueError, match="Unsupported"):
        GaussianFlowObjective()(None, dict(sample=x, loss_mask=x))
