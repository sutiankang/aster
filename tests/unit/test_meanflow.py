from copy import deepcopy
import pytest
import torch
from torch import nn

from aster.core import FieldOutput
from aster.models.interval_dit import IntervalDiTConfig, IntervalDiT
from aster.methods.meanflow import (
    MeanFlowObjective,
    meanflow_directional_derivative,
    sample_meanflow,
    sample_meanflow_times,
)
from aster.training import Trainer


class AnalyticField(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = IntervalDiTConfig(input_size=2, in_channels=1, num_classes=2)
        self.coefficients = nn.Parameter(torch.tensor([0.7, -0.2, 0.3, 0.12]))

    def forward(self, x, t, h, labels):
        a, b, c, d = self.coefficients
        return FieldOutput(
            a * x + (b * t + c * h + d * labels)[:, None, None, None], "average_velocity"
        )


@pytest.mark.parametrize("guided", [False, True])
def test_meanflow_analytic_guidance_jvp_all_derivatives_and_stopped_target(guided):
    torch.manual_seed(543)
    model = AnalyticField()
    clean, noise = torch.randn(3, 1, 2, 2), torch.randn(3, 1, 2, 2)
    t, r, labels = (
        torch.tensor([0.8, 0.7, 0.3]),
        torch.tensor([0.1, 0.7, 0.0]),
        torch.tensor([0, 1, 0]),
    )
    objective = MeanFlowObjective(
        guidance=guided, omega=1.7, kappa=0.3, norm_power=0.5, norm_epsilon=0.03
    )
    actual = objective(
        model,
        dict(sample=clean, noise=noise, labels=labels, time=t, reference_time=r, drop_count=1),
    )
    coefficients = model.coefficients.detach().clone().requires_grad_()
    a, b, c, d = coefficients
    z = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    velocity = noise - clean
    with torch.no_grad():
        uncond = a * z + (b * t + d * 2)[:, None, None, None]
        cond = a * z + (b * t + d * labels)[:, None, None, None]
        target_velocity = velocity.clone()
        if guided:
            target_velocity = 1.7 * velocity + (1 - 1.7 - 0.3) * uncond + 0.3 * cond
        target_velocity[0] = velocity[0]
        tangent = a * target_velocity + b + c
        target = target_velocity - (t - r)[:, None, None, None] * tangent
    actual_tangent = meanflow_directional_derivative(model, z, t, t - r, target_velocity, labels)
    torch.testing.assert_close(actual_tangent, tangent)
    active = labels.clone()
    active[0] = 2
    predicted = a * z + (b * t + c * (t - r) + d * active)[:, None, None, None]
    squared = (predicted - target).square().flatten(1).sum(1)
    expected = (squared / (squared.detach() + 0.03) ** 0.5).mean()
    torch.testing.assert_close(actual.mean, expected)
    actual.mean.backward()
    expected.backward()
    torch.testing.assert_close(model.coefficients.grad, coefficients.grad)


def test_meanflow_time_sampling_and_one_step_formula():
    generator, oracle = torch.Generator().manual_seed(431), torch.Generator().manual_seed(431)
    first, second = (
        (torch.randn(7, generator=oracle) - 0.4).sigmoid(),
        (torch.randn(7, generator=oracle) - 0.4).sigmoid(),
    )
    t, r = sample_meanflow_times(7, generator=generator)
    expected_t, expected_r = torch.maximum(first, second), torch.minimum(first, second)
    expected_r[:5] = expected_t[:5]
    torch.testing.assert_close(t, expected_t, atol=0, rtol=0)
    torch.testing.assert_close(r, expected_r, atol=0, rtol=0)
    model = AnalyticField()
    noise = torch.ones(2, 1, 2, 2)
    labels = torch.tensor([0, 1])
    expected = noise - model(noise, torch.ones(2), torch.ones(2), labels).prediction
    torch.testing.assert_close(sample_meanflow(model, noise, labels=labels), expected)
    with pytest.raises(ValueError, match="decrease"):
        sample_meanflow(model, noise, labels=labels, timesteps=(0.0, 1.0))


@pytest.mark.parametrize("stage,precision", [(0, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")])
def test_native_meanflow_jvp_training_all_zero_exact_resume(stage, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(391)
    config = IntervalDiTConfig(
        input_size=4, in_channels=1, hidden_size=16, num_layers=1, num_heads=2, num_classes=2
    )
    model = IntervalDiT(config)
    engine = Trainer(
        model, MeanFlowObjective(), zero_stage=stage, precision=precision, lr=0.001, ema_decay=0.9
    )
    batch = dict(sample=torch.randn(3, 1, 4, 4), labels=torch.tensor([0, 1, 0]))
    assert engine.step([batch]).updated
    path = engine.save_checkpoint(tmp_path / "complete")
    expected = engine.step([batch])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(path, trusted=True)
    actual = engine.step([batch])
    assert actual.loss == expected.loss
    for name, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    deployed = IntervalDiT(config)
    deployed.load_state_dict(weights)
    assert (
        sample_meanflow(
            deployed, batch["sample"], labels=batch["labels"], timesteps=(1.0, 0.5, 0.0)
        )
        .isfinite()
        .all()
    )


def test_meanflow_preflight_rejects_before_rng_and_ignores_no_fields():
    torch.manual_seed(548)
    model = IntervalDiT(
        IntervalDiTConfig(
            input_size=4, in_channels=1, hidden_size=16, num_layers=1, num_heads=2, num_classes=2
        )
    )
    objective = MeanFlowObjective()
    engine = Trainer(model, objective, zero_stage=3)
    batch = dict(sample=torch.randn(2, 1, 4, 4), labels=torch.tensor([0, 1]), drop_count=5)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="drop_count"):
        engine.step([batch])
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_meanflow_one_step_learning_regression():
    torch.set_num_threads(1)
    torch.manual_seed(810)
    config = IntervalDiTConfig(
        input_size=4, in_channels=1, hidden_size=32, num_layers=1, num_heads=2, num_classes=2
    )
    engine = Trainer(
        IntervalDiT(config),
        MeanFlowObjective(guidance=False, class_dropout=0.0, norm_power=0.5),
        lr=0.003,
    )
    labels = torch.tensor([0, 1] * 4)
    targets = (labels.float() - 0.5)[:, None, None, None].expand(8, 1, 4, 4) * 0.8
    noise = torch.randn_like(targets)
    initial = (sample_meanflow(engine.model, noise, labels=labels) - targets).square().mean().item()
    for _ in range(120):
        engine.step([dict(sample=targets + 0.02 * torch.randn_like(targets), labels=labels)])
    final = (sample_meanflow(engine.model, noise, labels=labels) - targets).square().mean().item()
    assert final < 0.12 and final < initial * 0.15
