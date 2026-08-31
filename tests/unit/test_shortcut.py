from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from torch import nn

from aster.core import FieldOutput
from aster.models.interval_dit import IntervalDiTConfig, IntervalDiT
from aster.methods.shortcut import (
    ShortcutMethod,
    shortcut_levels,
    shortcut_bootstrap_target,
    sample_shortcut,
)
from aster.training import Trainer


class AnalyticShortcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.config = IntervalDiTConfig(
            variant="shortcut", input_size=2, in_channels=1, num_classes=2
        )
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, value, time, levels, labels):
        self.calls += 1
        if labels is None:
            labels = torch.full_like(time, 2)
        return FieldOutput(
            self.scale * value + (time + levels + labels)[:, None, None, None], "average_velocity"
        )


def test_shortcut_exact_strata_and_bias_frequencies():
    levels, selected = shortcut_levels(10, 3)
    assert levels.tolist() == [2, 2, 2, 1, 1, 1, 0, 0, 0, 0] and selected == 3
    levels, selected = shortcut_levels(16, 4, bias=1)
    assert levels.tolist() == [3, 3, 2, 2] + [1] * 4 + [0] * 8 and selected == 2


def test_shortcut_two_half_steps_clips_and_selected_cfg_not_instant_flow():
    model = AnalyticShortcut()
    x = torch.tensor([[[[3.0, -3.0], [1.0, 2.0]]], [[[0.0, 1.0], [-1.0, 3.0]]]], requires_grad=True)
    t, levels, labels = torch.tensor([0.0, 0.5]), torch.tensor([0, 1]), torch.tensor([0, 1])
    mask = torch.tensor([True, False])
    cfg = 1.5

    def velocity(x, t):
        conditioned = 2 * x + (t + levels + 1 + labels)[:, None, None, None]
        unconditional = 2 * x + (t + levels + 1 + 2)[:, None, None, None]
        guided = unconditional + cfg * (conditioned - unconditional)
        return torch.where(mask[:, None, None, None], guided, conditioned)

    dt = torch.tensor([0.5, 0.25])[:, None, None, None]
    first = velocity(x, t)
    middle = (x + dt * first).clamp(-4, 4)
    expected = ((first + velocity(middle, t + dt.flatten())) / 2).clamp(-4, 4)
    actual = shortcut_bootstrap_target(
        model, x, t, levels, labels, guidance_mask=mask, guidance_scale=cfg
    )
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and model.calls == 2
    assert x.grad is None and model.scale.grad is None
    one_step = sample_shortcut(model, x.detach(), labels=labels)
    torch.testing.assert_close(one_step, x.detach() + 2 * x.detach() + labels[:, None, None, None])
    with pytest.raises(ValueError, match="power of two"):
        sample_shortcut(model, x.detach(), steps=3)


def _build(stage, ema, precision="fp32"):
    config = IntervalDiTConfig(
        variant="shortcut",
        input_size=4,
        in_channels=1,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_classes=2,
    )
    engine = Trainer(
        IntervalDiT(config), lr=0.001, zero_stage=stage, precision=precision, accumulation_steps=2
    )
    method = ShortcutMethod(
        engine,
        base_steps=8,
        bootstrap_every=2,
        bootstrap_ema=ema,
        ema_decay=0.8,
        bootstrap_cfg=True,
    )
    return engine, method


@pytest.mark.parametrize(
    "stage,ema,precision",
    [
        (0, False, "fp32"),
        (3, False, "fp32"),
        (0, True, "fp32"),
        (3, True, "fp32"),
        (3, True, "bf16"),
    ],
)
def test_shortcut_real_training_ema_accumulation_exact_restore(stage, ema, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(429)
    engine, method = _build(stage, ema, precision)
    batches = [
        dict(sample=torch.randn(4, 1, 4, 4), labels=torch.tensor([0, 1, 0, 1])),
        dict(sample=torch.randn(6, 1, 4, 4), labels=torch.tensor([0, 1, 0, 1, 1, 0])),
    ]
    old_target = deepcopy(method.target.state_dict()) if ema else None
    assert method.update(batches).updated
    if ema:
        weights = engine.export_state_dict()
        for key, value in method.target.state_dict().items():
            torch.testing.assert_close(
                value, torch.lerp(old_target[key], weights[key], 0.2), atol=1e-7, rtol=1e-6
            )
    checkpoint = engine.save_checkpoint(tmp_path / "complete")
    expected = method.update(batches)
    snapshots = {role: deepcopy(engine.export_state_dict(role=role)) for role in engine.roles}
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = method.update(batches)
    assert expected.loss == actual.loss
    for role in snapshots:
        for key, value in engine.export_state_dict(role=role).items():
            torch.testing.assert_close(value, snapshots[role][key], atol=0, rtol=0)
    deployed = IntervalDiT(engine.model.config)
    deployed.load_state_dict(engine.export_state_dict())
    assert (
        sample_shortcut(deployed, batches[0]["sample"], labels=batches[0]["labels"], steps=4)
        .isfinite()
        .all()
    )


def test_shortcut_preflight_and_interrupted_ema_boundary(tmp_path):
    engine, method = _build(0, True)
    batch = dict(sample=torch.randn(4, 1, 4, 4), labels=torch.tensor([0, 1, 0, 1]))
    method.update([batch, batch])
    checkpoint = engine.save_checkpoint(tmp_path / "complete")
    expected = method.update([batch, batch])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(checkpoint, trusted=True)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="exactly"):
        method.update([{**batch, "extra": 1}, batch])
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with patch.object(engine, "phase", side_effect=RuntimeError("interruption")):
        with pytest.raises(RuntimeError, match="interruption"):
            method.update([batch, batch])
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    engine.load_checkpoint(checkpoint, trusted=True)
    actual = method.update([batch, batch])
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
