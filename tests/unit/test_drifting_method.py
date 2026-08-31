from copy import deepcopy
import math
import pytest
import torch
from torch import nn

from aster.models import build_model, load_model
from aster.models.drifting import DriftingConfig, DriftingGenerator
from aster.methods.drifting import (
    ClassMemoryBank,
    DriftingMethod,
    SpatialFeatureStatistics,
    sample_training_cfg,
)
from aster.methods.generative_distillation import drifting_loss
from aster.training import Trainer


def _config():
    return DriftingConfig(
        input_size=4,
        in_channels=1,
        out_channels=1,
        hidden_size=16,
        cond_dim=12,
        patch_size=2,
        num_layers=1,
        num_heads=2,
        num_classes=2,
        n_cls_tokens=2,
        noise_classes=3,
        noise_coords=2,
    )


def _method(stage=0, precision="fp32"):
    engine = Trainer(
        build_model(_config()), lr=0.003, zero_stage=stage, precision=precision, ema_decay=0.9
    )
    features = SpatialFeatureStatistics(patch_sizes=(2,), use_std=False)
    method = DriftingMethod(
        engine,
        features,
        feature_identity="pixels-v1",
        positive_capacity=3,
        negative_capacity=5,
        positive_samples=2,
        negative_samples=2,
        generated_samples=3,
        seed=21,
    )
    return engine, method


def _batch():
    return dict(
        samples=torch.randn(3, 1, 4, 4) * 0.1
        + torch.tensor([-0.4, 0.4, -0.4])[:, None, None, None],
        labels=torch.tensor([0, 1, 0]),
    )


def test_class_bank_fifo_sampling_and_rng_restore():
    bank = ClassMemoryBank(2, 3, (2,))
    data = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    bank.add(data, torch.tensor([0, 0, 1, 0, 0]))
    assert bank.count.tolist() == [3, 1] and bank.cursor.tolist() == [1, 1]
    torch.testing.assert_close(bank.values[0], torch.stack((data[4], data[1], data[3])))
    generator = torch.Generator().manual_seed(5)
    state = generator.get_state()
    sample = bank.sample(torch.tensor([0, 1]), 3, generator=generator)
    assert sample[0].unique(dim=0).shape[0] == 3
    torch.testing.assert_close(sample[1], data[2].expand(3, 2))
    clone = ClassMemoryBank(2, 3, (2,))
    clone.load_state_dict(bank.state_dict())
    generator.set_state(state)
    torch.testing.assert_close(
        sample, clone.sample(torch.tensor([0, 1]), 3, generator=generator), atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="budget"):
        ClassMemoryBank(200, 100, (1000,), max_bytes=10)
    with pytest.raises(ValueError, match="nonempty"):
        ClassMemoryBank(2, 3, (2,)).sample(torch.tensor([1]), 2, generator=generator)


@pytest.mark.parametrize("power", [1.0, 5.0, -0.5])
def test_training_cfg_is_author_inverse_cdf_not_inference_interpolation(power):
    generator = torch.Generator().manual_seed(25)
    reference = torch.Generator().manual_seed(25)
    fraction = torch.rand(10, generator=reference)
    if power == 1:
        expected = (fraction * math.log(4.0)).exp()
    else:
        exponent = 1 - power
        expected = (1 + fraction * (4**exponent - 1)) ** (1 / exponent)
    expected[torch.rand(10, generator=reference) < 0.25] = 1.0
    actual = sample_training_cfg(10, power=power, no_cfg_fraction=0.25, generator=generator)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("stage,precision", [(0, "fp32"), (3, "fp32"), (0, "bf16"), (3, "bf16")])
def test_drifting_real_generator_queue_ema_and_exact_resume(stage, precision, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(381)
    engine, method = _method(stage, precision)
    data = _batch()
    assert method.update([data]).updated
    engine.save_checkpoint(tmp_path / "whole")
    expected = method.update([data])
    weights = deepcopy(engine.export_state_dict())
    queues = deepcopy(method.state_dict())
    engine.load_checkpoint(tmp_path / "whole", trusted=True)
    actual = method.update([data])
    assert actual.loss == expected.loss and method.updates == 2
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    torch.testing.assert_close(method.positive.values, queues["positive"]["values"], atol=0, rtol=0)
    torch.testing.assert_close(method.rng.get_state(), queues["rng"], atol=0, rtol=0)
    deployed = DriftingGenerator(_config())
    deployed.load_state_dict(engine.export_state_dict(ema=True))
    deployed.save_pretrained(tmp_path / "model")
    loaded = load_model(tmp_path / "model")
    labels = torch.tensor([0, 1])
    left = deployed.generate(labels, generator=torch.Generator().manual_seed(12))
    right = loaded.generate(labels, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_spatial_feature_axes_and_frozen_encoder_input_gradient():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, 1)

        def forward(self, value):
            return {"conv": self.conv(value)}

    feature = SpatialFeatureStatistics(Encoder(), patch_sizes=(2,)).eval().requires_grad_(False)
    x = torch.arange(16.0).reshape(1, 1, 4, 4).requires_grad_()
    result = feature(x)
    assert result["conv"].shape == (1, 16, 3) and result["conv_mean_2"].shape == (1, 4, 3)
    raw = feature.encoder(x)["conv"]
    expected = raw[:, :, :2, :2].mean((2, 3))
    torch.testing.assert_close(result["conv_mean_2"][:, 0], expected)
    sum(v.sum() for v in result.values()).backward()
    assert x.grad.abs().sum() > 0 and all(p.grad is None for p in feature.parameters())


def test_drifting_preflight_does_not_mutate_queues_or_rng():
    engine, method = _method()
    data = _batch()
    data["labels"][1] = 3
    rng = method.rng.get_state().clone()
    with pytest.raises(ValueError, match="preflight"):
        method.update([data])
    assert method.positive.count.sum() == 0 and engine.steps == 0 and not method._incomplete
    torch.testing.assert_close(method.rng.get_state(), rng, atol=0, rtol=0)


def test_drifting_generator_rejects_wrong_noise_and_cfg_contract():
    model = DriftingGenerator(_config())
    with pytest.raises(ValueError, match="noise_labels"):
        model(torch.zeros(2, 1, 4, 4), torch.ones(2), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="cfg_scale"):
        model(torch.zeros(2, 1, 4, 4), torch.zeros(2), torch.tensor([0, 1]))


def test_drifting_learns_class_conditional_samples_instead_of_only_finite_loss():
    torch.set_num_threads(1)
    torch.manual_seed(853)
    config = DriftingConfig(
        input_size=4,
        in_channels=1,
        out_channels=1,
        hidden_size=32,
        cond_dim=16,
        num_layers=1,
        num_heads=2,
        num_classes=2,
    )
    engine = Trainer(DriftingGenerator(config), lr=0.002, ema_decay=0.9)
    method = DriftingMethod(
        engine,
        SpatialFeatureStatistics(patch_sizes=(2,), use_std=False),
        feature_identity="pixels-learning-regression",
        positive_capacity=16,
        negative_capacity=32,
        positive_samples=4,
        negative_samples=4,
        generated_samples=8,
        cfg_min=1.0,
        cfg_max=1.0,
    )
    labels = torch.tensor([0, 1, 0, 1])
    target = (labels.float() - 0.5)[:, None, None, None].expand(4, 1, 4, 4) * 0.8
    noise = torch.randn(4, 1, 4, 4)
    initial = (engine.model(noise, 1.0, labels).prediction - target).square().mean().item()
    for _ in range(100):
        method.update([dict(samples=target + 0.05 * torch.randn_like(target), labels=labels)])
    final = (engine.model(noise, 1.0, labels).prediction - target).square().mean().item()
    assert final < 0.02 and final < initial * 0.15


def test_drifting_half_round_rejects_checkpoint_until_complete_restore(tmp_path):
    from unittest.mock import patch

    engine, method = _method()
    batch = _batch()
    method.update([batch])
    path = engine.save_checkpoint(tmp_path / "complete")
    expected = method.update([batch])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(path, trusted=True)
    with patch.object(engine, "phase", side_effect=RuntimeError("simulated interrupted update")):
        with pytest.raises(RuntimeError, match="interrupted"):
            method.update([batch])
    assert method._incomplete
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    engine.load_checkpoint(path, trusted=True)
    actual = method.update([batch])
    assert expected.loss == actual.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
