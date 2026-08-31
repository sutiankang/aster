from dataclasses import replace
import torch
import pytest
from aster.models import (
    CosmosPredict1Config,
    CosmosPredict1Condition,
    CosmosPredict1ModelConfig,
    build_model,
    load_model,
)
from aster.methods.cosmos_predict1 import CosmosPredict1Objective, sample_cosmos_predict1
from aster.methods.generation import EDMObjective
from aster.training import Trainer


def data(c):
    return dict(
        sample=torch.randn(2, c.in_channels, 2, 4, 6),
        sigma=torch.tensor([0.4, 0.8]),
        noise=torch.randn(2, c.in_channels, 2, 4, 6),
        condition=CosmosPredict1Condition(
            torch.randn(2, 3, c.crossattn_emb_channels),
            torch.tensor([24.0, 24.0]),
            torch.zeros(2, 2, 3) if c.concat_padding_mask else None,
        ),
    )


@pytest.mark.parametrize("lora,extra", [(True, False), (False, True)])
def test_models_cosmos_predict1_edm_training_and_net_export(tmp_path, lora, extra):
    torch.set_num_threads(1)
    torch.manual_seed(550)
    c = CosmosPredict1Config(use_adaln_lora=lora, extra_per_block_abs_pos_emb=extra)
    model = build_model(c)
    batch = data(c)
    engine = Trainer(model, EDMObjective(), lr=0.003)
    initial = engine.step([batch]).loss
    for _ in range(29):
        final = engine.step([batch]).loss
    assert final < initial * 0.3
    model.save_pretrained(tmp_path / "model")
    restored = load_model(tmp_path / "model")
    torch.testing.assert_close(
        model(batch["sample"], batch["sigma"], batch["condition"]).prediction,
        restored(batch["sample"], batch["sigma"], batch["condition"]).prediction,
        atol=0,
        rtol=0,
    )
    result = sample_cosmos_predict1(
        restored, batch["noise"], batch["condition"], steps=3, sigma_max=2, guidance=0
    )
    assert result.shape == batch["sample"].shape and torch.isfinite(result).all()
    with pytest.raises(ValueError, match="FPS"):
        model(batch["sample"], batch["sigma"], replace(batch["condition"], fps=None))
    with pytest.raises(ValueError, match="common"):
        model(
            batch["sample"],
            batch["sigma"],
            replace(batch["condition"], fps=torch.tensor([24.0, 30.0])),
        )


@pytest.mark.parametrize("stage", [0, 3])
def test_models_cosmos_predict1_kendall_shared_trainer_rng_restore_and_export(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(551)
    c = CosmosPredict1ModelConfig()
    model = build_model(c)
    batch = data(c.net)
    engine = Trainer(model, CosmosPredict1Objective(), zero_stage=stage, lr=0.0005)
    initial = engine.step([batch]).loss
    for _ in range(19):
        final = engine.step([batch]).loss
    assert final < initial * 0.55
    random_batch = dict(sample=batch["sample"], condition=batch["condition"])
    engine.step([random_batch])
    engine.save_checkpoint(tmp_path / "resume")
    expected = engine.step([random_batch])
    weights = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "resume", trusted=True)
    actual = engine.step([random_batch])
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    restored = build_model(c)
    restored.load_state_dict(weights, strict=True)
    restored.save_pretrained(tmp_path / "artifact")
    duplicate = load_model(tmp_path / "artifact")
    torch.testing.assert_close(
        restored.predict_logvar(batch["sigma"]),
        duplicate.predict_logvar(batch["sigma"]),
        atol=0,
        rtol=0,
    )


def test_models_cosmos_predict1_euler_is_not_heun_and_uses_official_guidance_scale():
    from aster.core import FieldOutput

    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, time, condition):
            self.calls.append((x.clone(), time.clone(), condition))
            return FieldOutput(x * 0.2 + condition, "edm_residual")

    noise = torch.tensor([[[[[0.2, -0.4]]]]])
    model = Residual()
    actual = sample_cosmos_predict1(
        model,
        noise,
        1.0,
        negative_condition=-0.5,
        steps=3,
        sigma_min=0.01,
        sigma_max=3,
        guidance=1.5,
    )
    sigma = (
        (
            3 ** (1 / 7)
            + torch.linspace(0, 1, 3, dtype=torch.float64) * (0.01 ** (1 / 7) - 3 ** (1 / 7))
        )
        .pow(7)
        .float()
    )
    sigma = torch.cat((sigma, torch.zeros(1)))
    expected = noise * (10**0.5)
    for s, next_s in zip(sigma[:-1], sigma[1:]):
        denominator = s * s + 0.25
        scaled = expected / denominator.sqrt()
        positive, negative = scaled * 0.2 + 1.0, scaled * 0.2 - 0.5
        residual = positive + 1.5 * (positive - negative)
        predicted = 0.25 / denominator * expected + s * 0.5 / denominator.sqrt() * residual
        expected = expected + (next_s - s) * (expected - predicted) / s
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(model.calls) == 6 and model.training
    torch.testing.assert_close(model.calls[0][0], noise * (10**0.5) / (9.25**0.5))
    torch.testing.assert_close(model.calls[0][1], torch.tensor([3.0]).log() / 4)
    with pytest.raises(ValueError, match="negative_condition"):
        sample_cosmos_predict1(model, noise, 1.0, steps=3)


def test_models_cosmos_predict1_bfloat16_finite_and_explicit_dtype_restore(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(555)
    config = CosmosPredict1Config()
    model = build_model(config).to(torch.bfloat16)
    batch = data(config)
    condition = replace(
        batch["condition"], text_embeddings=batch["condition"].text_embeddings.bfloat16()
    )
    output = model(batch["sample"].bfloat16(), batch["sigma"].bfloat16(), condition)
    assert output.prediction.dtype == torch.bfloat16 and torch.isfinite(output.prediction).all()
    output.prediction.float().square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.save_pretrained(tmp_path / "bf16")
    restored = load_model(tmp_path / "bf16")
    torch.testing.assert_close(
        output.prediction,
        restored(batch["sample"].bfloat16(), batch["sigma"].bfloat16(), condition).prediction,
        atol=0,
        rtol=0,
    )
    sampled = sample_cosmos_predict1(
        restored, batch["noise"].bfloat16(), condition, guidance=0, steps=3, sigma_max=2
    )
    assert sampled.dtype == torch.bfloat16 and torch.isfinite(sampled).all()
