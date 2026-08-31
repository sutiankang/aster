import pytest
import torch
from aster.models.generative import AutoencoderKL, AutoencoderConfig, UNet2D, UNetConfig
from aster.pipelines import LatentGenerationPipeline, LatentPipelineConfig, LatentFieldObjective
from aster.training import Trainer
from aster.methods.generation import DiffusionObjective, DiffusionSchedule


def components(prediction_type="velocity"):
    vae = AutoencoderKL(
        AutoencoderConfig(
            in_channels=3,
            latent_channels=2,
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            scaling_factor=0.5,
            shift_factor=0.2,
        )
    )
    field = UNet2D(
        UNetConfig(
            in_channels=2,
            model_channels=8,
            channel_mult=(1, 2),
            attention_levels=(1,),
            num_res_blocks=1,
            prediction_type=prediction_type,
        )
    )
    return vae, field


def test_latent_encode_train_sample_export_and_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    vae, field = components()
    pixels = torch.randn(2, 3, 8, 8)
    objective = LatentFieldObjective(vae, encoder_identity="vae-fixture", sample_posterior=False)
    trainer = Trainer(field, objective, lr=0.001)
    trainer.add_role("autoencoder", vae, trainable=False)
    before = [p.detach().clone() for p in vae.parameters()]
    assert trainer.step([{"pixels": pixels}]).updated
    for p, q in zip(before, vae.parameters()):
        torch.testing.assert_close(p, q)
    pipeline = LatentGenerationPipeline(vae, field, LatentPipelineConfig(steps=2))
    latent = pipeline.encode(pixels)
    torch.testing.assert_close(latent, (vae.encode(pixels).mode() - 0.2) * 0.5)
    torch.testing.assert_close(pipeline.decode(latent), vae.decode(vae.encode(pixels).mode()))
    noise = torch.randn_like(latent)
    output = pipeline.sample(noise)
    assert output.shape == pixels.shape and torch.isfinite(output).all()
    pipeline.save_pretrained(tmp_path / "pipeline")
    restored = LatentGenerationPipeline.from_pretrained(tmp_path / "pipeline")
    torch.testing.assert_close(restored.sample(noise), output)
    with pytest.raises(ValueError, match="different encoder"):
        objective(field, {"latent": latent, "encoder_identity": "wrong"})


@pytest.mark.parametrize(
    "method,solver", [("diffusion", "ddim"), ("diffusion", "ddpm"), ("edm", "heun")]
)
def test_latent_other_parameterizations(method, solver):
    torch.set_num_threads(1)
    vae, field = components("edm_residual" if method == "edm" else "epsilon")
    pipeline = LatentGenerationPipeline(
        vae,
        field,
        LatentPipelineConfig(
            method=method, solver=solver, steps=3, diffusion_steps=8, sigma_max=2.0
        ),
    )
    result = pipeline.sample(torch.randn(1, 2, 4, 4))
    assert result.shape == (1, 3, 8, 8) and torch.isfinite(result).all()
