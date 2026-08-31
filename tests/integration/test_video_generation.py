import copy

import pytest
import torch
import torch.nn.functional as F

from aster.models.video_world import WanVideoConfig, WanVideoDiT
from aster.models.video_vae import (
    WanVAEConfig,
    WanVideoVAE,
    CausalConv3D,
    causal_layer,
    VideoResample,
)
from aster.methods.video_generation import (
    WanVideoObjective,
    VideoGenerationPipeline,
    image_video_condition,
    sample_video_latents,
)
from aster.methods.generation import AutoencoderObjective
from aster.training import Trainer


def tiny_vae():
    return WanVideoVAE(
        WanVAEConfig(
            base_channels=4,
            latent_channels=2,
            channel_mult=(1, 2, 2),
            temporal_downsample=(True, True),
            num_res_blocks=1,
            latent_mean=(0.1, -0.2),
            latent_std=(0.8, 1.2),
        )
    )


def tiny_field(image=False):
    return WanVideoDiT(
        WanVideoConfig(
            latent_channels=2,
            condition_channels=6 if image else 0,
            image_conditioned=image,
            hidden_size=24,
            intermediate_size=32,
            num_heads=2,
            num_layers=1,
            text_dim=8,
            text_length=4,
            frequency_dim=8,
            image_dim=6,
        )
    )


def test_causal_convolution_and_temporal_resample_formulas():
    torch.manual_seed(24)
    torch.set_num_threads(1)
    conv = CausalConv3D(2, 3, 3, padding=1)
    video = torch.randn(1, 2, 7, 4, 4, requires_grad=True)
    full = F.conv3d(F.pad(video, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
    cache = {}
    parts = [causal_layer(conv, video[:, :, a:b], cache) for a, b in ((0, 1), (1, 5), (5, 7))]
    torch.testing.assert_close(torch.cat(parts, 2), full)
    up = VideoResample(2, up=True, temporal=True)
    x = torch.randn(1, 2, 3, 2, 2)
    state = {}
    chunks = [up(x[:, :, i : i + 1], state) for i in range(3)]
    assert [part.shape[2] for part in chunks] == [1, 2, 2]

    raw = F.conv3d(F.pad(x[:, :, 1:], (0, 0, 0, 0, 2, 0)), up.time_conv.weight, up.time_conv.bias)
    raw = raw.reshape(1, 2, 2, 2, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(1, 2, 4, 2, 2)
    frames = raw.permute(0, 2, 1, 3, 4).reshape(4, 2, 2, 2)
    expected = up.resample(frames).reshape(1, 4, 1, 4, 4).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(torch.cat(chunks[1:], 2), expected)


def test_video_vae_prefix_causality_streaming_gradients_and_training(tmp_path):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = tiny_vae().eval()
    video = torch.randn(1, 3, 9, 8, 8, requires_grad=True)
    latent = model.latent(video)
    prefix = model.latent(video[:, :, :5])
    assert latent.shape == (1, 2, 3, 2, 2)
    torch.testing.assert_close(latent[:, :, :2], prefix, atol=1e-6, rtol=1e-5)
    decoded = model.decode(latent, scaled=True)
    decoded_prefix = model.decode(latent[:, :, :2], scaled=True)
    torch.testing.assert_close(decoded[:, :, :5], decoded_prefix, atol=1e-6, rtol=1e-5)
    assert [v.shape[2] for v in model.decode_chunks(latent, scaled=True)] == [1, 4, 4]
    decoded[:, :, :1].sum().backward()
    assert video.grad[:, :, 1:].abs().sum() == 0 and video.grad[:, :, :1].abs().sum() > 0
    model.train()
    engine = Trainer(model, AutoencoderObjective(kl_weight=1e-6), lr=0.001)
    result = engine.step([{"sample": video.detach()}])
    assert result.updated
    model.save_pretrained(tmp_path / "vae")
    restored = WanVideoVAE.from_pretrained(tmp_path / "vae").eval()
    model.eval()
    torch.testing.assert_close(restored.latent(video), model.latent(video), rtol=0, atol=0)
    with pytest.raises(ValueError, match=r"1\+k"):
        model.encode(video[:, :, :8])


def test_video_latent_flow_fit_exact_resume_and_image_pipeline(tmp_path):
    torch.manual_seed(28)
    torch.set_num_threads(1)
    vae, field = tiny_vae().eval(), tiny_field(image=True)
    pipeline = VideoGenerationPipeline(field, vae)
    video, text, image = torch.randn(1, 3, 5, 8, 8), torch.randn(1, 3, 8), torch.randn(1, 3, 6)
    batch = pipeline.training_batch(video, text, image_features=image)
    changed_future = video.clone()
    changed_future[:, :, 1:] += 7
    alternative = pipeline.training_batch(changed_future, text, image_features=image)
    torch.testing.assert_close(
        batch["condition"]["video_condition"],
        alternative["condition"]["video_condition"],
        rtol=0,
        atol=0,
    )
    cond = batch["condition"]["video_condition"]
    assert cond.shape == (1, 6, 2, 2, 2)
    assert cond[:, :4, :1].eq(1).all() and cond[:, :4, 1:].eq(0).all()
    batch.update(noise=torch.randn_like(batch["sample"]), time=torch.tensor([0.6]))
    objective = WanVideoObjective()
    engine = Trainer(field, objective, lr=0.003)
    initial = float(objective(field, batch).mean.detach())
    for _ in range(50):
        assert engine.step([batch]).updated
    final = float(objective(field, batch).mean.detach())
    assert final < initial * 0.12, (initial, final)
    stochastic = {k: v for k, v in batch.items() if k not in {"noise", "time"}}
    engine.save_checkpoint(tmp_path / "flow")
    engine.step([stochastic])
    expected = copy.deepcopy(field.state_dict())
    engine.load_checkpoint(tmp_path / "flow", trusted=True)
    engine.step([stochastic])
    for key, value in field.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    pipeline.eval()
    generated = pipeline.generate(batch["noise"], batch["condition"], steps=3)
    assert generated.shape == video.shape and generated.isfinite().all()
    field.save_pretrained(tmp_path / "field")
    restored = WanVideoDiT.from_pretrained(tmp_path / "field").eval()
    torch.testing.assert_close(
        sample_video_latents(restored, batch["noise"], batch["condition"], steps=3),
        sample_video_latents(field, batch["noise"], batch["condition"], steps=3),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="negative"):
        sample_video_latents(field, batch["noise"], batch["condition"], guidance_scale=4.0)


@pytest.mark.parametrize("kind", ["field", "vae"])
def test_native_video_components_enter_real_zero3_update_and_export(kind):
    torch.manual_seed(61)
    torch.set_num_threads(1)
    if kind == "field":
        model, objective = tiny_field(), WanVideoObjective()
        batch = {"sample": torch.randn(1, 2, 2, 2, 2), "condition": {"text": torch.randn(1, 3, 8)}}
    else:
        model, objective = tiny_vae(), AutoencoderObjective()
        batch = {"sample": torch.randn(1, 3, 5, 8, 8)}
    config, model_type = model.config, type(model)
    before = copy.deepcopy(model.state_dict())
    engine = Trainer(model, objective, lr=0.001, zero_stage=3)
    assert engine.step([batch]).updated
    exported = engine.export_state_dict()
    assert any(not torch.equal(value, exported[key]) for key, value in before.items())
    deployed = model_type(config)
    deployed.load_state_dict(exported, strict=True)
    deployed.eval()
    if kind == "field":
        assert (
            deployed(batch["sample"], torch.tensor([0.5]), batch["condition"])
            .prediction.isfinite()
            .all()
        )
    else:
        assert deployed.latent(batch["sample"]).isfinite().all()
