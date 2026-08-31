from dataclasses import replace
import pytest
import torch
from aster.models import Cosmos3VLMConfig, Cosmos3VLM, Wan22VAEConfig, Wan22VideoVAE, load_model
from aster.models.cosmos3_audio import Cosmos3AudioConfig, Cosmos3AudioCodec
from aster.models.qwen_vl import pack_qwen_pixels
from aster.methods.cosmos3 import Cosmos3AudioVideoPipeline, Cosmos3VisualFlowObjective
from aster.training import Trainer


@pytest.mark.parametrize("stage", [0, 3])
def test_models_cosmos3_actual_pixels_waveform_video_joint_train_export_ode_decode(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(689)
    config = Cosmos3VLMConfig()
    model = Cosmos3VLM(config)
    vae = Wan22VideoVAE(Wan22VAEConfig())

    audio_codec = Cosmos3AudioCodec(Cosmos3AudioConfig(sampling_rate=240, normalize_volume=False))
    pipe = Cosmos3AudioVideoPipeline(model, vae, audio_codec)
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), config.vision_config)
    inputs = dict(
        input_ids=torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3]]),
        pixel_values=pixels,
        image_grid_thw=grid,
    )
    video = torch.randn(1, 3, 5, 16, 16).tanh()
    audio = torch.randn(1, 2, 50) * 0.2
    noise = dict(vision=torch.randn(1, 2, 2, 1, 1), sound=torch.randn(1, 5, 4))
    batch = pipe.training_batch(
        video,
        audio,
        inputs,
        sampling_rate=240,
        fps=24,
        noisy_frames=torch.tensor([[False, True]]),
        timesteps=torch.full((1, 2), 700.0),
        audio_timesteps=torch.full((1, 5), 700.0),
        noise=noise,
    )
    batch["labels"] = inputs["input_ids"]
    positions = batch["model_inputs"]["sound"].positions
    torch.testing.assert_close(
        positions[0, 0, 1:] - positions[0, 0, :-1],
        torch.full((4,), 24 / (240 / 12)),
        atol=0.001,
        rtol=0,
    )
    assert positions[0, 0, 0] == batch["model_inputs"]["vision"].positions[0, 0, 0]
    objective = Cosmos3VisualFlowObjective(
        text_weight=0.1, sound_weight=0.3, time_distribution="provided"
    )
    trainer = Trainer(model, objective, zero_stage=stage, lr=0.002)
    first = trainer.step([batch]).loss
    for _ in range(9):
        last = trainer.step([batch]).loss
    assert last < first * 0.7
    assert all(p.grad is None for codec in (vae, audio_codec) for p in codec.parameters())
    restored = Cosmos3VLM(config)
    restored.load_state_dict(trainer.export_state_dict())
    for name, module in (("mot", restored), ("video", vae), ("audio", audio_codec)):
        module.save_pretrained(tmp_path / name)
    pipe = Cosmos3AudioVideoPipeline(
        load_model(tmp_path / "mot"), load_model(tmp_path / "video"), load_model(tmp_path / "audio")
    )
    result = pipe.generate(
        inputs,
        noise["vision"],
        sound_noise=noise["sound"],
        fps=24,
        condition_video=video[:, :, :1],
        steps=3,
        solver="heun",
    )
    assert result.video.shape == video.shape and result.sound.shape == (1, 2, 60)
    assert result.sampling_rate == 240 and result.fps == 24 and torch.isfinite(result.sound).all()
    assert result.sound.abs().max() <= 1
    torch.testing.assert_close(
        result.latents["vision"][:, :, :1], pipe.encode_video(video[:, :, :1]), atol=0, rtol=0
    )

    other = pipe.generate(
        inputs,
        noise["vision"],
        sound_noise=noise["sound"] + 0.7,
        fps=24,
        condition_video=video[:, :, :1],
        steps=3,
        solver="heun",
    )
    assert not torch.allclose(result.latents["vision"][:, :, 1:], other.latents["vision"][:, :, 1:])
    with pytest.raises(ValueError, match="duration"):
        pipe.generate(inputs, noise["vision"], sound_noise=noise["sound"][:, :-1])
    with pytest.raises(ValueError, match="rate mismatch"):
        pipe.encode_audio(audio, sampling_rate=48000)


def test_models_cosmos3_audio_codec_explicit_precision_and_frozen_modes():
    torch.set_num_threads(1)
    torch.manual_seed(690)
    model = Cosmos3VLM(Cosmos3VLMConfig())
    vae = Wan22VideoVAE(Wan22VAEConfig())
    audio = Cosmos3AudioCodec(Cosmos3AudioConfig()).bfloat16().train()
    pipe = Cosmos3AudioVideoPipeline(model, vae, audio)
    waveform = torch.randn(1, 2, 47)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        latent = pipe.encode_audio(waveform, sampling_rate=48000)
    expected = (
        audio.eval().encode(waveform.bfloat16(), force_pad=True).mode().transpose(1, 2).float()
    )
    torch.testing.assert_close(latent, expected, atol=0, rtol=0)
    audio.train()
    decoded = pipe.decode_audio(latent)
    assert audio.training and decoded.dtype == torch.float32 and not decoded.requires_grad
