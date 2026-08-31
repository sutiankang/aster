import torch
import pytest
from aster.models import Cosmos3Config, Wan22VAEConfig, build_model, load_model
from aster.methods.cosmos3 import Cosmos3FlowObjective, Cosmos3VideoPipeline, sample_cosmos3
from aster.training import Trainer


def test_models_cosmos3_true_pixel_codec_flow_train_export_joint_sample_decode(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(563)
    vae = build_model(Wan22VAEConfig(latents_mean=(0.2, -0.1), latents_std=(0.7, 0.6)))
    model = build_model(Cosmos3Config())
    pipeline = Cosmos3VideoPipeline(model, vae)
    video = torch.randn(2, 3, 5, 16, 32).tanh()
    ids = torch.tensor([[1, 2, 0], [1, 5, 2]])
    inputs = dict(input_ids=ids, attention_mask=ids.ne(0))
    noise = torch.randn(2, 2, 2, 1, 2)
    noisy = torch.tensor([[False, True], [False, True]])
    batch = pipeline.training_batch(
        video,
        inputs,
        noisy_frames=noisy,
        timesteps=torch.full((2, 2), 600.0),
        noise={"vision": noise},
    )
    assert not batch["model_inputs"]["vision"].sample.requires_grad

    assert (
        batch["model_inputs"]["vision"].positions[0, 1, 0]
        - batch["model_inputs"]["vision"].positions[0, 0, 0]
        == 1
    )
    engine = Trainer(model, Cosmos3FlowObjective(time_distribution="provided"), lr=0.003)
    initial = engine.step([batch]).loss
    for _ in range(14):
        final = engine.step([batch]).loss
    assert final < initial * 0.4 and all(p.grad is None for p in vae.parameters())
    model.save_pretrained(tmp_path / "mot")
    vae.save_pretrained(tmp_path / "codec")
    reloaded = Cosmos3VideoPipeline(load_model(tmp_path / "mot"), load_model(tmp_path / "codec"))
    result = pipeline.generate(
        inputs, noise, condition_video=video[:, :, :1], steps=3, solver="euler"
    )
    restored = reloaded.generate(
        inputs, noise, condition_video=video[:, :, :1], steps=3, solver="euler"
    )
    assert (
        result.video.shape == video.shape
        and torch.isfinite(result.video).all()
        and result.video.abs().max() <= 1
    )
    torch.testing.assert_close(restored.video, result.video, atol=0, rtol=0)
    torch.testing.assert_close(
        result.latents["vision"][:, :, :1], pipeline.encode_video(video[:, :, :1]), atol=0, rtol=0
    )

    torch.testing.assert_close(
        pipeline.decode_video(result.latents["vision"]), result.video, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="latent channels"):
        Cosmos3VideoPipeline(build_model(Cosmos3Config(latent_channel=4)), vae)


def test_models_cosmos3_codec_official_invstd_rounding_and_amp_boundary():
    torch.set_num_threads(1)
    torch.manual_seed(564)
    vae = build_model(Wan22VAEConfig(latents_mean=(0.2, -0.1), latents_std=(0.7, 0.6))).bfloat16()
    model = build_model(Cosmos3Config()).bfloat16()
    pipeline = Cosmos3VideoPipeline(model, vae)
    video = torch.randn(1, 3, 5, 16, 16).tanh()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.bfloat16)[None, :, None, None, None]
    inv_std = (
        1.0 / torch.tensor(vae.config.latents_std, dtype=torch.bfloat16)[None, :, None, None, None]
    )
    expected = ((vae.encode(video.bfloat16()).mean - mean) * inv_std).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = pipeline.encode_video(video)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_video = vae.decode(actual.bfloat16() / inv_std + mean).float()
    torch.testing.assert_close(pipeline.decode_video(actual), expected_video, atol=0, rtol=0)
