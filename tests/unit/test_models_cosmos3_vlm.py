from dataclasses import replace
import pytest
import torch
from aster.models.cosmos3_vlm import Cosmos3VLMConfig, Cosmos3VLM
from aster.models.cosmos3 import Cosmos3Vision, cosmos3_positions
from aster.models.wan22_vae import Wan22VAEConfig, Wan22VideoVAE
from aster.models.qwen_vl import pack_qwen_pixels
from aster.methods.cosmos3 import Cosmos3VisualFlowObjective, Cosmos3VideoPipeline, sample_cosmos3
from aster.training import Trainer


def data(config):
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), config.vision_config)
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3, 2]])
    return dict(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)


@pytest.mark.parametrize("stage", [0, 3])
def test_models_cosmos3_visual_joint_train_rng_restore_shared_codec_sample(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(566)
    c = Cosmos3VLMConfig()
    model = Cosmos3VLM(c)
    inputs = data(c)
    vae = Wan22VideoVAE(Wan22VAEConfig())
    pipeline = Cosmos3VideoPipeline(model, vae)
    video = torch.randn(1, 3, 5, 16, 16).tanh()
    noise = torch.randn(1, 2, 2, 1, 1)
    batch = pipeline.training_batch(
        video,
        inputs,
        noisy_frames=torch.tensor([[False, True]]),
        timesteps=torch.full((1, 2), 700.0),
        noise={"vision": noise},
    )
    batch["labels"] = inputs["input_ids"]
    objective = Cosmos3VisualFlowObjective(text_weight=0.2, time_distribution="provided")
    trainer = Trainer(model, objective, zero_stage=stage, lr=0.002)
    initial = trainer.step([batch]).loss
    for _ in range(11):
        final = trainer.step([batch]).loss
    assert final < initial * 0.6 and all(parameter.grad is None for parameter in vae.parameters())
    random_batch = dict(batch)
    random_batch.pop("noise")
    random_objective = Cosmos3VisualFlowObjective(text_weight=0.2)

    checkpoint = trainer.save_checkpoint(tmp_path / "resume")
    expected = trainer.phase("joint", objective=random_objective, microbatches=[random_batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(checkpoint, trusted=True)
    actual = trainer.phase("joint", objective=random_objective, microbatches=[random_batch])
    assert actual.loss == expected.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    restored = Cosmos3VLM(c)
    restored.load_state_dict(weights, strict=True)
    pipeline = Cosmos3VideoPipeline(restored, vae)
    result = pipeline.generate(
        inputs, noise, condition_video=video[:, :, :1], steps=3, solver="euler"
    )
    assert result.video.shape == video.shape and torch.isfinite(result.video).all()
    torch.testing.assert_close(
        result.latents["vision"][:, :, :1], pipeline.encode_video(video[:, :, :1]), atol=0, rtol=0
    )


def test_models_cosmos3_visual_cache_fork_delta_and_joint_cached_recompute():
    torch.set_num_threads(1)
    torch.manual_seed(567)
    c = Cosmos3VLMConfig()
    model = Cosmos3VLM(c)
    inputs = data(c)
    ids = inputs["input_ids"]
    prefix = model.forward_text(**dict(inputs, input_ids=ids[:, :-1]), use_cache=True)
    state = prefix.state.fork()
    assert state.seen_tokens == ids.shape[1] - 1 and state.kind == "cosmos3_vlm_understanding"
    torch.testing.assert_close(
        model.forward_text(ids[:, -1:], state=state).logits,
        model.forward_text(**inputs).logits[:, -1:],
        atol=3e-6,
        rtol=4e-5,
    )
    assert (
        state.token_state.layers[0][0].data_ptr()
        != prefix.state.token_state.layers[0][0].data_ptr()
    )
    vision = Cosmos3Vision(
        torch.randn(1, 2, 2, 2, 2),
        cosmos3_positions((2, 1, 1)),
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    conditional = dict(inputs, vision=vision)
    cached = sample_cosmos3(model, conditional, steps=3, solver="heun")
    full = sample_cosmos3(model, conditional, steps=3, solver="heun", reuse_understanding=False)
    torch.testing.assert_close(cached["vision"], full["vision"], atol=4e-6, rtol=4e-5)
    with pytest.raises(ValueError, match="prefill"):
        model.forward_text(**inputs, state=state)
    with pytest.raises(ValueError, match="snapshot"):
        state.truncate(2)
    with pytest.raises(ValueError, match="config"):
        model.forward_text(ids[:, -1:], state=replace(state, model_key="wrong"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_cosmos3_visual_shared_factory_safe_save_reload(tmp_path, dtype):
    from aster.models import build_model, load_model

    torch.set_num_threads(1)
    torch.manual_seed(570)
    c = Cosmos3VLMConfig()
    model = build_model(c).to(dtype)
    inputs = data(c)
    output = model.forward_text(**inputs)
    assert output.logits.dtype == dtype and torch.isfinite(output.logits).all()
    output.logits.float().square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.save_pretrained(tmp_path / "visual")
    restored = load_model(tmp_path / "visual")
    assert isinstance(restored, Cosmos3VLM) and restored.config == c
    torch.testing.assert_close(
        output.logits, restored.forward_text(**inputs).logits, atol=0, rtol=0
    )
