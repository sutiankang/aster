import pytest
import torch
from aster.models import KimiK25Config, KimiK25VisionConfig, build_model, load_model
from aster.models.kimi import pack_kimi_patches, normalize_kimi_pixels


def test_models_kimi_image_train_cache_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = KimiK25Config()
    model = build_model(config).eval()
    patches, grid = pack_kimi_patches(torch.randn(1, 3, 8, 12), config.vision_config)
    tokens = torch.tensor([[1, 26] + [28] * 6 + [27, 3, 5]])
    full = model(tokens, pixel_values=patches, image_grid_thw=grid)
    prefix = model(tokens[:, :-1], pixel_values=patches, image_grid_thw=grid, use_cache=True)
    tail = model(tokens[:, -1:], state=prefix.state)
    torch.testing.assert_close(full.logits[:, -1:], tail.logits, atol=3e-6, rtol=3e-5)
    assert prefix.state.kind == "mla_latent"
    full.logits.square().mean().backward()
    assert model.model.vision_tower.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.model.mm_projector.in_proj.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(
        load_model(tmp_path).eval()(tokens, pixel_values=patches, image_grid_thw=grid).logits,
        full.logits,
        atol=0,
        rtol=0,
    )


def test_models_kimi_temporal_pooling_and_oov_visual_token():
    torch.set_num_threads(1)
    config = KimiK25Config(video_token_id=50)
    model = build_model(config)
    patches, grid = pack_kimi_patches(torch.randn(2, 3, 8, 8), config.vision_config)
    visual = model.model.vision_tower(patches, grid)
    assert visual.last_hidden_state.shape == (32, 32)
    assert visual.pooler_output.shape == (4, 4, 32)

    tokens = torch.tensor([[1, 26] + [50] * 4 + [27, 2]])
    assert model(tokens, pixel_values_videos=patches, video_grid_thw=grid).logits.shape == (
        1,
        8,
        32,
    )
    with pytest.raises(ValueError, match="pixel patches"):
        model(tokens)
    pixels = torch.tensor([0, 255], dtype=torch.uint8).reshape(2, 1, 1, 1).expand(2, 3, 1, 1)
    torch.testing.assert_close(normalize_kimi_pixels(pixels)[:, 0, 0, 0], torch.tensor([-1.0, 1.0]))
