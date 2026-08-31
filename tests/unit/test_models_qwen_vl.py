import pytest
import torch
from aster.models import Qwen3VLConfig, build_model, load_model
from aster.models.qwen_vl import pack_qwen_pixels, multimodal_positions


def _sample(config):
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 12), config.vision_config)
    tokens = torch.tensor([[1, 26] + [28] * 6 + [27, 3, 5]])
    modalities = torch.where(tokens == 28, 1, 0)
    return pixels, grid, tokens, modalities


def test_models_qwen_vl_train_cache_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    config = Qwen3VLConfig()
    model = build_model(config).eval()
    pixels, grid, tokens, modalities = _sample(config)
    full = model(tokens, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=modalities)
    prefix = model(
        tokens[:, :-1],
        pixel_values=pixels,
        image_grid_thw=grid,
        mm_token_type_ids=modalities[:, :-1],
        use_cache=True,
    )
    tail = model(tokens[:, -1:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(full.logits[:, -1:], tail.logits, atol=3e-6, rtol=3e-5)
    assert tail.state.seen_tokens == tokens.shape[1]
    with pytest.raises(ValueError, match="mRoPE delta"):
        prefix.state.truncate(1)
    full.logits.square().mean().backward()
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.model.visual.deepstack_merger_list[0].linear_fc2.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(
        load_model(tmp_path)
        .eval()(tokens, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=modalities)
        .logits,
        full.logits,
        atol=0,
        rtol=0,
    )


def test_models_qwen_vl_strict_grid_modality_contract():
    config = Qwen3VLConfig()
    pixels, grid, tokens, modalities = _sample(config)
    model = build_model(config)
    with pytest.raises(ValueError, match="explicit mm_token"):
        model(tokens, pixel_values=pixels, image_grid_thw=grid)
    with pytest.raises(ValueError, match="disagree"):
        model(
            tokens,
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=torch.zeros_like(tokens),
        )
    with pytest.raises(ValueError, match="span length"):
        multimodal_positions(tokens, modalities, 2, image_grid=torch.tensor([[1, 4, 4]]))
    packed, video_grid = pack_qwen_pixels(torch.randn(3, 3, 8, 8), config.vision_config)
    assert packed.shape == (32, 24) and video_grid.tolist() == [[2, 4, 4]]
