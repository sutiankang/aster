import pytest
import torch
from aster.models import CLIPVisionConfig, LlavaConfig, build_model, load_model
from aster.models.vision import normalize_clip_pixels
from aster.models.multimodal import replace_image_tokens


def test_models_vision_and_multimodal_storage_cache(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = build_model(LlavaConfig()).eval()
    pixels = normalize_clip_pixels(torch.randint(256, (2, 3, 16, 16), dtype=torch.uint8))
    tokens = torch.tensor([[1] + [31] * 16 + [2, 4], [1] + [31] * 16 + [3, 5]])
    full = model(tokens, pixel_values=pixels)
    prefix = model(tokens[:, :-1], pixel_values=pixels, use_cache=True)
    tail = model(tokens[:, -1:], state=prefix.state)
    torch.testing.assert_close(full.logits[:, -1:], tail.logits, rtol=2e-5, atol=2e-6)
    full.logits.square().mean().backward()
    assert model.model.vision_tower.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    assert model.model.multi_modal_projector.linear_1.weight.grad.abs().sum() > 0
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(
        load_model(tmp_path).eval()(tokens, pixel_values=pixels).logits, full.logits, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="prefill"):
        model(tokens[:, -1:], pixel_values=pixels, state=prefix.state)
    with pytest.raises(ValueError, match="placeholders require"):
        model(tokens)


def test_models_vision_feature_per_sample_alignment():
    embeds = torch.randn(2, 4, 8)
    features = torch.randn(2, 2, 8)

    with pytest.raises(ValueError, match="Every sample"):
        replace_image_tokens(
            embeds, features, torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="finite"):
        normalize_clip_pixels(torch.full((1, 3, 8, 8), float("nan")))


def test_models_clip_interpolation_and_local_reload(tmp_path):
    torch.set_num_threads(1)
    model = build_model(CLIPVisionConfig()).eval()
    pixels = torch.randn(2, 3, 24, 20)
    with pytest.raises(ValueError, match="explicit"):
        model(pixels)
    output = model(pixels, interpolate_pos_encoding=True, output_hidden_states=True)
    assert output.last_hidden_state.shape == (2, 31, 32)
    model.save_pretrained(tmp_path)
    reloaded = load_model(tmp_path).eval()(pixels, interpolate_pos_encoding=True)
    torch.testing.assert_close(output.pooler_output, reloaded.pooler_output, atol=0, rtol=0)
