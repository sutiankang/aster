from dataclasses import asdict
import pytest
import torch
from aster.models import Qwen3VLConfig, build_model
from aster.models.qwen_vl import Qwen3VLVisionModel, pack_qwen_pixels, multimodal_positions


def _oracle_config(tf, config):
    values = asdict(config.text_config)
    for key in ("rope", "mrope_section", "layer_types", "sliding_window"):
        values.pop(key)
    values["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": config.text_config.rope.theta,
        "mrope_section": list(config.text_config.mrope_section),
    }
    text = tf.Qwen3VLTextConfig(**values)
    vision = tf.Qwen3VLVisionConfig(**asdict(config.vision_config))
    return tf.Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
        tie_word_embeddings=False,
    )


@pytest.mark.oracle
def test_models_qwen3_vision_official_dynamic_grids_gradients():
    tf = pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as Oracle

    torch.set_num_threads(1)
    torch.manual_seed(22)
    config = Qwen3VLConfig().vision_config
    native, oracle = Qwen3VLVisionModel(config), Oracle(tf.Qwen3VLVisionConfig(**asdict(config)))
    oracle.config._attn_implementation = "eager"
    oracle.load_state_dict(native.state_dict(), strict=True)
    first, grid1 = pack_qwen_pixels(torch.randn(1, 3, 8, 12), config)
    second, grid2 = pack_qwen_pixels(torch.randn(4, 3, 12, 8), config)
    pixels, grid = torch.cat((first, second)), torch.cat((grid1, grid2))
    left, right = native(pixels, grid), oracle(pixels, grid)
    torch.testing.assert_close(
        left.last_hidden_state, right.last_hidden_state, atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(left.pooler_output, right.pooler_output, atol=3e-6, rtol=3e-5)
    for x, y in zip(left.deepstack_features, right.deepstack_features):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
    loss_left = left.pooler_output.square().sum() + sum(
        x.square().sum() for x in left.deepstack_features
    )
    loss_right = right.pooler_output.square().sum() + sum(
        x.square().sum() for x in right.deepstack_features
    )
    loss_left.backward()
    loss_right.backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=3e-5, rtol=3e-4, msg=name
        )


@pytest.mark.oracle
def test_models_qwen3_vl_official_logits_gradients_positions_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(23)
    config = Qwen3VLConfig()
    native = build_model(config).eval()
    reference_config = _oracle_config(tf, config)
    reference_config._attn_implementation = "eager"
    oracle = tf.Qwen3VLForConditionalGeneration(reference_config).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 12), config.vision_config)
    tokens = torch.tensor([[1, 26] + [28] * 6 + [27, 3, 5]])
    modalities = torch.where(tokens == 28, 1, 0)
    kwargs = dict(pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=modalities)
    left, right = native(tokens, **kwargs), oracle(tokens, **kwargs, use_cache=False)
    torch.testing.assert_close(left.logits, right.logits, atol=3e-6, rtol=3e-5)
    positions, delta = multimodal_positions(tokens, modalities, 2, grid)
    reference_positions, reference_delta = oracle.model.get_rope_index(
        tokens, modalities, image_grid_thw=grid
    )
    torch.testing.assert_close(positions, reference_positions)
    torch.testing.assert_close(delta, reference_delta)
    coefficients = torch.randn_like(left.logits)
    (left.logits * coefficients).sum().backward()
    (right.logits * coefficients).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=4e-5, rtol=4e-4, msg=name
        )
    kwargs["mm_token_type_ids"] = modalities[:, :-1]
    first = native(tokens[:, :-1], **kwargs, use_cache=True)
    reference = oracle(tokens[:, :-1], **kwargs, use_cache=True)
    torch.testing.assert_close(
        native(tokens[:, -1:], state=first.state).logits,
        oracle(tokens[:, -1:], past_key_values=reference.past_key_values).logits,
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.oracle
def test_models_qwen_image_patch_layout_official():
    tf = pytest.importorskip("transformers")

    processor = tf.Qwen2VLImageProcessorPil()
    config = Qwen3VLConfig().vision_config
    pixels = torch.randn(1, 3, 8, 12)
    native, _ = pack_qwen_pixels(pixels, config)
    reference, _, _ = processor.patchify(
        pixels[0].numpy(), config.patch_size, config.spatial_merge_size, config.temporal_patch_size
    )
    torch.testing.assert_close(native, torch.from_numpy(reference), atol=0, rtol=0)
