from dataclasses import asdict
import pytest
import torch
from aster.models import KimiK25Config, build_model
from aster.models.kimi import KimiK25VisionModel, pack_kimi_patches
from aster.nn.parameter_codec import public_parameter_names


def _vision_config(tf, config):
    values = asdict(config)
    values.pop("rope")
    values["rope_parameters"] = {"rope_type": "default", "rope_theta": config.rope.theta}
    return tf.Kimi_K25VisionConfig(**values)


@pytest.mark.oracle
@pytest.mark.parametrize("frames", [1, 2])
def test_models_kimi_vision_official_images_video_gradients(frames):
    tf = pytest.importorskip("transformers")
    from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25VisionModel

    torch.set_num_threads(1)
    torch.manual_seed(27)
    config = KimiK25Config().vision_config
    native, oracle = KimiK25VisionModel(config), Kimi_K25VisionModel(_vision_config(tf, config))
    oracle.config._attn_implementation = "eager"
    oracle.load_state_dict(native.state_dict(), strict=True)

    first, grid1 = pack_kimi_patches(torch.randn(frames, 3, 8, 12), config)
    video, grid2 = pack_kimi_patches(torch.randn(frames, 3, 12, 8), config)
    patches, grid = torch.cat((first, video)), torch.cat((grid1, grid2))
    left, right = native(patches, grid), oracle(patches, grid)
    torch.testing.assert_close(
        left.last_hidden_state, right.last_hidden_state, atol=4e-6, rtol=3e-5
    )
    torch.testing.assert_close(left.pooler_output, right.pooler_output, atol=4e-6, rtol=3e-5)
    factor = torch.randn_like(left.pooler_output)
    (left.pooler_output * factor).sum().backward()
    (right.pooler_output * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=6e-5, rtol=5e-4, msg=name
        )


@pytest.mark.oracle
def test_models_kimi_image_official_logits_gradients_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(28)
    config = KimiK25Config()
    text_values = asdict(config.text_config)
    text_values.pop("rope")
    text_values["rope_interleave"] = config.text_config.rope.interleaved
    text_values["rope_theta"] = config.text_config.rope.theta
    text_values["head_dim"] = config.text_config.qk_rope_head_dim
    text = tf.DeepseekV3Config(**text_values)
    reference_config = tf.Kimi_K25Config(
        text_config=text,
        vision_config=_vision_config(tf, config.vision_config),
        projection_hidden_size=config.projection_hidden_size,
        projection_layer_norm_eps=config.projection_layer_norm_eps,
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
        tie_word_embeddings=False,
    )
    reference_config._attn_implementation = "eager"
    native, oracle = (
        build_model(config).eval(),
        tf.Kimi_K25ForConditionalGeneration(reference_config).eval(),
    )
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels, grid = pack_kimi_patches(torch.randn(1, 3, 8, 12), config.vision_config)
    tokens = torch.tensor([[1, 26] + [28] * 6 + [27, 3, 5]])
    kwargs = dict(pixel_values=pixels, image_grid_thw=grid)
    left, right = native(tokens, **kwargs).logits, oracle(tokens, **kwargs, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    names = public_parameter_names(native)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=5e-5,
            rtol=4e-4,
            msg=name,
        )
    prefix = native(tokens[:, :-1], **kwargs, use_cache=True)
    reference = oracle(tokens[:, :-1], **kwargs, use_cache=True)
    torch.testing.assert_close(
        native(tokens[:, -1:], state=prefix.state).logits,
        oracle(tokens[:, -1:], past_key_values=reference.past_key_values).logits,
        atol=3e-6,
        rtol=3e-5,
    )
