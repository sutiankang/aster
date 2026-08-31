from dataclasses import asdict
import pytest
import torch
from aster.models import CLIPVisionConfig, LlavaConfig, build_model
from aster.models.vision import normalize_clip_pixels


@pytest.mark.oracle
def test_models_clip_official_forward_pixel_gradient():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = CLIPVisionConfig()
    native = build_model(config).eval()
    oracle = tf.CLIPVisionModel(tf.CLIPVisionConfig(**asdict(config))).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 3, 24, 20, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    left = native(x, interpolate_pos_encoding=True, output_hidden_states=True)
    right = oracle(y, interpolate_pos_encoding=True, output_hidden_states=True)
    torch.testing.assert_close(
        left.last_hidden_state, right.last_hidden_state, atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(left.pooler_output, right.pooler_output, atol=3e-6, rtol=3e-5)
    for a, b in zip(left.hidden_states, right.hidden_states):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    left.pooler_output.square().sum().backward()
    right.pooler_output.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-4)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=3e-5, rtol=3e-4, msg=name
        )


@pytest.mark.oracle
@pytest.mark.parametrize("layers,strategy", [(-2, "default"), ((-2, -1), "full")])
def test_models_llava_official_forward_gradient_cache(layers, strategy):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = LlavaConfig(vision_feature_layer=layers, vision_feature_select_strategy=strategy)
    text_kwargs = asdict(config.text_config)
    text_kwargs.pop("rope")
    text_kwargs["head_dim"] = config.text_config.attention_head_dim
    official_config = tf.LlavaConfig(
        text_config=tf.LlamaConfig(**text_kwargs),
        vision_config=tf.CLIPVisionConfig(**asdict(config.vision_config)),
        image_token_id=31,
        vision_feature_layer=list(layers) if isinstance(layers, tuple) else layers,
        vision_feature_select_strategy=strategy,
    )
    official_config._attn_implementation = "eager"
    native = build_model(config).eval()
    oracle = tf.LlavaForConditionalGeneration(official_config).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    count = 16 if strategy == "default" else 17
    tokens = torch.tensor([[1] + [31] * count + [2, 4]])
    pixels = torch.randn(1, 3, 16, 16)
    left, right = native(tokens, pixel_values=pixels), oracle(tokens, pixel_values=pixels)
    torch.testing.assert_close(left.logits, right.logits, atol=3e-6, rtol=3e-5)
    factors = torch.randn_like(left.logits)
    (left.logits * factors).sum().backward()
    (right.logits * factors).sum().backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(oracle.named_parameters())[name].grad,
            atol=3e-5,
            rtol=3e-4,
            msg=name,
        )
    prefix = native(tokens[:, :-1], pixel_values=pixels, use_cache=True)
    reference = oracle(tokens[:, :-1], pixel_values=pixels, use_cache=True)
    torch.testing.assert_close(
        native(tokens[:, -1:], state=prefix.state).logits,
        oracle(tokens[:, -1:], past_key_values=reference.past_key_values).logits,
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.oracle
def test_models_clip_pixel_normalization_official():
    tf = pytest.importorskip("transformers")
    import numpy as np

    pixels = torch.randint(256, (2, 3, 16, 16), dtype=torch.uint8)
    processor = tf.CLIPImageProcessor(do_resize=False, do_center_crop=False)
    reference = processor(images=[x.numpy() for x in pixels], return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(normalize_clip_pixels(pixels), reference, atol=3e-7, rtol=2e-6)
