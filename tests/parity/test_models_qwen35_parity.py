from dataclasses import asdict
import pytest
import torch
from aster.models import Qwen35TextConfig, Qwen35MoETextConfig, Qwen35Config, build_model
from aster.models.qwen_vl import pack_qwen_pixels
from aster.nn.parameter_codec import public_parameter_names


def official_text_config(tf, c):
    fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "tie_word_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    )
    values = {key: getattr(c, key) for key in fields}
    values["pad_token_id"] = None
    values["layer_types"] = list(c.layer_types)
    values["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": c.rope.theta,
        "partial_rotary_factor": c.partial_rotary_factor,
        "mrope_section": list(c.mrope_section),
        "mrope_interleaved": True,
    }
    if c.num_experts:
        values.update(
            {
                key: getattr(c, key)
                for key in (
                    "num_experts",
                    "num_experts_per_tok",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                )
            }
        )
    cls = tf.Qwen3_5MoeTextConfig if c.num_experts else tf.Qwen3_5TextConfig
    result = cls(**values)
    result._attn_implementation = "eager"
    return result


@pytest.mark.oracle
@pytest.mark.parametrize("moe", [False, True])
def test_models_qwen35_text_logits_gradients_mrope_and_cache(moe):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(53)
    c = Qwen35MoETextConfig() if moe else Qwen35TextConfig()
    rc = official_text_config(tf, c)
    native = build_model(c)
    oracle = (tf.Qwen3_5MoeForCausalLM if moe else tf.Qwen3_5ForCausalLM)(rc)
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 5, 7, 2], [1, 4, 6, 8, 2]])
    positions = torch.arange(5)[None, None].expand(3, 2, -1).clone()
    positions[1, :, 2:] += 2
    positions[2, :, 1:] += 1
    left, right = (
        native(ids, position_ids=positions).logits,
        oracle(ids, position_ids=positions, use_cache=False).logits,
    )
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    names = public_parameter_names(native)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=6e-5,
            rtol=7e-4,
            msg=name,
        )
    l = native(ids[:, :3], use_cache=True)
    r = oracle(ids[:, :3], use_cache=True)
    torch.testing.assert_close(
        native(ids[:, 3:], state=l.state, use_cache=True).logits,
        oracle(ids[:, 3:], past_key_values=r.past_key_values, use_cache=True).logits,
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.oracle
def test_models_qwen35_full_image_gradients_and_hybrid_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(54)
    c = Qwen35Config()
    vc = asdict(c.vision_config)
    vc.pop("deepstack_visual_indexes")
    rc = tf.Qwen3_5Config(
        text_config=official_text_config(tf, c.text_config),
        vision_config=vc,
        image_token_id=c.image_token_id,
        video_token_id=c.video_token_id,
        vision_start_token_id=c.vision_start_token_id,
        vision_end_token_id=c.vision_end_token_id,
        tie_word_embeddings=c.text_config.tie_word_embeddings,
    )
    rc._attn_implementation = "eager"
    native, oracle = build_model(c), tf.Qwen3_5ForConditionalGeneration(rc)
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.vision_config)
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 2]])
    kinds = torch.where(ids == 28, 1, 0)
    left_pixels = pixels.detach().requires_grad_()
    right_pixels = pixels.detach().clone().requires_grad_()
    kwargs = dict(image_grid_thw=grid, mm_token_type_ids=kinds)
    left = native(ids, pixel_values=left_pixels, **kwargs).logits
    right = oracle(ids, pixel_values=right_pixels, use_cache=False, **kwargs).logits
    torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    torch.testing.assert_close(left_pixels.grad, right_pixels.grad, atol=3e-5, rtol=5e-4)
    names = public_parameter_names(native)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=8e-5,
            rtol=8e-4,
            msg=name,
        )
    l = native(ids, pixel_values=pixels, use_cache=True, **kwargs)
    r = oracle(ids, pixel_values=pixels, use_cache=True, **kwargs)
    suffix = torch.tensor([[3, 4]])
    torch.testing.assert_close(
        native(suffix, state=l.state, use_cache=True).logits,
        oracle(suffix, past_key_values=r.past_key_values, use_cache=True).logits,
        atol=4e-6,
        rtol=4e-5,
    )
