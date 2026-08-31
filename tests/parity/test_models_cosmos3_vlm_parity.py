from dataclasses import asdict
import pytest
import torch
from aster.models.cosmos3_vlm import Cosmos3VLM, Cosmos3VLMConfig
from aster.models.qwen_vl import pack_qwen_pixels


@pytest.mark.oracle
@pytest.mark.parametrize("video", [False, True])
def test_models_cosmos3_qwen_visual_actual_transformers_logits_all_understanding_gradients_cache(
    video,
):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(565)
    c = Cosmos3VLMConfig()
    native = Cosmos3VLM(c)
    m = c.mot
    text_config = tf.Qwen3VLTextConfig(
        vocab_size=m.vocab_size,
        hidden_size=m.hidden_size,
        intermediate_size=m.intermediate_size,
        num_hidden_layers=m.num_hidden_layers,
        num_attention_heads=m.num_attention_heads,
        num_key_value_heads=m.num_key_value_heads,
        head_dim=m.head_dim,
        rms_norm_eps=m.rms_norm_eps,
        rope_parameters=dict(
            rope_type="default", rope_theta=m.rope_theta, mrope_section=list(m.rope_axes_dim)
        ),
    )
    config = tf.Qwen3VLConfig(
        text_config=text_config,
        vision_config=tf.Qwen3VLVisionConfig(**asdict(c.vision_config)),
        image_token_id=c.image_token_id,
        video_token_id=c.video_token_id,
        vision_start_token_id=c.vision_start_token_id,
        vision_end_token_id=c.vision_end_token_id,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    official = tf.Qwen3VLForConditionalGeneration(config)
    mapping = {}
    for name in official.state_dict():
        target = name.replace("model.language_model.", "transformer.").replace(
            "model.visual.", "visual."
        )
        if name.startswith("lm_head."):
            target = "transformer." + name
        for old, new in (
            ("q_proj", "to_q"),
            ("k_proj", "to_k"),
            ("v_proj", "to_v"),
            ("o_proj", "to_out"),
            ("q_norm", "norm_q"),
            ("k_norm", "norm_k"),
        ):
            target = target.replace(".self_attn." + old + ".", ".self_attn." + new + ".")
        mapping[name] = target
    official.load_state_dict(
        {key: native.state_dict()[value] for key, value in mapping.items()}, strict=True
    )
    frames = torch.randn(4 if video else 1, 3, 8, 12)
    pixels, grid = pack_qwen_pixels(frames, c.vision_config)
    pixels.requires_grad_()
    oracle_pixels = pixels.detach().clone().requires_grad_()
    kind = 2 if video else 1
    placeholder = c.video_token_id if video else c.image_token_id
    ids = torch.tensor([[1] + ([26] + [placeholder] * 6 + [27, 4]) * (2 if video else 1) + [3, 5]])
    kwargs = (
        dict(pixel_values_videos=pixels, video_grid_thw=grid)
        if video
        else dict(pixel_values=pixels, image_grid_thw=grid)
    )
    oracle_kwargs = {**kwargs, "pixel_values_videos" if video else "pixel_values": oracle_pixels}
    types = torch.where(ids == placeholder, kind, 0)
    left = native.forward_text(ids, **kwargs, output_hidden_states=True)
    right = official(
        ids, **oracle_kwargs, mm_token_type_ids=types, output_hidden_states=True, use_cache=False
    )
    torch.testing.assert_close(left.logits, right.logits, atol=4e-6, rtol=4e-5)
    coefficient = torch.randn_like(left.logits) / left.logits.numel()
    (left.logits * coefficient).sum().backward()
    (right.logits * coefficient).sum().backward()
    for name, parameter in official.named_parameters():
        torch.testing.assert_close(
            dict(native.named_parameters())[mapping[name]].grad,
            parameter.grad,
            atol=5e-6,
            rtol=8e-4,
            msg=name,
        )
    torch.testing.assert_close(pixels.grad, oracle_pixels.grad, atol=4e-6, rtol=5e-4)
    prefix = native.forward_text(ids[:, :-1], **kwargs, use_cache=True)
    cache = official(
        ids[:, :-1], **oracle_kwargs, mm_token_type_ids=types[:, :-1], use_cache=True
    ).past_key_values
    torch.testing.assert_close(
        native.forward_text(ids[:, -1:], state=prefix.state).logits,
        official(ids[:, -1:], past_key_values=cache).logits,
        atol=4e-6,
        rtol=4e-5,
    )
