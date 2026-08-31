from dataclasses import asdict, replace
import pytest
import torch
from aster.models import (
    Gemma4VisionConfig,
    Gemma4TextConfig,
    Gemma4Config,
    build_model,
    pack_gemma4_images,
)


def vision_oracle_config(tf, c):
    values = asdict(c)
    values.pop("rope_theta")
    values["rope_parameters"] = {"rope_type": "default", "rope_theta": c.rope_theta}
    result = tf.Gemma4VisionConfig(**values)
    result._attn_implementation = "eager"
    return result


@pytest.mark.oracle
@pytest.mark.parametrize("clipped", [False, True])
def test_models_gemma4_vision_official_pixels_all_weights_gradients(clipped):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(21)
    c = Gemma4VisionConfig(use_clipped_linears=clipped, standardize=clipped)
    model = build_model(c).eval()
    with torch.no_grad():
        model.patch_embedder.position_embedding_table.normal_(std=0.03)
        if clipped:
            model.std_bias.normal_(std=0.2)
            model.std_scale.uniform_(0.5, 1.5)
            for name, value in model.named_buffers():
                if name.endswith("input_min"):
                    value.fill_(-0.6)
                elif name.endswith("input_max"):
                    value.fill_(0.7)
                elif name.endswith("output_min"):
                    value.fill_(-0.1)
                elif name.endswith("output_max"):
                    value.fill_(0.12)
    oracle = tf.Gemma4VisionModel(vision_oracle_config(tf, c)).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    batch = pack_gemma4_images([torch.rand(3, 8, 8), torch.rand(3, 4, 8)], c)
    pixels = batch["pixel_values"].requires_grad_()
    reference_pixels = pixels.detach().clone().requires_grad_()
    native = model(**batch)
    reference = oracle(reference_pixels, batch["pixel_position_ids"])
    assert native.counts == (4, 2)
    torch.testing.assert_close(
        native.last_hidden_state, reference.last_hidden_state, atol=8e-6, rtol=7e-5
    )
    coeff = torch.randn_like(native.last_hidden_state)
    (native.last_hidden_state * coeff).sum().backward()
    (reference.last_hidden_state * coeff).sum().backward()
    torch.testing.assert_close(pixels.grad, reference_pixels.grad, atol=5e-5, rtol=8e-4)
    ref_parameters = dict(oracle.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(
            value.grad, ref_parameters[name].grad, atol=8e-5, rtol=8e-4, msg=name
        )


def full_oracle_config(tf, c):
    values = asdict(c.text_config)
    for name in (
        "local_rope_theta",
        "global_rope_theta",
        "global_rotary_fraction",
        "global_rope_factor",
    ):
        values.pop(name)
    values["layer_types"] = list(values["layer_types"])
    values["rope_parameters"] = {
        "sliding_attention": {"rope_type": "default", "rope_theta": c.text_config.local_rope_theta},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": c.text_config.global_rope_theta,
            "partial_rotary_factor": c.text_config.global_rotary_fraction,
            "factor": c.text_config.global_rope_factor,
        },
    }
    text = tf.Gemma4TextConfig(**values)
    text._attn_implementation = "eager"
    result = tf.Gemma4Config(
        text_config=text,
        vision_config=vision_oracle_config(tf, c.vision_config),
        audio_config=None,
        image_token_id=c.image_token_id,
        video_token_id=c.video_token_id,
        tie_word_embeddings=c.text_config.tie_word_embeddings,
    )
    result._attn_implementation = "eager"
    return result


@pytest.mark.oracle
@pytest.mark.parametrize("bidirectional", [False, True])
def test_models_gemma4_official_full_image_video_ple_and_shared_cache(bidirectional):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(109)
    text = replace(
        Gemma4TextConfig(),
        use_bidirectional_attention="vision" if bidirectional else None,
        enable_moe_block=bidirectional,
    )
    c = Gemma4Config(text_config=text)
    model = build_model(c).eval()
    oracle = tf.Gemma4ForConditionalGeneration(full_oracle_config(tf, c)).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randint(1, 30, (2, 14))
    tokens[0, 1:5], tokens[1, 1:3], tokens[:, 7:11] = (
        c.image_token_id,
        c.image_token_id,
        c.video_token_id,
    )
    mm = (
        torch.zeros_like(tokens)
        .masked_fill(tokens == c.image_token_id, 1)
        .masked_fill(tokens == c.video_token_id, 2)
    )
    batch = pack_gemma4_images([torch.rand(3, 8, 8), torch.rand(3, 4, 8)], c.vision_config)
    movie = pack_gemma4_images(torch.rand(4, 3, 4, 8), c.vision_config)
    images = batch["pixel_values"].requires_grad_()
    videos = movie["pixel_values"].reshape(2, 2, 8, 12).requires_grad_()
    kwargs = dict(
        pixel_values=images,
        image_position_ids=batch["pixel_position_ids"],
        pixel_values_videos=videos,
        video_position_ids=movie["pixel_position_ids"].reshape(2, 2, 8, 2),
        mm_token_type_ids=mm,
    )
    ref_images, ref_videos = (
        images.detach().clone().requires_grad_(),
        videos.detach().clone().requires_grad_(),
    )
    ref_kwargs = {**kwargs, "pixel_values": ref_images, "pixel_values_videos": ref_videos}
    left, right = (
        model(tokens, **kwargs).logits,
        oracle(tokens, **ref_kwargs, use_cache=False).logits,
    )
    torch.testing.assert_close(left, right, atol=5e-6, rtol=8e-5)
    coefficient = torch.randn_like(left)
    (left * coefficient).sum().backward()
    (right * coefficient).sum().backward()
    reference = dict(oracle.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.grad, reference[name].grad, atol=1e-4, rtol=9e-4, msg=name)
    torch.testing.assert_close(images.grad, ref_images.grad, atol=5e-5, rtol=8e-4)
    torch.testing.assert_close(videos.grad, ref_videos.grad, atol=5e-5, rtol=8e-4)
    prefix = {**kwargs, "mm_token_type_ids": mm[:, :12]}
    left = model(tokens[:, :12], **prefix, use_cache=True)
    right = oracle(tokens[:, :12], **prefix, use_cache=True)
    torch.testing.assert_close(left.logits, right.logits, atol=5e-6, rtol=8e-5)
    left = model(tokens[:, 12:], state=left.state, use_cache=True)
    right = oracle(tokens[:, 12:], past_key_values=right.past_key_values, use_cache=True)
    torch.testing.assert_close(left.logits, right.logits, atol=5e-6, rtol=8e-5)
