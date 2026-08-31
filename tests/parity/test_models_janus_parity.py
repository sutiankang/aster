from dataclasses import asdict
import pytest
import torch
from aster.models import JanusConfig, JanusVQConfig, JanusVisionConfig, build_model


def _compare_grad(native, oracle, left, right, atol=5e-6):
    torch.testing.assert_close(left, right, atol=atol, rtol=5e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=2e-4, rtol=2e-3, msg=name
        )


@pytest.mark.oracle
def test_models_janus_official_vision_forward_gradient():
    tf = pytest.importorskip("transformers")
    from transformers.models.janus.modeling_janus import JanusVisionModel

    torch.set_num_threads(1)
    torch.manual_seed(37)
    config = JanusVisionConfig()
    rc = tf.JanusVisionConfig(**asdict(config))
    rc._attn_implementation = "eager"
    native, oracle = build_model(config), JanusVisionModel(rc)
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 12)
    _compare_grad(
        native,
        oracle,
        native(pixels, interpolate_pos_encoding=True).last_hidden_state,
        oracle(pixels, interpolate_pos_encoding=True).last_hidden_state,
    )


@pytest.mark.oracle
def test_models_janus_official_vq_values_gradient_and_decode():
    tf = pytest.importorskip("transformers")
    from transformers.models.janus.modeling_janus import JanusVQVAE

    torch.set_num_threads(1)
    torch.manual_seed(38)
    config = JanusVQConfig()
    native, oracle = (
        build_model(config).eval(),
        JanusVQVAE(tf.JanusVQVAEConfig(**asdict(config))).eval(),
    )
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 8)
    left, right = native.encode(pixels), oracle.encode(pixels)
    torch.testing.assert_close(left.image_tokens.flatten(), right.image_tokens)
    torch.testing.assert_close(
        left.quantized_last_hidden_state, right.quantized_last_hidden_state, atol=3e-6, rtol=5e-5
    )
    loss = left.commitment_errors.mean() + config.beta * left.codebook_errors.mean()
    torch.testing.assert_close(loss, right.embedding_loss, atol=1e-6, rtol=5e-5)

    _compare_grad(
        native,
        oracle,
        native.decode(left.image_tokens) + loss,
        oracle.decode(left.image_tokens) + right.embedding_loss,
        atol=2e-5,
    )


@pytest.mark.oracle
def test_models_janus_official_multimodal_and_image_head_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(39)
    config = JanusConfig()
    tc = asdict(config.text_config)
    tc.pop("rope")
    tc["head_dim"] = config.text_config.attention_head_dim
    rc = tf.JanusConfig(
        text_config=tf.LlamaConfig(**tc),
        vision_config=asdict(config.vision_config),
        vq_config=asdict(config.vq_config),
        image_token_id=31,
        tie_word_embeddings=False,
    )
    rc._attn_implementation = "eager"
    native, oracle = build_model(config).eval(), tf.JanusForConditionalGeneration(rc).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    pixels = torch.randn(1, 3, 8, 8)
    ids = torch.tensor([[1] + [31] * 16 + [4, 7]])
    _compare_grad(
        native,
        oracle,
        native(ids, pixel_values=pixels).logits,
        oracle(ids, pixel_values=pixels).logits,
    )
    native.zero_grad()
    oracle.zero_grad()
    codes = torch.tensor([[1, 4, 6]])
    le, re = (
        native.prepare_embeddings_for_image_generation(codes),
        oracle.prepare_embeddings_for_image_generation(codes),
    )
    left = native(inputs_embeds=le, output_kind="image_codes", use_cache=True)
    right = oracle.model.language_model(inputs_embeds=re, use_cache=True)
    _compare_grad(
        native, oracle, left.logits, oracle.model.generation_head(right.last_hidden_state)
    )
    torch.testing.assert_close(
        native.decode_image_tokens(torch.ones(1, 16, dtype=torch.long)),
        oracle.decode_image_tokens(torch.ones(1, 16, dtype=torch.long)).permute(0, 3, 1, 2),
        atol=2e-5,
        rtol=5e-5,
    )
    next_left = native(
        inputs_embeds=native.prepare_embeddings_for_image_generation(codes[:, :1]),
        state=left.state,
        output_kind="image_codes",
    )
    next_right = oracle.model.language_model(
        inputs_embeds=oracle.prepare_embeddings_for_image_generation(codes[:, :1]),
        past_key_values=right.past_key_values,
    )
    torch.testing.assert_close(
        next_left.logits,
        oracle.model.generation_head(next_right.last_hidden_state),
        atol=3e-6,
        rtol=5e-5,
    )
