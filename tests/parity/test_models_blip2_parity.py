from dataclasses import asdict, replace
import pytest
import torch
from aster.models import (
    Blip2Config,
    Blip2QFormerConfig,
    Blip2VisionConfig,
    LlamaConfig,
    build_model,
)


def same_gradients(model, oracle, left, right, inputs=()):
    torch.testing.assert_close(left, right, atol=5e-6, rtol=6e-5)
    coeff = torch.randn_like(left)
    (left * coeff).sum().backward()
    (right * coeff).sum().backward()
    expected = dict(oracle.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.grad, expected[name].grad, atol=5e-5, rtol=8e-4, msg=name)
    for x, y in inputs:
        torch.testing.assert_close(x.grad, y.grad, atol=5e-5, rtol=8e-4)


@pytest.mark.oracle
@pytest.mark.parametrize("text_queries", [(False, None), (True, 3), (True, 0)])
def test_models_blip2_qformer_official_query_cross_and_separate_text_ffn(text_queries):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(54)
    enabled, qlen = text_queries
    c = Blip2QFormerConfig(use_qformer_text_input=enabled)
    model = build_model(c).eval()
    reference_config = tf.Blip2QFormerConfig(**asdict(c))
    reference_config._attn_implementation = "eager"
    oracle = tf.Blip2QFormerModel(reference_config).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    query = torch.randn(2, 5, c.hidden_size, requires_grad=True)
    ref_query = query.detach().clone().requires_grad_()
    visual = torch.randn(2, 7, c.encoder_hidden_size, requires_grad=True)
    ref_visual = visual.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    visual_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    kwargs = dict(query_length=qlen, attention_mask=mask, encoder_attention_mask=visual_mask)
    left = model(query, encoder_hidden_states=visual, **kwargs).last_hidden_state
    if qlen == 3:
        self_mask = torch.zeros_like(mask, dtype=torch.float).masked_fill(~mask.bool(), -torch.inf)[
            :, None, None
        ]
        cross_mask = torch.zeros_like(visual_mask, dtype=torch.float).masked_fill(
            ~visual_mask.bool(), -torch.inf
        )[:, None, None]
        right = oracle.encoder(
            oracle.dropout(oracle.layernorm(ref_query)),
            attention_mask=self_mask,
            encoder_hidden_states=ref_visual,
            encoder_attention_mask=cross_mask,
            query_length=qlen,
        ).last_hidden_state
    else:
        right = oracle(ref_query, encoder_hidden_states=ref_visual, **kwargs).last_hidden_state
    same_gradients(model, oracle, left, right, [(query, ref_query), (visual, ref_visual)])


@pytest.mark.oracle
@pytest.mark.parametrize("interpolate", [False, True])
def test_models_blip2_vision_actual_cls_qkv_interpolation(interpolate):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(61)
    c = Blip2VisionConfig(qkv_bias=not interpolate)
    model = build_model(c).eval()
    reference_config = tf.Blip2VisionConfig(**asdict(c))
    reference_config._attn_implementation = "eager"
    oracle = tf.Blip2VisionModel(reference_config).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 12 if interpolate else 8, requires_grad=True)
    reference_pixels = pixels.detach().clone().requires_grad_()
    left, right = (
        model(pixels, interpolate_pos_encoding=interpolate),
        oracle(reference_pixels, interpolate_pos_encoding=interpolate),
    )
    torch.testing.assert_close(left.pooler_output, right.pooler_output, atol=3e-6, rtol=5e-5)
    same_gradients(
        model, oracle, left.last_hidden_state, right.last_hidden_state, [(pixels, reference_pixels)]
    )


def full_config(tf, c):
    vision = tf.Blip2VisionConfig(**asdict(c.vision_config))
    vision._attn_implementation = "eager"
    queries = tf.Blip2QFormerConfig(**asdict(c.qformer_config))
    queries._attn_implementation = "eager"
    values = asdict(c.text_config)
    if c.text_config.architecture == "t5":
        text = tf.T5Config(**values)
    else:
        values.pop("rope")
        text = tf.LlamaConfig(**values, rope_theta=c.text_config.rope.theta)
    text._attn_implementation = "eager"
    return tf.Blip2Config(
        vision_config=vision,
        qformer_config=queries,
        text_config=text,
        num_query_tokens=c.num_query_tokens,
        image_token_id=c.image_token_id,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["t5", "llama"])
def test_models_blip2_official_whole_vlm_gradient_and_cache(family):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(80)
    c = Blip2Config()
    if family == "llama":
        c = replace(c, text_config=LlamaConfig())
    model = build_model(c).eval()
    oracle = tf.Blip2ForConditionalGeneration(full_config(tf, c)).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    pixels = torch.randn(2, 3, 8, 8, requires_grad=True)
    ref_pixels = pixels.detach().clone().requires_grad_()
    ids = torch.tensor([[31, 31, 31, 31, 1, 3, 5], [31, 31, 31, 31, 2, 4, 6]])
    padding = torch.ones_like(ids)
    padding[1, -1] = 0
    kwargs = dict(attention_mask=padding)
    if family == "t5":
        kwargs["decoder_input_ids"] = torch.tensor([[0, 2, 8, 9], [0, 4, 7, 3]])
    left = model(ids, pixel_values=pixels, **kwargs).logits
    right = oracle(input_ids=ids, pixel_values=ref_pixels, **kwargs, use_cache=False).logits
    same_gradients(model, oracle, left, right, [(pixels, ref_pixels)])
    if family == "t5":
        prefix_args = {**kwargs, "decoder_input_ids": kwargs["decoder_input_ids"][:, :2]}
        prefix = model(ids, pixel_values=pixels, **prefix_args, use_cache=True)
        ref_prefix = oracle(
            input_ids=ids, pixel_values=pixels, **prefix_args, use_cache=True
        ).language_model_outputs
        actual = model(
            decoder_input_ids=kwargs["decoder_input_ids"][:, 2:], state=prefix.state, use_cache=True
        ).logits
        expected = oracle.language_model(
            encoder_outputs=(ref_prefix.encoder_last_hidden_state,),
            attention_mask=padding,
            decoder_input_ids=kwargs["decoder_input_ids"][:, 2:],
            past_key_values=ref_prefix.past_key_values,
        ).logits
        torch.testing.assert_close(actual, left[:, 2:], atol=3e-6, rtol=5e-5)
    else:
        prefix = model(ids, pixel_values=pixels, **kwargs, use_cache=True)
        ref_prefix = oracle(
            input_ids=ids, pixel_values=pixels, **kwargs, use_cache=True
        ).language_model_outputs
        next_ids = torch.tensor([[8, 9], [10, 11]])
        mask = torch.cat((padding, torch.ones_like(next_ids)), 1)
        actual = model(next_ids, attention_mask=mask, state=prefix.state, use_cache=True).logits
        expected = oracle.language_model(
            next_ids, attention_mask=mask, past_key_values=ref_prefix.past_key_values
        ).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-5)
