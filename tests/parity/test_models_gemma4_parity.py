from dataclasses import asdict, replace
import pytest
import torch
from aster.models import Gemma4TextConfig, build_model


def oracle_config(tf, c):
    values = asdict(c)
    for key in (
        "local_rope_theta",
        "global_rope_theta",
        "global_rotary_fraction",
        "global_rope_factor",
    ):
        values.pop(key)
    values["layer_types"] = list(values["layer_types"])
    values["rope_parameters"] = {
        "sliding_attention": {"rope_type": "default", "rope_theta": c.local_rope_theta},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": c.global_rope_theta,
            "partial_rotary_factor": c.global_rotary_fraction,
            "factor": c.global_rope_factor,
        },
    }
    config = tf.Gemma4TextConfig(**values)
    config._attn_implementation = "eager"
    return config


@pytest.mark.oracle
@pytest.mark.parametrize("variant", ["shared", "moe", "independent", "no_ple"])
def test_models_gemma4_official_all_weights_gradients_and_cache(variant):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(123)
    c = Gemma4TextConfig()
    if variant == "moe":
        c = replace(c, enable_moe_block=True, attention_bias=True, final_logit_softcapping=3.0)
    if variant == "independent":
        c = replace(c, num_kv_shared_layers=0, attention_k_eq_v=False, global_rope_factor=2.0)
    if variant == "no_ple":
        c = replace(
            c, hidden_size_per_layer_input=0, hidden_activation="silu", tie_word_embeddings=False
        )
    model = build_model(c).eval()
    oracle = tf.Gemma4ForCausalLM(oracle_config(tf, c)).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randint(1, c.vocab_size, (2, 9))
    padding = torch.ones_like(tokens)
    padding[1, 2] = 0
    left = model(tokens, attention_mask=padding).logits
    right = oracle(tokens, attention_mask=padding, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=3e-6, rtol=5e-5)
    coefficients = torch.randn_like(left)
    (left * coefficients).sum().backward()
    (right * coefficients).sum().backward()
    reference = dict(oracle.named_parameters())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.grad, reference[name].grad, atol=6e-5, rtol=7e-4, msg=name)
    p, q = (
        model(tokens[:, :3], attention_mask=padding[:, :3], use_cache=True),
        oracle(tokens[:, :3], attention_mask=padding[:, :3], use_cache=True),
    )
    for start, end in ((3, 7), (7, 9)):
        p = model(
            tokens[:, start:end], attention_mask=padding[:, :end], state=p.state, use_cache=True
        )
        q = oracle(
            tokens[:, start:end],
            attention_mask=padding[:, :end],
            past_key_values=q.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(p.logits, q.logits, atol=3e-6, rtol=5e-5)
    if c.hidden_size_per_layer_input:
        inputs = model.get_input_embeddings()(tokens).detach().requires_grad_()
        ple = model.get_per_layer_inputs(tokens).detach().requires_grad_()
        ref_inputs, ref_ple = (
            inputs.detach().clone().requires_grad_(),
            ple.detach().clone().requires_grad_(),
        )
        result = model(inputs_embeds=inputs, per_layer_inputs=ple).logits
        ref_result = oracle(
            inputs_embeds=ref_inputs, per_layer_inputs=ref_ple, use_cache=False
        ).logits
        torch.testing.assert_close(result, ref_result, atol=3e-6, rtol=5e-5)
        (result * coefficients).sum().backward()
        (ref_result * coefficients).sum().backward()
        torch.testing.assert_close(inputs.grad, ref_inputs.grad, atol=6e-5, rtol=7e-4)
        torch.testing.assert_close(ple.grad, ref_ple.grad, atol=6e-5, rtol=7e-4)
