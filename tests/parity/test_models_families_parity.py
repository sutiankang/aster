from dataclasses import asdict
import pytest
import torch
from aster.models import Gemma3TextConfig, Llama4TextConfig, build_model


@pytest.mark.oracle
@pytest.mark.parametrize(
    "family,config",
    [
        ("Gemma3", Gemma3TextConfig(sliding_window=3)),
        ("Gemma3", Gemma3TextConfig(sliding_window=3, final_logit_softcapping=1.0)),
        ("Llama4", Llama4TextConfig(attention_chunk_size=3, floor_scale=2)),
        ("Llama4", Llama4TextConfig(moe_layers=(1,), use_qk_norm=False, num_experts_per_tok=2)),
    ],
)
def test_models_gemma_llama4_official_forward_gradient_cache(family, config):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(21)
    values = asdict(config)
    values.pop("rope")
    values["rope_theta"] = config.rope.theta
    values["pad_token_id"] = None
    if family == "Gemma3":
        values.pop("rope_local")
        values["rope_local_base_freq"] = config.rope_local.theta
        values["layer_types"] = list(config.layer_types)
    else:
        values["moe_layers"] = list(config.moe_layers)
        values["no_rope_layers"] = list(config.no_rope_layers)
    reference_config = getattr(tf, family + "TextConfig")(**values)
    reference_config._attn_implementation = "eager"
    native, oracle = (
        build_model(config).eval(),
        getattr(tf, family + "ForCausalLM")(reference_config).eval(),
    )
    oracle.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.tensor([[1, 3, 5, 7, 4, 9]])
    left, right = native(tokens).logits, oracle(tokens, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    coefficients = torch.randn_like(left)
    (left * coefficients).sum().backward()
    (right * coefficients).sum().backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(oracle.named_parameters())[name].grad,
            atol=5e-5,
            rtol=4e-4,
            msg=name,
        )
    first = native(tokens[:, :2], use_cache=True)
    reference = oracle(tokens[:, :2], use_cache=True)
    torch.testing.assert_close(
        native(tokens[:, 2:], state=first.state).logits,
        oracle(tokens[:, 2:], past_key_values=reference.past_key_values).logits,
        atol=3e-6,
        rtol=3e-5,
    )
