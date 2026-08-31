import pytest
import torch
from aster.models import LLaDAConfig, build_model


def _mapping(name):
    if name == "model.transformer.wte.weight":
        return "model.embed_tokens.weight"
    if name == "model.transformer.ln_f.weight":
        return "model.norm.weight"
    if name == "model.transformer.ff_out.weight":
        return "lm_head.weight"
    name = name.replace("model.transformer.blocks.", "model.layers.")
    for old, new in (
        (".attn_norm.", ".input_layernorm."),
        (".ff_norm.", ".post_attention_layernorm."),
        (".attn_out.", ".self_attn.o_proj."),
        (".q_proj.", ".self_attn.q_proj."),
        (".k_proj.", ".self_attn.k_proj."),
        (".v_proj.", ".self_attn.v_proj."),
        (".ff_proj.", ".mlp.gate_proj."),
        (".up_proj.", ".mlp.up_proj."),
        (".ff_out.", ".mlp.down_proj."),
    ):
        name = name.replace(old, new)
    return name


@pytest.mark.oracle
def test_models_llada_public_backbone_weight_map_forward_gradient():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(45)
    config = LLaDAConfig(n_kv_heads=2)
    rc = tf.LlamaConfig(
        vocab_size=config.embedding_size,
        hidden_size=config.d_model,
        intermediate_size=config.mlp_hidden_size,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_sequence_length,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=False,
    )
    rc._attn_implementation = "eager"
    native, oracle = build_model(config), tf.LlamaForCausalLM(rc)
    oracle.load_state_dict({_mapping(k): v for k, v in native.state_dict().items()}, strict=True)
    ids = torch.tensor([[1, 31, 4, 7], [1, 5, 31, 2]])

    left, right = native(ids).logits, oracle(ids, attention_mask=torch.zeros(2, 1, 4, 4)).logits
    torch.testing.assert_close(left, right, atol=2e-5, rtol=3e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[_mapping(name)].grad,
            atol=1e-4,
            rtol=5e-4,
            msg=name,
        )
