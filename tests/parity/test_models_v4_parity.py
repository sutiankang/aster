from types import SimpleNamespace
import pytest
import torch
from aster.nn.hyperconnection import HyperConnection, HyperHead
from dataclasses import asdict, replace
from aster.models import DeepSeekV4Config, build_model
from aster.nn.position import RopeConfig


@pytest.mark.oracle
def test_models_mhc_official_forward_gradients():
    tf = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
    torch.set_num_threads(1)
    torch.manual_seed(49)
    config = SimpleNamespace(
        hc_mult=3, hidden_size=8, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6
    )
    native = HyperConnection(8, 3)
    reference = tf.DeepseekV4HyperConnection(config)
    with torch.no_grad():
        native.base.normal_()
        native.fn.normal_(std=0.1)
    reference.load_state_dict(native.state_dict(), strict=True)
    left_input = torch.randn(2, 4, 3, 8, requires_grad=True)
    right_input = left_input.detach().clone().requires_grad_()
    left = native(left_input)
    right = reference(right_input)
    for l, r in zip(left, right):
        torch.testing.assert_close(l, r, atol=1e-7, rtol=1e-6)
    factors = [torch.randn_like(value) for value in left]
    sum((value * f).sum() for value, f in zip(left, factors)).backward()
    sum((value * f).sum() for value, f in zip(right, factors)).backward()
    torch.testing.assert_close(left_input.grad, right_input.grad)
    for name, p in native.named_parameters():
        torch.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad)
    torch.testing.assert_close(left[1].sum(-1), torch.ones_like(left[1].sum(-1)), atol=3e-6, rtol=0)
    torch.testing.assert_close(left[1].sum(-2), torch.ones_like(left[1].sum(-2)), atol=3e-6, rtol=0)
    head, official_head = HyperHead(8, 3), tf.DeepseekV4HyperHead(config)
    official_head.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(head(left_input.detach()), official_head(left_input.detach()))


@pytest.mark.oracle
@pytest.mark.parametrize("yarn", [False, True])
def test_models_v4_official_logits_gradients_and_compressor_cache(yarn):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(50)
    c = DeepSeekV4Config()
    if yarn:
        c = replace(
            c,
            compress_rope=RopeConfig(
                kind="yarn",
                theta=160000,
                factor=4,
                original_max_position_embeddings=32,
                attention_factor=1.0,
            ),
        )
    values = asdict(c)
    values.pop("rope")
    values.pop("compress_rope")
    values.pop("qk_rope_head_dim")
    values["moe_intermediate_size"] = values.pop("intermediate_size")
    values["n_routed_experts"] = values.pop("num_local_experts")
    values["layer_types"], values["mlp_layer_types"] = list(c.layer_types), list(c.mlp_layer_types)
    values["partial_rotary_factor"] = c.qk_rope_head_dim / c.head_dim
    values["rope_parameters"] = {
        "main": {
            "rope_type": "default",
            "rope_theta": c.rope.theta,
            "partial_rotary_factor": c.qk_rope_head_dim / c.head_dim,
        },
        "compress": {
            "rope_type": c.compress_rope.kind,
            "rope_theta": c.compress_rope.theta,
            "partial_rotary_factor": c.qk_rope_head_dim / c.head_dim,
        },
    }
    if yarn:
        values["rope_parameters"]["compress"].update(
            factor=4,
            original_max_position_embeddings=32,
            attention_factor=1.0,
            beta_fast=32,
            beta_slow=1,
        )
    rc = tf.DeepseekV4Config(**values)
    rc._attn_implementation = "eager"
    native, official = build_model(c), tf.DeepseekV4ForCausalLM(rc)
    routes = torch.stack(
        (
            torch.arange(c.vocab_size) % c.num_local_experts,
            (torch.arange(c.vocab_size) + 1) % c.num_local_experts,
        ),
        -1,
    )
    native.set_hash_routes(0, routes)
    official.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 4, 7, 5, 8, 9, 11, 2, 6], [1, 5, 3, 7, 2, 6, 4, 8, 9]])
    left, right = native(ids).logits, official(ids, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(official.named_parameters())[name].grad, atol=1e-4, rtol=1e-3, msg=name
        )
    native.eval()
    official.eval()

    left_state = right_state = None
    for start in (0, 3, 6):
        l = native(ids[:, start : start + 3], state=left_state, use_cache=True)
        r = official(ids[:, start : start + 3], past_key_values=right_state, use_cache=True)
        torch.testing.assert_close(l.logits, r.logits, atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(l.logits, left[:, start : start + 3], atol=5e-6, rtol=5e-5)
        left_state, right_state = l.state, r.past_key_values
