from dataclasses import asdict
import pytest
import torch
from aster.models import Qwen3NextConfig, build_model
from aster.nn.delta import gated_delta_rule
from aster.nn.parameter_codec import public_parameter_names


@pytest.mark.oracle
def test_models_delta_official_chunk_values_gradients():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule

    torch.set_num_threads(1)
    torch.manual_seed(14)
    native = [torch.randn(1, 9, 2, 4, requires_grad=True) for _ in range(3)]
    oracle = [x.detach().clone().requires_grad_() for x in native]
    decay, beta = -torch.rand(1, 9, 2), torch.rand(1, 9, 2)
    left, state = gated_delta_rule(*native, decay, beta)
    right, reference_state = torch_chunk_gated_delta_rule(
        *oracle, decay, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state, reference_state, atol=2e-6, rtol=2e-5)
    left.square().sum().backward()
    right.square().sum().backward()
    for x, y in zip(native, oracle):
        torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5)


@pytest.mark.oracle
@pytest.mark.parametrize("sparse", [False, True])
def test_models_qwen_next_official_forward_gradient_cache(sparse):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(15)
    config = Qwen3NextConfig(num_experts=4 if sparse else 0)
    values = asdict(config)
    values.pop("rope")
    values["layer_types"] = list(config.layer_types)
    values["mlp_only_layers"] = list(config.mlp_only_layers)
    values["rope_theta"] = config.rope.theta
    reference_config = tf.Qwen3NextConfig(**values)
    reference_config._attn_implementation = "eager"
    native, oracle = build_model(config).eval(), tf.Qwen3NextForCausalLM(reference_config).eval()
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 6, 4, 8, 7], [1, 6, 5, 3, 7, 2]])
    left, right = native(ids).logits, oracle(ids, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    coefficients = torch.randn_like(left)
    (left * coefficients).sum().backward()
    (right * coefficients).sum().backward()
    names = public_parameter_names(native)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=4e-5,
            rtol=4e-4,
            msg=name,
        )
    first = native(ids[:, :3], use_cache=True)
    reference = oracle(ids[:, :3], use_cache=True)
    torch.testing.assert_close(
        native(ids[:, 3:], state=first.state).logits,
        oracle(ids[:, 3:], past_key_values=reference.past_key_values).logits,
        atol=3e-6,
        rtol=3e-5,
    )
