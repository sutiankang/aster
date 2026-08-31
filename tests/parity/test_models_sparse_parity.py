from dataclasses import asdict
import pytest
import torch
from aster.models import DeepSeekV32Config, build_model
from aster.nn.parameter_codec import public_parameter_names


@pytest.mark.oracle
@pytest.mark.parametrize("topk", [2, 128])
def test_models_dsa_official_sparse_logits_gradients_and_cache(topk):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(47)
    config = DeepSeekV32Config(index_topk=topk)
    values = asdict(config)
    values.pop("rope")
    values["rope_parameters"] = {"rope_type": "default", "rope_theta": config.rope.theta}
    values["head_dim"] = config.qk_rope_head_dim
    rc = tf.DeepseekV32Config(**values)
    rc._attn_implementation = "eager"
    native, oracle = build_model(config), tf.DeepseekV32ForCausalLM(rc)
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 4, 7, 2], [1, 6, 5, 8, 2]])
    left, right = native(ids).logits, oracle(ids, use_cache=False).logits
    torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    names = public_parameter_names(native)
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad,
            dict(oracle.named_parameters())[names[name]].grad,
            atol=7e-5,
            rtol=7e-4,
            msg=names[name],
        )
    native.eval()
    oracle.eval()
    left = native(ids[:, :3], use_cache=True)
    right = oracle(ids[:, :3], use_cache=True)
    torch.testing.assert_close(
        native(ids[:, 3:], state=left.state).logits,
        oracle(ids[:, 3:], past_key_values=right.past_key_values, use_cache=True).logits,
        atol=4e-6,
        rtol=4e-5,
    )
