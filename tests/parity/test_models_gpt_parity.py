from dataclasses import asdict
import pytest
import torch
from aster.models import GPT2Config, build_model


@pytest.mark.oracle
@pytest.mark.parametrize("scaled,upcast", [(False, False), (True, True)])
def test_models_gpt_official_weights_gradients_cache(scaled, upcast):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(41)
    config = GPT2Config(scale_attn_by_inverse_layer_idx=scaled, reorder_and_upcast_attn=upcast)
    rc = tf.GPT2Config(**asdict(config))
    rc._attn_implementation = "eager"
    native, oracle = build_model(config), tf.GPT2LMHeadModel(rc)
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 3, 7, 2], [2, 8, 4, 6]])
    types = torch.tensor([[1, 1, 2, 2], [1, 1, 1, 1]])
    left, right = native(ids, token_type_ids=types).logits, oracle(ids, token_type_ids=types).logits
    torch.testing.assert_close(left, right, atol=2e-6, rtol=3e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=3e-5, rtol=3e-4, msg=name
        )
    left = native(ids[:, :2], use_cache=True)
    right = oracle(ids[:, :2], use_cache=True)
    torch.testing.assert_close(
        native(ids[:, 2:], state=left.state).logits,
        oracle(ids[:, 2:], past_key_values=right.past_key_values).logits,
        atol=2e-6,
        rtol=3e-5,
    )
