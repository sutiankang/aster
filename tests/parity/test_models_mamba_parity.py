from dataclasses import asdict
import pytest
import torch
from aster.models import MambaConfig, build_model


@pytest.mark.oracle
@pytest.mark.parametrize("bias", [False, True])
def test_models_mamba_official_weights_gradients_single_token_state(bias):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    config = MambaConfig(use_bias=bias)
    native, oracle = (
        build_model(config),
        tf.MambaForCausalLM(tf.MambaConfig(**asdict(config), use_associative_scan=False)),
    )
    oracle.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[0, 1, 3, 5, 2], [1, 3, 4, 6, 2]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    left, right = (
        native(ids, attention_mask=mask).logits,
        oracle(ids, attention_mask=mask, use_cache=False).logits,
    )
    torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-5)
    factor = torch.randn_like(left)
    (left * factor).sum().backward()
    (right * factor).sum().backward()
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=6e-5, rtol=5e-4, msg=name
        )
    native.eval()
    oracle.eval()
    left = native(ids[:, :3], use_cache=True)
    right = oracle(ids[:, :3], use_cache=True)
    for i in (3, 4):
        left = native(ids[:, i : i + 1], state=left.state)
        right = oracle(ids[:, i : i + 1], cache_params=right.cache_params, use_cache=True)
        torch.testing.assert_close(left.logits, right.logits, atol=4e-6, rtol=4e-5)
