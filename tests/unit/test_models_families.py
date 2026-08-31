import pytest
import torch
from aster.models import Gemma3TextConfig, Llama4TextConfig, build_model, load_model
from aster.models.families import Llama4Experts


@pytest.mark.parametrize(
    "config",
    [
        Gemma3TextConfig(sliding_window=3, final_logit_softcapping=2.0),
        Llama4TextConfig(attention_chunk_size=3, floor_scale=2),
        Llama4TextConfig(moe_layers=(), use_qk_norm=False),
    ],
)
def test_models_modern_family_cache_storage(tmp_path, config):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    model = build_model(config).eval()
    tokens = torch.tensor([[1, 4, 7, 5, 8, 9]])
    full = model(tokens).logits
    prefix = model(tokens[:, :2], use_cache=True)
    continuation = model(tokens[:, 2:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(full[:, 2:], continuation.logits, atol=3e-6, rtol=3e-5)
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(load_model(tmp_path).eval()(tokens).logits, full, atol=0, rtol=0)
    full.square().mean().backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_models_llama4_router_is_input_gating():
    config = Llama4TextConfig()
    experts = Llama4Experts(config)
    x = torch.randn(2, config.hidden_size)
    indices = torch.zeros(2, 1, dtype=torch.long)
    weights = torch.full((2, 1), 0.3)
    actual = experts(x, indices, weights)
    gate, up = (x * 0.3 @ experts.gate_up_proj[0]).chunk(2, -1)
    torch.testing.assert_close(actual, (torch.nn.functional.silu(gate) * up) @ experts.down_proj[0])
    gate, up = (x @ experts.gate_up_proj[0]).chunk(2, -1)
    assert not torch.allclose(
        actual, ((torch.nn.functional.silu(gate) * up) @ experts.down_proj[0]) * 0.3
    )
