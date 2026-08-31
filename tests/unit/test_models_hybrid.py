from dataclasses import replace
import pytest
import torch
from aster.models import build_model, load_model, Qwen3NextConfig, LlamaConfig
from aster.nn.delta import gated_delta_rule


def test_models_delta_chunk_recurrence_gradient_and_state():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    q, k, v = [torch.randn(2, 7, 3, 4, requires_grad=True) for _ in range(3)]
    decay = -torch.rand(2, 7, 3)
    beta = torch.rand(2, 7, 3)
    full, final = gated_delta_rule(q, k, v, decay, beta)
    first, initial = gated_delta_rule(q[:, :3], k[:, :3], v[:, :3], decay[:, :3], beta[:, :3])
    last, restored = gated_delta_rule(
        q[:, 3:], k[:, 3:], v[:, 3:], decay[:, 3:], beta[:, 3:], initial
    )
    torch.testing.assert_close(torch.cat((first, last), 1), full, atol=0, rtol=0)
    torch.testing.assert_close(restored, final, atol=0, rtol=0)
    full.square().mean().backward()
    assert all(x.grad.isfinite().all() and x.grad.abs().sum() > 0 for x in (q, k, v))


@pytest.mark.parametrize("sparse", [False, True])
def test_models_qwen_next_full_chunk_cache_and_reload(tmp_path, sparse):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    config = Qwen3NextConfig(num_experts=4 if sparse else 0)
    model = build_model(config).eval()
    ids = torch.tensor([[1, 4, 5, 3, 8, 7], [1, 3, 4, 7, 5, 9]])
    full = model(ids).logits
    first = model(ids[:, :3], use_cache=True)
    tail = model(ids[:, 3:], state=first.state, use_cache=True)
    torch.testing.assert_close(tail.logits, full[:, 3:], atol=3e-6, rtol=3e-5)
    assert first.state.seen_tokens == 3 and tail.state.seen_tokens == 6
    assert first.state.layers[0][0].shape[-1] == config.linear_conv_kernel_dim - 1
    assert first.state.layers[0][1].shape == (2, 4, 4, 4)
    with pytest.raises(ValueError, match="cannot be truncated"):
        first.state.truncate(2)
    fork = first.state.fork()
    fork.layers[0][0].zero_()
    assert first.state.layers[0][0].abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(load_model(tmp_path).eval()(ids).logits, full, atol=0, rtol=0)


def test_models_storage_preserves_dtype_ties_and_rejects_broken_cache(tmp_path):
    model = build_model(LlamaConfig(tie_word_embeddings=True)).double().eval()
    model.save_pretrained(tmp_path)
    result = load_model(tmp_path)
    assert next(result.parameters()).dtype == torch.float64
    assert result.lm_head.weight is result.model.embed_tokens.weight
    state = result(torch.tensor([[1, 2]]), use_cache=True).state
    with pytest.raises(ValueError, match="KV tensor layout"):
        result(torch.tensor([[3]]), state=replace(state, seen_tokens=4))
