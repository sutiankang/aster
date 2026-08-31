import pytest
import torch
from aster.models import (
    build_model,
    load_model,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    MistralConfig,
    MixtralConfig,
    DeepSeekV3Config,
)
from aster.nn import RMSNorm, RopeConfig, RotaryEmbedding


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "config",
    [
        LlamaConfig(),
        Qwen2Config(),
        Qwen3Config(),
        MistralConfig(sliding_window=3),
        MixtralConfig(sliding_window=3),
        DeepSeekV3Config(),
    ],
)
def test_models_decoder_cache_and_roundtrip(config, tmp_path):
    torch.manual_seed(7)
    model = build_model(config).eval()
    ids = torch.tensor([[1, 5, 4, 6, 2], [1, 7, 8, 9, 2]])
    full = model(ids, output_hidden_states=True)
    prefix = model(ids[:, :3], use_cache=True)
    snapshot = prefix.state.fork()
    tail = model(ids[:, 3:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(full.logits[:, 3:], tail.logits, atol=1e-6, rtol=1e-5)
    assert len(full.hidden_states) == config.num_hidden_layers + 1
    assert prefix.state.seen_tokens == 3 and tail.state.seen_tokens == 5
    for old, saved in zip(prefix.state.layers, snapshot.layers):
        assert torch.equal(old[0], saved[0])
    model.save_pretrained(tmp_path)
    reloaded = load_model(tmp_path).eval()
    assert torch.equal(model(ids).logits, reloaded(ids).logits)


def test_models_causality_embeddings_and_zero_padding():
    model = build_model(LlamaConfig()).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    changed = ids.clone()
    changed[:, 3] = 5
    torch.testing.assert_close(
        model(ids).logits[:, :3], model(changed).logits[:, :3], atol=0, rtol=0
    )
    torch.testing.assert_close(
        model(ids).logits, model(inputs_embeds=model.get_input_embeddings()(ids)).logits
    )
    padded = model(ids, attention_mask=torch.zeros_like(ids))
    assert torch.isfinite(padded.logits).all()
    with pytest.raises(ValueError, match="cover"):
        model(ids, attention_mask=torch.ones(1, 3))


def test_models_state_rejects_config_change_and_window_rollback():
    first = build_model(LlamaConfig())
    state = first(torch.tensor([[1, 3, 2]]), use_cache=True).state
    with pytest.raises(ValueError, match="mismatch"):
        build_model(LlamaConfig(rms_norm_eps=1e-5))(torch.tensor([[3]]), state=state)
    short = state.truncate(2)
    assert short.seen_tokens == 2 and short.layers[0][0].shape[-2] == 2
    windowed = build_model(MistralConfig(sliding_window=2))(
        torch.tensor([[1, 2, 3]]), use_cache=True
    ).state
    assert windowed.layers[0][0].shape[-2] == 1
    with pytest.raises(ValueError, match="replay"):
        windowed.truncate(1)


def test_models_norm_and_rope_math():
    layer = RMSNorm(4).double()
    value = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, value)
    state = torch.randn(2, 3, 4, 8)
    positions = torch.arange(4)[None].expand(2, -1)
    regular = RotaryEmbedding(8, RopeConfig())
    linear = RotaryEmbedding(8, RopeConfig(kind="linear", factor=2))
    torch.testing.assert_close(linear(state, positions * 2), regular(state, positions))
    with pytest.raises(ValueError, match="Unsupported"):
        RopeConfig(kind="unimplemented")


def test_models_mla_absorption_and_compressed_storage():
    from aster.nn.latent_attention import MultiheadLatentAttention

    config = DeepSeekV3Config()
    layer = MultiheadLatentAttention(config).eval()
    hidden = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    position = torch.arange(5)[None].expand(2, -1)
    left, cache = layer(hidden, position, use_cache=True, implementation="absorbed")
    right, _ = layer(hidden, position, implementation="expanded")
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(left)
    a = torch.autograd.grad(
        (left * weights).sum(), (hidden, layer.kv_b_proj.weight), retain_graph=True
    )
    b = torch.autograd.grad((right * weights).sum(), (hidden, layer.kv_b_proj.weight))
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    assert cache[0].shape == (2, 1, 5, config.kv_lora_rank)
    assert cache[1].shape == (2, 1, 5, config.qk_rope_head_dim)


def test_models_router_bias_changes_selection_not_mixing_scores():
    from aster.nn.experts import TopKRouter

    router = TopKRouter(2, 4, 2, sigmoid=True, scale=2.5)
    hidden = torch.randn(3, 2)
    with torch.no_grad():
        router.e_score_correction_bias.copy_(torch.tensor([100.0, 100.0, 0.0, 0.0]))
    logits, weights, indices = router(hidden)
    assert (indices < 2).all()
    expected = logits.sigmoid().gather(-1, indices)
    torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True) * 2.5)
