import asyncio
import copy
import types
import numpy as np
import pytest
import torch

from aster.inference import (
    KVQuantization,
    PagedStatePool,
    PrefixCache,
    PrefixIdentity,
    PagedAttentionRunner,
    InferenceEngine,
    SamplingConfig,
)
from aster.optimization.kv_quantization import QuantizedKV, quantize_kv
from aster.optimization.online_attention import AttentionBlock, online_attention
from aster.models import build_model, LlamaConfig, Qwen2Config, Qwen3Config


@pytest.mark.parametrize("format", ["int8", "fp8_e4m3fn", "fp8_e5m2"])
def test_real_codes_scale_and_error_bound_against_independent_scalar_oracle(format):
    torch.manual_seed(76)
    x = torch.randn(2, 3, 9, 32)
    x[0, 0, 0] = 0
    profile = KVQuantization(format)
    packed = quantize_kv(x, profile)
    scale = np.max(np.abs(x.numpy()), axis=-1, keepdims=True) / profile.maximum
    scale = np.where(scale == 0, 1, scale)
    normalized = np.clip(x.numpy() / scale, -profile.maximum, profile.maximum)
    codes = torch.from_numpy(np.rint(normalized) if format == "int8" else normalized).to(
        profile.dtype
    )
    assert packed.values.dtype == profile.dtype
    torch.testing.assert_close(packed.scales, torch.from_numpy(scale), atol=0, rtol=0)
    torch.testing.assert_close(packed.values.float(), codes.float(), atol=0, rtol=0)
    expected = codes.float() * torch.from_numpy(scale)
    torch.testing.assert_close(packed.dequantize(), expected, atol=0, rtol=0)
    assert packed.nbytes < x.numel() * x.element_size()
    if format == "int8":
        assert ((x - expected).abs() <= packed.scales * 0.5001).all()


@pytest.mark.parametrize("format", ["int8", "fp8_e4m3fn", "fp8_e5m2"])
def test_live_pages_cow_prefix_references_masks_and_reclaim(format):
    torch.manual_seed(90)
    pool = PagedStatePool(block_size=3, max_blocks=12, quantization=KVQuantization(format))
    identity = PrefixIdentity("packed-policy")
    state = pool.create(identity.fingerprint())
    key, value = torch.randn(1, 2, 5, 32), torch.randn(1, 2, 5, 32)
    mask = torch.tensor([True, False, True, True, True]).reshape(1, 1, 5, 1)
    pool.append_delta(state, ((key, value), (mask,)))
    reference = pool.materialize(state)
    fork = pool.fork(state)
    tail = fork.pages[-1]
    cache = PrefixCache(pool)
    cache.publish(identity, list(range(5)), state)
    with pool.read_pages(fork) as views:
        assert all(isinstance(p.payload[0], QuantizedKV) for p in views)
        assert views[0].payload[-1].dtype == torch.bool
        pool.append_delta(state, ((key[..., :1, :], value[..., :1, :]), (mask[..., :1, :],)))
        assert state.pages[-1] != tail == fork.pages[-1]
        torch.testing.assert_close(pool.materialize(fork)[0][0], reference[0][0], atol=0, rtol=0)
        pool.release(fork)
        assert views[-1].payload[0].dequantize().shape[-2] == 2
    pool.truncate(state, 4)
    torch.testing.assert_close(
        pool.materialize(state)[0][0], reference[0][0][..., :4, :], atol=0, rtol=0
    )
    with pytest.raises(AttributeError):
        pool.quantization = None
    pool.release(state)
    cache.clear()
    assert pool.storage_metrics()["allocated_tensor_bytes"] == 0


@pytest.mark.parametrize("format", ["int8", "fp8_e4m3fn", "fp8_e5m2"])
def test_quantized_tile_attention_matches_full_dequantized_dense_softmax(format, monkeypatch):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    q, k, v = torch.randn(1, 4, 5, 32), torch.randn(1, 2, 23, 32), torch.randn(1, 2, 23, 24)
    profile = KVQuantization(format)
    pk, pv = quantize_kv(k, profile), quantize_kv(v, profile)
    dk, dv = pk.dequantize().repeat_interleave(2, 1), pv.dequantize().repeat_interleave(2, 1)
    pos = torch.arange(18, 23)[None]
    visible = (torch.arange(23)[None, None, :] <= pos[:, :, None]) & (
        torch.arange(23)[None, None, :] > pos[:, :, None] - 12
    )
    scores = (q @ dk.transpose(-1, -2)) / 32**0.5
    expected = scores.masked_fill(~visible[:, None], -torch.inf).softmax(-1) @ dv
    maximum = []
    original = QuantizedKV.dequantize

    def bounded(self, **kwargs):
        maximum.append(self.shape[-2])
        assert self.shape[-2] <= 3
        return original(self, **kwargs)

    monkeypatch.setattr(QuantizedKV, "dequantize", bounded)
    blocks = [
        AttentionBlock(pk.narrow(2, i, min(4, 23 - i)), pv.narrow(2, i, min(4, 23 - i)), i)
        for i in range(0, 23, 4)
    ]
    actual = online_attention(
        q, blocks, query_positions=pos, window=12, query_block_size=2, key_block_size=3
    )
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=3e-5)
    assert maximum and max(maximum) <= 3


@pytest.mark.parametrize("format", ["int8", "fp8_e4m3fn", "fp8_e5m2"])
@pytest.mark.parametrize("family", [LlamaConfig, Qwen2Config, Qwen3Config])
def test_whole_model_chunking_and_continuous_scheduler_match_quantized_dense_oracle(format, family):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = build_model(
        family(
            vocab_size=24,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).eval()
    oracle = copy.deepcopy(model)
    profile = KVQuantization(format)

    def forward(
        self, hidden, position_ids, padding=None, previous=None, *, seen_tokens=0, use_cache=False
    ):
        assert previous is None and not use_cache and seen_tokens == 0
        b, n, _ = hidden.shape
        split = lambda x, h: x.reshape(b, n, h, self.head_dim).transpose(1, 2)
        q = self.rope(self.q_norm(split(self.q_proj(hidden), self.num_heads)), position_ids)
        k = self.rope(self.k_norm(split(self.k_proj(hidden), self.num_kv_heads)), position_ids)
        v = split(self.v_proj(hidden), self.num_kv_heads)

        def independent(x):
            scale = x.abs().amax(-1, keepdim=True) / profile.maximum
            scale = torch.where(scale == 0, 1.0, scale)
            code = (x / scale).clamp(-profile.maximum, profile.maximum)
            if format == "int8":
                code = code.round()
            return code.to(profile.dtype).float() * scale

        k, v = (
            independent(k).repeat_interleave(self.num_heads // self.num_kv_heads, 1),
            independent(v).repeat_interleave(self.num_heads // self.num_kv_heads, 1),
        )
        scores = (q @ k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), -torch.inf)
        result = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, -1)
        return self.o_proj(result), None

    for layer in oracle.model.layers:
        layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
    runner = PagedAttentionRunner(
        model,
        policy_artifact_id="packed",
        backend="torch_online_paged",
        block_size=3,
        max_blocks=40,
        query_block_size=2,
        key_block_size=3,
        kv_quantization=profile,
    )
    runner.pool.materialize = lambda *a: pytest.fail(
        "Hot quantized attention must never rebuild float history"
    )
    sequence = runner.pool.create("test")
    inputs = [1, 2, 3, 4, 5, 6, 7]
    with torch.no_grad():
        expected = oracle(torch.tensor([inputs])).logits[0]
    actual = torch.cat(
        [
            runner.forward_batch([sequence], [chunk], return_all_logits=True)[0]
            for chunk in [inputs[:4], inputs[4:5], inputs[5:]]
        ]
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    runner.pool.release(sequence)

    async def run():
        engine = InferenceEngine(runner, max_active=2, max_batch_tokens=8, prefill_chunk_size=4)
        try:
            results = []
            for _ in range(2):
                results.append(
                    await (
                        await engine.submit(inputs, SamplingConfig(max_new_tokens=4, temperature=0))
                    ).collect()
                )
            tokens = inputs[:]
            greedy = []
            with torch.no_grad():
                for _ in range(4):
                    token = int(oracle(torch.tensor([tokens])).logits[0, -1].argmax())
                    tokens.append(token)
                    greedy.append(token)
            assert all(r.token_ids == tuple(greedy) and r.stop_reason == "length" for r in results)
            assert results[1].prefix_hit_tokens >= 3
        finally:
            await engine.close()
        assert runner.pool.storage_metrics()["allocated_tensor_bytes"] == 0

    asyncio.run(run())


def test_nonfinite_suffix_rejected_before_storage_mutation():
    pool = PagedStatePool(block_size=2, max_blocks=8, quantization=KVQuantization())
    state = pool.create("x")
    good = torch.ones(1, 1, 3, 16)
    pool.append_delta(state, ((good, good),))
    before = pool.storage_metrics()
    with pytest.raises(ValueError):
        pool.append_delta(state, ((good, good * float("nan")),))
    assert state.length == 3 and pool.storage_metrics() == before
