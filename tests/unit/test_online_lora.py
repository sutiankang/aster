import asyncio
import copy
import json
import pytest
import torch
from aster.inference import (
    MultiLoRARunner,
    LoRAWeights,
    ModelRunner,
    PagedAttentionRunner,
    PagedStateArchive,
    KVQuantization,
    InferenceEngine,
    SamplingConfig,
    PrefixIdentity,
    HTTPServer,
)
from aster.inference.state import StateError
from aster.models import build_model, LlamaConfig


def setup(kind="paged", quantized=False, capacity=40):
    torch.set_num_threads(1)
    torch.manual_seed(179)
    model = build_model(
        LlamaConfig(
            vocab_size=24,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).eval()
    make = lambda m: (
        PagedAttentionRunner(
            m,
            policy_artifact_id="base",
            backend="torch_online_paged",
            block_size=3,
            max_blocks=capacity,
            kv_quantization=KVQuantization() if quantized else None,
        )
        if kind == "paged"
        else ModelRunner(m, policy_artifact_id="base", block_size=3, max_blocks=capacity)
    )
    runner = MultiLoRARunner(make(model))
    weights = []
    for rank in (2, 3):
        targets = {}
        for name in ("model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj", "lm_head"):
            layer = model.get_submodule(name)
            targets[name] = LoRAWeights(
                torch.randn(rank, layer.in_features) * 0.03,
                torch.randn(layer.out_features, rank) * 0.1,
                2 * rank,
            )
        weights.append(targets)
    identifiers = [runner.register_adapter(w, base_artifact_id="base") for w in weights]
    return model, runner, weights, identifiers, make


@pytest.mark.parametrize("kind", ["dense", "paged"])
def test_entire_model_logits_match_independently_merged_weights(kind):
    model, runner, weights, identifiers, _ = setup(kind)
    base_weights = [p.detach().clone() for p in runner.model.parameters()]
    prompt = [1, 2, 3, 4, 5]
    for adapter, values in zip(identifiers, weights):
        reference = copy.deepcopy(model)
        with torch.no_grad():
            for name, w in values.items():
                reference.get_submodule(name).weight.add_((w.b @ w.a) * (w.alpha / w.a.shape[0]))
            expected = reference(torch.tensor([prompt])).logits[0]
        identity = runner.resolve_model_identity(adapter)
        runner.prepare_request(prompt, identity, None, max_prefill_tokens=4)
        sequence = runner.pool.create(identity.fingerprint())
        try:
            actual = torch.cat(
                [
                    runner.forward_batch([sequence], [chunk], return_all_logits=True)[0]
                    for chunk in (prompt[:3], prompt[3:])
                ]
            )
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            with pytest.raises(StateError):
                runner.remove_adapter(adapter)
        finally:
            runner.pool.release(sequence)
            runner.release_request(identity)
    for a, b in zip(base_weights, runner.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(layer.active is None for layer in runner._layers.values())
    assert runner.adapter_metrics()["pinned_requests"] == 0


@pytest.mark.parametrize("quantized", [False, True])
def test_shared_online_model_identity_prefix_swap_and_cancel(quantized):
    _, runner, _, identifiers, _ = setup(quantized=quantized, capacity=4)

    async def run():
        engine = InferenceEngine(
            runner,
            max_active=3,
            max_batch_tokens=9,
            prefill_chunk_size=3,
            offload_archive=PagedStateArchive(runner.pool),
        )
        prompt = [1, 2, 3, 4, 5]
        config = SamplingConfig(max_new_tokens=5, temperature=0)
        try:
            handles = [
                await engine.submit(prompt, config, identity=runner.resolve_model_identity(adapter))
                for adapter in ["base", *identifiers]
            ]
            with pytest.raises(StateError):
                runner.remove_adapter(identifiers[0])
            results = await asyncio.gather(*(h.collect() for h in handles))
            assert all(r.stop_reason == "length" for r in results)
            assert [r.adapter_id for r in results] == ["none", *identifiers]
            assert sum(r.preemption_count for r in results) > 0
            assert runner.adapter_metrics()["request_domains"] == 0

            for adapter, expected in zip(["base", *identifiers], results):
                result = await (
                    await engine.submit(
                        prompt, config, identity=runner.resolve_model_identity(adapter)
                    )
                ).collect()
                assert result.token_ids == expected.token_ids
            pending = await engine.submit(
                prompt, config, identity=runner.resolve_model_identity(identifiers[0])
            )
            assert (await pending.cancel()).stop_reason == "cancelled"
        finally:
            await engine.close()
        assert runner.pool.used_blocks == engine.offload_archive.stored_bytes == 0
        assert runner.adapter_metrics()["pinned_requests"] == 0
        for adapter in identifiers:
            runner.remove_adapter(adapter)
        assert runner.resident_bytes == 0

    asyncio.run(run())


def test_content_identity_copies_inputs_validation_and_failed_forward_restores_selection(
    monkeypatch,
):
    model, runner, weights, ids, _ = setup()
    assert runner.register_adapter(weights[0], base_artifact_id="base") == ids[0]
    weights[0]["lm_head"].b.add_(0.1)
    changed = runner.register_adapter(weights[0], base_artifact_id="base")
    assert changed != ids[0]
    for value, base in [
        ({"lm_head": LoRAWeights(torch.ones(2, 65), torch.ones(24, 2), 4)}, "base"),
        (weights[0], "wrong-base"),
    ]:
        with pytest.raises(ValueError):
            runner.register_adapter(value, base_artifact_id=base)
    assert runner.adapter_metrics()["adapters"] == 3
    identity = PrefixIdentity("base", adapter=ids[0], tenant="other")
    runner.prepare_request([1], identity, None, max_prefill_tokens=1)
    state = runner.pool.create(identity.fingerprint())

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(runner._runner, "forward_batch", fail)
    with pytest.raises(RuntimeError):
        runner.forward_batch([state], [[1]])
    assert all(layer.active is None for layer in runner._layers.values())
    runner.release_request(identity)
    runner.pool.release(state)


def test_real_http_sse_selects_registered_adapter_and_returns_lineage():
    _, runner, _, identifiers, _ = setup(quantized=True)

    async def request(server, payload):
        reader, writer = await asyncio.open_connection(*server.address)
        body = json.dumps(payload).encode()
        writer.write(
            f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 10)
        writer.close()
        await writer.wait_closed()
        return response

    async def run():
        async with HTTPServer(InferenceEngine(runner)) as server:
            payload = dict(
                model=identifiers[0], prompt_token_ids=[1, 2, 3], max_tokens=3, temperature=0
            )
            response = await request(server, payload)
            assert response.startswith(b"HTTP/1.1 200")
            result = json.loads(response.split(b"\r\n\r\n")[1])
            assert result["model"] == "base" and result["aster"]["adapter_id"] == identifiers[0]
            streamed = await request(server, dict(payload, stream=True))
            messages = [
                json.loads(line[6:])
                for line in streamed.splitlines()
                if line.startswith(b"data: {")
            ]
            assert [m["aster"]["token_id"] for m in messages[:-1]] == result["aster"]["token_ids"]
            assert messages[-1]["aster"]["adapter_id"] == identifiers[0]
            bad = await request(server, dict(payload, model="unregistered"))
            assert bad.startswith(b"HTTP/1.1 400")
        assert runner.adapter_metrics()["pinned_requests"] == runner.pool.used_blocks == 0

    asyncio.run(run())


def test_shared_trainer_lora_update_deploy_and_full_weight_oracle():
    from aster.methods.distillation import inject_lora, merge_lora
    from aster.methods.supervised import CrossEntropyObjective
    from aster.training import Trainer

    model, runner, _, _, _ = setup()
    trained = inject_lora(copy.deepcopy(model), targets=["lm_head"], rank=4, alpha=8.0)
    trainer = Trainer(trained, CrossEntropyObjective(), lr=0.01)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
    }
    for _ in range(3):
        trainer.step([batch])
    adapter = runner.register_trained_adapter(trainer.model, base_artifact_id="base")
    reference = merge_lora(trainer.model)
    identity = runner.resolve_model_identity(adapter)
    runner.prepare_request([1, 2, 3], identity, None, max_prefill_tokens=3)
    state = runner.pool.create(identity.fingerprint())
    try:
        actual = runner.forward_batch([state], [[1, 2, 3]])[0]
        with torch.no_grad():
            expected = reference(torch.tensor([[1, 2, 3]])).logits[0, -1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        runner.release_request(identity)
        runner.pool.release(state)
    with torch.no_grad():
        trainer.model.model.embed_tokens.weight.add_(1.0)
    with pytest.raises(ValueError, match="base weights differ"):
        runner.register_trained_adapter(trainer.model, base_artifact_id="base")
