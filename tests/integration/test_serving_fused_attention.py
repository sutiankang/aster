import asyncio
from copy import deepcopy

import pytest
import torch

from aster.models import build_model, LlamaConfig, Qwen3Config
from aster.optimization.fused_attention import set_attention_backend, UnsupportedAttentionBackend
from aster.methods.supervised import CrossEntropyObjective
from aster.training import Trainer
from aster.inference import PagedAttentionRunner, InferenceEngine, SamplingConfig


def native(config=LlamaConfig, **kwargs):
    torch.set_num_threads(1)
    torch.manual_seed(680)
    return build_model(
        config(
            vocab_size=24,
            hidden_size=24,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=128,
            **kwargs,
        )
    )


@pytest.mark.parametrize("zero_stage", [0, 3])
def test_real_shared_trainer_update_checkpoint_resume_and_backend_identity(tmp_path, zero_stage):
    model = native()
    fused = set_attention_backend(deepcopy(model), query_block_size=3, key_block_size=2)
    objective = CrossEntropyObjective()
    reference = Trainer(model, objective, lr=0.001, zero_stage=zero_stage)
    trainer = Trainer(fused, objective, lr=0.001, zero_stage=zero_stage)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])}
    expected, actual = reference.step([batch]), trainer.step([batch])
    assert (
        expected.updated
        and actual.updated
        and actual.loss == pytest.approx(expected.loss, abs=2e-6)
    )
    left, right = reference.export_state_dict(), trainer.export_state_dict()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=3e-5, rtol=3e-4)
    checkpoint = trainer.save_checkpoint(tmp_path / "native-attention.json")
    trainer.step([batch])
    next_state = trainer.export_state_dict()
    restored = Trainer(
        set_attention_backend(native(), query_block_size=3, key_block_size=2),
        objective,
        lr=0.001,
        zero_stage=zero_stage,
    )
    restored.load_checkpoint(checkpoint)
    restored.step([batch])
    for key, value in restored.export_state_dict().items():
        torch.testing.assert_close(value, next_state[key], atol=0, rtol=0)
    wrong = Trainer(
        set_attention_backend(native(), query_block_size=4, key_block_size=2),
        objective,
        lr=0.001,
        zero_stage=zero_stage,
    )
    with pytest.raises(ValueError, match="配置"):
        wrong.load_checkpoint(checkpoint)


def test_bf16_shared_trainer_uses_actual_tiled_backward():
    model = set_attention_backend(native(), query_block_size=3, key_block_size=2)
    trainer = Trainer(model, CrossEntropyObjective(), lr=0.001, precision="bf16")
    assert trainer.step([{"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}]).updated
    assert model.model.layers[0].self_attn.attention_backend.work.backward_tiles > 0


def make_paged(model, *, blocks=40):
    return PagedAttentionRunner(
        model.eval(),
        policy_artifact_id="fixed-native-weight-id",
        backend="triton_fused_paged",
        attention_fallback="torch_tiled",
        query_block_size=2,
        key_block_size=3,
        block_size=3,
        max_blocks=blocks,
    )


def test_explicit_fused_runner_fallback_cow_rollback_absolute_window_and_error_lease(monkeypatch):
    model = native(
        Qwen3Config, sliding_window=4, layer_types=("sliding_attention", "full_attention")
    ).eval()
    paged = make_paged(model)
    assert (
        paged.attention_work.backend == "torch_tiled"
        and "CUDA" in paged.attention_work.fallback_reason
    )

    def forbid(*args, **kwargs):
        raise AssertionError("No dense cache materialization")

    paged.pool.materialize = forbid
    sequence = paged.pool.create("same-domain")
    prompt = [1, 3, 5, 7]
    paged.forward_batch([sequence], [prompt])
    branch = paged.pool.fork(sequence)
    old = sequence.pages[-1]
    got = paged.forward_batch([sequence], [[9, 11, 13, 15]], return_all_logits=True)[0]
    assert old == branch.pages[-1] and old != sequence.pages[1]
    with torch.no_grad():
        expected = model(torch.tensor([prompt + [9, 11, 13, 15]])).logits[0, 4:]
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)
    paged.pool.truncate(sequence, 3)
    got = paged.forward_batch([sequence], [[2, 4]])[0]
    with torch.no_grad():
        expected = model(torch.tensor([prompt[:3] + [2, 4]])).logits[0, -1]
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)
    import aster.inference.paged_attention as runner_module

    checks = []

    def check_sync():
        checks.append(sum(page.readers for page in paged.pool._pages if page is not None))

    paged._synchronize = check_sync

    def fail(*args, **kwargs):
        raise RuntimeError("A later computation failed after possible GPU submission")

    monkeypatch.setattr(runner_module, "paged_fused_attention", fail)
    with pytest.raises(RuntimeError, match="later computation"):
        paged.forward_batch([sequence], [[6]])

    assert checks[-1] > 0 and all(
        page.readers == 0 for page in paged.pool._pages if page is not None
    )
    assert sequence.length == 5
    paged.pool.release(sequence)
    paged.pool.release(branch)
    assert paged.pool.used_blocks == 0


def greedy(model, prompt, count):
    inputs, generated = list(prompt), []
    with torch.no_grad():
        for _ in range(count):
            token = int(model(torch.tensor([inputs])).logits[0, -1].argmax())
            inputs.append(token)
            generated.append(token)
    return tuple(generated)


def test_continuous_scheduler_prefix_and_preemption_through_explicit_fallback():
    model = native().eval()

    async def run():
        runner = make_paged(model, blocks=4)
        engine = InferenceEngine(runner, max_active=2, max_batch_tokens=12, prefill_chunk_size=4)
        prompt, config = [1, 2, 3, 4], SamplingConfig(temperature=0, max_new_tokens=5)
        try:
            first = await (await engine.submit(prompt, config)).collect()
            assert first.token_ids == greedy(model, prompt, 5)
            handles = [await engine.submit(prompt, config) for _ in range(2)]
            results = await asyncio.wait_for(asyncio.gather(*(x.collect() for x in handles)), 10)
            assert all(
                x.token_ids == first.token_ids and x.stop_reason == "length" for x in results
            )
            assert any(x.prefix_hit_tokens > 0 for x in results)
            assert any(x.preemption_count > 0 for x in results)
        finally:
            await engine.close()
        assert runner.pool.used_blocks == 0

    asyncio.run(run())


def test_unsupported_fused_runner_is_rejected_before_serving():
    with pytest.raises(UnsupportedAttentionBackend, match="CUDA"):
        PagedAttentionRunner(native(), policy_artifact_id="x", backend="triton_fused_paged")


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA: native fused model/pool integration awaits real hardware",
)
def test_actual_cuda_native_model_fused_forward_backward_and_paged_prefix():
    pytest.importorskip("triton")
    torch.manual_seed(17)
    model = (
        build_model(
            LlamaConfig(
                vocab_size=24,
                hidden_size=128,
                intermediate_size=160,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=2,
            )
        )
        .cuda()
        .half()
        .eval()
    )
    test = set_attention_backend(deepcopy(model), backend="triton_fused")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device="cuda")
    expected, actual = model(ids).logits, test(ids).logits
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    actual.float().square().sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in test.parameters() if x.grad is not None)
    paged = PagedAttentionRunner(
        model,
        policy_artifact_id="cuda-fused-native",
        backend="triton_fused_paged",
        query_block_size=32,
        key_block_size=32,
        block_size=3,
    )
    sequence = paged.pool.create("bound-cuda")
    paged.forward_batch([sequence], [[1, 2, 3, 4]])
    branch = paged.pool.fork(sequence)
    got = paged.forward_batch([sequence], [[5, 6, 7]])[0]
    torch.testing.assert_close(got, expected[0, -1].float().cpu(), atol=0.02, rtol=0.03)
    assert paged.attention_work.page_launches > 0 and paged.attention_work.backend == "triton_fused"
    paged.pool.release(sequence)
    paged.pool.release(branch)
