import asyncio
import threading
import pytest
from aster.core.async_work import settle_thread
from aster.agents import EventLog, PermissionBroker, ToolExecutor, ToolSpec, replay


@pytest.mark.parametrize("fails", [False, True])
def test_thread_barrier_preserves_completion_and_exceptions_after_repeated_cancel(fails):
    entered, finish = threading.Event(), threading.Event()

    def work():
        entered.set()
        assert finish.wait(5)
        if fails:
            raise ValueError("actual worker error")
        return 17

    async def run():
        task = asyncio.create_task(settle_thread(work))
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finish.set()
        result = await task
        assert result.cancelled
        if fails:
            with pytest.raises(ValueError, match="actual worker error"):
                result.unwrap()
        else:
            assert result.unwrap() == 17

    try:
        asyncio.run(run())
    finally:
        finish.set()


def test_scheduler_does_not_release_forward_state_on_second_cancellation(monkeypatch):
    import torch
    from aster.models import LlamaConfig, build_model
    from aster.inference import ModelRunner, InferenceEngine, SamplingConfig

    torch.set_num_threads(1)
    model = build_model(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    runner = ModelRunner(model, policy_artifact_id="cancel-fixture")
    original = runner.forward_batch
    entered, finish = threading.Event(), threading.Event()
    held = []

    def slow(sequences, chunks):
        held.extend(sequences)
        entered.set()
        assert finish.wait(5)
        assert all(not state.released for state in sequences)
        return original(sequences, chunks)

    monkeypatch.setattr(runner, "forward_batch", slow)

    async def run():
        engine = InferenceEngine(runner)
        handle = await engine.submit([1, 2, 3], SamplingConfig(max_new_tokens=2, temperature=0))
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(3):
            engine._worker.cancel()
            await asyncio.sleep(0)
            assert not engine._worker.done() and all(not state.released for state in held)
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await engine._worker
        assert (await handle.collect()).stop_reason == "cancelled"
        assert all(state.released for state in held) and runner.pool.used_blocks == 0

    try:
        asyncio.run(run())
    finally:
        finish.set()


@pytest.mark.parametrize("failure", [None, "read", "external"])
def test_agent_tool_commits_receipt_before_propagating_repeated_cancellation(tmp_path, failure):
    entered, finish = threading.Event(), threading.Event()
    effects = []

    def tool(arguments):
        entered.set()
        assert finish.wait(5)
        effects.append("once")
        if failure is not None:
            raise ValueError("operation failed after cancellation")
        return {"result": "done"}

    async def run():
        with EventLog(tmp_path / "events.jsonl") as log:
            log.append("thread.started", thread_id="t")
            log.append("turn.started", thread_id="t", turn_id="u")
            broker = PermissionBroker(tmp_path, external_authorizer=lambda call: True)
            executor = ToolExecutor(broker, log, tmp_path / "receipts")
            executor.register(
                ToolSpec(
                    "fixture.effect", "1", "a" * 64, failure or "external", "controlled test effect"
                ),
                tool,
            )
            call = executor.prepare("fixture.effect", {}, thread_id="t", turn_id="u")
            task = asyncio.create_task(
                executor.execute(call, broker.approve(call), thread_id="t", turn_id="u")
            )
            assert await asyncio.to_thread(entered.wait, 5)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert effects == ["once"]
        state = replay(tmp_path / "events.jsonl")
        assert state.items[call.call_id]["status"] == (
            "ambiguous" if failure == "external" else "result_committed"
        )
        assert (tmp_path / "receipts" / (call.call_id + ".json")).is_file() == (
            failure != "external"
        )

    try:
        asyncio.run(run())
    finally:
        finish.set()
