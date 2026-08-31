import asyncio
import json
import pytest
import torch

from aster.core import ArtifactStore
from aster.models import Gemma4TextConfig, build_model, load_model
from aster.inference import Gemma4SnapshotRunner, InferenceEngine, HTTPServer


def test_gemma4_artifact_loopback_sse_and_cancel(tmp_path):
    async def exercise():
        torch.set_num_threads(1)
        torch.manual_seed(76)
        model = build_model(Gemma4TextConfig()).eval()
        model.save_pretrained(tmp_path / "model")
        store = ArtifactStore(tmp_path / "store")
        artifact = store.publish(tmp_path / "model", kind="native_gemma4", metadata={})
        runner = Gemma4SnapshotRunner.from_artifact(store, artifact.id, loader=load_model)
        engine = InferenceEngine(runner, prefill_chunk_size=2)
        server = await HTTPServer(engine).start()
        reader, writer = await asyncio.open_connection(*server.address)
        body = json.dumps(
            {
                "prompt_token_ids": [1, 2, 3, 4, 5, 6],
                "max_tokens": 3,
                "temperature": 0,
                "stream": True,
            }
        ).encode()
        writer.write(
            f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), 10)
        writer.close()
        await writer.wait_closed()
        assert response.startswith(b"HTTP/1.1 200") and response.endswith(b"data: [DONE]\n\n")
        events = [
            json.loads(line[6:]) for line in response.splitlines() if line.startswith(b"data: {")
        ]
        assert (
            len(events) == 4 and len([e for e in events if "token_id" in e.get("aster", {})]) == 3
        )
        assert all(e["model"] == artifact.id for e in events)
        reader, writer = await asyncio.open_connection(*server.address)
        body = json.dumps(
            {
                "prompt_token_ids": [1, 2, 3, 4, 5, 6],
                "max_tokens": 100,
                "temperature": 0,
                "stream": True,
            }
        ).encode()
        writer.write(
            f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if engine.completed_count >= 2:
                break
            await asyncio.sleep(0.01)
        assert engine.completed_count == 2 and not engine.active_count
        await server.close()
        assert runner.pool.used_bytes == 0 and not runner._bindings and engine._worker.done()

    asyncio.run(exercise())
