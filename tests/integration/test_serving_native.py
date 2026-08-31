import asyncio
import json
import pytest
import torch

from aster.inference import HTTPServer, InferenceEngine, ModelRunner
from aster.models import build_model, LlamaConfig


def make_server(**engine_options):
    torch.set_num_threads(1)
    model = build_model(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    engine = InferenceEngine(
        ModelRunner(model, policy_artifact_id="native-artifact", block_size=4), **engine_options
    )
    return HTTPServer(engine)


async def request(server, method, path, payload=None):
    reader, writer = await asyncio.open_connection(*server.address)
    body = json.dumps(payload).encode() if payload is not None else b""
    writer.write(
        f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    response = await asyncio.wait_for(reader.read(), 10)
    writer.close()
    await writer.wait_closed()
    return response


def test_http_native_health_completion_sse_and_errors():
    async def exercise():
        async with make_server() as server:
            health = await request(server, "GET", "/ready")
            assert health.startswith(b"HTTP/1.1 200")
            result = await request(
                server,
                "POST",
                "/v1/completions",
                {"prompt_token_ids": [1, 2], "max_tokens": 3, "temperature": 0},
            )
            payload = json.loads(result.split(b"\r\n\r\n", 1)[1])
            assert len(payload["aster"]["token_ids"]) == 3
            assert payload["aster"]["behavior_logprobs"] == [0.0, 0.0, 0.0]
            assert payload["model"] == "native-artifact"
            streamed = await request(
                server,
                "POST",
                "/v1/completions",
                {"prompt_token_ids": [1, 2], "max_tokens": 3, "temperature": 0, "stream": True},
            )
            body = streamed.split(b"\r\n\r\n", 1)[1]
            events = [
                json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")
            ]
            assert len(events) == 4 and body.endswith(b"data: [DONE]\n\n")
            assert [event["aster"]["token_id"] for event in events[:-1]] == payload["aster"][
                "token_ids"
            ]
            bad = await request(
                server, "POST", "/v1/completions", {"prompt_token_ids": [1], "unknown": True}
            )
            assert bad.startswith(b"HTTP/1.1 400")
            bad_model = await request(
                server,
                "POST",
                "/v1/completions",
                {"prompt_token_ids": [1], "model": "mutable-alias"},
            )
            assert bad_model.startswith(b"HTTP/1.1 400")
        assert server.engine.runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_http_disconnect_cancels_and_server_drains_workers():
    async def exercise():
        server = await make_server().start()
        reader, writer = await asyncio.open_connection(*server.address)
        payload = json.dumps(
            {
                "prompt_token_ids": [1, 2],
                "max_tokens": 100,
                "stream": True,
                "request_id": "disconnect",
            }
        ).encode()
        writer.write(
            f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if server.engine.completed_count:
                break
            await asyncio.sleep(0.01)
        assert server.engine.active_count == 0 and server.engine.completed_count == 1
        await server.close()
        assert server.engine._worker.done()
        assert server.engine.runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_public_binding_rejected_without_gateway():
    server = make_server()
    with pytest.raises(ValueError, match="Public"):
        HTTPServer(server.engine, host="0.0.0.0")


def test_http_rejects_duplicate_json_and_numeric_overflow_before_model_execution():
    async def exercise():
        async with make_server() as server:
            for body in (
                b'{"prompt_token_ids":[1],"temperature":1e999}',
                b'{"prompt_token_ids":[1],"max_tokens":2,"max_tokens":3}',
            ):
                reader, writer = await asyncio.open_connection(*server.address)
                writer.write(
                    f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 5)
                writer.close()
                await writer.wait_closed()
                assert response.startswith(b"HTTP/1.1 400")
            assert server.engine.runner.forward_calls == 0

    asyncio.run(exercise())
