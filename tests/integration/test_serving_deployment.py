import asyncio
import json
import pytest
import torch

from aster.core import ArtifactStore
from aster.models import build_model, LlamaConfig, load_model
from aster.data import ByteTokenizer, load_tokenizer
from aster.inference import (
    DeploymentRouter,
    HTTPServer,
    ChatTemplate,
    SamplingConfig,
    measure_http,
    ModelRunner,
    InferenceEngine,
)


def publish(store, directory, seed):
    torch.manual_seed(seed)
    model = build_model(
        LlamaConfig(
            vocab_size=259,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
    )
    model.save_pretrained(directory)
    ByteTokenizer().save_pretrained(directory)
    ChatTemplate().save_pretrained(directory)
    return store.publish(directory, kind="native_token_model", metadata={"seed": seed})


def test_artifact_deploy_atomic_switch_rollback_and_client_measurement(tmp_path):
    torch.set_num_threads(1)
    store = ArtifactStore(tmp_path / "store")
    first = publish(store, tmp_path / "one", 1)
    second = publish(store, tmp_path / "two", 2)

    async def exercise():
        router = DeploymentRouter(
            store,
            loader=load_model,
            tokenizer_loader=load_tokenizer,
            chat_template_loader=ChatTemplate.from_pretrained,
        )
        initial = await router.deploy(first.id, warmup_prompt_ids=[1, 5])
        assert initial.warmup_input_tokens == 2
        active = await router.submit([1, 2, 3], SamplingConfig(max_new_tokens=8, temperature=0))
        await router.deploy(second.id, warmup_prompt_ids=[1, 5])
        newer = await router.submit([1, 2, 3], SamplingConfig(max_new_tokens=4, temperature=0))
        a, b = await asyncio.gather(active.collect(), newer.collect())
        assert a.policy_artifact_id == first.id and b.policy_artifact_id == second.id
        await router.rollback(first.id)
        assert router.runner.policy_artifact_id == first.id
        async with HTTPServer(router) as server:
            observation = await measure_http(
                server.url, [[1, 2], [1, 3], [1, 4]], max_new_tokens=3, concurrency=2
            )
            assert observation["successful_requests"] == 3 and observation["failed_requests"] == 0
            assert observation["throughput_tokens_per_second"] > 0
            for record in observation["records"]:
                assert len(record["itl_seconds"]) == 2 and record["ttft_seconds"] >= 0
                assert record["clock"] if "clock" in record else True
        assert all(engine._worker.done() for engine in router._versions.values())

    asyncio.run(exercise())


def test_chat_structured_sse_has_no_duplicate_final_content(tmp_path):
    torch.set_num_threads(1)
    store = ArtifactStore(tmp_path / "store")
    artifact = publish(store, tmp_path / "model", 7)

    async def exercise():
        router = DeploymentRouter(
            store,
            loader=load_model,
            tokenizer_loader=load_tokenizer,
            chat_template_loader=ChatTemplate.from_pretrained,
        )
        await router.deploy(artifact.id, warmup_prompt_ids=[1])
        async with HTTPServer(router) as server:
            reader, writer = await asyncio.open_connection(*server.address)
            data = {
                "messages": [{"role": "user", "content": "给出结果"}],
                "max_tokens": 64,
                "stream": True,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"result": {"const": "成功"}},
                            "required": ["result"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            body = json.dumps(data).encode()
            writer.write(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 10)
            writer.close()
            await writer.wait_closed()
            assert response.startswith(b"HTTP/1.1 200")
            events = [
                json.loads(line[6:])
                for line in response.split(b"\r\n\r\n", 1)[1].splitlines()
                if line.startswith(b"data: {")
            ]
            combined = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
            assert json.loads(combined) == {"result": "成功"}
            assert events[-1]["choices"][0]["delta"] == {}

    asyncio.run(exercise())
