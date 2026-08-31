import asyncio
import copy
import json

import torch

from aster.agents import (
    AgentConfig,
    AgentLoop,
    EventLog,
    NativeAgentPolicy,
    PermissionBroker,
    ToolExecutor,
)
from aster.core import digest_json, read_json
from aster.data import ByteTokenizer
from aster.inference import ModelRunner, InferenceEngine
from aster.methods.agent_learning import verified_agent_corpus, AgentSFTMethod
from aster.models import LlamaConfig, build_model
from aster.training import Trainer
from test_agents_native import FiniteStateControlModel


def test_verified_native_agent_actions_feed_real_student_training(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    tokenizer = ByteTokenizer()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("工具观察绝不能当成生成动作学习", encoding="utf-8")
    path = tmp_path / "events.jsonl"
    processor_id = digest_json({"fixture_renderer": "compact-json-then-END-v1"})

    async def collect():
        engine = InferenceEngine(
            ModelRunner(
                FiniteStateControlModel(),
                policy_artifact_id="control-teacher",
                tokenizer=tokenizer,
                block_size=32,
                max_blocks=1024,
            ),
            max_batch_tokens=4096,
            prefill_chunk_size=4096,
        )
        policy = NativeAgentPolicy(
            engine,
            tokenizer,
            processor_fingerprint=processor_id,
            render_messages=lambda value: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n<END>\n"
            ),
        )
        with EventLog(path) as log:
            loop = AgentLoop(
                policy,
                ToolExecutor(PermissionBroker(workspace), log, tmp_path / "receipts"),
                log,
                config=AgentConfig(max_steps=3, max_action_tokens=128, max_context_tokens=16000),
            )
            result = await loop.run(
                "读取note.txt", verifier=lambda text: {"passed": text == "已读取"}
            )
            assert result.status == "verified"

            await loop.run("再次读取note.txt")
        await engine.close()

    asyncio.run(collect())
    corpus = verified_agent_corpus(
        path,
        expected_policy_id="control-teacher",
        tokenizer_fingerprint=digest_json(tokenizer.to_dict()),
        processor_fingerprint=processor_id,
        vocab_size=259,
    )
    assert len(corpus.examples) == 2 and len(corpus.receipts) == 2
    assert [row["accepted"] for row in corpus.receipts] == [True, False]
    for row in corpus.examples:
        first_action = next(index for index, value in enumerate(row["labels"]) if value != -100)
        assert row["labels"][:first_action] == [-100] * first_action
        assert row["labels"][first_action:] == row["input_ids"][first_action:]
    assert "工具观察" in tokenizer.decode(corpus.examples[1]["input_ids"])
    corpus.save(tmp_path / "corpus.json")
    assert read_json(tmp_path / "corpus.json")["identity"] == corpus.identity
    model = build_model(
        LlamaConfig(
            vocab_size=259,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8192,
        )
    )
    engine = Trainer(model, lr=0.001, accumulation_steps=2)
    method = AgentSFTMethod(engine, corpus, tokenizer=tokenizer, processor_fingerprint=processor_id)
    before = copy.deepcopy(model.state_dict())
    assert method.update([0, 1]).updated
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    engine.save_checkpoint(tmp_path / "checkpoint")
    method.update([1, 0])
    expected = copy.deepcopy(model.state_dict())
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    method.update([1, 0])
    for key, value in expected.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    rejected = verified_agent_corpus(
        path,
        expected_policy_id="wrong-policy",
        tokenizer_fingerprint=corpus.tokenizer_fingerprint,
        processor_fingerprint=processor_id,
        vocab_size=259,
    )
    assert not rejected.examples and all(not row["accepted"] for row in rejected.receipts)
