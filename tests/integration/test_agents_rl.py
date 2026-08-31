import asyncio
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from aster.agents.agent_rl import AgentRolloutBatch, NativeAgentRLMethod, ReadFileTask
from aster.agents.runtime import AgentConfig
from aster.core import digest_json
from aster.inference import SamplingConfig
from aster.models import LlamaConfig, build_model
from aster.training import Trainer


class ToolActionTokenizer:
    pad_token_id = 3
    eos_token_id = 3
    actions = {
        0: '{"type":"tool","name":"workspace.read","arguments":{"path":"note.txt"}}',
        1: '{"type":"final","text":"blue"}',
        2: '{"type":"final","text":"wrong"}',
        3: "invalid-json",
    }

    def to_dict(self):
        return {
            "type": "test.finite-read-state-processor.v1",
            "vocab": self.actions,
            "context": "BOS + 0 iff a real read result is visible, else BOS + 3",
        }

    def encode(self, text):
        messages = json.loads(text)
        read = any(
            row["role"] == "tool"
            and isinstance(row["content"], dict)
            and row["content"].get("trust") == "untrusted_tool_data"
            and '"sha256"' in row["content"].get("content", "")
            for row in messages
        )
        return [3, 0 if read else 3]

    def decode(self, ids):
        return "".join(self.actions[token] for token in ids)


def setup(tmp_path, *, algorithm="rloo", seed=19, group_size=16, accumulation_steps=2):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    content = b"blue"
    (workspace / "note.txt").write_bytes(content)
    task = ReadFileTask(
        "read-note",
        "Read note.txt and report the exact content",
        str(workspace),
        "note.txt",
        hashlib.sha256(content).hexdigest(),
        "blue",
    )
    tokenizer = ToolActionTokenizer()
    model = build_model(
        LlamaConfig(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
    )
    reference = copy.deepcopy(model)
    trainer = Trainer(model, lr=0.012, max_grad_norm=1.0, accumulation_steps=accumulation_steps)
    method = NativeAgentRLMethod(
        trainer,
        reference,
        tokenizer,
        work_directory=tmp_path / "rollouts",
        reference_tokenizer_fingerprint=digest_json(tokenizer.to_dict()),
        algorithm=algorithm,
        group_size=group_size,
        agent_config=AgentConfig(
            max_steps=2,
            max_action_tokens=1,
            max_total_action_tokens=2,
            max_context_tokens=8,
            timeout_seconds=30.0,
        ),
        sampling=SamplingConfig(max_new_tokens=1, seed=37, eos_token_ids=()),
        kl_weight=0.01,
    )
    return model, trainer, method, task


def success_probability(model):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([[3, 3], [3, 0]])).logits[:, -1].softmax(-1)
    return float(logits[0, 0] * logits[1, 1])


def semantics(records):
    return [
        {
            "reward": row["reward"],
            "seed": row["seed"],
            "status": row["result"]["status"],
            "traces": row["traces"],
        }
        for row in records
    ]


@pytest.mark.parametrize("algorithm", ["rloo", "grpo"])
def test_real_tool_rollout_learning_and_exact_new_controller_resume(tmp_path, algorithm):
    model, trainer, method, task = setup(tmp_path, algorithm=algorithm)
    baseline = success_probability(model)
    frozen = copy.deepcopy(method.reference.state_dict())
    before = copy.deepcopy(model.state_dict())
    result = asyncio.run(method.update([task]))
    assert result.updated and method.updates == 1
    assert len(method.last_records) == 16
    assert any(row["reward"] == 1 and row["successful_reads"] for row in method.last_records)
    assert any(row["reward"] == 0 for row in method.last_records)
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    for row in method.last_records:
        for trace in row["traces"]:
            assert trace["loss_mask"] == [0, 0, 1]
            assert trace["sampling_config"]["temperature"] == 1
            assert trace["raw_model_logprobs"] == trace["behavior_logprobs"]
        if row["reward"]:
            assert [trace["action_token_ids"] for trace in row["traces"]] == [[0], [1]]
            assert row["traces"][1]["prompt_token_ids"] == [3, 0]
    for _ in range(3):
        asyncio.run(method.update([task]))
    assert success_probability(model) > baseline * 1.8
    checkpoint = tmp_path / "checkpoint.json"
    trainer.save_checkpoint(checkpoint)
    asyncio.run(method.update([task]))
    expected_weights = copy.deepcopy(model.state_dict())
    expected_trajectories = semantics(method.last_records)

    fresh_model, fresh_trainer, fresh_method, fresh_task = setup(
        tmp_path, algorithm=algorithm, seed=999
    )
    fresh_trainer.load_checkpoint(checkpoint, trusted=True)
    assert fresh_method.updates == 4
    asyncio.run(fresh_method.update([fresh_task]))
    assert semantics(fresh_method.last_records) == expected_trajectories
    for key, value in expected_weights.items():
        torch.testing.assert_close(value, fresh_model.state_dict()[key], rtol=0, atol=0)
    for key, value in frozen.items():
        torch.testing.assert_close(value, fresh_method.reference.state_dict()[key], rtol=0, atol=0)


def test_reject_forgery_stale_policy_and_incomplete_checkpoint(tmp_path):
    model, trainer, method, task = setup(tmp_path, group_size=4)
    cohort = asyncio.run(method.rollout([task]))
    data = json.loads(cohort.payload_json)
    data["records"][0]["reward"] = 999.0
    with pytest.raises(ValueError, match="Forged"):
        method.optimize(AgentRolloutBatch(json.dumps(data), cohort.seal))
    with pytest.raises(RuntimeError, match="transaction boundary"):
        method.state_dict()
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="Stale policy"):
        method.optimize(cohort)
    method.discard(cohort)
    assert method.state_dict()["attempts"] == 1 and trainer.steps == 0


def test_reject_rewritten_tool_receipt_and_changed_environment(tmp_path):
    _, trainer, method, task = setup(tmp_path, group_size=16)
    cohort = asyncio.run(method.rollout([task]))
    row = next(row for row in cohort.records if row["successful_reads"])
    receipt = Path(row["receipt_dir"]) / (row["successful_reads"][0] + ".json")
    receipt.write_text('{"forged":"tool said blue"}', encoding="utf-8")
    with pytest.raises(ValueError, match="receipt identity/hash"):
        method.optimize(cohort)
    assert trainer.steps == 0
    method.discard(cohort)
    Path(task.workspace, "note.txt").write_text("red", encoding="utf-8")
    with pytest.raises(ValueError, match="Task file/answer changed"):
        asyncio.run(method.rollout([task]))


def test_consumed_cohort_cannot_be_replayed_and_greedy_is_not_on_policy(tmp_path):
    _, trainer, method, task = setup(tmp_path, group_size=4)
    cohort = asyncio.run(method.rollout([task]))
    method.optimize(cohort)
    with pytest.raises(ValueError, match="consumed"):
        method.optimize(cohort)
    with pytest.raises(ValueError, match="untruncated"):
        NativeAgentRLMethod(
            trainer,
            copy.deepcopy(trainer.model),
            ToolActionTokenizer(),
            work_directory=tmp_path / "invalid",
            reference_tokenizer_fingerprint=method.fingerprint,
            sampling=SamplingConfig(temperature=0),
        )


def test_rehashed_event_log_is_not_live_rollout_provenance(tmp_path):
    from aster.agents.events import canonical_json, digest, read_events

    _, trainer, method, task = setup(tmp_path, group_size=4)
    cohort = asyncio.run(method.rollout([task]))
    path = Path(cohort.records[0]["log_path"])
    events = read_events(path)
    events[0]["payload"]["workspace"] = "forged-workspace"
    previous = "0" * 64
    for event in events:
        event["previous"] = previous
        event["hash"] = digest({key: value for key, value in event.items() if key != "hash"})
        previous = event["hash"]
    path.write_text("".join(canonical_json(event) + "\n" for event in events), encoding="utf-8")
    assert read_events(path)
    with pytest.raises(ValueError, match="log changed"):
        method.optimize(cohort)
    assert trainer.steps == 0


def test_finite_token_budget_failure_is_retained_not_selected_out(tmp_path):
    _, _, method, task = setup(tmp_path, group_size=4)

    trainer = Trainer(copy.deepcopy(method.trainer.model), accumulation_steps=1)
    limited = NativeAgentRLMethod(
        trainer,
        copy.deepcopy(method.reference),
        method.tokenizer,
        work_directory=tmp_path / "limited",
        reference_tokenizer_fingerprint=method.fingerprint,
        group_size=4,
        agent_config=AgentConfig(
            max_steps=3, max_action_tokens=1, max_total_action_tokens=1, max_context_tokens=8
        ),
        sampling=SamplingConfig(eos_token_ids=()),
    )
    cohort = asyncio.run(limited.rollout([task]))
    assert len(cohort.records) == 4 and all(row["reward"] == 0 for row in cohort.records)
    assert all(row["result"]["status"] == "token_budget" for row in cohort.records)
    assert limited.optimize(cohort).updated


def test_collection_refuses_disk_rewrite_before_host_sealing(tmp_path, monkeypatch):
    from aster.agents.agent_rl import _CapturedLog
    from aster.agents.events import canonical_json, digest, read_events

    _, trainer, method, task = setup(tmp_path, group_size=2, accumulation_steps=1)
    original = _CapturedLog.close

    def malicious_close(log):
        original(log)
        events = read_events(log.path)
        events[0]["payload"]["workspace"] = "forged-during-rollout"
        previous = "0" * 64
        for event in events:
            event["previous"] = previous
            event["hash"] = digest({key: value for key, value in event.items() if key != "hash"})
            previous = event["hash"]
        log.path.write_text(
            "".join(canonical_json(event) + "\n" for event in events), encoding="utf-8"
        )

    monkeypatch.setattr(_CapturedLog, "close", malicious_close)
    with pytest.raises(ValueError, match="live host-captured"):
        asyncio.run(method.rollout([task]))
    assert trainer.steps == 0 and method.state_dict()["attempts"] == 1
