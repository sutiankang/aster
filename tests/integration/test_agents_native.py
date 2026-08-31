import asyncio
import json

import torch
from torch import nn

from aster.core import TokenOutput, digest_json
from aster.data import ByteTokenizer
from aster.inference import ModelRunner, InferenceEngine
from aster.agents import (
    EventLog,
    PermissionBroker,
    ToolExecutor,
    NativeAgentPolicy,
    AgentLoop,
    AgentConfig,
    replay,
)
from aster.agents import AgentPlanExecutor, PlanNode, MemoryStore


class FiniteStateControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.tokenizer = ByteTokenizer()
        self.marker = self.tokenizer.encode("\n<END>\n", add_special_tokens=False)

    def forward(
        self, input_ids, *, state=None, use_cache=False, attention_mask=None, position_ids=None
    ):
        prefix = state[0][0][:, 0, :, 0].long() if state is not None else input_ids[:, :0]
        full = torch.cat((prefix, input_ids), dim=1)
        logits = torch.full((*input_ids.shape, 259), -100.0, device=input_ids.device)
        for batch, row in enumerate(full.tolist()):
            boundary = None
            for index in range(len(row) - len(self.marker) + 1):
                if row[index : index + len(self.marker)] == self.marker:
                    boundary = index + len(self.marker)
                    break
            token = 2
            if boundary is not None:
                messages = json.loads(self.tokenizer.decode(row[: boundary - len(self.marker)]))
                has_result = any(
                    message.get("role") == "tool"
                    and isinstance(message.get("content"), dict)
                    and message["content"].get("trust") == "untrusted_tool_data"
                    for message in messages
                )
                action = (
                    {"type": "final", "text": "已读取"}
                    if has_result
                    else {
                        "type": "tool",
                        "name": "workspace.read",
                        "arguments": {"path": "note.txt"},
                    }
                )
                response = self.tokenizer.encode(
                    json.dumps(action, ensure_ascii=False, separators=(",", ":")),
                    add_special_tokens=False,
                ) + [2]
                index = len(row) - boundary
                token = response[index] if index < len(response) else 2
            logits[batch, -1, token] = 100.0
        cached = full[:, None, :, None].float()
        return TokenOutput(logits, ((cached, cached.clone()),) if use_cache else None)


def test_agent_native_tool_loop_exact_trajectory_and_readonly_replay(tmp_path):
    torch.set_num_threads(1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("你好, local-agent", encoding="utf-8")
    log_path = tmp_path / "events.jsonl"

    async def exercise():
        tokenizer = ByteTokenizer()
        runner = ModelRunner(
            FiniteStateControlModel(),
            policy_artifact_id="protocol-control-fixture",
            tokenizer=tokenizer,
            block_size=32,
            max_blocks=1024,
        )
        engine = InferenceEngine(runner, max_batch_tokens=4096, prefill_chunk_size=4096)
        policy = NativeAgentPolicy(
            engine,
            tokenizer,
            render_messages=lambda value: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n<END>\n"
            ),
        )
        with EventLog(log_path) as log:
            broker = PermissionBroker(workspace)
            executor = ToolExecutor(broker, log, tmp_path / "receipts")
            agent = AgentLoop(
                policy,
                executor,
                log,
                config=AgentConfig(max_steps=3, max_action_tokens=128, max_context_tokens=16000),
            )
            result = await agent.run(
                "读取note.txt后回答", verifier=lambda text: {"passed": text == "已读取"}
            )
            assert result.status == "verified" and result.text == "已读取"
            assert result.steps == 2 and len(result.tool_call_ids) == 1
        await engine.close()
        recovered = replay(log_path)
        assert recovered.items[result.tool_call_ids[0]]["status"] == "result_committed"
        assert len(recovered.model_traces) == 2
        for trace in recovered.model_traces:
            prompts, actions = trace["prompt_token_ids"], trace["action_token_ids"]
            assert trace["loss_mask"] == [0] * len(prompts) + [1] * len(actions)
            assert len(trace["behavior_logprobs"]) == len(actions)
            assert trace["tokenizer_fingerprint"] == digest_json(tokenizer.to_dict())
            assert trace["sampling_config"]["temperature"] == 0.0
            assert trace["sampling_config"]["max_new_tokens"] == 128
            assert trace["sampling_config"]["seed"] in {0, 1}
            assert trace["sampling_config"]["eos_token_ids"] == [tokenizer.eos_token_id]
        assert runner.input_tokens_computed > 0 and runner.pool.used_blocks == 0

    asyncio.run(exercise())


def test_bounded_dependency_plan_runs_real_child_agent_loops(tmp_path):
    torch.set_num_threads(1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("bounded planning evidence", encoding="utf-8")

    async def exercise():
        tokenizer = ByteTokenizer()
        runner = ModelRunner(
            FiniteStateControlModel(),
            policy_artifact_id="plan-control-fixture",
            tokenizer=tokenizer,
            block_size=32,
            max_blocks=1024,
        )
        engine = InferenceEngine(runner, max_batch_tokens=4096, prefill_chunk_size=4096)
        policy = NativeAgentPolicy(
            engine,
            tokenizer,
            render_messages=lambda value: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n<END>\n"
            ),
        )
        with EventLog(tmp_path / "events.jsonl") as log:
            memory = MemoryStore(log)

            def factory(node):
                executor = ToolExecutor(
                    PermissionBroker(workspace), log, tmp_path / ("receipts-" + node.id)
                )
                return AgentLoop(
                    policy,
                    executor,
                    log,
                    memory_store=memory,
                    config=AgentConfig(
                        max_steps=3,
                        max_action_tokens=128,
                        max_total_action_tokens=256,
                        max_context_tokens=16000,
                    ),
                )

            planner = AgentPlanExecutor(
                factory, workspace=workspace, max_parallel=2, max_total_action_tokens=512
            )
            result = await planner.run(
                [PlanNode("verify", "再次读取", ("read",)), PlanNode("read", "先读取文件")],
                verifiers={
                    key: (lambda text: {"passed": text == "已读取"}) for key in ("read", "verify")
                },
            )
            assert all(value["status"] == "ok" for value in result.values())
            assert len(memory._items) == 2
        await engine.close()

    asyncio.run(exercise())


def test_agent_evaluation_runs_native_model_tools_and_independent_verifier(tmp_path):
    from dataclasses import asdict
    import time
    from aster.evaluation import ComparisonProtocol
    from aster.evaluation.suites import (
        AgentCase,
        EvaluationGrant,
        evaluate_agents,
        workspace_fingerprint,
    )

    torch.set_num_threads(1)
    workspace = tmp_path / "task"
    workspace.mkdir()
    (workspace / "note.txt").write_text("evaluation protocol fixture", encoding="utf-8")
    case = AgentCase(
        "read", "读取note.txt", 17, workspace_fingerprint(workspace), "file-read-verifier-v1"
    )
    protocol = ComparisonProtocol(
        "agent-read-fixture",
        "task-fixture-v1",
        "native_agent_loop",
        "1",
        {"cases": [asdict(case)]},
        (case.id,),
        "resolved",
    )

    async def exercise():
        tokenizer = ByteTokenizer()
        engine = InferenceEngine(
            ModelRunner(
                FiniteStateControlModel(),
                policy_artifact_id="native-eval-fixture",
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
            render_messages=lambda value: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n<END>\n"
            ),
        )
        with EventLog(tmp_path / "events.jsonl") as log:

            def factory(selected):
                return AgentLoop(
                    policy,
                    ToolExecutor(PermissionBroker(workspace), log, tmp_path / "receipts"),
                    log,
                    config=AgentConfig(
                        seed=selected.seed,
                        max_steps=3,
                        max_action_tokens=128,
                        max_context_tokens=16000,
                    ),
                )

            def verifier_factory(selected):
                def verify(text):
                    return {
                        "passed": text == "已读取"
                        and (workspace / "note.txt").read_text(encoding="utf-8")
                        == "evaluation protocol fixture"
                    }

                verify.verifier_id = selected.verifier_id
                return verify

            result = await evaluate_agents(
                protocol,
                "native-eval-fixture",
                [case],
                agent_factory=factory,
                verifier_factory=verifier_factory,
                grant=EvaluationGrant(protocol.id, ("agent",), time.monotonic() + 30),
                environment={"kind": "native-protocol-fixture-not-public-benchmark"},
            )
            assert (
                result.scores().tolist() == [1.0]
                and result.records["read"].details["tool_call_ids"]
            )
        await engine.close()

    asyncio.run(exercise())
