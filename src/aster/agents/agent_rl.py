"""On-policy, multi-turn agent learning with verified tool receipts and action-token masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import math
from pathlib import Path
import secrets
import uuid

import torch
from torch import nn

from aster.core import LossTerm, atomic_json, digest_json, file_digest
from aster.data import causal_collate
from aster.inference import InferenceEngine, ModelRunner, SamplingConfig
from aster.methods.policy_gradient import leave_one_out_advantages
from aster.methods.reinforcement import group_relative_advantages
from aster.methods.rollout_distillation import tensor_state_identity
from aster.methods.supervised import sequence_logprobs
from aster.models import build_model
from .events import EventLog, canonical_json, digest, read_events, replay, strict_loads
from .permissions import PermissionBroker, PermissionDenied, Workspace
from .runtime import AgentConfig, AgentLoop, NativeAgentPolicy
from .tools import ToolExecutor, sanitize


@dataclass(frozen=True)
class ReadFileTask:
    """Require a real read of an explicitly scoped file followed by its correct UTF-8 content."""

    id: str
    prompt: str
    workspace: str
    path: str
    sha256: str
    expected_answer: str
    revision: str = "local-read-task-v1"

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in asdict(self).values()):
            raise ValueError("Read task fields must be nonempty strings")
        if (
            len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
            or Path(self.path).is_absolute()
            or ".." in Path(self.path).parts
        ):
            raise ValueError("Read task needs a relative path and exact SHA256")

    @property
    def identity(self):
        return digest_json(asdict(self))

    def verify_file(self):
        workspace = Workspace(self.workspace)
        path = workspace.resolve(self.path)
        if not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("Agent RL fixture must be a bounded regular file")
        data = path.read_bytes()
        if (
            len(data) > 65536
            or hashlib.sha256(data).hexdigest() != self.sha256
            or data.decode("utf-8").strip() != self.expected_answer
        ):
            raise ValueError("Task file/answer changed from the declared environment")
        return workspace


class _ReadTaskExecutor(ToolExecutor):
    def __init__(self, task, log, receipt_dir):
        super().__init__(
            PermissionBroker(task.verify_file()), log, receipt_dir, max_file_bytes=65536
        )
        self.task = task

        self._tools = {"workspace.read": self._tools["workspace.read"]}

    def prepare(self, name, arguments, **context):
        if name != "workspace.read" or arguments != {"path": self.task.path}:
            raise PermissionDenied("This RL environment allows only its declared whole-file read")
        self.task.verify_file()
        return super().prepare(name, arguments, **context)


class _CapturedLog(EventLog):
    def __init__(self, path):
        super().__init__(path)
        self._captured = []

    def append(self, kind, **kwargs):
        event = super().append(kind, **kwargs)
        self._captured.append(canonical_json(event))
        return event

    def captured_events(self):
        return [strict_loads(value) for value in self._captured]


@dataclass(frozen=True)
class AgentRolloutBatch:
    """Authenticate canonical JSON bytes, not mutable dictionary references."""

    payload_json: str
    seal: str

    @property
    def records(self):
        return strict_loads(self.payload_json)["records"]


def collate_agent_trajectories(records, *, pad_token_id, device):
    """Represent each decision as a row while retaining trajectory_index across tool turns.
    Earlier prompts, observations, and actions in context are masked with label -100."""
    examples, traces, indices = [], [], []
    for trajectory, record in enumerate(records):
        if not record["traces"]:
            raise ValueError("Empty trajectory cannot disappear from an RL cohort")
        for trace in record["traces"]:
            prompt, actions = trace["prompt_token_ids"], trace["action_token_ids"]
            if (
                not prompt
                or not actions
                or trace["loss_mask"] != [0] * len(prompt) + [1] * len(actions)
            ):
                raise ValueError("Only actual action tokens may carry policy loss")
            if any(
                len(trace[key]) != len(actions)
                for key in ("raw_model_logprobs", "behavior_logprobs")
            ):
                raise ValueError("Each action needs raw and behavior probability")
            if not all(
                math.isfinite(x)
                for key in ("raw_model_logprobs", "behavior_logprobs")
                for x in trace[key]
            ):
                raise ValueError("Action log probabilities must be finite")
            examples.append(
                {"input_ids": prompt + actions, "labels": [-100] * len(prompt) + actions}
            )
            traces.append(trace)
            indices.append(trajectory)
    batch = {
        key: value.to(device)
        for key, value in causal_collate(examples, pad_token_id=pad_token_id).items()
    }
    old = torch.zeros((len(examples), batch["input_ids"].shape[1] - 1), device=device)
    raw = torch.zeros_like(old)
    for row, trace in enumerate(traces):
        start = len(trace["prompt_token_ids"]) - 1
        end = start + len(trace["action_token_ids"])
        old[row, start:end] = torch.tensor(trace["behavior_logprobs"], device=device)
        raw[row, start:end] = torch.tensor(trace["raw_model_logprobs"], device=device)
    batch.update(
        old_behavior_log_probs=old,
        old_raw_log_probs=raw,
        trajectory_index=torch.tensor(indices, device=device),
        advantages=torch.zeros(len(records), device=device),
    )
    return batch


class AgentPolicyObjective(nn.Module):
    """Use full-trajectory RLOO ratios or trajectory-normalized token GRPO.
    All tool turns share the trajectory advantage; observations never receive action loss."""

    def __init__(self, algorithm="rloo", *, clip_low=0.2, clip_high=0.2, kl_weight=0.0):
        super().__init__()
        if (
            algorithm not in {"rloo", "grpo"}
            or not 0 <= clip_low < 1
            or not math.isfinite(clip_high)
            or clip_high < 0
            or not math.isfinite(kl_weight)
            or kl_weight < 0
        ):
            raise ValueError("Invalid Agent RL objective")
        self.algorithm, self.clip_low, self.clip_high, self.kl_weight = (
            algorithm,
            clip_low,
            clip_high,
            kl_weight,
        )

    def config_dict(self):
        return {
            "type": "agent_trajectory_" + self.algorithm,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "kl_weight": self.kl_weight,
        }

    def forward(self, model, batch):
        logp, valid = sequence_logprobs(model, batch)
        old = batch["old_behavior_log_probs"].detach()
        indices, advantage = batch["trajectory_index"], batch["advantages"].detach()
        count = len(advantage)
        if (
            old.shape != logp.shape
            or indices.shape != (len(logp),)
            or set(indices.tolist()) != set(range(count))
            or not torch.isfinite(advantage).all()
        ):
            raise ValueError("Loss requires complete trajectories and aligned decision rows")
        lengths = logp.new_zeros(count).index_add(0, indices, valid.sum(-1).to(logp))
        if (lengths <= 0).any():
            raise ValueError("Every trajectory needs at least one action")
        difference = (logp - old) * valid
        if self.algorithm == "rloo":
            ratio = logp.new_zeros(count).index_add(0, indices, difference.sum(-1)).exp()
            values = -torch.minimum(
                ratio * advantage, ratio.clamp(1 - self.clip_low, 1 + self.clip_high) * advantage
            )
        else:
            ratio = difference.exp()
            advantages = advantage[indices, None]
            token_loss = -torch.minimum(
                ratio * advantages, ratio.clamp(1 - self.clip_low, 1 + self.clip_high) * advantages
            )
            if self.kl_weight:
                delta = batch["reference_log_probs"].detach() - logp
                token_loss = token_loss + self.kl_weight * (delta.exp() - 1 - delta)
            values = (
                logp.new_zeros(count).index_add(0, indices, (token_loss * valid).sum(-1)) / lengths
            )
        if not torch.isfinite(values).all():
            raise ValueError(
                "Policy ratio/KL overflow; refusing to alter the objective by clamping logs"
            )
        return LossTerm(
            values.sum(), values.new_tensor(count), "trajectory", "agent_" + self.algorithm
        )


def _verify_read_receipts(events, task, receipt_dir):
    """Verify committed tool receipts; model-generated text is not execution evidence."""
    prepared, successful = {}, []
    for event in events:
        if event["kind"] == "tool.prepared":
            body = event["payload"]
            if (
                body["tool"]["name"] != "workspace.read"
                or body["tool"]["effect"] != "read"
                or body["arguments_digest"] != digest({"path": task.path})
            ):
                raise ValueError("Tool trace is outside the pinned read environment")
            prepared[event["item_id"]] = body
        elif event["kind"] == "tool.result_committed":
            body = event["payload"]
            path = Path(body["raw_receipt_path"])
            expected = Path(receipt_dir) / (event["item_id"] + ".json")
            Workspace._reject_links(path)
            if path.absolute() != expected.absolute() or file_digest(path) != body["raw_sha256"]:
                raise ValueError("Tool receipt identity/hash mismatch")
            raw = strict_loads(path.read_text(encoding="utf-8"))
            preparation = prepared.get(event["item_id"])
            if (
                not preparation
                or raw["binding"] != preparation["binding"]
                or raw["call_id"] != event["item_id"]
                or raw["status"] != body["status"]
                or body["model_view"] != sanitize(raw["result"])
            ):
                raise ValueError("Committed tool response differs from its original bound call")
            if raw["status"] == "ok":
                value = raw["result"]
                if (
                    value["path"] != str(Path(task.path))
                    or value["sha256"] != task.sha256
                    or value["content"].strip() != task.expected_answer
                    or value["truncated"]
                ):
                    raise ValueError("Read receipt is not complete evidence for the task file")
                successful.append(event["item_id"])
    return successful


class NativeAgentRLMethod:
    """Consume each verified on-policy cohort once through the shared trainer."""

    def __init__(
        self,
        trainer,
        reference,
        tokenizer,
        *,
        work_directory,
        reference_tokenizer_fingerprint,
        algorithm="rloo",
        group_size=4,
        agent_config=None,
        sampling=None,
        render_messages=None,
        processor_fingerprint=None,
        kl_weight=0.0,
        clip_low=0.2,
        clip_high=0.2,
    ):
        self.config = agent_config or AgentConfig()
        if (
            any(
                type(getattr(self.config, name)) is not int or getattr(self.config, name) <= 0
                for name in (
                    "max_steps",
                    "max_action_tokens",
                    "max_total_action_tokens",
                    "max_context_tokens",
                )
            )
            or type(self.config.seed) is not int
            or not math.isfinite(self.config.timeout_seconds)
            or self.config.timeout_seconds <= 0
        ):
            raise ValueError("Agent RL needs finite explicit horizon/timeout budgets")
        self.sampling = sampling or SamplingConfig(eos_token_ids=(tokenizer.eos_token_id,))
        if (
            self.sampling.temperature != 1
            or self.sampling.top_k
            or self.sampling.top_p != 1
            or self.sampling.repetition_penalty != 1
            or self.sampling.logit_bias
            or self.sampling.min_new_tokens
        ):
            raise ValueError("Agent RL requires untruncated unbiased temperature-one sampling")
        if type(group_size) is not int or group_size < 2:
            raise ValueError("Agent RL needs at least two trajectories per task")
        if trainer.parallel.world.size != 1 or trainer.precision != "fp32":
            raise ValueError("Agent RL controller currently supports single-rank FP32 only")
        if not callable(getattr(tokenizer, "to_dict", None)):
            raise ValueError("RL tokenizer must export its exact token semantics")
        self.fingerprint = digest_json(tokenizer.to_dict())
        if self.fingerprint != reference_tokenizer_fingerprint:
            raise ValueError("Reference policy uses a different tokenizer")
        if render_messages is not None and not processor_fingerprint:
            raise ValueError("Custom message processing needs an explicit semantic fingerprint")
        self.processor = processor_fingerprint or digest_json(
            {"renderer": "aster.agent.canonical_json.v1"}
        )
        self.objective = AgentPolicyObjective(
            algorithm, clip_low=clip_low, clip_high=clip_high, kl_weight=kl_weight
        )

        def dropout_config(value):
            if isinstance(value, dict):
                return any(
                    ("dropout" in key.lower() and isinstance(child, (int, float)) and child != 0)
                    or dropout_config(child)
                    for key, child in value.items()
                )
            return isinstance(value, (list, tuple)) and any(
                dropout_config(child) for child in value
            )

        for model in (trainer.model, reference):
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in model.state_dict().values()
            ):
                raise ValueError("This exact Agent RL snapshot path requires FP32 model storage")
            if any(
                isinstance(module, nn.Dropout) and module.p for module in model.modules()
            ) or dropout_config(model.config.to_dict()):
                raise ValueError("Training/inference policy likelihood requires disabled dropout")
        self.trainer, self.tokenizer, self.render_messages = trainer, tokenizer, render_messages
        self.reference = trainer.add_role("agent_rl_reference", reference, trainable=False)
        self.root = Path(work_directory).absolute()
        Workspace._reject_links(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.group_size, self.algorithm, self.kl_weight = group_size, algorithm, kl_weight
        self.settings = {
            "objective": self.objective.config_dict(),
            "group_size": group_size,
            "agent_config": asdict(self.config),
            "sampling": asdict(self.sampling),
            "tokenizer": self.fingerprint,
            "processor": self.processor,
            "controller_source": file_digest(__file__),
            "runtime_sources": {
                name: file_digest(Path(__file__).with_name(name))
                for name in ("runtime.py", "events.py", "tools.py", "permissions.py")
            },
            "verifier": "aster.whole-file-read-and-exact-answer.v1",
        }

        self.settings = strict_loads(canonical_json(self.settings))
        self.updates, self.attempts, self._busy = 0, 0, False
        self._pending, self._key, self._tasks = None, secrets.token_bytes(32), {}
        self.last_records = ()
        trainer.register_state("native_agent_rl", self)

    def _identity(self):
        weights = self.trainer.export_state_dict(only_rank_zero=False)
        return weights, tensor_state_identity(weights, self.trainer.model.config.to_dict())

    def _seal(self, encoded):
        return hmac.new(self._key, encoded.encode("utf-8"), hashlib.sha256).hexdigest()

    async def rollout(self, tasks):
        if self._busy or self._pending is not None:
            raise RuntimeError("Finish or explicitly discard the outstanding Agent RL cohort")
        tasks = tuple(tasks)
        if (
            not tasks
            or any(not isinstance(task, ReadFileTask) for task in tasks)
            or len({task.id for task in tasks}) != len(tasks)
            or len(tasks) * self.group_size < self.trainer.accumulation_steps
        ):
            raise ValueError("A complete unique-task cohort must fill every accumulation slot")
        for task in tasks:
            task.verify_file()
            if task.id in self._tasks and self._tasks[task.id] != task.identity:
                raise ValueError("A task ID cannot silently change its environment/reward revision")
        if digest_json(self.tokenizer.to_dict()) != self.fingerprint:
            raise ValueError("Tokenizer changed before rollout")
        self._busy, inference = True, None
        try:
            weights, policy_id = self._identity()
            initial_step = self.trainer.steps
            snapshot = build_model(self.trainer.model.config)
            snapshot.load_state_dict(weights, strict=True)
            snapshot.to(self.trainer.device)
            blocks = max(
                64,
                4 * ((self.config.max_context_tokens + self.config.max_action_tokens + 15) // 16),
            )
            runner = ModelRunner(
                snapshot,
                policy_artifact_id=policy_id,
                tokenizer=self.tokenizer,
                block_size=16,
                max_blocks=blocks,
            )
            inference = InferenceEngine(
                runner,
                max_prompt_tokens=self.config.max_context_tokens,
                max_generation_tokens=self.config.max_action_tokens,
                max_batch_tokens=min(256, self.config.max_context_tokens),
                prefill_chunk_size=min(256, self.config.max_context_tokens),
            )
            policy = NativeAgentPolicy(
                inference,
                self.tokenizer,
                render_messages=self.render_messages,
                processor_fingerprint=self.processor,
                sampling_config=self.sampling,
            )
            attempt = self.attempts
            self.attempts += 1
            directory = self.root / (f"rollout-{attempt:08d}-" + uuid.uuid4().hex)
            directory.mkdir()
            records = []
            for group, task in enumerate(tasks):
                self._tasks[task.id] = task.identity
                for sample in range(self.group_size):
                    seed = (
                        self.sampling.seed
                        + self.config.seed
                        + int.from_bytes(
                            hashlib.sha256(
                                canonical_json([attempt, task.id, task.revision, sample]).encode()
                            ).digest()[:7],
                            "big",
                        )
                    )
                    selected = replace(self.config, seed=seed)
                    run_dir = directory / f"{group}-{sample}"
                    run_dir.mkdir()
                    log_path, receipt_dir = run_dir / "events.jsonl", run_dir / "receipts"
                    result, error = None, None
                    with _CapturedLog(log_path) as log:
                        executor = _ReadTaskExecutor(task, log, receipt_dir)
                        loop = AgentLoop(policy, executor, log, config=selected)

                        def verifier(text):
                            task.verify_file()
                            reads = _verify_read_receipts(log.captured_events(), task, receipt_dir)
                            return {
                                "passed": text == task.expected_answer and bool(reads),
                                "rule": self.settings["verifier"],
                            }

                        try:
                            result = await loop.run(task.prompt, verifier=verifier)
                        except Exception as exc:
                            error = type(exc).__name__
                    events = log.captured_events()
                    if read_events(log_path) != events:
                        raise ValueError(
                            "Event log differs from live host-captured tool/model events"
                        )
                    traces = [
                        event["payload"] for event in events if event["kind"] == "model.trace"
                    ]
                    reads = _verify_read_receipts(events, task, receipt_dir)
                    record = {
                        "task": asdict(task),
                        "task_identity": task.identity,
                        "group": group,
                        "sample": sample,
                        "seed": seed,
                        "result": asdict(result) if result else None,
                        "error": error,
                        "traces": traces,
                        "log_path": str(log_path),
                        "log_sha256": file_digest(log_path),
                        "receipt_dir": str(receipt_dir),
                        "successful_reads": reads,
                        "reward": float(
                            result is not None and result.status == "verified" and bool(reads)
                        ),
                    }
                    records.append(record)
                    self.last_records = tuple(records)
            encoded = canonical_json(
                {
                    "schema": 1,
                    "attempt": attempt,
                    "policy_id": policy_id,
                    "policy_step": initial_step,
                    "settings": self.settings,
                    "records": records,
                }
            )
            batch = AgentRolloutBatch(encoded, self._seal(encoded))
            atomic_json(
                directory / "cohort.json", {"payload": strict_loads(encoded), "seal": batch.seal}
            )
            self._pending = hashlib.sha256(encoded.encode()).hexdigest()
            return batch
        finally:
            if inference is not None:
                await inference.close()
            self._busy = False

    def _validate(self, cohort):
        if not isinstance(cohort, AgentRolloutBatch) or not hmac.compare_digest(
            self._seal(cohort.payload_json), cohort.seal
        ):
            raise ValueError("Forged or foreign Agent RL cohort seal")
        if self._pending != hashlib.sha256(cohort.payload_json.encode()).hexdigest():
            raise ValueError("This cohort is not outstanding or was already consumed")
        payload = strict_loads(cohort.payload_json)
        if (
            payload["settings"] != self.settings
            or digest_json(self.tokenizer.to_dict()) != self.fingerprint
        ):
            raise ValueError("Processor/tokenizer/learning configuration changed")
        _, policy_id = self._identity()
        if self.trainer.steps != payload["policy_step"] or policy_id != payload["policy_id"]:
            raise ValueError(
                "Stale policy: rollout must match both current parameters and training step"
            )
        groups = {}
        for record in payload["records"]:
            task = ReadFileTask(**record["task"])
            task.verify_file()
            if (
                task.identity != record["task_identity"]
                or self._tasks.get(task.id) != task.identity
            ):
                raise ValueError("Task environment/reward identity mismatch")
            groups.setdefault(record["group"], []).append((record["sample"], task.identity))
            log_path = Path(record["log_path"])
            Workspace._reject_links(log_path)
            if (
                not log_path.is_relative_to(self.root)
                or file_digest(log_path) != record["log_sha256"]
            ):
                raise ValueError("Agent event log changed after live collection")
            events = read_events(log_path)
            replay(log_path)
            traces = [event["payload"] for event in events if event["kind"] == "model.trace"]
            completed = [event for event in events if event["kind"] == "turn.completed"]
            if (
                record["error"]
                or len(completed) != 1
                or completed[0]["payload"] != record["result"]
                or traces != record["traces"]
                or not traces
            ):
                raise ValueError("Failed/incomplete trajectory cannot be removed or fabricated")
            result = record["result"]
            if result["status"] not in {"verified", "step_budget", "token_budget"}:
                raise ValueError(
                    "Only completed or explicit finite-horizon failure trajectories are trainable"
                )
            reads = _verify_read_receipts(events, task, record["receipt_dir"])
            reward = float(
                result["status"] == "verified"
                and result["text"] == task.expected_answer
                and bool(reads)
            )
            if reads != record["successful_reads"] or reward != record["reward"]:
                raise ValueError(
                    "Trajectory reward differs from independently verified tool evidence"
                )

            positions = [
                index for index, event in enumerate(events) if event["kind"] == "model.trace"
            ]
            for ordinal, (trace, start) in enumerate(zip(traces, positions)):
                expected_sampling = strict_loads(
                    canonical_json(
                        asdict(
                            replace(
                                self.sampling,
                                max_new_tokens=min(
                                    self.config.max_action_tokens,
                                    self.config.max_total_action_tokens
                                    - sum(
                                        len(previous["action_token_ids"])
                                        for previous in traces[:ordinal]
                                    ),
                                ),
                                seed=record["seed"] + ordinal,
                            )
                        )
                    )
                )
                if (
                    trace["policy_artifact_id"] != policy_id
                    or trace["tokenizer_fingerprint"] != self.fingerprint
                    or trace["processor_fingerprint"] != self.processor
                    or trace["sampling_config"] != expected_sampling
                    or trace["stop_reason"] not in {"eos", "length"}
                    or trace["sampling_transform_order"] != list(self.sampling.transform_order)
                ):
                    raise ValueError(
                        "Action provenance/sampling law differs from the frozen policy"
                    )
                end = positions[ordinal + 1] if ordinal + 1 < len(positions) else len(events)
                tool_events = [
                    event for event in events[start + 1 : end] if event["kind"] == "tool.prepared"
                ]
                try:
                    action = AgentLoop._parse_action(
                        self.tokenizer.decode(trace["action_token_ids"])
                    )
                except (ValueError, TypeError):
                    action = None
                if tool_events and (
                    len(tool_events) != 1
                    or action is None
                    or action["type"] != "tool"
                    or action["name"] != "workspace.read"
                    or action["arguments"] != {"path": task.path}
                ):
                    raise ValueError("Forged tool turn: no matching sampled model action")
        if not groups or sorted(groups) != list(range(len(groups))):
            raise ValueError("Cohort must contain its full declared task groups")
        for values in groups.values():
            if (
                sorted(sample for sample, _ in values) != list(range(self.group_size))
                or len({identity for _, identity in values}) != 1
            ):
                raise ValueError("Missing/duplicate trajectory changes the on-policy group")
        return payload

    def optimize(self, cohort):
        if self._busy:
            raise RuntimeError("Agent rollout/update is already active")
        payload = self._validate(cohort)
        records = payload["records"]
        batch = collate_agent_trajectories(
            records, pad_token_id=self.tokenizer.pad_token_id, device=self.trainer.device
        )
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        self.reference.eval()
        try:
            with torch.no_grad():
                current, valid = sequence_logprobs(self.trainer.model, batch)
                reference, _ = sequence_logprobs(self.reference, batch)
        finally:
            self.trainer.model.train(was_training)
        old, raw = batch["old_behavior_log_probs"], batch["old_raw_log_probs"]
        if not torch.allclose(old, raw, atol=2e-5, rtol=2e-5) or not torch.allclose(
            current, raw, atol=5e-5, rtol=5e-5
        ):
            raise ValueError(
                "Live action behavior logp cannot be reproduced by the exact current policy"
            )
        rewards = torch.tensor([record["reward"] for record in records], device=self.trainer.device)
        groups = torch.tensor([record["group"] for record in records], device=self.trainer.device)
        index = batch["trajectory_index"]
        if self.algorithm == "rloo":
            penalty = rewards.new_zeros(len(records)).index_add(
                0, index, ((raw - reference) * valid).sum(-1)
            )
            advantages = leave_one_out_advantages(rewards - self.kl_weight * penalty, groups)
        else:
            advantages = group_relative_advantages(rewards, groups)
        batch.update(advantages=advantages, reference_log_probs=reference)
        microbatches = []
        for slot in range(self.trainer.accumulation_steps):
            chosen = torch.arange(
                slot, len(records), self.trainer.accumulation_steps, device=index.device
            )

            row_mask = torch.isin(index, chosen)
            mapping = {int(old_id): new_id for new_id, old_id in enumerate(chosen.tolist())}
            selected = {key: value[row_mask] for key, value in batch.items() if key != "advantages"}
            selected["trajectory_index"] = torch.tensor(
                [mapping[int(value)] for value in index[row_mask]], device=index.device
            )
            selected["advantages"] = advantages[chosen]
            microbatches.append(selected)

        self._pending, self._busy = None, True
        try:
            result = self.trainer.phase(
                "native_agent_" + self.algorithm,
                objective=self.objective,
                microbatches=microbatches,
            )
            self.updates += int(result.updated)
            self.last_records = tuple(records)
            return result
        finally:
            self._busy = False

    async def update(self, tasks):
        return self.optimize(await self.rollout(tasks))

    def discard(self, cohort):

        if (
            self._busy
            or not isinstance(cohort, AgentRolloutBatch)
            or self._pending != hashlib.sha256(cohort.payload_json.encode()).hexdigest()
        ):
            raise ValueError("No matching outstanding cohort to discard")
        self._pending = None

    def state_dict(self):
        if self._busy or self._pending is not None:
            raise RuntimeError("Checkpoint requires a completed Agent RL transaction boundary")
        return {
            "schema": 1,
            "settings": self.settings,
            "updates": self.updates,
            "attempts": self.attempts,
            "tasks": dict(self._tasks),
            "seal_key": self._key.hex(),
        }

    def load_state_dict(self, state):
        if self._busy or self._pending is not None:
            raise RuntimeError("Finish/discard pending Agent RL work before checkpoint restoration")
        if (
            state.get("schema") != 1
            or state.get("settings") != self.settings
            or any(
                type(state.get(key)) is not int or state[key] < 0 for key in ("updates", "attempts")
            )
            or state["updates"] > state["attempts"]
        ):
            raise ValueError("Agent RL checkpoint configuration/counters differ")
        key = bytes.fromhex(state["seal_key"])
        if len(key) != 32 or not isinstance(state["tasks"], dict):
            raise ValueError("Invalid Agent RL identity state")
        self.updates, self.attempts = state["updates"], state["attempts"]
        self._key, self._tasks = key, dict(state["tasks"])
        self.last_records = ()
