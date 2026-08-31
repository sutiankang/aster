"""Complete episode and agent-task evaluation with independently determined success."""

from __future__ import annotations
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch

from ..core import digest_json, atomic_json
from .protocol import ComparisonProtocol, EvaluationRecord, EvaluationRun


@dataclass(frozen=True)
class EvaluationGrant:
    """Trusted-host permission for the current evaluation protocol, not model-generated approval."""

    protocol_id: str
    effects: tuple[str, ...]
    expires_at: float

    def require(self, protocol, effect):
        if (
            self.protocol_id != protocol.id
            or effect not in self.effects
            or not math.isfinite(self.expires_at)
            or time.monotonic() >= self.expires_at
        ):
            raise PermissionError(
                "Evaluation execution is not currently authorized for this protocol/effect"
            )


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Observation/action must have finite numeric values")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Evaluation data cannot be converted to strict JSON without losing semantics")


@dataclass(frozen=True)
class EpisodeCase:
    id: str
    task_id: str
    seed: int
    max_steps: int
    initial_state_id: str = "reset_seed"
    options: dict | None = None

    def __post_init__(self):
        if (
            not self.id
            or not self.task_id
            or not self.initial_state_id
            or type(self.seed) is not int
            or type(self.max_steps) is not int
            or self.max_steps < 1
        ):
            raise ValueError("Episode identity/seed/horizon must be explicit")
        digest_json(asdict(self))


def episode_protocol(
    cases,
    *,
    dataset_fingerprint,
    simulator,
    simulator_version,
    dataset_revision,
    split,
    action_spec,
    success_rule_id,
    metric="success",
    failure_score=0.0,
    controls=None,
):

    if (
        metric not in {"success", "episode_return"}
        or not split
        or not success_rule_id
        or dataset_revision in {"", "main", "master", "latest"}
    ):
        raise ValueError(
            "Fixed split/revision/success rule and supported episode metric are required"
        )
    fixed = {
        "dataset_revision": dataset_revision,
        "split": split,
        "episodes": [asdict(case) for case in cases],
        "action_spec": action_spec,
        "success_rule_id": success_rule_id,
        "success_aggregation": "any_post_step",
        "timeout_semantics": "cooperative_between_steps",
    }
    if controls and set(controls) & fixed.keys():
        raise ValueError("Extra controls cannot replace episode identity")
    fixed.update(controls or {})
    return ComparisonProtocol(
        "control_episodes",
        dataset_fingerprint,
        simulator,
        simulator_version,
        fixed,
        tuple(case.id for case in cases),
        metric,
        True,
        failure_score,
    )


def evaluate_episodes(
    protocol,
    candidate_artifact_id,
    cases,
    *,
    environment_factory,
    policy_factory,
    success,
    grant,
    environment,
    output_directory=None,
    episode_timeout_seconds=300.0,
):

    grant.require(protocol, "environment")
    if tuple(case.id for case in cases) != protocol.expected_ids or protocol.controls.get(
        "episodes"
    ) != [asdict(case) for case in cases]:
        raise ValueError("Episode manifest differs from the frozen comparison protocol")
    if not math.isfinite(episode_timeout_seconds) or episode_timeout_seconds <= 0:
        raise ValueError("Episode timeout must be positive and finite")
    run = EvaluationRun(protocol, candidate_artifact_id, environment=environment)
    directory = Path(output_directory) if output_directory is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    for case in cases:
        env, trace, record = None, [], None
        started, total_reward, reached = time.monotonic(), 0.0, False
        try:
            grant.require(protocol, "environment")
            env = environment_factory(case)
            policy = policy_factory(case)
            if getattr(policy, "policy_artifact_id", None) != candidate_artifact_id:
                raise ValueError("Actual control policy differs from evaluated artifact")
            observation, info = env.reset(seed=case.seed, options=case.options)
            if not isinstance(info, dict):
                raise ValueError("Environment reset must return an info mapping")
            observation_hash = digest_json(_json_value(observation))
            for step in range(case.max_steps):
                if time.monotonic() - started >= episode_timeout_seconds:
                    raise TimeoutError("episode_deadline_between_steps")
                action = policy(observation, info)
                encoded_action = _json_value(action)
                outcome = env.step(action)
                if not isinstance(outcome, tuple) or len(outcome) != 5:
                    raise ValueError(
                        "Episode environment must use terminated/truncated five-tuple semantics"
                    )
                observation, reward, terminated, truncated, info = outcome
                if (
                    type(terminated) not in {bool, np.bool_}
                    or type(truncated) not in {bool, np.bool_}
                    or not isinstance(info, dict)
                ):
                    raise ValueError("Environment termination/info schema mismatch")
                reward = float(reward)
                if not math.isfinite(reward):
                    raise ValueError("Non-finite environment reward")
                total_reward += reward
                accepted = success(observation, info)
                if type(accepted) not in {bool, np.bool_}:
                    raise ValueError(
                        "Success evaluator must return an explicit boolean, not model self-judgment"
                    )
                reached = reached or bool(accepted)
                next_hash = digest_json(_json_value(observation))
                trace.append(
                    {
                        "step": step,
                        "observation_hash": observation_hash,
                        "action": encoded_action,
                        "reward": reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(accepted),
                        "next_observation_hash": next_hash,
                    }
                )
                observation_hash = next_hash
                if terminated or truncated:
                    break
            if time.monotonic() - started >= episode_timeout_seconds:
                raise TimeoutError("episode_deadline_between_steps")
            record = EvaluationRecord(
                case.id,
                "ok",
                {"success": float(reached), "episode_return": total_reward},
                details={
                    "steps": len(trace),
                    "elapsed_seconds": time.monotonic() - started,
                    "horizon_exhausted": len(trace) == case.max_steps
                    and not (trace[-1]["terminated"] or trace[-1]["truncated"]),
                    "trajectory_sha256": digest_json(trace),
                },
            )
        except TimeoutError:
            record = EvaluationRecord(
                case.id, "timeout", error="episode_timeout", details={"completed_steps": len(trace)}
            )
        except Exception as error:
            record = EvaluationRecord(
                case.id,
                "error",
                error=type(error).__name__,
                details={"completed_steps": len(trace)},
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    record = EvaluationRecord(case.id, "error", error="environment_close_failed")
        run.add(record)
        if directory is not None:
            atomic_json(
                directory / (hashlib.sha256(case.id.encode()).hexdigest() + ".trajectory.json"),
                {"case": asdict(case), "steps": trace},
            )
    run.finalize()
    if directory is not None:
        run.save(directory)
    return run


@dataclass(frozen=True)
class AgentCase:
    id: str
    prompt: str
    seed: int
    workspace_fingerprint: str
    verifier_id: str

    def __post_init__(self):
        if (
            not all((self.id, self.prompt, self.workspace_fingerprint, self.verifier_id))
            or type(self.seed) is not int
        ):
            raise ValueError("Agent case needs fixed prompt/seed/workspace/verifier identity")


def workspace_fingerprint(root):

    from ..agents.permissions import Workspace

    workspace = Workspace(root)
    files = {}
    for candidate in sorted(workspace.root.rglob("*")):
        workspace._reject_links(candidate)
        if candidate.is_file():
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[candidate.relative_to(workspace.root).as_posix()] = digest.hexdigest()
    return digest_json(files)


async def evaluate_agents(
    protocol,
    candidate_artifact_id,
    cases,
    *,
    agent_factory,
    verifier_factory,
    grant,
    environment,
    approval_handler=None,
    output_directory=None,
):

    from ..agents.runtime import AgentLoop

    grant.require(protocol, "agent")
    if tuple(case.id for case in cases) != protocol.expected_ids or protocol.metric != "resolved":
        raise ValueError("Agent cases must cover the exact frozen resolved-score sample set")
    if protocol.controls.get("cases") != [asdict(case) for case in cases]:
        raise ValueError("Agent task/seed/workspace/verifier manifest differs from protocol")
    run = EvaluationRun(protocol, candidate_artifact_id, environment=environment)
    for case in cases:
        agent = None
        try:
            grant.require(protocol, "agent")
            agent = agent_factory(case)
            if not isinstance(agent, AgentLoop) or agent.config.seed != case.seed:
                raise ValueError("Agent factory must return a real loop with the fixed seed")
            if (
                workspace_fingerprint(agent.executor.broker.workspace.root)
                != case.workspace_fingerprint
            ):
                raise ValueError("Agent workspace differs from the fixed benchmark snapshot")
            live_policy = getattr(
                getattr(getattr(agent.policy, "engine", None), "runner", None),
                "policy_artifact_id",
                None,
            )
            if live_policy != candidate_artifact_id:
                raise ValueError(
                    "Actual Agent inference policy differs from the evaluated artifact"
                )
            verifier = verifier_factory(case)
            if not callable(verifier) or getattr(verifier, "verifier_id", None) != case.verifier_id:
                raise ValueError("An independent task verifier is mandatory")
            if isinstance(verifier, SandboxTestVerifier):
                grant.require(protocol, "untrusted_code")
            result = await agent.run(
                case.prompt, thread_id=case.id, approval_handler=approval_handler, verifier=verifier
            )
            if result.status not in {"verified", "completed_unverified", "verification_failed"}:
                record = EvaluationRecord(
                    case.id,
                    "error",
                    error="agent_" + result.status,
                    details={"action_tokens": result.action_tokens, "steps": result.steps},
                )
            else:
                record = EvaluationRecord(
                    case.id,
                    "ok",
                    {"resolved": float(result.status == "verified")},
                    details={
                        "action_tokens": result.action_tokens,
                        "steps": result.steps,
                        "tool_call_ids": list(result.tool_call_ids),
                        "trace_sequences": list(result.trace_sequences),
                    },
                )
        except asyncio.CancelledError:
            run.add(EvaluationRecord(case.id, "error", error="evaluation_cancelled"))
            run.finalize()
            if output_directory is not None:
                run.save(output_directory)
            raise
        except Exception as error:
            record = EvaluationRecord(case.id, "error", error=type(error).__name__)
        run.add(record)
    run.finalize()
    if output_directory is not None:
        run.save(output_directory)
    return run


class SandboxTestVerifier:
    """Bind test-file hashes before running inside the Linux isolation backend;
    refuse execution when isolation is unavailable."""

    def __init__(
        self, backend, *, argv, test_file_hashes, timeout_seconds=60.0, expected_stdout_sha256=None
    ):
        from ..agents.sandbox import BubblewrapSandbox

        if (
            not isinstance(backend, BubblewrapSandbox)
            or backend.allow_network
            or backend.allow_workspace_write
        ):
            raise PermissionError(
                "Coding verification requires actual read-only, no-network OS isolation"
            )
        if not test_file_hashes or not argv:
            raise ValueError("Verifier needs immutable tests and explicit command argv")
        self.backend, self.argv, self.test_file_hashes = (
            backend,
            tuple(argv),
            dict(test_file_hashes),
        )
        self.timeout_seconds, self.expected_stdout_sha256 = timeout_seconds, expected_stdout_sha256
        self.receipts = []
        self.verifier_id = digest_json(
            {
                "kind": "sandbox_test",
                "argv": self.argv,
                "tests": self.test_file_hashes,
                "timeout_seconds": timeout_seconds,
                "stdout_sha256": expected_stdout_sha256,
                "isolation": backend.isolation_kind,
            }
        )

    def _verify_files(self):
        for name, expected in self.test_file_hashes.items():
            path = self.backend.workspace.resolve(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen benchmark tests were modified")

    def __call__(self, final_text):
        self._verify_files()
        receipt = self.backend.run(
            argv=list(self.argv),
            cwd=str(self.backend.workspace.root),
            timeout_seconds=self.timeout_seconds,
            environment={},
        )
        self._verify_files()
        self.receipts.append(receipt)
        passed = (
            receipt["exit_code"] == 0
            and receipt["stop_reason"] == "exited"
            and (
                self.expected_stdout_sha256 is None
                or hashlib.sha256(receipt["stdout"].encode()).hexdigest()
                == self.expected_stdout_sha256
            )
        )
        return {
            "passed": passed,
            "receipt_sha256": digest_json(receipt),
            "verifier_id": self.verifier_id,
        }
