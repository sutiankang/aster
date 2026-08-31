from dataclasses import asdict
import time
import numpy as np
import pytest

from aster.evaluation import ComparisonProtocol
from aster.evaluation.suites import (
    EpisodeCase,
    EvaluationGrant,
    episode_protocol,
    evaluate_episodes,
    SandboxTestVerifier,
)
from aster.evaluation.adapters import OfficialModulePin, normalize_swebench_report


class PointEnvironment:
    def __init__(self, fail=False):
        self.fail, self.closed = fail, False

    def reset(self, *, seed, options=None):
        self.x = 0
        return np.array([self.x], dtype=np.float32), {}

    def step(self, action):
        self.x += float(action[0])
        return (
            np.array([self.x]),
            float("nan") if self.fail else 1.0,
            self.x >= 2,
            False,
            {"success": self.x >= 2},
        )

    def close(self):
        self.closed = True


def test_episode_real_steps_complete_denominator_failure_and_trajectory(tmp_path):
    cases = [
        EpisodeCase("ok", "point", 1, 3),
        EpisodeCase("failed", "point", 2, 3),
        EpisodeCase("unfinished", "point", 3, 1),
    ]
    protocol = episode_protocol(
        cases,
        dataset_fingerprint="fixture-data",
        simulator="point-fixture",
        simulator_version="1",
        dataset_revision="fixture-v1",
        split="test",
        action_spec={"units": ["m"], "frame": "world"},
        success_rule_id="x>=2",
    )
    grant = EvaluationGrant(protocol.id, ("environment",), time.monotonic() + 30)
    environments = []

    def factory(case):
        env = PointEnvironment(fail=case.id == "failed")
        environments.append(env)
        return env

    def policy(case):
        def control(obs, info):
            return np.array([1.0])

        control.policy_artifact_id = "native-controller"
        return control

    result = evaluate_episodes(
        protocol,
        "native-controller",
        cases,
        environment_factory=factory,
        policy_factory=policy,
        success=lambda obs, info: info["success"],
        grant=grant,
        environment={"kind": "protocol-fixture"},
        output_directory=tmp_path / "report",
    )
    assert result.scores().tolist() == [1.0, 0.0, 0.0]
    assert result.summary()["denominator"] == 3 and result.summary()["statuses"]["error"] == 1
    assert all(env.closed for env in environments)
    assert len(list((tmp_path / "report").glob("*.trajectory.json"))) == 3
    assert result.records["unfinished"].details["horizon_exhausted"]


def test_episode_authorization_and_manifest_fail_before_environment_construction():
    case = EpisodeCase("0", "point", 1, 2)
    protocol = episode_protocol(
        [case],
        dataset_fingerprint="fixture",
        simulator="point",
        simulator_version="1",
        dataset_revision="revision-1",
        split="test",
        action_spec={},
        success_rule_id="test",
    )
    with pytest.raises(PermissionError):
        evaluate_episodes(
            protocol,
            "model",
            [case],
            environment_factory=lambda _: pytest.fail("not authorized"),
            policy_factory=None,
            success=None,
            grant=EvaluationGrant(protocol.id, (), time.monotonic() + 1),
            environment={"fixture": True},
        )
    with pytest.raises(ValueError, match="manifest"):
        evaluate_episodes(
            protocol,
            "model",
            [EpisodeCase("0", "point", 8, 2)],
            environment_factory=None,
            policy_factory=None,
            success=None,
            grant=EvaluationGrant(protocol.id, ("environment",), time.monotonic() + 1),
            environment={"fixture": True},
        )
    with pytest.raises(PermissionError):
        SandboxTestVerifier(
            object(), argv=["python", "untrusted.py"], test_file_hashes={"test.py": "0" * 64}
        )


def test_official_pin_never_imports_before_authorization():
    protocol = ComparisonProtocol("vlm", "data", "lmms", "a" * 40, {}, ("x",), "acc")
    pin = OfficialModulePin("untrusted.module", "does-not-exist", "0" * 64, "none", "0", "a" * 40)
    with pytest.raises(PermissionError):
        pin.load(protocol, EvaluationGrant(protocol.id, (), time.monotonic() + 1))


def test_swebench_report_fixture_missing_errors_not_dropped_or_aggregate_copied():
    protocol = ComparisonProtocol(
        "coding", "data", "swebench", "revision", {}, ("a", "b", "c", "d"), "resolved"
    )
    result = normalize_swebench_report(
        protocol,
        "policy",
        {"resolved_ids": ["a"], "error_ids": ["b"], "empty_patch_ids": ["c"]},
        environment={"kind": "report-parser-fixture-not-benchmark"},
    )
    assert result.scores().tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result.records["d"].error == "missing_official_result"
    with pytest.raises(ValueError, match="contradictory"):
        normalize_swebench_report(
            protocol,
            "policy",
            {"resolved_ids": ["a"], "error_ids": ["a"]},
            environment={"fixture": True},
        )
