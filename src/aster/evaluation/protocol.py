"""Predeclared paired evaluation protocols and report-based candidate promotion."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import numpy as np
from ..core import digest_json, atomic_json, read_json


@dataclass(frozen=True)
class ComparisonProtocol:
    task: str
    dataset_fingerprint: str
    evaluator: str
    evaluator_version: str
    controls: dict
    expected_ids: tuple[str, ...]
    metric: str
    higher_is_better: bool = True
    failure_score: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "expected_ids", tuple(self.expected_ids))
        if (
            not all(
                (
                    self.task,
                    self.dataset_fingerprint,
                    self.evaluator,
                    self.evaluator_version,
                    self.metric,
                )
            )
            or not self.expected_ids
            or len(set(self.expected_ids)) != len(self.expected_ids)
        ):
            raise ValueError("Evaluation requires complete task/data/version/sample identity")
        if not math.isfinite(self.failure_score):
            raise ValueError("Failure score must be explicitly finite")
        if any(
            key in self.controls for key in ("candidate_artifact_id", "candidate", "model_weights")
        ):
            raise ValueError("Candidate belongs to EvaluationRun, not comparison controls")
        digest_json(self.to_dict())

    def to_dict(self):
        return asdict(self)

    @property
    def id(self):
        return digest_json(self.to_dict())


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.sample_id or self.status not in {"ok", "error", "timeout", "skipped"}:
            raise ValueError("Invalid evaluation record")
        if any(not math.isfinite(v) for v in self.metrics.values()):
            raise ValueError("NaN/Inf metrics must be reported as errors, not averaged away")
        if self.status != "ok" and not self.error:
            raise ValueError("Unsuccessful sample requires an explanation")


class EvaluationRun:
    def __init__(self, protocol, candidate_artifact_id, *, environment, transforms=()):
        if not candidate_artifact_id or not environment:
            raise ValueError("Run needs candidate and actual environment")
        self.protocol, self.candidate_artifact_id, self.environment = (
            protocol,
            candidate_artifact_id,
            environment,
        )
        self.transforms, self.records = tuple(transforms), {}
        self.protocol_id = protocol.id

    def add(self, record):
        if self.protocol.id != self.protocol_id:
            raise ValueError("Protocol mutated during run")
        if record.sample_id not in self.protocol.expected_ids or record.sample_id in self.records:
            raise ValueError("Unexpected or duplicate evaluated sample")
        if record.status == "ok" and self.protocol.metric not in record.metrics:
            raise ValueError("Successful record lacks the primary metric")
        self.records[record.sample_id] = record

    def finalize(self):
        for sample_id in self.protocol.expected_ids:
            if sample_id not in self.records:
                self.add(EvaluationRecord(sample_id, "error", error="missing_result"))
        return self

    def scores(self):
        if self.protocol.id != self.protocol_id:
            raise ValueError("Protocol mutated during run")
        if len(self.records) != len(self.protocol.expected_ids):
            raise ValueError("Run incomplete; finalize explicitly accounts for missing results")
        scores = np.asarray(
            [
                self.records[key].metrics[self.protocol.metric]
                if self.records[key].status == "ok"
                else self.protocol.failure_score
                for key in self.protocol.expected_ids
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scores).all():
            raise ValueError("Evaluation metrics mutated to NaN/Inf")
        return scores

    def summary(self):
        scores = self.scores()
        statuses = {
            name: sum(record.status == name for record in self.records.values())
            for name in ("ok", "error", "timeout", "skipped")
        }
        return {
            "protocol_id": self.protocol.id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "metric": self.protocol.metric,
            "mean": float(scores.mean()),
            "denominator": len(scores),
            "statuses": statuses,
        }

    def save(self, directory):
        self.scores()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if (target / "report.json").exists():
            raise FileExistsError("Evaluation report is append-free; choose a new run directory")
        payload = {
            "schema_version": 1,
            "protocol": self.protocol.to_dict(),
            "candidate_artifact_id": self.candidate_artifact_id,
            "environment": self.environment,
            "transforms": self.transforms,
            "records": [asdict(self.records[key]) for key in self.protocol.expected_ids],
            "summary": self.summary(),
        }
        atomic_json(target / "report.json", {"payload": payload, "sha256": digest_json(payload)})
        return target / "report.json"

    @classmethod
    def load(cls, path):
        envelope = read_json(path)
        payload = envelope["payload"]
        if digest_json(payload) != envelope["sha256"] or payload["schema_version"] != 1:
            raise ValueError("Invalid evaluation evidence checksum/schema")
        result = cls(
            ComparisonProtocol(**payload["protocol"]),
            payload["candidate_artifact_id"],
            environment=payload["environment"],
            transforms=payload["transforms"],
        )
        for record in payload["records"]:
            result.add(EvaluationRecord(**record))
        if result.summary() != payload["summary"]:
            raise ValueError("Evaluation summary differs from underlying raw records")
        return result


def paired_bootstrap(baseline, candidate, *, repetitions=2000, confidence=0.95, seed=0):
    if baseline.protocol.id != candidate.protocol.id:
        raise ValueError("Evaluation comparison protocol differs")
    if repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    difference = candidate.scores() - baseline.scores()
    if not baseline.protocol.higher_is_better:
        difference = -difference
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        estimates[i] = difference[rng.integers(0, len(difference), len(difference))].mean()
    tail = (1 - confidence) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return {
        "improvement": float(difference.mean()),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "paired_samples": len(difference),
        "seed": seed,
    }


def quality_gate(
    baseline_report, candidate_report, *, max_regression=0.0, max_failure_rate=0.0, confidence=0.95
):
    """Recompute acceptance from actual reports rather than trusting a supplied passed flag."""
    if max_regression < 0 or not 0 <= max_failure_rate <= 1:
        raise ValueError("Invalid gate thresholds")
    baseline, candidate = EvaluationRun.load(baseline_report), EvaluationRun.load(candidate_report)
    comparison = paired_bootstrap(baseline, candidate, confidence=confidence)
    failed = sum(record.status != "ok" for record in candidate.records.values()) / len(
        candidate.records
    )
    passed = comparison["low"] >= -max_regression and failed <= max_failure_rate
    return {
        "passed": passed,
        "comparison": comparison,
        "failure_rate": failed,
        "max_regression": max_regression,
        "max_failure_rate": max_failure_rate,
    }
