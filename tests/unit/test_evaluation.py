import math
import numpy as np
import pytest
import torch
from aster.evaluation import *


def protocol():
    return ComparisonProtocol(
        "test",
        "dataset-sha256",
        "local-accuracy",
        "1",
        {"seed": 3, "split": "test"},
        ("a", "b", "c", "d"),
        "accuracy",
    )


def test_evaluation_missing_samples_denominator_and_report(tmp_path):
    run = EvaluationRun(protocol(), "candidate-one", environment={"device": "cpu"})
    run.add(EvaluationRecord("a", "ok", {"accuracy": 1.0}))
    run.add(EvaluationRecord("b", "skipped", error="missing_asset"))
    run.finalize()
    assert run.summary()["mean"] == 0.25 and run.summary()["denominator"] == 4
    report = run.save(tmp_path / "one")
    assert EvaluationRun.load(report).summary() == run.summary()
    with pytest.raises(ValueError):
        run.add(EvaluationRecord("a", "ok", {"accuracy": 1.0}))
    with pytest.raises(ValueError):
        EvaluationRecord("x", "ok", {"accuracy": float("nan")})


def test_paired_gate_compares_different_candidates_not_protocols(tmp_path):
    baseline = EvaluationRun(protocol(), "teacher", environment={"device": "cpu"})
    candidate = EvaluationRun(protocol(), "student", environment={"device": "cpu"})
    for key in protocol().expected_ids:
        baseline.add(EvaluationRecord(key, "ok", {"accuracy": 0.5}))
        candidate.add(EvaluationRecord(key, "ok", {"accuracy": 0.6}))
    result = paired_bootstrap(baseline, candidate)
    assert result["low"] == pytest.approx(0.1) and result["high"] == pytest.approx(0.1)
    assert quality_gate(baseline.save(tmp_path / "teacher"), candidate.save(tmp_path / "student"))[
        "passed"
    ]


def test_public_metric_math():
    assert perplexity(2 * math.log(2), 2) == pytest.approx(2)
    assert pass_at_k(10, 2, 1) == pytest.approx(0.2)
    assert pass_at_k(10, 2, 9) == 1
    assert word_error_rate("a b c", "a b") == pytest.approx(1 / 3)
    assert exact_match("The CAT!", ["cat"], normalize=True) == 1
    assert interquartile_mean([0.0, 1.0, 2.0, 100.0]) == 1.5
    features = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [-1.0, 3.0]])
    assert frechet_distance(features, features) < 1e-10
    assert frechet_distance(features, features + 2) == pytest.approx(8.0)
    assert math.isfinite(kernel_inception_distance(features, features))
    x = torch.rand(2, 3, 12, 12)
    torch.testing.assert_close(ssim(x, x), torch.ones(2, dtype=torch.float64))
