import json
import sys
import types
import pytest
from aster.evaluation.language import evaluate_official_language


def fake_harness(monkeypatch):
    calls = []

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"local": {"acc,none": 0.5}}, "samples": {"local": []}}

    monkeypatch.setitem(
        sys.modules, "lm_eval", types.SimpleNamespace(simple_evaluate=simple_evaluate)
    )
    monkeypatch.setattr("aster.evaluation.language.lm_eval_adapter", lambda evaluator: evaluator)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "contract-test-not-official-run")
    return calls


def test_four_seeds_and_unsafe_policy_forwarded_and_persisted(monkeypatch, tmp_path):
    calls = fake_harness(monkeypatch)
    evaluator = object()
    evaluate_official_language(
        evaluator, tasks=["local"], output_directory=tmp_path / "run", seed=12, fewshot=3, limit=2
    )
    assert calls[0]["model"] is evaluator
    assert all(
        calls[0][key] == 12
        for key in ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")
    )
    assert calls[0]["confirm_run_unsafe_code"] is False and calls[0]["log_samples"] is True
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["subset_only"] and report["fewshot_random_seed"] == 12
    with pytest.raises(FileExistsError):
        evaluate_official_language(evaluator, tasks=["local"], output_directory=tmp_path / "run")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "settings",
    [
        dict(tasks="local"),
        dict(tasks=["a", "a"]),
        dict(tasks=[]),
        dict(limit=True),
        dict(limit=0),
        dict(limit=1.1),
        dict(fewshot=-1),
        dict(seed=True),
        dict(seed=2**32),
    ],
)
def test_invalid_protocol_rejected_before_evaluator(monkeypatch, tmp_path, settings):
    calls = fake_harness(monkeypatch)
    with pytest.raises(ValueError):
        evaluate_official_language(
            object(), output_directory=tmp_path / "run", **{"tasks": ["local"], **settings}
        )
    assert not calls and not (tmp_path / "run").exists()
