from pathlib import Path
import torch
import pytest
from aster.core import ArtifactStore
from aster.core.workflow import Stage, Workflow
from aster.recipes import BUILTIN_STAGES, gate_candidate, load_predictor_artifact
from aster.inference.optimization import PackedLinear


def test_native_train_quantize_eval_gate(tmp_path):
    torch.set_num_threads(1)
    data = str(Path(__file__).parents[2] / "examples/data/tiny_text.jsonl")
    store = ArtifactStore(tmp_path / "artifacts")
    model = {
        "architecture": "llama",
        "vocab_size": 259,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
    }
    stages = [
        Stage(
            "teacher",
            "language_fit",
            {
                "model": model,
                "data": data,
                "training": {"steps": 2, "batch_size": 2, "max_length": 64},
            },
        ),
        Stage("baseline", "language_evaluate", {"data": data, "max_length": 64}, ("teacher",)),
        Stage(
            "packed",
            "language_quantize",
            {
                "data": data,
                "max_length": 64,
                "targets": ["model.layers.0.mlp.down_proj"],
                "algorithm": "gptq",
                "bits": 4,
                "group_size": 8,
                "max_rows": 64,
            },
            ("teacher",),
        ),
        Stage("candidate", "language_evaluate", {"data": data, "max_length": 64}, ("packed",)),
        Stage(
            "gate",
            "quality_gate",
            {"baseline": "baseline", "candidate": "candidate", "max_regression": 1.0},
            ("baseline", "candidate"),
        ),
    ]
    workflow = Workflow(stages, BUILTIN_STAGES, artifact_store=store, directory=tmp_path / "run")
    result = workflow.run()
    assert result["gate"]["details"]["quality_gate"]["passed"]
    assert (
        result["candidate"]["details"]["protocol_id"]
        == result["baseline"]["details"]["protocol_id"]
    )
    assert result["gate"]["artifacts"]["model"] == result["packed"]["artifacts"]["model"]
    packed, tokenizer = load_predictor_artifact(store.get(result["packed"]["artifacts"]["model"]))
    assert isinstance(packed.model.layers[0].mlp.down_proj, PackedLinear)
    assert packed(torch.tensor([[1, 3, 4]])).logits.shape[-1] == tokenizer.vocab_size
    assert workflow.run() == result


def test_failed_gate_cannot_emit_candidate(tmp_path):
    from aster.evaluation import ComparisonProtocol, EvaluationRun, EvaluationRecord

    store = ArtifactStore(tmp_path / "artifacts")
    inputs = {}
    protocol = ComparisonProtocol("test", "fixture-v1", "fixture", "1", {}, ("a", "b"), "score")
    for name, score in [("baseline", 1.0), ("candidate", -10.0)]:
        run = EvaluationRun(protocol, name, environment={"test": True})
        for sample in protocol.expected_ids:
            run.add(EvaluationRecord(sample, "ok", {"score": score}))
        report = run.save(tmp_path / name)
        evidence = store.publish(report.parent, kind="evaluation", metadata={})
        inputs[name] = {"artifacts": {"evaluation": evidence.id}}
    with pytest.raises(RuntimeError, match="failed quality gate"):
        gate_candidate(
            {"baseline": "baseline", "candidate": "candidate"}, inputs, tmp_path / "gate", store
        )
    assert (tmp_path / "gate/quality-gate.json").is_file()
