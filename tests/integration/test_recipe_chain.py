from pathlib import Path
import torch
from aster.core import ArtifactStore
from aster.core.workflow import Stage, Workflow
from aster.recipes import BUILTIN_STAGES
from aster.models import load_model
from aster.data import load_tokenizer
from aster.evaluation.language import LanguageEvaluator


def test_train_distill_artifact_evaluate_and_reenter(tmp_path):
    torch.set_num_threads(1)
    data = Path(__file__).parents[2] / "examples/data/tiny_text.jsonl"
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
    base = {
        "model": model,
        "data": str(data),
        "training": {"steps": 3, "batch_size": 2, "max_length": 64, "learning_rate": 0.002},
    }
    stages = [
        Stage("teacher", "language_fit", base),
        Stage("student", "language_fit", {**base, "distillation": {"weight": 0.7}}, ("teacher",)),
        Stage("evaluate", "language_evaluate", {"data": str(data), "max_length": 64}, ("student",)),
    ]
    workflow = Workflow(stages, BUILTIN_STAGES, artifact_store=store, directory=tmp_path / "run")
    results = workflow.run()
    assert results["evaluate"]["metrics"]["perplexity"] > 0
    assert workflow.run() == results
    student = store.get(results["student"]["artifacts"]["model"])
    assert student.parents == (results["teacher"]["artifacts"]["model"],)
    model = load_model(student.path / "model")
    tokenizer = load_tokenizer(student.path / "tokenizer")
    evaluator = LanguageEvaluator(model, tokenizer, max_length=64)
    assert evaluator.score("red ", "green")[0] < 0
    assert evaluator.rolling("hello world " * 10) < 0
    assert isinstance(evaluator.generate("red", max_new_tokens=3), str)
