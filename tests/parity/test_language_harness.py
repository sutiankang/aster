import json
from pathlib import Path
import tempfile
import shutil
import gc

import pytest
import torch

from aster.core import ArtifactStore
from aster.data import ByteTokenizer
from aster.models import CausalLM, LlamaConfig
from aster.evaluation import EvaluationRun


@pytest.fixture
def short_dataset_cache():

    root = Path(tempfile.gettempdir()).resolve()
    directory = Path(tempfile.mkdtemp(prefix="aster-lm-", dir=root)).resolve()
    assert directory.parent == root and directory.name.startswith("aster-lm-")
    try:
        yield directory
    finally:
        gc.collect()
        shutil.rmtree(directory)


def test_actual_official_harness_native_artifact_samples_and_reproducible_fewshot(
    tmp_path, monkeypatch, record_property, short_dataset_cache
):
    pytest.importorskip(
        "lm_eval",
        reason="Actual official language harness is a separately installed evaluation dependency",
    )
    from lm_eval.tasks import TaskManager
    from importlib.metadata import version
    from aster.evaluation.language import evaluate_language_artifact

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    torch.set_num_threads(1)
    torch.manual_seed(333)
    record_property("lm_eval_version", version("lm-eval"))
    record_property(
        "scope",
        "actual official multiple_choice evaluator; local synthetic fixture, no benchmark score claim",
    )
    data = [{"question": f"Letter {i}:", "choices": [" A", " B"], "gold": i % 2} for i in range(4)]
    records = tmp_path / "data.jsonl"
    records.write_text("\n".join(json.dumps(x) for x in data) + "\n", encoding="utf-8")
    definitions = tmp_path / "tasks"
    definitions.mkdir()
    task_name = "aster_local_multiple_choice"
    config = dict(
        task=task_name,
        dataset_path="json",
        dataset_kwargs={
            "data_files": {"test": str(records), "train": str(records)},
            "cache_dir": str(short_dataset_cache),
            "keep_in_memory": True,
        },
        test_split="test",
        training_split="train",
        output_type="multiple_choice",
        doc_to_text="{{question}}",
        doc_to_target="gold",
        doc_to_choice="choices",
        metric_list=[dict(metric="acc", aggregation="mean", higher_is_better=True)],
        metadata={"version": 1},
    )
    (definitions / "fixture.yaml").write_text(json.dumps(config), encoding="utf-8")
    manager = TaskManager(include_path=definitions, include_defaults=False)
    tokenizer = ByteTokenizer()
    model = CausalLM(
        LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(tmp_path / "source/model")
    tokenizer.save_pretrained(tmp_path / "source/tokenizer")
    store = ArtifactStore(tmp_path / "store")
    source = store.publish(
        tmp_path / "source",
        kind="token_predictor",
        metadata={"purpose": "synthetic_evaluator_fixture"},
    )
    reports = []
    for index in range(2):
        result = evaluate_language_artifact(
            store,
            source.id,
            task_name=task_name,
            dataset_revision="local_synthetic_v1",
            output_directory=tmp_path / f"run-{index}",
            max_length=128,
            fewshot=1,
            seed=21,
            task_manager=manager,
        )
        report = EvaluationRun.load(result["report"])
        reports.append(report)
        assert report.candidate_artifact_id == source.id and report.summary()["denominator"] == 4
        assert report.summary()["statuses"]["ok"] == 4
        official = json.loads(
            (tmp_path / f"run-{index}/official/official-results.json").read_text(encoding="utf-8")
        )
        independent = []
        for row in official["samples"][task_name]:
            choice = max(
                range(len(row["filtered_resps"])), key=lambda k: row["filtered_resps"][k][0]
            )
            independent.append(float(choice == row["doc"]["gold"]))
        assert report.summary()["mean"] == sum(independent) / len(independent)
        assert store.get(result["artifact_id"]).parents == (source.id,)
        controls = json.loads((tmp_path / f"run-{index}/official/run.json").read_text())
        assert controls["fewshot_random_seed"] == 21 and not controls["unsafe_task_code_authorized"]
    assert reports[0].protocol.id == reports[1].protocol.id
    assert reports[0].records == reports[1].records
    subset = evaluate_language_artifact(
        store,
        source.id,
        task_name=task_name,
        dataset_revision="local_synthetic_v1",
        output_directory=tmp_path / "subset",
        max_length=128,
        limit=2,
        fewshot=1,
        seed=21,
        task_manager=manager,
    )
    assert subset["summary"]["denominator"] == 2
    assert EvaluationRun.load(subset["report"]).protocol.controls["subset_only"]
