from pathlib import Path
import runpy

from aster.core import ArtifactStore
from aster.recipes import load_predictor_artifact


def test_two_stage_dsa_real_training_artifacts_and_cached_deployment(tmp_path):
    run = runpy.run_path(str(Path(__file__).parents[2] / "examples/dsa_pipeline.py"))[
        "run_pipeline"
    ]
    report = run(tmp_path / "pipeline", steps=1)
    assert report["cache_equivalent"] and not report["public_quality_validated"]
    warmup, sparse = report["stages"]
    assert warmup["stage"] == "dense_warmup" and sparse["stage"] == "sparse_training"
    assert (
        sparse["parent_id"] == warmup["artifact_id"] and warmup["updates"] == sparse["updates"] == 2
    )
    store = ArtifactStore(tmp_path / "pipeline/store")
    final = store.get(report["final_artifact_id"])
    assert final.parents == (warmup["artifact_id"],)
    model, _ = load_predictor_artifact(final)
    assert all(layer.self_attn.indexer_stage is None for layer in model.model.layers)
