import json
import pytest
import torch
from aster.core import (
    ArtifactStore,
    LossTerm,
    LossBundle,
    RunLock,
    atomic_json,
    read_json,
    digest_json,
    SourceLock,
    file_digest,
)


def test_atomic_json_strict(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"x": [1, "中"]})
    assert read_json(path) == {"x": [1, "中"]}
    with pytest.raises(ValueError):
        atomic_json(path, {"x": float("nan")})
    assert read_json(path) == {"x": [1, "中"]}
    path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)
    with RunLock(tmp_path / "lock"):
        with pytest.raises(FileExistsError):
            with RunLock(tmp_path / "lock"):
                pass
    assert not (tmp_path / "lock").exists()


def test_artifact_snapshot_lineage_and_mutation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    atomic_json(source / "config.json", {"value": 1})
    store = ArtifactStore(tmp_path / "store")
    artifact = store.publish(source, kind="fixture", metadata={"domain": "llm"})
    assert store.get(artifact.id).id == artifact.id
    atomic_json(source / "config.json", {"value": 2})
    assert read_json(artifact.path / "config.json")["value"] == 1
    child = store.publish(source, kind="fixture", metadata={}, parents=[artifact.id])
    assert child.parents == (artifact.id,)
    assert store.publish(source, kind="fixture", metadata={}, parents=[artifact.id]).id == child.id
    atomic_json(artifact.path / "config.json", {"value": 99})
    with pytest.raises(ValueError, match="integrity"):
        store.get(artifact.id)
    with pytest.raises(ValueError):
        store.get("../escape")


def test_loss_terms_preserve_units():
    first = LossTerm(torch.tensor(2.0, requires_grad=True), torch.tensor(2.0), "token", name="ce")
    second = LossTerm(
        torch.tensor(10.0, requires_grad=True), torch.tensor(1.0), "episode", name="reward"
    )
    bundle = LossBundle((first, second))
    assert len(bundle.terms) == 2
    with pytest.raises(ValueError):
        LossBundle((first, first))
    with pytest.raises(ValueError):
        LossTerm(torch.tensor(1.0), torch.tensor(1.0, requires_grad=True), "token")


def test_source_lock_requires_actual_files(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("source", encoding="utf-8")
    lock = SourceLock(
        "https://github.com/org/repo",
        "a" * 40,
        "Apache-2.0",
        {"model.py": file_digest(path)},
        "pinned-env",
    )
    assert lock.verify_files(tmp_path)
    with pytest.raises(ValueError):
        SourceLock("https://github.com/org/repo", "main", "Apache-2.0", {}, "env")
