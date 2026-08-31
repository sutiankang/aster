import os
from pathlib import Path

import pytest

from aster.core.artifacts import ArtifactStore, _filesystem_path, _regular_tree
from aster.core import digest_json


def test_artifact_deep_paths_preserve_identity_and_integrity(tmp_path):

    source = tmp_path / "source"
    source.mkdir()
    relative = "nested/" + "a" * 64 + ".bin"
    (source / "nested").mkdir()
    (source / relative).write_bytes(b"bounded-native-artifact")
    deep = tmp_path / "store"
    while len(str(deep)) < 220:
        deep = deep / ("level_" + "x" * 24)
    store = ArtifactStore(deep)
    artifact = store.publish(source, kind="long-path-test", metadata={"schema_version": 1})
    assert len(str(artifact.path / relative)) > 300
    assert (artifact.path / relative).read_bytes() == b"bounded-native-artifact"
    manifest = {
        "schema_version": 1,
        "kind": "long-path-test",
        "metadata": {"schema_version": 1},
        "parents": [],
        "files": _regular_tree(source),
    }
    assert artifact.id == digest_json(manifest)
    assert store.get(artifact.id).id == artifact.id

    assert (
        ArtifactStore(tmp_path / "short")
        .publish(source, kind=artifact.kind, metadata=artifact.metadata)
        .id
        == artifact.id
    )
    (artifact.path / relative).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        store.get(artifact.id)


def test_artifact_long_source_tree_and_store_containment(tmp_path):
    plain_source = tmp_path / "source"
    while len(str(plain_source)) < 290:
        plain_source = plain_source / ("source_" + "z" * 26)
    source = _filesystem_path(plain_source)
    source.mkdir(parents=True)
    (source / "data.txt").write_text("long source", encoding="utf-8")
    artifact = ArtifactStore(tmp_path / "store").publish(plain_source, kind="text", metadata={})
    assert (artifact.path / "data.txt").read_text(encoding="utf-8") == "long source"
    with pytest.raises(ValueError, match="nested"):
        ArtifactStore(source / "bad-store").publish(plain_source, kind="text", metadata={})


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace syntax")
def test_windows_long_path_is_idempotent_and_device_namespace_rejected(tmp_path):
    path = _filesystem_path(tmp_path)
    assert str(path).startswith("\\\\?\\")
    assert _filesystem_path(path) == path
    assert _filesystem_path(tmp_path / "child" / "..") == path
    for forbidden in (
        "\\\\.\\PhysicalDrive0",
        "\\\\?\\GLOBALROOT\\Device\\HarddiskVolumeShadowCopy1",
    ):
        with pytest.raises(ValueError, match="namespace|filesystem"):
            _filesystem_path(forbidden)
