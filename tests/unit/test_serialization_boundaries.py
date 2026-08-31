import pytest
from aster.core import read_json, atomic_json
from aster.core.serialization import RunLock


@pytest.mark.parametrize("payload", ['{"x":1e999}', '{"x":-1e999}', '{"x":NaN}', '{"x":1,"x":2}'])
def test_strict_json_rejects_overflow_and_duplicate_keys(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_run_lock_cannot_remove_replacement_owners_lock(tmp_path):
    path = tmp_path / "run.lock"
    lock = RunLock(path)
    lock.__enter__()
    with pytest.raises(FileExistsError):
        RunLock(path).__enter__()
    atomic_json(path, {"pid": 4, "owner": "different-writer"})
    with pytest.raises(RuntimeError, match="ownership"):
        lock.__exit__(None, None, None)
    assert path.exists() and read_json(path)["owner"] == "different-writer"


def test_run_lock_normal_release_and_reuse(tmp_path):
    path = tmp_path / "run.lock"
    lock = RunLock(path)
    with lock:
        assert read_json(path)["owner"]
    assert not path.exists()
    with lock:
        assert path.exists()
    assert not path.exists()
