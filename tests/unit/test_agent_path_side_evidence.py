from pathlib import Path
from types import SimpleNamespace
import stat
import pytest
from aster.agents import Workspace, PermissionDenied


@pytest.mark.parametrize("kind", ["symlink_mode", "windows_reparse_attribute"])
def test_real_workspace_rejection_logic_with_explicit_os_metadata_double(
    tmp_path, monkeypatch, kind
):
    workspace = Workspace(tmp_path)
    file = tmp_path / "entry.txt"
    file.write_text("fixture", encoding="utf-8")
    original = Path.lstat

    def metadata(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == file:
            return SimpleNamespace(
                st_mode=stat.S_IFLNK if kind == "symlink_mode" else value.st_mode,
                st_file_attributes=0x400 if kind == "windows_reparse_attribute" else 0,
            )
        return value

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(PermissionDenied, match="Symlink/junction/reparse"):
        workspace.resolve(str(file))
