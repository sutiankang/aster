"""Immutable, content-addressed deployment artifacts with weights, processors, and lineage."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from .serialization import atomic_json, digest_json, file_digest, read_json


MANIFEST = ".aster-artifact.json"


def _filesystem_path(value):

    path = Path(os.path.abspath(os.fspath(value)))
    if os.name != "nt":
        return path
    name = str(path)
    if name.startswith("\\\\.\\"):
        raise ValueError("Artifact paths cannot use a device namespace")
    if name.startswith("\\\\?\\"):
        tail = name[4:]
        if not re.match(r"^[A-Za-z]:\\", tail) and not tail.upper().startswith("UNC\\"):
            raise ValueError("Artifact paths require a filesystem drive or UNC share")
        return path
    if name.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + name[2:])
    return Path("\\\\?\\" + name)


def _regular_tree(root):
    root = _filesystem_path(root)
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Artifact roots cannot be links/junctions")
    files = {}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            path = Path(directory) / name
            attributes = path.stat(follow_symlinks=False)
            if path.is_symlink() or getattr(attributes, "st_file_attributes", 0) & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024
            ):
                raise ValueError("Artifacts cannot contain links, junctions or reparse points")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative != MANIFEST:
                    files[relative] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
            elif not path.is_dir():
                raise ValueError("Artifact trees may contain only regular files/directories")
    return dict(sorted(files.items()))


@dataclass(frozen=True)
class Artifact:
    id: str
    path: Path
    kind: str
    metadata: dict
    parents: tuple[str, ...]


class ArtifactStore:
    def __init__(self, root):
        self.root = _filesystem_path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, directory, *, kind, metadata, parents=()):
        source = _filesystem_path(directory)
        if not source.is_dir() or not isinstance(kind, str) or not kind:
            raise ValueError("Publish a directory with an explicit kind/format")
        if source.resolve() == self.root or self.root.is_relative_to(source.resolve()):
            raise ValueError("The artifact store cannot be nested in the source tree")
        if (source / MANIFEST).exists():
            raise ValueError("Input contains the reserved manifest; use the existing artifact ID")
        parents = tuple(parents)
        if len(set(parents)) != len(parents):
            raise ValueError("Duplicate parent artifact")
        for parent in parents:
            self.get(parent)
        files = _regular_tree(source)
        if not files:
            raise ValueError("Cannot publish an empty artifact")
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "metadata": copy.deepcopy(metadata),
            "parents": list(parents),
            "files": files,
        }
        artifact_id = digest_json(manifest)
        destination = self.root / artifact_id
        if destination.exists():
            return self.get(artifact_id)
        temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=self.root))
        try:
            for relative in files:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
            if _regular_tree(temporary) != files or _regular_tree(source) != files:
                raise RuntimeError("Source changed during artifact snapshot")
            atomic_json(temporary / MANIFEST, manifest)
            try:
                os.rename(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise

                self.get(artifact_id)
            return self.get(artifact_id)
        finally:
            if (
                temporary.exists()
                and temporary.parent == self.root
                and temporary.name.startswith(".publish-")
            ):
                shutil.rmtree(temporary)

    def get(self, artifact_id, verify=True):
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[a-f0-9]{64}", artifact_id):
            raise ValueError("Artifact ID must be a SHA256 digest")
        path = self.root / artifact_id
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Artifact path cannot redirect outside its store")
        manifest = read_json(path / MANIFEST)
        if (
            set(manifest) != {"schema_version", "kind", "metadata", "parents", "files"}
            or manifest["schema_version"] != 1
        ):
            raise ValueError("Unknown artifact manifest schema")
        if digest_json(manifest) != artifact_id:
            raise ValueError("Artifact manifest identity mismatch")
        if verify and _regular_tree(path) != manifest["files"]:
            raise ValueError("Artifact payload integrity mismatch")
        return Artifact(
            artifact_id,
            path,
            manifest["kind"],
            copy.deepcopy(manifest["metadata"]),
            tuple(manifest["parents"]),
        )
