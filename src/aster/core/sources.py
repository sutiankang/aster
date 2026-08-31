"""Verify local reference-source revisions and file hashes without downloading code."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from .serialization import file_digest


@dataclass(frozen=True)
class SourceLock:
    repository: str
    commit: str
    license: str
    files: dict[str, str]
    oracle_environment: str

    def __post_init__(self):
        if not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ValueError("An explicit canonical source repository is required")
        if (
            not re.fullmatch(r"[a-f0-9]{40}", self.commit)
            or not self.license
            or not self.oracle_environment
        ):
            raise ValueError("Pin a full commit, license and oracle environment")
        if not self.files:
            raise ValueError("A source lock must identify the actual implementation files")
        for name, digest in self.files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Source file paths must be repository-relative")
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("Every referenced source file needs a SHA256")

    def verify_files(self, root):
        root = Path(root).resolve(strict=True)
        for name, digest in self.files.items():
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or file_digest(path) != digest:
                raise ValueError(f"Reference source mismatch: {name}")
        return True
