"""Deterministic JSON encoding, atomic publication, and single-writer locking."""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"JSON floating-point value is outside finite range: {value}")
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(
            stream,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicates,
        )


def atomic_json(path, value):
    """Write and fsync a same-directory temporary file before atomic replacement."""
    payload = canonical_json(value).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunLock:
    """Single-writer ownership; an existing lock is not automatically treated as stale."""

    def __init__(self, path):
        self.path = Path(path)
        self.owned = False
        self._nonce = None

    def __enter__(self):
        if self.owned:
            raise RuntimeError("A RunLock instance cannot be entered twice")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json({"pid": os.getpid(), "owner": nonce}))
            stream.flush()
            os.fsync(stream.fileno())
        self._nonce = nonce
        self.owned = True
        return self

    def __exit__(self, *_):
        if self.owned:
            try:
                if (
                    self.path.is_symlink()
                    or not self.path.is_file()
                    or read_json(self.path).get("owner") != self._nonce
                ):
                    raise RuntimeError(
                        "Run lock ownership changed; refusing to remove another writer's lock"
                    )
                self.path.unlink()
            finally:
                self.owned = False
                self._nonce = None
