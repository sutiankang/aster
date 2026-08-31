"""Single-writer, hash-chained event storage and side-effect-free replay."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def strict_loads(text):
    def reject(value):
        raise ValueError("Non-finite JSON value")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed

    return json.loads(
        text, parse_constant=reject, parse_float=finite_float, object_pairs_hook=object_pairs
    )


def read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("Event log cannot be a symlink")
    result, previous = [], "0" * 64
    with path.open("r", encoding="utf-8") as stream:
        for sequence, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError("Incomplete event; explicit recovery is required")
            event = strict_loads(line)
            if set(event) != {
                "sequence",
                "kind",
                "thread_id",
                "turn_id",
                "item_id",
                "payload",
                "wall_time",
                "previous",
                "hash",
            }:
                raise ValueError("Unexpected event schema")
            body = {key: value for key, value in event.items() if key != "hash"}
            if (
                event["sequence"] != sequence
                or event["previous"] != previous
                or digest(body) != event["hash"]
            ):
                raise ValueError("Event hash chain or sequence mismatch")
            result.append(event)
            previous = event["hash"]
    return result


class EventLog:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lease_path = self.path.with_suffix(self.path.suffix + ".writer")
        self._mutex = threading.RLock()

        descriptor = os.open(self._lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self._closed = False
        try:
            events = read_events(self.path)
            self._sequence = len(events)
            self._previous = events[-1]["hash"] if events else "0" * 64
            self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        except BaseException:
            self._lease_path.unlink()
            raise

    def append(self, kind, *, thread_id, turn_id=None, item_id=None, payload=None):
        with self._mutex:
            if self._closed or not kind or not thread_id:
                raise ValueError("Closed log or invalid event identity")
            body = {
                "sequence": self._sequence + 1,
                "kind": kind,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "item_id": item_id,
                "payload": payload or {},
                "wall_time": time.time(),
                "previous": self._previous,
            }
            event = {**body, "hash": digest(body)}
            encoded = canonical_json(event) + "\n"
            self._stream.write(encoded)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._sequence, self._previous = body["sequence"], event["hash"]
            return event

    def close(self):
        with self._mutex:
            if not self._closed:
                self._stream.close()
                self._closed = True
                self._lease_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class ReplayState:
    threads: dict = field(default_factory=dict)
    turns: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    model_traces: list = field(default_factory=list)
    approvals_active: bool = False


def replay(path):
    """Read events without executing tools or restoring approvals. Uncommitted tool starts
    remain ambiguous and are never automatically retried."""
    state = ReplayState()
    transitions = {
        "tool.prepared": (None, "prepared"),
        "tool.approved": ("prepared", "approved"),
        "tool.started": ("approved", "started"),
        "tool.result_committed": ("started", "result_committed"),
        "tool.ambiguous": ("started", "ambiguous"),
        "tool.denied": ("prepared", "denied"),
    }
    for event in read_events(path):
        kind, thread_id, turn_id, item_id = (
            event[key] for key in ("kind", "thread_id", "turn_id", "item_id")
        )
        if kind == "thread.started":
            if thread_id in state.threads:
                raise ValueError("Duplicate thread start")
            state.threads[thread_id] = event["payload"]
        elif kind == "turn.started":
            if thread_id not in state.threads or turn_id in state.turns:
                raise ValueError("Invalid turn start")
            state.turns[turn_id] = {"status": "running", "thread_id": thread_id, **event["payload"]}
        elif kind == "turn.completed":
            if turn_id not in state.turns or state.turns[turn_id]["status"] != "running":
                raise ValueError("Invalid turn completion")
            state.turns[turn_id].update(event["payload"])
            state.turns[turn_id]["outcome"] = event["payload"].get("status")
            state.turns[turn_id]["status"] = "completed"
        elif kind in transitions:
            old, new = transitions[kind]
            current = state.items.get(item_id)
            if (current["status"] if current else None) != old or turn_id not in state.turns:
                raise ValueError("Invalid tool event transition")
            state.items[item_id] = {
                **(current or {}),
                **event["payload"],
                "receipt_status": event["payload"].get("status"),
                "status": new,
                "turn_id": turn_id,
                "thread_id": thread_id,
            }
        elif kind == "model.trace":
            state.model_traces.append(event["payload"])
    for item in state.items.values():
        if item["status"] == "started":
            item["status"] = "ambiguous"
    return state
