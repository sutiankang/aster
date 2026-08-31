"""Random state, EMA, and committed checkpoint boundaries."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG 拓扑不匹配；不能声明精确恢复")
        torch.cuda.set_rng_state_all(state["cuda"])


class EMA:
    """Update moving averages only after successful optimizer updates."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay 必须在 [0,1)")
        self.decay, self.updates = float(decay), 0
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA 模型结构发生变化")
        for name, value in current.items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "shadow": deepcopy(self.shadow)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["decay"] != self.decay or state["shadow"].keys() != self.shadow.keys():
            raise ValueError("EMA 配置/参数名称不匹配")
        for key, value in state["shadow"].items():
            self.shadow[key].copy_(value)
        self.updates = state["updates"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True)
    fd, temporary = tempfile.mkstemp(prefix=".aster-manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_payload(directory: Path, stem: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    name = f".{stem}.{uuid.uuid4().hex}.pt"
    path = directory / name
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {"file": name, "sha256": file_hash(path), "bytes": path.stat().st_size}


def read_payload(directory: Path, entry: Mapping[str, Any], *, trusted: bool) -> dict[str, Any]:
    name = entry["file"]
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("checkpoint payload 路径必须在 manifest 同目录")
    path = directory / name
    if path.is_symlink() or bool(getattr(os.path, "isjunction", lambda _: False)(path)):
        raise ValueError("checkpoint 不允许链接 payload")
    if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
        raise ValueError("checkpoint payload 完整性校验失败")

    return torch.load(path, map_location="cpu", weights_only=not trusted)
