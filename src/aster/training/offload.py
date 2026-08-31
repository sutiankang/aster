"""CPU and per-parameter disk optimizer-state offload."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import uuid
import torch
from torch import nn
from .state import file_hash, read_payload
from .muon import MuonWithAuxAdam, rebind_matrix_layout


class CPUOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer):
        if type(optimizer) not in {
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.RAdam,
            torch.optim.SGD,
            MuonWithAuxAdam,
        }:
            raise TypeError(
                "CPU optimizer offload 当前仅支持 Adam/AdamW/RAdam/SGD/显式MuonWithAuxAdam"
            )
        if optimizer.state:
            raise ValueError("offload 需在首次更新前建立，已有状态使用 checkpoint 恢复")
        self.originals, self.masters = [], []
        groups = []
        for group in optimizer.param_groups:
            masters = []
            for parameter in group["params"]:
                dtype = (
                    torch.float32
                    if parameter.dtype in {torch.float16, torch.bfloat16}
                    else parameter.dtype
                )
                data = parameter.detach().to(device="cpu", dtype=dtype).clone()
                if parameter.device.type == "cuda":
                    data = data.pin_memory()
                master = nn.Parameter(data, requires_grad=True)
                rebind_matrix_layout(parameter, master)
                self.originals.append(parameter)
                self.masters.append(master)
                masters.append(master)
            groups.append(
                {
                    **{key: value for key, value in group.items() if key != "params"},
                    "params": masters,
                }
            )
        self.optimizer = type(optimizer)(groups)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @torch.no_grad()
    def step(self):
        for original, master in zip(self.originals, self.masters):
            master.grad = (
                original.grad.detach().to(device="cpu", dtype=master.dtype)
                if original.grad is not None
                else None
            )
        self.optimizer.step()
        for original, master in zip(self.originals, self.masters):
            original.copy_(master.to(device=original.device, dtype=original.dtype))

    def state_dict(self):
        return {
            "kind": "cpu_optimizer",
            "optimizer": self.optimizer.state_dict(),
            "masters": [p.detach().clone() for p in self.masters],
        }

    def load_state_dict(self, state):
        if state.get("kind") != "cpu_optimizer" or len(state["masters"]) != len(self.masters):
            raise ValueError("CPU optimizer 状态不兼容")
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for original, master, saved in zip(self.originals, self.masters, state["masters"]):
                master.copy_(saved)
                original.copy_(master.to(device=original.device, dtype=original.dtype))


class DiskOptimizer(CPUOptimizer):
    """Offload optimizer state per parameter to a caller-selected disk directory."""

    def __init__(self, optimizer, directory):
        if isinstance(optimizer, MuonWithAuxAdam) and any(
            group["missing_grad"] != "skip" for group in optimizer.param_groups
        ):
            raise ValueError(
                "Disk Muon per-parameter eviction requires explicit missing_grad=skip; zero policy would update other parameters repeatedly"
            )
        super().__init__(optimizer)
        root = Path(directory).absolute()
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ValueError("offload 根目录不能是符号链接")

        self.directory = Path(tempfile.mkdtemp(prefix="aster-optimizer-", dir=root))
        self.records = {}
        self.indices = {id(parameter): index for index, parameter in enumerate(self.masters)}
        self.peak_resident_state_elements = 0
        self.optimizer._aster_state_loader = self.state_for

    def state_for(self, parameter):
        if parameter in self.optimizer.state:
            return self.optimizer.state[parameter]
        entry = self.records.get(id(parameter))
        return read_payload(self.directory, entry, trusted=False) if entry is not None else {}

    def _evict(self, parameter):
        state = self.optimizer.state.pop(parameter, None)
        if state is None:
            return
        self.peak_resident_state_elements = max(
            self.peak_resident_state_elements,
            sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)),
        )
        target = self.directory / f"state-{self.indices[id(parameter)]}.pt"
        temporary = self.directory / f".pending-{uuid.uuid4().hex}.pt"
        try:
            with temporary.open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.records[id(parameter)] = {
            "file": target.name,
            "sha256": file_hash(target),
            "bytes": target.stat().st_size,
        }

    def evict_all(self):
        for parameter in self.masters:
            self._evict(parameter)

    @torch.no_grad()
    def step(self):

        for master in self.masters:
            master.grad = None
        for original, master in zip(self.originals, self.masters):
            if original.grad is None:
                continue
            self.optimizer.state[master] = self.state_for(master)
            master.grad = original.grad.detach().to(device="cpu", dtype=master.dtype)
            try:
                self.optimizer.step()
                original.copy_(master.to(device=original.device, dtype=original.dtype))
                self._evict(master)
            finally:
                master.grad = None

    @contextmanager
    def _materialized(self):
        for parameter in self.masters:
            self.optimizer.state[parameter] = self.state_for(parameter)
        try:
            yield
        finally:
            self.optimizer.state.clear()

    def state_dict(self):
        with self._materialized():
            return {
                "kind": "disk_optimizer",
                "optimizer": self.optimizer.state_dict(),
                "masters": [p.detach().clone() for p in self.masters],
            }

    def load_state_dict(self, state):
        if state.get("kind") != "disk_optimizer":
            raise ValueError("磁盘 optimizer 状态类型不一致")
        super().load_state_dict({**state, "kind": "cpu_optimizer"})
        self.records.clear()
        self.evict_all()
