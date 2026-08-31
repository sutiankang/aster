"""Bounded host state archives and raw-code-preserving paged KV swap."""

from collections import OrderedDict
from dataclasses import dataclass
import threading
import uuid
import torch

from .state import StateError, CacheCapacityError
from .task_runners import _tree_map
from .state import PagedStatePool
from aster.optimization.kv_quantization import QuantizedKV, clone_kv
from aster.core.async_work import settle_thread


@dataclass(frozen=True)
class _Int8Tensor:
    values: torch.Tensor
    scales: torch.Tensor
    dtype: torch.dtype


class StateArchive:
    """Bind handles to policy/processor/tenant identity and return independent tensors."""

    evidence_kind = "native_storage_reference"

    def __init__(self, *, max_bytes=256 * 1024**2):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Archive byte capacity must be a positive integer")
        self.max_bytes, self.stored_bytes = max_bytes, 0
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def put(self, state, *, identity, quantize=False):
        kind = getattr(state, "kind", getattr(getattr(state, "capabilities", None), "kind", None))
        if (
            not isinstance(identity, str)
            or not identity
            or kind
            not in {
                "dense_kv",
                "window_kv",
                "mla_latent",
                "indexed_mla",
                "hybrid_delta",
                "mamba_ssm",
                "compressed_window_mqa",
                "qwen3_vl_kv",
                "gemma4_shared_kv",
                "rssm",
            }
        ):
            raise StateError("Archive requires an explicit identity and supported typed state")
        if quantize and kind not in {"dense_kv", "window_kv", "mla_latent"}:
            raise StateError("Lossy quantization for recurrent/multimodal state is not implemented")
        size = 0

        def pack(tensor):
            nonlocal size
            value = tensor.detach().cpu().contiguous().clone()
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise StateError("Cannot archive non-finite state")
            if quantize and value.is_floating_point() and value.numel() and value.ndim:
                scale = value.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
                packed = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
                size += packed.numel() + scale.numel() * scale.element_size()
                return _Int8Tensor(packed, scale, value.dtype)
            size += value.numel() * value.element_size()
            return value

        stored = _tree_map(state, pack)
        if size > self.max_bytes:
            raise CacheCapacityError("Single snapshot exceeds archive capacity")
        handle = uuid.uuid4().hex
        with self._lock:
            while self._entries and self.stored_bytes + size > self.max_bytes:
                _, (_, _, removed) = self._entries.popitem(last=False)
                self.stored_bytes -= removed
            self._entries[handle] = identity, stored, size
            self.stored_bytes += size
        return handle

    def get(self, handle, *, identity, device="cpu"):
        from dataclasses import fields, is_dataclass

        def restore(value):
            if isinstance(value, _Int8Tensor):
                return (value.values.float() * value.scales).to(device=device, dtype=value.dtype)
            if isinstance(value, torch.Tensor):
                return value.to(device).clone()
            if is_dataclass(value):
                return type(value)(
                    **{field.name: restore(getattr(value, field.name)) for field in fields(value)}
                )
            if isinstance(value, tuple):
                return tuple(restore(item) for item in value)
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            return value

        with self._lock:
            if handle not in self._entries or self._entries[handle][0] != identity:
                raise StateError("Missing/evicted snapshot or identity mismatch")
            self._entries.move_to_end(handle)
            return restore(self._entries[handle][1])

    def release(self, handle, *, identity):
        with self._lock:
            if handle not in self._entries or self._entries[handle][0] != identity:
                raise StateError("Missing snapshot or identity mismatch")
            _, _, size = self._entries.pop(handle)
            self.stored_bytes -= size


@dataclass(frozen=True)
class _PagedSnapshot:
    identity: str
    length: int
    pages: tuple
    nbytes: int


class PagedStateArchive:
    """Bounded host storage preserving KV codes, scales, and padding exactly."""

    def __init__(self, pool, *, max_bytes=256 * 1024**2, pin_memory=False):
        if type(pool) is not PagedStatePool or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Archive requires the actual paged pool and a positive byte budget")
        if type(pin_memory) is not bool or (pin_memory and not torch.cuda.is_available()):
            raise ValueError("Pinned host archive requires a provisioned CUDA allocator")
        self.pool, self.max_bytes, self.pin_memory = pool, max_bytes, pin_memory
        self._entries = {}
        self._lock = threading.RLock()
        self.stored_bytes = 0
        self.offloaded_tokens = self.restored_tokens = 0

    def put(self, sequence):
        with self.pool.read_pages(sequence) as views:
            if not views:
                raise StateError("Cannot offload an empty request state")
            length = sum(page.payload[0].shape[self.pool._dims[0]] for page in views)
            size = sum(
                x.nbytes if isinstance(x, QuantizedKV) else x.numel() * x.element_size()
                for page in views
                for x in page.payload
            )
            with self._lock:
                if self.stored_bytes + size > self.max_bytes:
                    raise CacheCapacityError("Host KV archive is full")
                self.stored_bytes += size
            devices = {x.device for page in views for x in page.payload}
            try:
                for device in devices:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                pages = tuple(
                    tuple(
                        clone_kv(x, device="cpu", pin_memory=self.pin_memory) for x in view.payload
                    )
                    for view in views
                )
                for device in devices:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                handle = uuid.uuid4().hex
                with self._lock:
                    self._entries[handle] = _PagedSnapshot(sequence.identity, length, pages, size)
                    self.offloaded_tokens += length
                return handle
            except BaseException:
                for device in devices:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                with self._lock:
                    self.stored_bytes -= size
                raise

    def restore(self, handle, *, identity):
        with self._lock:
            snapshot = self._entries.get(handle)
            if snapshot is None or snapshot.identity != identity:
                raise StateError("Missing archived KV or identity mismatch")

        state = self.pool.restore_pages(snapshot.pages, identity=identity, length=snapshot.length)
        with self._lock:
            self.restored_tokens += snapshot.length
        return state

    def release(self, handle, *, identity):
        with self._lock:
            snapshot = self._entries.get(handle)
            if snapshot is None or snapshot.identity != identity:
                raise StateError("Missing archived KV or identity mismatch")
            self._entries.pop(handle)
            self.stored_bytes -= snapshot.nbytes

    async def put_async(self, sequence):
        import asyncio

        work = await settle_thread(self.put, sequence)
        if work.cancelled:
            if work.error is None:
                self.release(work.value, identity=sequence.identity)
            raise asyncio.CancelledError from work.error
        return work.unwrap()

    async def restore_async(self, handle, *, identity):
        import asyncio

        work = await settle_thread(self.restore, handle, identity=identity)
        if work.cancelled:
            if work.error is None:
                self.pool.release(work.value)
            raise asyncio.CancelledError from work.error
        return work.unwrap()

    def metrics(self):
        with self._lock:
            return {
                "host_bytes": self.stored_bytes,
                "snapshots": len(self._entries),
                "offloaded_tokens": self.offloaded_tokens,
                "restored_tokens": self.restored_tokens,
                "pinned_memory": self.pin_memory,
            }
