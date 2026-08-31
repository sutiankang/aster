"""Page storage, ownership, copy-on-write, and compressed radix prefix reuse."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
from typing import Any

import torch

from aster.core.contracts import StateCapabilities
from aster.optimization.kv_quantization import (
    KVQuantization,
    QuantizedKV,
    quantize_kv,
    allocate_kv_like,
    copy_kv,
    clone_kv,
)


class StateError(RuntimeError):
    pass


class CacheCapacityError(StateError):
    pass


@dataclass(frozen=True)
class _TypedTree:
    state_type: Any
    model_key: str
    kind: str
    layers_tree: Any
    sequence_dim: int


@dataclass(frozen=True)
class KVStateCodec:
    """Declare the sequence axis of every tensor leaf; latent and indexer leaves may
    have different feature dimensions."""

    kind: str = "dense_kv"
    sequence_dim: int = 2
    sequence_dims: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.kind not in {"dense_kv", "gqa_kv", "mla_latent_rope", "mla_latent", "indexed_mla"}:
            raise ValueError("An explicit supported KV codec tag is required")
        if self.sequence_dim < 1 or (self.sequence_dims and min(self.sequence_dims) < 1):
            raise ValueError("Batch axis 0 cannot also be the sequence axis")

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, truncatable=True, reorderable=True)

    def flatten(self, state):
        leaves = []

        def visit(value):
            if isinstance(value, torch.Tensor):
                index = len(leaves)
                leaves.append(value)
                return index
            if isinstance(value, tuple) and value:
                return tuple(visit(child) for child in value)
            raise StateError("KV state must be a nonempty tuple tree of tensors")

        typed = all(hasattr(state, name) for name in ("layers", "seen_tokens", "model_key", "kind"))
        if typed:
            if state.kind != self.kind:
                raise StateError("Native state tag does not match the selected codec")
            tree = visit(state.layers)
            tree = _TypedTree(type(state), state.model_key, state.kind, tree, self.sequence_dim)
        else:
            tree = visit(state)
        dims = self.sequence_dims or (self.sequence_dim,) * len(leaves)
        if len(dims) != len(leaves):
            raise StateError("Codec sequence axes do not match state leaves")
        lengths = []
        for tensor, dim in zip(leaves, dims):
            if tensor.ndim <= dim or tensor.shape[0] < 1:
                raise StateError("Invalid KV tensor layout")
            lengths.append(tensor.shape[dim])
        if len(set(lengths)) != 1:
            raise StateError("Every cache leaf must describe the same token span")
        if typed and state.seen_tokens != lengths[0]:
            raise StateError("Truncated/window/recurrent state needs its own codec")
        return tuple(leaves), tree, tuple(dims), lengths[0]

    @staticmethod
    def unflatten(leaves, tree):
        if isinstance(tree, _TypedTree):
            return tree.state_type(
                layers=KVStateCodec.unflatten(leaves, tree.layers_tree),
                seen_tokens=leaves[0].shape[tree.sequence_dim],
                model_key=tree.model_key,
                kind=tree.kind,
            )
        if isinstance(tree, int):
            return leaves[tree]
        return tuple(KVStateCodec.unflatten(leaves, branch) for branch in tree)

    def concatenate_batch(self, states):
        flattened = [self.flatten(state) for state in states]
        first = flattened[0]
        if any(item[1:] != first[1:] for item in flattened[1:]):
            raise StateError("Only matching codecs/layouts/cache lengths can share a batch")
        leaves = [
            torch.cat([item[0][i] for item in flattened], dim=0) for i in range(len(first[0]))
        ]
        return self.unflatten(leaves, first[1])

    def split_batch(self, state):
        leaves, tree, _, _ = self.flatten(state)
        sizes = {tensor.shape[0] for tensor in leaves}
        if len(sizes) != 1:
            raise StateError("Inconsistent cache batch dimensions")
        return [
            self.unflatten([leaf[i : i + 1] for leaf in leaves], tree)
            for i in range(leaves[0].shape[0])
        ]


@dataclass(frozen=True)
class PageRef:
    index: int
    generation: int


@dataclass
class _Page:
    generation: int = 0
    owners: int = 0
    readers: int = 0
    valid: int = 0
    payload: tuple[torch.Tensor, ...] | None = None


@dataclass
class PagedSequence:
    """A request or prefix block table modified only by its owning pool."""

    owner: int
    kind: str
    identity: str
    pages: list[PageRef] = field(default_factory=list)
    length: int = 0
    released: bool = False


@dataclass(frozen=True)
class PageView:
    """A borrowed page view; offset is an absolute logical token position, not a page number."""

    offset: int
    payload: tuple[torch.Tensor, ...]


class PagedStatePool:
    evidence_kind = "native_storage_reference"

    def __init__(self, *, block_size=16, max_blocks=256, codec=None, quantization=None):
        if block_size < 1 or max_blocks < 1:
            raise ValueError("Cache capacities must be positive")
        self.block_size, self.max_blocks = block_size, max_blocks
        self.codec = codec or KVStateCodec()
        if quantization is not None and not isinstance(quantization, KVQuantization):
            raise ValueError("An explicit immutable KVQuantization profile is required")
        self._quantization = quantization
        self._pages = [_Page() for _ in range(max_blocks)]
        self._lock = threading.RLock()
        self._tree = self._dims = self._signature = None

    def create(self, identity: str):
        if not identity:
            raise ValueError("Cache identity must bind an immutable policy")
        return PagedSequence(id(self), self.codec.kind, identity)

    @property
    def quantization(self):
        return self._quantization

    def _check(self, sequence):
        if sequence.owner != id(self) or sequence.kind != self.codec.kind or sequence.released:
            raise StateError("Foreign, mistagged or released sequence")

    def _page(self, ref):
        page = self._pages[ref.index]
        if page.generation != ref.generation or (page.owners == 0 and page.readers == 0):
            raise StateError("Stale page reference")
        return page

    def _collect(self, page):
        if page.owners == 0 and page.readers == 0:
            page.payload = None
            page.valid = 0

    @property
    def used_blocks(self):
        with self._lock:
            return sum(page.owners > 0 or page.readers > 0 for page in self._pages)

    def fork(self, sequence):
        with self._lock:
            self._check(sequence)
            for ref in sequence.pages:
                self._page(ref).owners += 1
            return PagedSequence(
                id(self), sequence.kind, sequence.identity, list(sequence.pages), sequence.length
            )

    def truncate(self, sequence, length):
        with self._lock:
            self._check(sequence)
            if not 0 <= length <= sequence.length:
                raise ValueError("Cannot extend state by truncation")
            count = math.ceil(length / self.block_size)
            for ref in sequence.pages[count:]:
                page = self._page(ref)
                page.owners -= 1
                self._collect(page)
            sequence.pages = sequence.pages[:count]
            sequence.length = length

    def release(self, sequence):
        with self._lock:
            if sequence.released:
                return
            self._check(sequence)
            for ref in sequence.pages:
                page = self._page(ref)
                page.owners -= 1
                self._collect(page)
            sequence.released = True
            sequence.pages = []
            sequence.length = 0

    @contextmanager
    def borrow(self, sequence):
        """Retain read references until the device/worker finishes; release must not reuse
        pages still owned by in-flight computation."""
        with self._lock:
            self._check(sequence)
            refs = tuple(sequence.pages)
            for ref in refs:
                self._page(ref).readers += 1
        try:
            yield sequence
        finally:
            with self._lock:
                for ref in refs:
                    page = self._page(ref)
                    page.readers -= 1
                    self._collect(page)

    def append(self, sequence, full_state):
        """Append the new suffix of returned state atomically; allocation failure leaves
        the existing page table unchanged."""
        leaves, tree, dims, length = self.codec.flatten(full_state)
        return self._append_leaves(sequence, leaves, tree, dims, length, source_start=0)

    def append_delta(self, sequence, suffix_state):
        """Commit only new-token layer KV without constructing full history tensors."""
        leaves, tree, dims, count = self.codec.flatten(suffix_state)
        with self._lock:
            self._check(sequence)
            return self._append_leaves(
                sequence, leaves, tree, dims, sequence.length + count, source_start=sequence.length
            )

    def _append_leaves(self, sequence, leaves, tree, dims, length, *, source_start):
        if any(tensor.shape[0] != 1 for tensor in leaves):
            raise StateError("Page storage accepts single-sequence states only")
        signature = tuple(
            (tuple(t.shape[:d]) + tuple(t.shape[d + 1 :]), t.dtype, t.device)
            for t, d in zip(leaves, dims)
        )
        if self.quantization is not None and any(
            t.is_floating_point() and d != t.ndim - 2 for t, d in zip(leaves, dims)
        ):
            raise StateError(
                "Quantized KV requires token/head vectors, not an arbitrary state tensor"
            )

        leaves = tuple(quantize_kv(t, self.quantization) for t in leaves)
        with self._lock:
            self._check(sequence)
            if length < sequence.length:
                raise StateError("Append cannot shrink state")
            if self._signature is not None and (
                signature != self._signature or tree != self._tree or dims != self._dims
            ):
                raise StateError("Cache layout changed within a model pool")
            if length == sequence.length:
                return
            old_pages = list(sequence.pages)
            needed = math.ceil(length / self.block_size)
            tail = sequence.length % self.block_size
            cow = bool(
                tail
                and old_pages
                and (self._page(old_pages[-1]).owners > 1 or self._page(old_pages[-1]).readers > 0)
            )
            allocation_count = needed - len(old_pages) + int(cow)
            free = [
                i for i, page in enumerate(self._pages) if page.owners == 0 and page.readers == 0
            ]
            if len(free) < allocation_count:
                raise CacheCapacityError("State page budget exhausted")
            allocated = []
            try:
                for index in free[:allocation_count]:
                    page = self._pages[index]
                    payload = []
                    for tensor, dim in zip(leaves, dims):
                        payload.append(allocate_kv_like(tensor, dim, self.block_size))
                    page.generation += 1
                    page.payload = tuple(payload)
                    page.owners, page.valid = 1, 0
                    allocated.append(PageRef(index, page.generation))
                new_pages = list(old_pages)
                cursor = 0
                if cow:
                    old = self._page(old_pages[-1])
                    new_pages[-1] = allocated[cursor]
                    cursor += 1
                    replacement = self._page(new_pages[-1])
                    for target, source, dim in zip(replacement.payload, old.payload, dims):
                        copy_kv(target, source, dim, 0, 0, tail)
                    replacement.valid = tail
                new_pages.extend(allocated[cursor:])
                for offset in range(sequence.length, length):
                    page_index, slot = divmod(offset, self.block_size)
                    page = self._page(new_pages[page_index])
                    for destination, source, dim in zip(page.payload, leaves, dims):
                        copy_kv(destination, source, dim, slot, offset - source_start, 1)
                    page.valid = max(page.valid, slot + 1)
            except BaseException:
                for ref in allocated:
                    page = self._pages[ref.index]
                    page.owners = 0
                    self._collect(page)
                raise
            if cow:
                old = self._page(old_pages[-1])
                old.owners -= 1
                self._collect(old)
            sequence.pages, sequence.length = new_pages, length
            self._signature, self._tree, self._dims = signature, tree, dims

    @contextmanager
    def read_pages(self, sequence):
        """Borrow zero-copy page views with a fixed logical length and generation."""
        with self._lock:
            self._check(sequence)
            refs, length, views = tuple(sequence.pages), sequence.length, []
            if len(refs) != math.ceil(length / self.block_size):
                raise StateError("Block table length differs from committed tokens")
            for index, ref in enumerate(refs):
                page = self._page(ref)
                count = min(self.block_size, length - index * self.block_size)
                if page.payload is None or page.valid < count:
                    raise StateError("Cannot expose an uninitialized page")
                views.append(
                    PageView(
                        index * self.block_size,
                        tuple(
                            tensor.narrow(dim, 0, count)
                            for tensor, dim in zip(page.payload, self._dims)
                        ),
                    )
                )

            for ref in refs:
                self._page(ref).readers += 1
        try:
            yield tuple(views)
        finally:
            with self._lock:
                for ref in refs:
                    page = self._page(ref)
                    page.readers -= 1
                    self._collect(page)

    def materialize(self, sequence):
        with self._lock:
            self._check(sequence)
            if not sequence.length:
                return None
            leaves = []
            for index, dim in enumerate(self._dims):
                parts = []
                for i, ref in enumerate(sequence.pages):
                    count = min(self.block_size, sequence.length - i * self.block_size)
                    page = self._page(ref)
                    if page.valid < count:
                        raise StateError("Uninitialized cache page")
                    part = page.payload[index].narrow(dim, 0, count)
                    parts.append(part.dequantize() if isinstance(part, QuantizedKV) else part)

                leaves.append(torch.cat(parts, dim=dim))
            return self.codec.unflatten(leaves, self._tree)

    def storage_metrics(self):
        """Count persistent tensor storage, including quantization scales; not allocator
        peaks or temporary forward activations."""
        with self._lock:
            leaves = [t for p in self._pages if p.payload is not None for t in p.payload]
            packed = [t for t in leaves if isinstance(t, QuantizedKV)]
            return {
                "format": self.quantization.format if self.quantization else "native",
                "allocated_tensor_bytes": sum(
                    t.nbytes if isinstance(t, QuantizedKV) else t.numel() * t.element_size()
                    for t in leaves
                ),
                "quantized_leaves": len(packed),
                "physical_pages": self.used_blocks,
            }

    def restore_pages(self, payloads, *, identity, length):
        """Reserve every destination page before transfer and publish only after copy
        completion; preserve archived codes without requantization."""
        sequence = self.create(identity)
        if (
            type(length) is not int
            or length < 1
            or len(payloads) != math.ceil(length / self.block_size)
        ):
            raise StateError("Archived page count and token length differ")
        with self._lock:
            if self._signature is None:
                raise StateError("Restore requires this initialized model pool")
            for index, leaves in enumerate(payloads):
                count = min(self.block_size, length - index * self.block_size)
                if len(leaves) != len(self._signature):
                    raise StateError("Archived leaf plan differs")
                for leaf, dim, (shape, dtype, device) in zip(leaves, self._dims, self._signature):
                    actual = tuple(leaf.shape[:dim]) + tuple(leaf.shape[dim + 1 :])
                    if actual != shape or leaf.dtype != dtype or leaf.shape[dim] != count:
                        raise StateError("Archived KV layout differs from model pool")
                    expected_quantized = self.quantization is not None and dtype.is_floating_point
                    if isinstance(leaf, QuantizedKV) != expected_quantized:
                        raise StateError("Archived low-bit format differs")
                    if expected_quantized and leaf.values.dtype != self.quantization.dtype:
                        raise StateError("Archived quantization format changed")
            free = [i for i, p in enumerate(self._pages) if p.owners == 0 and p.readers == 0]
            if len(free) < len(payloads):
                raise CacheCapacityError("Insufficient pages for complete restore")
            refs = []
            for index in free[: len(payloads)]:
                page = self._pages[index]
                page.generation += 1
                page.owners = 1
                page.valid = 0
                refs.append(PageRef(index, page.generation))
        devices = {signature[2] for signature in self._signature}
        try:
            restored = []
            for leaves in payloads:
                targets = []
                for leaf, dim, signature in zip(leaves, self._dims, self._signature):
                    transferred = clone_kv(leaf, device=signature[2])
                    target = allocate_kv_like(transferred, dim, self.block_size)
                    copy_kv(target, transferred, dim, 0, 0, leaf.shape[dim])
                    targets.append(target)
                restored.append(tuple(targets))
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            with self._lock:
                for i, (ref, leaves) in enumerate(zip(refs, restored)):
                    page = self._page(ref)
                    page.payload = leaves
                    page.valid = min(self.block_size, length - i * self.block_size)
                sequence.pages, sequence.length = refs, length
            return sequence
        except BaseException:
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            with self._lock:
                for ref in refs:
                    page = self._pages[ref.index]
                    page.owners = 0
                    self._collect(page)
            raise


@dataclass(frozen=True)
class PrefixIdentity:
    policy_artifact_id: str
    adapter: str = "none"
    processor: str = "none"
    position: str = "absolute_1d"
    multimodal_digest: str = "none"
    tenant: str = "local"

    def fingerprint(self):
        if any(not isinstance(value, str) or not value for value in self.__dict__.values()):
            raise ValueError("Prefix identity fields cannot be implicit or empty")
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


@dataclass(eq=False)
class _PrefixNode:
    key: tuple[int, ...] = ()
    pages: list[PageRef] = field(default_factory=list)
    parent: "_PrefixNode | None" = None
    children: dict = field(default_factory=dict)
    domain: str = ""


class PrefixCache:
    """Page-aligned compressed radix prefixes with edge ownership, splitting, and COW."""

    def __init__(self, pool, *, max_entries=128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Prefix cache needs positive integer capacity")
        self.pool, self.max_entries = pool, max_entries
        self._roots, self._lru = {}, OrderedDict()
        self._lock = threading.RLock()
        self._lookups = self._hits = self._matched_tokens = self._evictions = 0

    @staticmethod
    def _tokens(token_ids):
        values = tuple(token_ids)
        if any(type(x) is not int or x < 0 for x in values):
            raise ValueError("Prefix tokens must be explicit nonnegative integer IDs")
        return values

    def _touch(self, node):
        self._lru[node] = None
        self._lru.move_to_end(node)

    def _common(self, edge, tokens, offset, limit):
        end = min(len(edge), limit - offset)
        count = 0
        while count < end and edge[count] == tokens[offset + count]:
            count += 1
        return count // self.pool.block_size * self.pool.block_size

    def publish(self, identity, token_ids, sequence):
        tokens, domain = self._tokens(token_ids), identity.fingerprint()
        block = self.pool.block_size
        with self._lock, self.pool._lock:
            self.pool._check(sequence)
            if sequence.identity != domain:
                raise StateError("Prefix domain does not match state identity")
            length = min(len(tokens), sequence.length) // block * block
            if not length:
                return
            refs = list(sequence.pages[: length // block])

            if len(refs) != length // block or any(
                self.pool._page(ref).valid < block for ref in refs
            ):
                raise StateError("Cannot publish incomplete prefix pages")
            node = self._roots.setdefault(domain, _PrefixNode(domain=domain))
            offset = 0
            while offset < length:
                key = tokens[offset : offset + block]
                child = node.children.get(key)
                if child is None:
                    suffix = refs[offset // block :]
                    for ref in suffix:
                        self.pool._page(ref).owners += 1

                    if node.parent is not None and not node.children:
                        node.key += tokens[offset:length]
                        node.pages.extend(suffix)
                        self._touch(node)
                    else:
                        child = _PrefixNode(tokens[offset:length], suffix, node)
                        node.children[key] = child
                        self._touch(child)
                    break
                common = self._common(child.key, tokens, offset, length)
                if common < len(child.key):
                    cut = common // block
                    split = _PrefixNode(child.key[:common], child.pages[:cut], node)
                    child.key, child.pages = child.key[common:], child.pages[cut:]
                    child.parent = split
                    split.children[child.key[:block]] = child
                    node.children[key] = split
                    self._touch(split)
                    child = split
                self._touch(child)
                node, offset = child, offset + common
            while len(self._lru) > self.max_entries:
                self._evict_one_locked()

    def lookup(self, identity, token_ids, *, leave_last_token=True):
        tokens, domain = self._tokens(token_ids), identity.fingerprint()
        if type(leave_last_token) is not bool:
            raise ValueError("leave_last_token must be boolean")
        block = self.pool.block_size
        limit = max(0, len(tokens) - int(leave_last_token)) // block * block
        with self._lock, self.pool._lock:
            self._lookups += 1
            node, refs, offset = self._roots.get(domain), [], 0
            while node is not None and offset < limit:
                child = node.children.get(tokens[offset : offset + block])
                if child is None:
                    break
                common = self._common(child.key, tokens, offset, limit)
                refs.extend(child.pages[: common // block])
                self._touch(child)
                offset += common
                if common < len(child.key):
                    break
                node = child

            pages = [self.pool._page(ref) for ref in refs]
            for page in pages:
                page.owners += 1
            self._hits += bool(offset)
            self._matched_tokens += offset
            return PagedSequence(id(self.pool), self.pool.codec.kind, domain, refs, offset)

    def _evict_one_locked(self):

        leaf = next((node for node in self._lru if not node.children), None)
        if leaf is None:
            return False
        del self._lru[leaf]
        parent = leaf.parent
        del parent.children[leaf.key[: self.pool.block_size]]
        for ref in leaf.pages:
            page = self.pool._page(ref)
            page.owners -= 1
            self.pool._collect(page)
        if parent.parent is None and not parent.children:
            del self._roots[parent.domain]
        self._evictions += 1
        return True

    def evict_one(self):
        with self._lock, self.pool._lock:
            return self._evict_one_locked()

    def clear(self):
        with self._lock, self.pool._lock:
            while self._evict_one_locked():
                pass

    def stats(self):

        with self._lock:
            return dict(
                radix_nodes=len(self._lru),
                domains=len(self._roots),
                cached_page_references=sum(len(node.pages) for node in self._lru),
                stored_token_ids=sum(len(node.key) for node in self._lru),
                lookups=self._lookups,
                hits=self._hits,
                matched_tokens=self._matched_tokens,
                evictions=self._evictions,
            )
