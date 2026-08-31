"""Lazy local datasets, deterministic shuffling, and resumable sampling."""

from __future__ import annotations
from array import array
import json
from pathlib import Path
import random
import numpy as np
import torch
from ..core import digest_json, file_digest


class JsonlDataset:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.offsets = array("Q")
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.fingerprint = file_digest(self.path)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            value = json.loads(stream.readline().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Every JSONL record must be an object")
        return value

    def verify(self):
        if file_digest(self.path) != self.fingerprint:
            raise ValueError("Dataset contents changed after indexing")


class TokenDataset:
    """Memory-map contiguous tokens and emit length+1 windows for objective-owned shifting."""

    def __init__(self, path, *, sequence_length, dtype="uint32"):
        if sequence_length < 2 or dtype not in {"uint16", "uint32"}:
            raise ValueError("Need a valid sequence length and unsigned token storage")
        self.path, self.sequence_length, self.dtype = (
            Path(path).resolve(strict=True),
            sequence_length,
            dtype,
        )
        if self.path.stat().st_size % np.dtype(dtype).itemsize:
            raise ValueError("Token file ends in a partial element")
        self.tokens = np.memmap(self.path, dtype=dtype, mode="r")
        self.fingerprint = digest_json(
            {"bytes": file_digest(path), "dtype": dtype, "sequence_length": sequence_length}
        )

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.sequence_length)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        return {
            "input_ids": self.tokens[start : start + self.sequence_length + 1]
            .astype(np.int64)
            .tolist()
        }


class StatefulSampler:
    """Create one global permutation before strided sharding to avoid overlapping rank shuffles."""

    def __init__(self, dataset, *, seed=0, rank=0, world_size=1, shuffle=True):
        if not 0 <= rank < world_size or len(dataset) < 1:
            raise ValueError("Invalid sampler topology or empty dataset")
        self.dataset, self.seed, self.rank, self.world_size, self.shuffle = (
            dataset,
            seed,
            rank,
            world_size,
            shuffle,
        )
        self.epoch = self.cursor = 0
        if rank >= len(dataset):
            raise ValueError(
                "This rank would have an empty data shard; choose explicit padding or fewer replicas"
            )
        self._cached_epoch, self._cached_indices = None, None

    def _indices(self):

        if self._cached_epoch == self.epoch:
            return self._cached_indices
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        self._cached_epoch, self._cached_indices = self.epoch, indices[self.rank :: self.world_size]
        return self._cached_indices

    def take(self, count):
        if count < 1:
            raise ValueError("Batch size must be positive")
        indices = self._indices()[self.cursor : self.cursor + count]
        self.cursor += len(indices)
        return [self.dataset[index] for index in indices]

    def next_epoch(self):
        if self.cursor != len(self._indices()):
            raise ValueError("Current shard has unconsumed data")
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return {
            "fingerprint": self.dataset.fingerprint,
            "length": len(self.dataset),
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected) or any(
            state[k] != expected[k] for k in expected if k not in {"epoch", "cursor"}
        ):
            raise ValueError("Sampler data/config/topology mismatch")
        if any(type(state[k]) is not int or state[k] < 0 for k in ("epoch", "cursor")) or state[
            "cursor"
        ] > len(self._indices()):
            raise ValueError("Invalid sampler cursor")
        if hasattr(self.dataset, "verify"):
            self.dataset.verify()
        self.epoch, self.cursor = state["epoch"], state["cursor"]


def causal_collate(records, *, pad_token_id=0, multiple_of=1, max_length=None):
    if not records or multiple_of < 1:
        raise ValueError("Need records and a positive padding multiple")
    lengths = [len(record["input_ids"]) for record in records]
    if min(lengths) < 2:
        raise ValueError("Causal samples require at least two tokens")
    width = (max(lengths) + multiple_of - 1) // multiple_of * multiple_of
    if max_length is not None and width > max_length:
        raise ValueError("Do not silently truncate a training answer")
    tokens = torch.full((len(records), width), pad_token_id, dtype=torch.long)
    labels, mask = torch.full_like(tokens, -100), torch.zeros_like(tokens)
    for row, record in enumerate(records):
        values, targets = record["input_ids"], record.get("labels", record["input_ids"])
        if len(values) != len(targets) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Invalid aligned token IDs")
        if any(type(v) is not int or (v < 0 and v != -100) for v in targets):
            raise ValueError("Only -100 is an ignored label")
        tokens[row, : len(values)] = torch.tensor(values)
        labels[row, : len(values)] = torch.tensor(targets)
        labels[row, 0] = -100
        mask[row, : len(values)] = 1
    return {"input_ids": tokens, "labels": labels, "attention_mask": mask}


def pack_documents(documents, *, length, eos_token_id=2):
    """Pack a continuous pretraining stream with explicit EOS, not document-isolated attention."""
    if length < 2:
        raise ValueError("Packing length must be at least two")
    pending = []
    for document in documents:
        values = list(document)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Documents must contain token IDs")
        pending.extend(values + [eos_token_id])
        while len(pending) > length:
            yield {"input_ids": pending[: length + 1]}

            pending = pending[length:]
    if len(pending) > 1:
        yield {"input_ids": pending}
