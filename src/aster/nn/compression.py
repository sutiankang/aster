"""Closed-window compression with overlapping CSA and non-overlapping HCA state."""

from dataclasses import dataclass
import torch
from aster.core import StateCapabilities


@dataclass(frozen=True)
class WindowCompressionState:
    entries: torch.Tensor
    pending_kv: torch.Tensor
    pending_gate: torch.Tensor
    overlap_kv: torch.Tensor | None = None
    overlap_gate: torch.Tensor | None = None

    def map(self, function):
        return type(self)(
            *(
                function(value) if value is not None else None
                for value in (
                    self.entries,
                    self.pending_kv,
                    self.pending_gate,
                    self.overlap_kv,
                    self.overlap_gate,
                )
            )
        )


@dataclass(frozen=True)
class CompressedLayerState:
    window_kv: torch.Tensor
    compressor: WindowCompressionState | None = None
    indexer: WindowCompressionState | None = None

    def map(self, function):
        return type(self)(
            function(self.window_kv),
            self.compressor.map(function) if self.compressor else None,
            self.indexer.map(function) if self.indexer else None,
        )


@dataclass(frozen=True)
class CompressedAttentionState:
    layers: tuple[CompressedLayerState, ...]
    seen_tokens: int
    model_key: str
    kind: str = "compressed_window_mqa"

    @property
    def capabilities(self):
        return StateCapabilities(
            self.kind, forkable=True, truncatable=False, reorderable=True, replayable=True
        )

    def fork(self):
        return type(self)(
            tuple(layer.map(torch.clone) for layer in self.layers), self.seen_tokens, self.model_key
        )

    def reorder(self, indices):
        return type(self)(
            tuple(layer.map(lambda x: x.index_select(0, indices)) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError(
            "Closed/overlapping compressed windows cannot be cut as dense KV; replay from a retained prefix"
        )


def compress_windows(
    kv, gate, position_bias, ratio, dimension, norm, rotate, *, overlap=False, previous=None
):
    """Return closed-window entries and only the state needed by the next chunk."""
    if kv.ndim != 3 or kv.shape != gate.shape or kv.shape[-1] != dimension * (2 if overlap else 1):
        raise ValueError("Invalid window projection layout")
    batch = kv.shape[0]
    completed = 0 if previous is None else previous.entries.shape[1]
    if previous is not None:
        kv = torch.cat((previous.pending_kv, kv), 1)
        gate = torch.cat((previous.pending_gate, gate), 1)
    usable = kv.shape[1] // ratio * ratio
    count = usable // ratio
    prior_kv = previous.overlap_kv if previous else None
    prior_gate = previous.overlap_gate if previous else None
    if count:
        values = kv[:, :usable].reshape(batch, count, ratio, -1)
        gates = gate[:, :usable].reshape(batch, count, ratio, -1) + position_bias
        if overlap:
            before_kv = (
                values.new_zeros(batch, 1, ratio, dimension)
                if prior_kv is None
                else prior_kv[:, None]
            )
            before_gate = (
                gates.new_full((batch, 1, ratio, dimension), -torch.inf)
                if prior_gate is None
                else prior_gate[:, None]
            )
            left_kv = torch.cat((before_kv, values[:, :-1, :, :dimension]), 1)
            left_gate = torch.cat((before_gate, gates[:, :-1, :, :dimension]), 1)
            pooled_values = torch.cat((left_kv, values[..., dimension:]), 2)
            pooled_gates = torch.cat((left_gate, gates[..., dimension:]), 2)
            prior_kv, prior_gate = (
                values[:, -1, :, :dimension].clone(),
                gates[:, -1, :, :dimension].clone(),
            )
        else:
            pooled_values, pooled_gates = values, gates
        new_entries = norm(
            (
                pooled_values * pooled_gates.softmax(2, dtype=torch.float32).to(pooled_values.dtype)
            ).sum(2)
        )
        positions = (torch.arange(count, device=kv.device) + completed) * ratio
        new_entries = rotate(new_entries[:, None], positions[None].expand(batch, -1))[:, 0]
    else:
        new_entries = kv.new_zeros(batch, 0, dimension)
    entries = new_entries if previous is None else torch.cat((previous.entries, new_entries), 1)
    return entries, WindowCompressionState(
        entries, kv[:, usable:], gate[:, usable:], prior_kv, prior_gate
    )
