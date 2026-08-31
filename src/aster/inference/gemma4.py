"""Budgeted snapshot inference for Gemma4 shared-owner and sliding-window state."""

from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import threading

import torch

from ..core import StateCapabilities
from ..models.gemma4 import Gemma4ForCausalLM, Gemma4State
from ..models.gemma4_vl import Gemma4ForConditionalGeneration
from .runner import ModelRunner
from .state import CacheCapacityError, PrefixIdentity, StateError
from .task_runners import _PrivateRunner, _fingerprint, _tree_map


class Gemma4SnapshotCodec:
    kind = "gemma4_shared_kv"
    capabilities = StateCapabilities(kind, forkable=True, reorderable=True, replayable=True)

    def __init__(self, model):
        self.config, self.model_key = model.text_config, model.model_key

    def validate(self, state, length):
        c = self.config
        if (
            type(state) is not Gemma4State
            or state.kind != self.kind
            or state.model_key != self.model_key
            or state.seen_tokens != length
            or len(state.layers) != c.independent_layers
        ):
            raise StateError("Gemma4 snapshot has wrong owner count/model/history identity")
        size = 0
        for index, pair in enumerate(state.layers):
            local = c.layer_types[index] == "sliding_attention"
            heads = (
                c.num_global_key_value_heads
                if c.attention_k_eq_v and not local
                else c.num_key_value_heads
            )
            width = c.head_dim if local else c.global_head_dim
            history = min(length, c.sliding_window - 1) if local else length
            if len(pair) != 2:
                raise StateError("Each actual owner stores one K/V pair")
            for tensor in pair:
                if (
                    tensor.shape != (1, heads, history, width)
                    or not tensor.is_floating_point()
                    or not torch.isfinite(tensor).all()
                ):
                    raise StateError("Gemma4 owner local/global layout mismatch")
                size += tensor.numel() * tensor.element_size()
        return size


@dataclass
class _Snapshot:
    state: Gemma4State
    tokens: tuple[int, ...]
    size: int
    identity: str
    owners: int = 1
    readers: int = 0


@dataclass
class SnapshotSequence:
    owner: int
    identity: str
    snapshot: int | None = None
    length: int = 0
    released: bool = False


class Gemma4SnapshotPool:
    evidence_kind = "native_snapshot_storage_reference"
    block_size = 1

    def __init__(self, runner, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Snapshot storage needs a positive byte capacity")
        self.runner, self.codec, self.max_bytes = runner, runner.codec, max_bytes
        self._snapshots, self._next = {}, 0
        self._lock = threading.RLock()
        self.cow_commits = self.replay_rollbacks = 0

    @property
    def used_bytes(self):
        return sum(snapshot.size for snapshot in self._snapshots.values())

    @property
    def used_blocks(self):
        return len(self._snapshots)

    def _check(self, sequence):
        if (
            not isinstance(sequence, SnapshotSequence)
            or sequence.owner != id(self)
            or sequence.released
        ):
            raise StateError("Foreign/released Gemma4 sequence")
        if sequence.snapshot is not None and sequence.snapshot not in self._snapshots:
            raise StateError("Stale Gemma4 snapshot handle")
        if sequence.snapshot is None and sequence.length != 0:
            raise StateError("Empty snapshot cannot claim cached tokens")
        if sequence.snapshot is not None:
            snapshot = self._snapshots[sequence.snapshot]
            if snapshot.identity != sequence.identity or len(snapshot.tokens) != sequence.length:
                raise StateError("Snapshot domain/history metadata was changed")

    def create(self, identity):
        self.runner._retain_binding(identity)
        return SnapshotSequence(id(self), identity)

    def _collect(self, key):
        snapshot = self._snapshots[key]
        if snapshot.owners == 0 and snapshot.readers == 0:
            del self._snapshots[key]

    def fork(self, sequence):
        with self._lock:
            self._check(sequence)
            result = self.create(sequence.identity)
            result.snapshot, result.length = sequence.snapshot, sequence.length
            if sequence.snapshot is not None:
                self._snapshots[sequence.snapshot].owners += 1
            return result

    def tokens(self, sequence):
        self._check(sequence)
        return () if sequence.snapshot is None else self._snapshots[sequence.snapshot].tokens

    def materialize(self, sequence):
        with self._lock:
            self._check(sequence)
            return (
                None
                if sequence.snapshot is None
                else self._snapshots[sequence.snapshot].state.fork()
            )

    @contextmanager
    def borrow(self, sequence):
        with self._lock:
            self._check(sequence)
            key = sequence.snapshot
            if key is not None:
                self._snapshots[key].readers += 1
        try:
            yield sequence
        finally:
            with self._lock:
                if key is not None:
                    self._snapshots[key].readers -= 1
                    self._collect(key)

    def commit_batch(self, sequences, states, tokens):

        sizes = [self.codec.validate(state, len(history)) for state, history in zip(states, tokens)]
        if (
            len(sequences) != len(states)
            or len(states) != len(tokens)
            or len({id(s) for s in sequences}) != len(sequences)
        ):
            raise StateError("Snapshot batch must align unique sequences and states")
        with self._lock:
            for sequence in sequences:
                self._check(sequence)
            removed = {}
            for sequence in sequences:
                if sequence.snapshot is not None:
                    removed[sequence.snapshot] = removed.get(sequence.snapshot, 0) + 1
            reclaim = sum(
                self._snapshots[key].size
                for key, count in removed.items()
                if self._snapshots[key].owners == count and self._snapshots[key].readers == 0
            )
            if self.used_bytes - reclaim + sum(sizes) > self.max_bytes:
                raise CacheCapacityError("Gemma4 snapshot byte budget exhausted")

            new = [
                _Snapshot(state.fork(), tuple(history), size, sequence.identity)
                for sequence, state, history, size in zip(sequences, states, tokens, sizes)
            ]
            for sequence, snapshot in zip(sequences, new):
                old = sequence.snapshot
                if old is not None:
                    self.cow_commits += int(
                        self._snapshots[old].owners > 1 or self._snapshots[old].readers > 0
                    )
                    self._snapshots[old].owners -= 1
                    self._collect(old)
                key = self._next
                self._next += 1
                self._snapshots[key] = snapshot
                sequence.snapshot, sequence.length = key, len(snapshot.tokens)

    def truncate(self, sequence, length):
        self._check(sequence)
        if type(length) is not int or not 0 <= length <= sequence.length:
            raise ValueError("Invalid snapshot rollback length")
        if length == sequence.length:
            return
        if length == 0:
            with self._lock:
                key = sequence.snapshot
                if key is not None:
                    self._snapshots[key].owners -= 1
                    self._collect(key)
                sequence.snapshot, sequence.length = None, 0
            return
        history = self.tokens(sequence)[:length]

        output = self.runner._native_forward(sequence.identity, history, None)
        self.commit_batch([sequence], [output.state], [history])
        self.replay_rollbacks += 1

    def release(self, sequence):
        with self._lock:
            if sequence.released:
                return
            self._check(sequence)
            if sequence.snapshot is not None:
                key = sequence.snapshot
                self._snapshots[key].owners -= 1
                self._collect(key)
            sequence.snapshot, sequence.length, sequence.released = None, 0, True
            self.runner._release_binding(sequence.identity)


class Gemma4PrefixCache:
    def __init__(self, pool, *, max_entries=128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Invalid prefix capacity")
        self.pool, self.max_entries, self._entries = pool, max_entries, OrderedDict()

    def publish(self, identity, token_ids, sequence):
        domain = identity.fingerprint()
        if sequence.identity != domain:
            raise StateError("Prefix domain mismatch")
        if sequence.length > len(token_ids) or not sequence.length:
            return
        if self.pool.tokens(sequence) != tuple(token_ids[: sequence.length]):
            raise StateError("Prefix token labels differ from the state actually computed")
        end = sequence.length

        if end == len(token_ids) and self.pool.runner.safe_boundary(domain, end - 1):
            end -= 1
        if end < 1 or not self.pool.runner.safe_boundary(domain, end):
            return
        key = domain, tuple(token_ids[:end])
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        owned = self.pool.fork(sequence)
        try:
            self.pool.truncate(owned, end)
        except CacheCapacityError:
            self.pool.release(owned)

            return
        except BaseException:
            self.pool.release(owned)
            raise
        self._entries[key] = owned
        while len(self._entries) > self.max_entries:
            self.evict_one()

    def lookup(self, identity, token_ids, *, leave_last_token=True):
        domain, limit = identity.fingerprint(), len(token_ids) - int(leave_last_token)
        candidates = [
            key
            for key in self._entries
            if key[0] == domain
            and len(key[1]) <= limit
            and tuple(token_ids[: len(key[1])]) == key[1]
        ]
        if not candidates:
            return self.pool.create(domain)
        key = max(candidates, key=lambda value: len(value[1]))
        self._entries.move_to_end(key)
        return self.pool.fork(self._entries[key])

    def evict_one(self):
        if not self._entries:
            return False
        _, sequence = self._entries.popitem(last=False)
        self.pool.release(sequence)
        return True

    def clear(self):
        while self.evict_one():
            pass


class Gemma4SnapshotRunner(_PrivateRunner):
    kind = "token_predictor"
    sampling_prefix_transforms = (
        "exclude_declared_out_of_vocab_visual_placeholders_from_repetition_context",
    )
    decode, stream_text = ModelRunner.decode, ModelRunner.stream_text

    def __init__(
        self,
        model,
        *,
        policy_artifact_id,
        processor_id="none",
        tokenizer=None,
        max_cache_bytes=256 * 1024**2,
        max_modality_bytes=256 * 1024**2,
    ):
        if type(model) not in {Gemma4ForCausalLM, Gemma4ForConditionalGeneration}:
            raise TypeError("This codec/runner is only for native Gemma4 text/image/video")
        if (
            not isinstance(processor_id, str)
            or not processor_id
            or type(max_modality_bytes) is not int
            or max_modality_bytes < 1
        ):
            raise ValueError("Processor identity and modality memory capacity must be explicit")
        super().__init__(model, policy_artifact_id=policy_artifact_id)
        self.processor_id, self.tokenizer, self.chat_template = processor_id, tokenizer, None
        self.codec = Gemma4SnapshotCodec(self.model)
        self.pool = Gemma4SnapshotPool(self, max_cache_bytes)
        self.max_modality_bytes, self.modality_bytes = max_modality_bytes, 0
        self._bindings = {}
        self.position_protocol = "gemma4_shared_owner/" + self.model.model_key
        self.input_tokens_computed = self.forward_calls = 0
        self.model_execution_seconds = 0.0

    def create_prefix_cache(self, *, max_entries):
        return Gemma4PrefixCache(self.pool, max_entries=max_entries)

    def sampling_context_ids(self, tokens):

        special = (
            {self.model.config.image_token_id, self.model.config.video_token_id}
            if type(self.model) is Gemma4ForConditionalGeneration
            else set()
        )
        vocab = self.model.text_config.vocab_size
        if any(token >= vocab and token not in special for token in tokens):
            raise ValueError("Gemma4 context has an undeclared out-of-vocabulary token")
        return tuple(token for token in tokens if token < vocab)

    def prepare_request(self, prompt, identity, modality_inputs=None, *, max_prefill_tokens=None):
        if (
            identity.policy_artifact_id != self.policy_artifact_id
            or identity.processor not in {"none", self.processor_id}
            or identity.adapter != "none"
            or identity.position not in {"absolute_1d", self.position_protocol}
        ):
            raise StateError(
                "Gemma4 request has a foreign policy/processor/adapter/position contract"
            )
        allowed = {
            "pixel_values",
            "image_position_ids",
            "image_batch_indices",
            "pixel_values_videos",
            "video_position_ids",
            "video_batch_indices",
            "mm_token_type_ids",
        }
        supplied = dict(modality_inputs or {})
        if set(supplied) - allowed:
            raise ValueError("Unsupported Gemma4 modality input")
        if supplied and type(self.model) is not Gemma4ForConditionalGeneration:
            raise ValueError("Text-only Gemma4 cannot accept visual data")
        if any(not isinstance(value, torch.Tensor) for value in supplied.values()):
            raise ValueError("Modality inputs must be explicit tensors")
        layout = ()
        if type(self.model) is Gemma4ForConditionalGeneration:
            layout = tuple(
                (i, 1 if token == self.model.config.image_token_id else 2)
                for i, token in enumerate(prompt)
                if token in {self.model.config.image_token_id, self.model.config.video_token_id}
            )
        types = supplied.pop("mm_token_type_ids", None)
        expected = torch.zeros((1, len(prompt)), dtype=torch.long)
        for index, kind in layout:
            expected[0, index] = kind
        if types is not None and (
            types.dtype != torch.long or not torch.equal(types.cpu(), expected)
        ):
            raise ValueError("Gemma4 visual token layout differs from placeholder IDs")
        if bool(layout) != bool(supplied):
            raise ValueError("Visual placeholders and explicit media must be supplied together")
        if layout and self.processor_id == "none":
            raise ValueError("Visual processing must have an explicit immutable processor identity")
        minimum = max((index + 1 for index, _ in layout), default=1)
        if max_prefill_tokens is not None and minimum > max_prefill_tokens:
            raise ValueError(
                "Complete Gemma4 visual block exceeds the scheduler prefill token budget"
            )
        size = 0

        def freeze(value):
            nonlocal size
            if (
                value.layout != torch.strided
                or value.device.type == "meta"
                or (value.is_floating_point() and not torch.isfinite(value).all())
            ):
                raise ValueError("Invalid Gemma4 modality tensor")
            size += value.numel() * value.element_size()
            return value.detach().cpu().clone()

        media = _tree_map(supplied, freeze)
        digest = _fingerprint({"media": media, "visual_layout": layout}) if layout else "none"
        if identity.multimodal_digest not in {"none", digest}:
            raise StateError("Caller multimodal digest does not match actual tensors/layout")
        identity = replace(
            identity,
            processor=self.processor_id,
            position=self.position_protocol,
            multimodal_digest=digest,
        )
        domain = identity.fingerprint()
        with self._lock:
            if domain not in self._bindings:
                if self.modality_bytes + size > self.max_modality_bytes:
                    raise CacheCapacityError("Bound modality input byte budget exhausted")
                self._bindings[domain] = {
                    "media": media,
                    "layout": layout,
                    "minimum": minimum,
                    "size": size,
                    "refs": 0,
                }
                self.modality_bytes += size
            self._bindings[domain]["refs"] += 1
        return identity

    def _retain_binding(self, domain):
        with self._lock:
            if domain not in self._bindings:
                raise StateError("Request domain was not prepared or is released")
            self._bindings[domain]["refs"] += 1

    def _release_binding(self, domain):
        with self._lock:
            if domain not in self._bindings or self._bindings[domain]["refs"] < 1:
                raise StateError("Stale modality binding lease")
            self._bindings[domain]["refs"] -= 1
            if self._bindings[domain]["refs"] == 0:
                self.modality_bytes -= self._bindings.pop(domain)["size"]

    def release_request(self, identity):
        self._release_binding(identity.fingerprint())

    def create_sequence(self, prompt, *, identity=None, modality_inputs=None):
        identity = self.prepare_request(
            tuple(prompt), identity or PrefixIdentity(self.policy_artifact_id), modality_inputs
        )
        sequence = self.pool.create(identity.fingerprint())
        self.release_request(identity)
        return sequence, identity

    def safe_boundary(self, domain, length):
        binding = self._bindings[domain]
        return length > 0 and (not binding["layout"] or length >= binding["minimum"])

    def plan_chunk_length(self, sequence, context, proposed):
        if sequence.length:
            return proposed
        return max(proposed, self._bindings[sequence.identity]["minimum"])

    def _native_forward(self, domain, chunk, state, *, output_hidden_states=False):
        binding, start = self._bindings[domain], 0 if state is None else state.seen_tokens
        inputs = {}
        if state is None:
            if not self.safe_boundary(domain, len(chunk)):
                raise StateError("Cannot prefill/replay only part of a Gemma4 visual block")
            inputs = _tree_map(binding["media"], lambda tensor: tensor.to(self.device).clone())
            if binding["layout"]:
                types = torch.zeros((1, len(chunk)), device=self.device, dtype=torch.long)
                for index, kind in binding["layout"]:
                    expected_id = (
                        self.model.config.image_token_id
                        if kind == 1
                        else self.model.config.video_token_id
                    )
                    if chunk[index] != expected_id:
                        raise StateError("Visual layout does not match replayed tokens")
                    types[0, index] = kind
                inputs["mm_token_type_ids"] = types
        ids = torch.tensor([chunk], device=self.device, dtype=torch.long)
        if output_hidden_states:
            inputs["output_hidden_states"] = True
        output = self._call(
            self.model,
            input_ids=ids,
            state=state,
            use_cache=True,
            position_ids=torch.arange(start, start + len(chunk), device=self.device)[None],
            attention_mask=torch.ones(
                (1, start + len(chunk)), device=self.device, dtype=torch.bool
            ),
            **inputs,
        )
        self.forward_calls += 1
        self.input_tokens_computed += len(chunk)
        self.model_execution_seconds = self.model_seconds
        self.codec.validate(output.state, start + len(chunk))
        if (
            output.logits.shape[:2] != (1, len(chunk))
            or output.logits.ndim != 3
            or not torch.isfinite(output.logits).all()
        ):
            raise StateError("Gemma4 cached logits are invalid")
        return output

    def forward_batch(self, sequences, chunks, *, return_all_logits=False):
        return self._forward_batch(sequences, chunks, return_all_logits=return_all_logits)

    def forward_feature_batch(self, sequences, chunks, *, hidden_state_indices):
        if not hidden_state_indices or any(
            type(i) is not int or i < 0 for i in hidden_state_indices
        ):
            raise ValueError("Explicit nonnegative Gemma4 hidden-state indices are required")
        return self._forward_batch(
            sequences,
            chunks,
            return_all_logits=True,
            hidden_state_indices=tuple(hidden_state_indices),
        )

    def _forward_batch(self, sequences, chunks, *, return_all_logits, hidden_state_indices=None):
        if not sequences or len(sequences) != len(chunks) or any(not chunk for chunk in chunks):
            raise ValueError("Empty/misaligned Gemma4 batch")
        states, histories, logits, features = [], [], [], []
        for sequence, chunk in zip(sequences, chunks):
            self.pool._check(sequence)
            history = self.pool.tokens(sequence) + tuple(chunk)
            with self.pool.borrow(sequence):
                output = self._native_forward(
                    sequence.identity,
                    tuple(chunk),
                    self.pool.materialize(sequence),
                    output_hidden_states=hidden_state_indices is not None,
                )
                if hidden_state_indices is not None:
                    if (
                        output.hidden_states is None
                        or max(hidden_state_indices) >= len(output.hidden_states)
                        or any(
                            output.hidden_states[i].shape[:2] != (1, len(chunk))
                            for i in hidden_state_indices
                        )
                    ):
                        raise StateError("Gemma4 did not return aligned target features")
                    features.append(
                        torch.cat([output.hidden_states[i] for i in hidden_state_indices], -1)[
                            0
                        ].detach()
                    )
            states.append(output.state)
            histories.append(history)
            logits.append(
                (output.logits[0] if return_all_logits else output.logits[0, -1])
                .detach()
                .float()
                .cpu()
            )
        self.pool.commit_batch(sequences, states, histories)
        return list(zip(logits, features)) if hidden_state_indices is not None else logits
