"""Sequential truncated BPTT with checkpointed data, random streams, and hidden state."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import random

import torch
from torch import nn

from ..models.vmc import MDNRNN, MDNState
from .vmc import MDNRNNObjective


class VMCSequenceStream:
    """Shuffle episodes, concatenate global batch rows, and partition rows across DP;
    time remains sequential within each row."""

    def __init__(
        self,
        episodes,
        *,
        batch_size=100,
        sequence_length=500,
        seed=0,
        rank=0,
        world_size=1,
        shuffle=True,
    ):
        if (
            any(
                type(value) is not int
                for value in (batch_size, sequence_length, seed, rank, world_size)
            )
            or batch_size < 1
            or sequence_length < 2
            or seed < 0
            or world_size < 1
            or not 0 <= rank < world_size
            or batch_size < world_size
            or type(shuffle) is not bool
        ):
            raise ValueError("Invalid VMC stream dimensions/seed/DP lane partition")
        rows, mode, dimensions = [], None, None
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise TypeError("VMC episodes must be tensor mappings")
            keys = set(episode)
            current = "latents" if keys == {"latents", "actions"} else "distribution"
            if current == "distribution" and keys != {"mean", "logvar", "actions"}:
                raise ValueError(
                    "Episode requires latents/actions OR mean/logvar/actions; restart is generated per episode"
                )
            if mode is not None and mode != current:
                raise ValueError("Do not mix sampled and distribution episodes")
            mode = current
            copied = {}
            for key, value in episode.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 2
                    or min(value.shape) < 1
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        "VMC episode fields must be finite nonempty float [T,D] tensors"
                    )
                copied[key] = (
                    value.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
                )
            latent = copied["latents" if mode == "latents" else "mean"]
            if len(copied["actions"]) != len(latent) or (
                mode == "distribution" and copied["logvar"].shape != latent.shape
            ):
                raise ValueError("VMC episode lengths/distribution shapes differ")
            shape = (latent.shape[1], copied["actions"].shape[1])
            if dimensions is not None and shape != dimensions:
                raise ValueError("VMC episode feature dimensions differ")
            dimensions = shape
            copied["restart"] = torch.zeros(len(latent), dtype=torch.bool)
            copied["restart"][0] = True
            rows.append(copied)
        if not rows:
            raise ValueError("VMC stream requires at least one episode")
        total = sum(len(row["actions"]) for row in rows)
        self.num_chunks = total // (batch_size * sequence_length)
        if not self.num_chunks:
            raise ValueError("Not enough frames for one global B*T chunk")
        digest = hashlib.sha256()
        for row in rows:
            for key, value in sorted(row.items()):
                digest.update(json.dumps([key, list(value.shape), str(value.dtype)]).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
        self._identity = dict(
            schema_version=1,
            protocol="legacy_disjoint",
            dataset_sha256=digest.hexdigest(),
            episodes=len(rows),
            frames=total,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
            world_size=world_size,
            shuffle=shuffle,
            mode=mode,
            latent_size=dimensions[0],
            action_dim=dimensions[1],
            num_chunks=self.num_chunks,
            dropped_frames=total - self.num_chunks * batch_size * sequence_length,
            sampling="torch_cpu_fp32_per_chunk",
            storage="cpu_fp32",
        )
        self.rank, self._episodes = rank, rows

        self.lane_start = batch_size * rank // world_size
        self.lane_end = batch_size * (rank + 1) // world_size
        self._shuffle_rng = random.Random(seed)
        self._latent_rng = torch.Generator().manual_seed((seed + rank) % (2**63 - 1))
        self.epoch, self.cursor, self.order = 0, 0, list(range(len(rows)))
        if shuffle:
            self._shuffle_rng.shuffle(self.order)
        self._rows = self._materialize(self.order)

    @property
    def local_batch_size(self):
        return self.lane_end - self.lane_start

    @property
    def remaining(self):
        return self.num_chunks - self.cursor

    def config_dict(self):
        return deepcopy(self._identity)

    def _materialize(self, order):
        b, t = self._identity["batch_size"], self.num_chunks * self._identity["sequence_length"]
        return {
            key: torch.cat([self._episodes[index][key] for index in order])[: b * t].reshape(
                b, t, *value.shape[1:]
            )
            for key, value in self._episodes[0].items()
        }

    def preview(self, count=1, *, device="cpu"):
        """Build a candidate window without changing cursor or generator state; commit
        after a successful optimizer update."""
        if type(count) is not int or not 1 <= count <= self.remaining:
            raise ValueError("VMC stream has insufficient chunks for this accumulation window")
        generator = torch.Generator()
        generator.set_state(self._latent_rng.get_state())
        length, batches = self._identity["sequence_length"], []
        for index in range(self.cursor, self.cursor + count):
            batch = {
                key: value[
                    self.lane_start : self.lane_end, index * length : (index + 1) * length
                ].clone()
                for key, value in self._rows.items()
            }
            if self._identity["mode"] == "distribution":
                batch["latents"] = batch.pop("mean") + (
                    0.5 * batch.pop("logvar")
                ).exp() * torch.randn(
                    self.local_batch_size,
                    length,
                    self._identity["latent_size"],
                    generator=generator,
                )
                if not torch.isfinite(batch["latents"]).all():
                    raise ValueError("VMC latent sampling overflow")
            batches.append({key: value.to(device) for key, value in batch.items()})
        return batches, generator.get_state().clone()

    def _commit(self, count, generator_state):
        self._latent_rng.set_state(generator_state)
        self.cursor += count

    def state_dict(self):
        return dict(
            schema_version=1,
            identity=self.config_dict(),
            rank=self.rank,
            epoch=self.epoch,
            cursor=self.cursor,
            order=list(self.order),
            shuffle_rng=self._shuffle_rng.getstate(),
            latent_rng=self._latent_rng.get_state().clone(),
        )

    def _validated(self, state):
        if (
            not isinstance(state, Mapping)
            or set(state)
            != {
                "schema_version",
                "identity",
                "rank",
                "epoch",
                "cursor",
                "order",
                "shuffle_rng",
                "latent_rng",
            }
            or state["schema_version"] != 1
            or state["identity"] != self._identity
            or state["rank"] != self.rank
        ):
            raise ValueError("VMC stream checkpoint dataset/configuration/rank identity differs")
        if (
            type(state["epoch"]) is not int
            or state["epoch"] < 0
            or type(state["cursor"]) is not int
            or not 0 <= state["cursor"] <= self.num_chunks
            or not isinstance(state["order"], list)
            or any(type(index) is not int for index in state["order"])
            or sorted(state["order"]) != list(range(len(self._episodes)))
        ):
            raise ValueError("Invalid VMC stream epoch/cursor/permutation")
        shuffle_rng, latent_rng = random.Random(), torch.Generator()
        shuffle_rng.setstate(state["shuffle_rng"])
        latent_rng.set_state(state["latent_rng"].cpu())
        rows = self._materialize(state["order"])
        return shuffle_rng, latent_rng, rows

    def load_state_dict(self, state):
        shuffle_rng, latent_rng, rows = self._validated(state)
        self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], list(state["order"])
        self._shuffle_rng, self._latent_rng, self._rows = shuffle_rng, latent_rng, rows

    def _advance_epoch(self):

        order = list(self.order)
        generator = random.Random()
        generator.setstate(self._shuffle_rng.getstate())
        if self._identity["shuffle"]:
            generator.shuffle(order)
        rows = self._materialize(order)
        self.epoch += 1
        self.cursor = 0
        self.order, self._shuffle_rng, self._rows = order, generator, rows


class _StreamObjective(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method

    def config_dict(self):
        return deepcopy(self.method._contract)

    def preflight_microbatches(self, model, batches):
        return self.method.objective.preflight_microbatches(model, batches)

    def forward(self, model, batch):
        owner = self.method
        if not owner._running:
            raise RuntimeError("Stream objective is only valid inside MDNStreamMethod.step")
        owner._executed += 1
        terms, state = owner.objective.loss_and_state(model, batch, state=owner._working_state)

        owner._working_state = state.detach()
        return terms


class MDNStreamMethod:
    """Advance sequential chunks through shared DP/ZeRO training and checkpoint state."""

    def __init__(
        self, engine, stream: VMCSequenceStream, *, restart_factor=10.0, state_name="vmc_stream"
    ):
        error, declaration = None, None
        try:
            if engine._busy or engine._failed:
                raise RuntimeError("VMC stream requires a successful idle Trainer")
            if type(engine.model) is not MDNRNN or not isinstance(stream, VMCSequenceStream):
                raise TypeError("VMC stream requires the native MDNRNN and VMCSequenceStream")
            if any(
                getattr(engine.parallel.config, key) != 1
                for key in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "gtp_remat",
                    "expert_parallel",
                    "expert_tensor_parallel",
                )
            ):
                raise ValueError(
                    "VMC sequential TBPTT supports DP only; TP/PP/CP/GTP/EP are not implemented"
                )
            if (
                stream._identity["world_size"] != engine.parallel.dp.size
                or stream.rank != engine.parallel.dp.rank
            ):
                raise ValueError("VMC stream DP lane partition differs from Trainer")
            if (stream._identity["latent_size"], stream._identity["action_dim"]) != (
                engine.model.config.latent_size,
                engine.model.config.action_dim,
            ):
                raise ValueError("VMC stream features differ from MDNRNN configuration")
            if stream.cursor != 0 or stream.epoch != 0:
                raise ValueError(
                    "Attach a fresh stream, then restore the complete Trainer checkpoint"
                )
            if not isinstance(state_name, str) or not state_name or state_name in engine.states:
                raise ValueError("VMC registered state name must be nonempty and unique")
            objective = MDNRNNObjective(
                sequence_length=stream._identity["sequence_length"], restart_factor=restart_factor
            )
            declaration = dict(
                type="vmc_stream_method",
                version=1,
                stream=stream.config_dict(),
                objective=objective.config_dict(),
                model=engine.model.config.to_dict(),
                accumulation=engine.accumulation_steps,
                state_name=state_name,
                chunk_boundary="legacy_disjoint_T_frames_T_minus_1_inputs",
                truncation="detach_every_chunk",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective(engine, error, declaration)
        self.engine, self.stream, self.objective = engine, stream, objective
        self._contract, self._objective = declaration, _StreamObjective(self)
        self._initial_role_updates = engine.roles["model"].updates
        self.updates = self.dropped_chunks = 0
        self.state = self._working_state = None
        self._running = self._incomplete = False
        self._executed = 0
        engine.register_state(state_name, self)

    @staticmethod
    def _collective(engine, error, declaration=None):
        records = engine.parallel.world.gather_objects((error, declaration))
        if any(record[0] for record in records) or any(
            record[1] != records[0][1] for record in records
        ):
            raise ValueError(f"VMC stream collective preflight failed: {records}")

    def _check(self):
        if self._running or self._incomplete or self.engine._busy or self.engine._failed:
            raise RuntimeError(
                "VMC stream update incomplete/busy; restore last complete checkpoint"
            )
        if (
            self.stream.config_dict() != self._contract["stream"]
            or self.objective.config_dict() != self._contract["objective"]
            or self.engine.model.config.to_dict() != self._contract["model"]
            or self.engine.accumulation_steps != self._contract["accumulation"]
        ):
            raise ValueError("VMC stream settings changed outside the declared identity")
        if self.engine.roles["model"].updates != self._initial_role_updates + self.updates:
            raise ValueError("MDNRNN optimizer clock changed outside its stream method")
        if (self.state is None) != (self.stream.cursor == 0):
            raise ValueError("VMC stream carry/cursor boundary differs")
        if self.state is not None:
            self._validate_carry(self.state)

    def _validate_carry(self, state):
        shape = (self.stream.local_batch_size, self.engine.model.config.hidden_size)
        if not isinstance(state, MDNState) or state.config_key != self.engine.model.config_key:
            raise ValueError("Invalid VMC stream carry configuration")
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
            or value.device != self.engine.device
            or value.requires_grad
            or not torch.isfinite(value).all()
            for value in (state.cell, state.hidden)
        ):
            raise ValueError(
                "VMC stream carry must be finite detached FP32 tensors on the Trainer device"
            )

    def _declaration(self):
        return (
            self._contract,
            self.stream.epoch,
            self.stream.cursor,
            self.stream.order,
            self.updates,
            self.dropped_chunks,
        )

    def step(self):
        error, batches, generator_state = None, None, None
        try:
            self._check()
            batches, generator_state = self.stream.preview(
                self.engine.accumulation_steps, device=self.engine.device
            )
            self.objective.preflight_microbatches(self.engine.model, batches)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective(self.engine, error, self._declaration())
        self._working_state = None if self.state is None else self.state.fork().detach()
        self._executed = 0
        self._running = True
        try:
            result = self.engine.phase("vmc_tbptt", objective=self._objective, microbatches=batches)
            if not result.updated:
                raise RuntimeError(
                    "VMC stream optimizer update skipped; restore last complete checkpoint"
                )
            error = None
            try:
                if self._executed != len(batches):
                    raise RuntimeError("VMC stream did not execute every declared chunk")
                self._validate_carry(self._working_state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._collective(self.engine, error, self._executed)
            self.stream._commit(len(batches), generator_state)
            self.state = self._working_state.detach()
            self.updates += 1
            return result
        except BaseException:
            if self._executed or self.engine._failed:
                self._incomplete = True
                self.engine._failed = True
            raise
        finally:
            self._running = False
            self._working_state = None

    def advance_epoch(self, *, drop_remaining=False):
        error = None
        try:
            self._check()
            if type(drop_remaining) is not bool:
                raise ValueError("drop_remaining must be explicit bool")
            if self.stream.remaining and not drop_remaining:
                raise ValueError("Unconsumed chunks require explicit drop_remaining=True")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective(self.engine, error, (self._declaration(), drop_remaining))
        self.dropped_chunks += self.stream.remaining
        self.stream._advance_epoch()
        self.state = None

    def state_dict(self):
        self._check()
        state = (
            None
            if self.state is None
            else dict(
                cell=self.state.cell.detach().cpu().clone(),
                hidden=self.state.hidden.detach().cpu().clone(),
                config_key=self.state.config_key,
            )
        )
        return dict(
            schema_version=1,
            contract=deepcopy(self._contract),
            stream=self.stream.state_dict(),
            carry=state,
            updates=self.updates,
            initial_role_updates=self._initial_role_updates,
            dropped_chunks=self.dropped_chunks,
        )

    def load_state_dict(self, state):
        expected = {
            "schema_version",
            "contract",
            "stream",
            "carry",
            "updates",
            "initial_role_updates",
            "dropped_chunks",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state["schema_version"] != 1
            or state["contract"] != self._contract
        ):
            raise ValueError("VMC stream method checkpoint identity differs")
        if any(
            type(state[key]) is not int or state[key] < 0
            for key in ("updates", "initial_role_updates", "dropped_chunks")
        ):
            raise ValueError("Invalid VMC stream update counters")
        if self.engine.roles["model"].updates != state["initial_role_updates"] + state["updates"]:
            raise ValueError("VMC stream checkpoint optimizer/state clocks differ")
        self.stream._validated(state["stream"])
        if (
            state["updates"] * self._contract["accumulation"] + state["dropped_chunks"]
            != state["stream"]["epoch"] * self.stream.num_chunks + state["stream"]["cursor"]
        ):
            raise ValueError("VMC stream consumed/dropped chunk clocks differ")
        carry = state["carry"]
        if (carry is None) != (state["stream"]["cursor"] == 0):
            raise ValueError("VMC stream carry/cursor boundary differs")
        if carry is not None:
            if (
                not isinstance(carry, Mapping)
                or set(carry) != {"cell", "hidden", "config_key"}
                or carry["config_key"] != self.engine.model.config_key
            ):
                raise ValueError("Invalid VMC stream hidden-state configuration")
            shape = (self.stream.local_batch_size, self.engine.model.config.hidden_size)
            if any(
                not isinstance(carry[key], torch.Tensor)
                or carry[key].shape != shape
                or carry[key].dtype != torch.float32
                or not torch.isfinite(carry[key]).all()
                for key in ("cell", "hidden")
            ):
                raise ValueError("Invalid VMC stream hidden-state values")
            carry = MDNState(
                carry["cell"].to(self.engine.device).detach().clone(),
                carry["hidden"].to(self.engine.device).detach().clone(),
                carry["config_key"],
            )
        self.stream.load_state_dict(state["stream"])
        self.state, self.updates = carry, state["updates"]
        self._initial_role_updates, self.dropped_chunks = (
            state["initial_role_updates"],
            state["dropped_chunks"],
        )
        self._running = self._incomplete = False
        self._working_state = None
        self._executed = 0
