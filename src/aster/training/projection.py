"""Persistent embedding-row projection across parameter, master-weight, and shard storage."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import torch
from torch import nn

from .portable import logical_tensors, gather_tensor, local_tensor, optimizer_mapping
from .sharding import ShardOptimizer, Zero3Unit


class EmbeddingProjectionRegistry:
    def __init__(self, engine):
        self.engine = engine
        self.policies = {}
        self.events = {}

    def describe(self):
        return [deepcopy(self.policies[key]) for key in sorted(self.policies)]

    def _entry(self, role, path):
        engine = self.engine
        if engine._busy or engine._failed:
            raise RuntimeError("Projection requires an idle successful phase boundary")
        if role not in engine.roles or not isinstance(path, str):
            raise ValueError("Projection role/path must identify a registered model")
        if any(
            getattr(engine.parallel.config, axis) != 1
            for axis in ("tensor_parallel", "pipeline_parallel", "context_parallel", "gtp_remat")
        ):
            raise ValueError(
                "Embedding projection currently supports DP only; TP/PP/CP/GTP row/column ownership needs a separate implementation"
            )
        model = engine.roles[role].model
        unit = model.get_submodule(path)
        original = unit.module if isinstance(unit, Zero3Unit) else unit
        if not isinstance(original, (nn.Embedding, nn.EmbeddingBag)):
            raise TypeError("Projection path must point to an explicit Embedding/EmbeddingBag leaf")
        if original.max_norm is not None:
            raise ValueError(
                "Explicit projection requires max_norm=None; never silently alter a user module"
            )
        name = path + ".weight" if path else "weight"
        entries = [entry for entry in logical_tensors(model, engine.parallel) if entry.name == name]
        if len(entries) != 1:
            raise ValueError("Projection must resolve exactly one logical weight")
        entry = entries[0]
        if (
            entry.shape != (original.num_embeddings, original.embedding_dim)
            or entry.tp_dimension is not None
        ):
            raise ValueError(
                "Projection requires an unpartitioned logical row/feature shape; TP weights are unsupported"
            )
        if entry.group.ranks != engine.parallel.dp.ranks or hasattr(
            entry.tensor, "_aster_extra_gradient_group"
        ):
            raise ValueError("Projection currently requires the default DP storage/gradient group")
        if entry.tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError(
                "Embedding projection requires a supported floating-point weight dtype"
            )
        return entry

    def register(self, role, path, *, max_norm, norm_type):
        error, policy = None, None
        try:
            self._entry(role, path)
            if (
                isinstance(max_norm, bool)
                or isinstance(norm_type, bool)
                or not math.isfinite(max_norm)
                or not math.isfinite(norm_type)
                or min(max_norm, norm_type) <= 0
            ):
                raise ValueError("Projection maximum and norm type must be finite positive numbers")
            key = (role, path)
            if key in self.policies:
                raise ValueError("Projection policy is already registered")
            if (
                "_embedding_projection" in self.engine.states
                and self.engine.states["_embedding_projection"] is not self
            ):
                raise ValueError("Reserved embedding projection state name is already occupied")
            policy = {
                "role": role,
                "path": path,
                "max_norm": float(max_norm),
                "norm_type": float(norm_type),
                "formula": "accessed_row_union_lpnorm_gt_max_scale_max_over_norm_plus_1e-7",
                "version": 1,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        declarations = self.engine.parallel.world.gather_objects((error, policy))
        if any(item[0] for item in declarations):
            raise ValueError(
                "Projection registration failed collectively: "
                + str([item[0] for item in declarations])
            )
        if any(item[1] != policy for item in declarations):
            raise ValueError("All ranks must register the same projection policy")
        self.policies[(role, path)] = policy
        self.events[(role, path)] = 0

    @torch.no_grad()
    def project(self, role, path, indices):
        error, local_indices, entry, policy = None, None, None, None
        try:
            entry = self._entry(role, path)
            key = (role, path)
            if key not in self.policies:
                raise ValueError("Register an explicit projection policy before applying it")
            policy = self.policies[key]
            if not isinstance(indices, torch.Tensor) or indices.dtype not in {
                torch.int32,
                torch.int64,
            }:
                raise ValueError(
                    "Projection indices must be an int32/int64 Tensor; empty tensors are allowed"
                )
            local_indices = indices.detach().flatten().unique().cpu().tolist()
            if any(index < 0 or index >= entry.shape[0] for index in local_indices):
                raise ValueError("Projection row index is outside the embedding vocabulary")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        declaration = (error, policy, None if error else self.events[(role, path)], local_indices)
        declarations = self.engine.parallel.world.gather_objects(declaration)
        if any(item[0] for item in declarations):
            raise ValueError(
                "Projection event failed collectively: " + str([item[0] for item in declarations])
            )
        if any(item[1:3] != declaration[1:3] for item in declarations):
            raise ValueError("Projection call order/policy/event counter differs across ranks")
        rows = sorted({index for item in declarations for index in item[3]})

        values = gather_tensor(entry.tensor, entry, self.engine.parallel)
        selected = values[rows].contiguous()
        signature = (
            str(selected.dtype),
            tuple(selected.shape),
            hashlib.sha256(selected.view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
        signatures = self.engine.parallel.world.gather_objects(signature)
        if any(item != signature for item in signatures):
            raise ValueError(
                "DP embedding replicas already differ before projection; refusing to hide an ownership error"
            )
        error = None
        try:
            if not torch.isfinite(selected).all():
                raise ValueError("Cannot project nonfinite embedding rows")
            accumulation = (
                selected.double() if selected.dtype == torch.float64 else selected.float()
            )
            norms = torch.linalg.vector_norm(accumulation, ord=policy["norm_type"], dim=1)
            changed = norms > policy["max_norm"]
            changed_rows = torch.tensor(rows, dtype=torch.long)[changed]
            if changed.any():
                projected = accumulation[changed] * (
                    policy["max_norm"] / (norms[changed] + 1e-7)
                ).unsqueeze(1)
                values[changed_rows] = projected.to(values.dtype)
            mask = torch.zeros(values.shape, dtype=torch.bool)
            mask[changed_rows] = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.engine._collective_error(error)
        error = None
        try:
            if len(changed_rows):
                self._copy_masked(entry.tensor, values, mask, entry)
                role_state = self.engine.roles[role]
                if role_state.trainable:
                    _, owners, sharded = optimizer_mapping(role_state)

                    owner = owners.get(id(entry.tensor))
                    if owner is not None and owner is not entry.tensor:
                        self._copy_masked(owner, values, mask, entry, optimizer_sharded=sharded)
                    if owner is not None and isinstance(role_state.optimizer, ShardOptimizer):
                        wrapper = role_state.optimizer
                        index = next(
                            i
                            for i, original in enumerate(wrapper.originals)
                            if original is entry.tensor
                        )
                        shard = wrapper.shards[index]
                        if shard is not owner:
                            self._copy_masked(shard, values, mask, entry, optimizer_sharded=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        try:
            self.engine._collective_error(error)
        except Exception:
            self.engine._failed = True
            raise
        self.events[(role, path)] += 1
        return {
            "rows": len(rows),
            "changed_rows": len(changed_rows),
            "event": self.events[(role, path)],
        }

    def _copy_masked(self, destination, values, mask, entry, *, optimizer_sharded=False):
        local_values = local_tensor(
            values, entry, self.engine.parallel, optimizer_sharded=optimizer_sharded
        ).to(destination.device, destination.dtype)
        local_mask = local_tensor(
            mask, entry, self.engine.parallel, optimizer_sharded=optimizer_sharded
        ).to(destination.device)
        destination.masked_scatter_(local_mask, local_values[local_mask])

    def state_dict(self):
        return {
            "policies": self.describe(),
            "events": [
                {"role": role, "path": path, "count": self.events[(role, path)]}
                for role, path in sorted(self.events)
            ],
        }

    def load_state_dict(self, state):
        if set(state) != {"policies", "events"} or state["policies"] != self.describe():
            raise ValueError("Embedding projection policy changed across resume")
        restored = {}
        for record in state["events"]:
            if (
                set(record) != {"role", "path", "count"}
                or type(record["count"]) is not int
                or record["count"] < 0
            ):
                raise ValueError("Invalid projection event checkpoint")
            key = (record["role"], record["path"])
            if key in restored:
                raise ValueError("Duplicate projection event checkpoint")
            restored[key] = record["count"]
        if restored.keys() != self.events.keys():
            raise ValueError("Projection checkpoint role/path layout differs")
        self.events = restored
