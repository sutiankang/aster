"""Muon and auxiliary Adam with unique parameter ownership and explicit algorithm profiles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn


SOURCES = {
    "keller": {
        "repository": "https://github.com/KellerJordan/Muon",
        "commit": "f98f1cacc0263b04290753e32be8d498c1efc806",
        "path": "muon.py",
        "sha256": "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d",
        "license": "MIT",
    },
    "moonlight": {
        "repository": "https://github.com/MoonshotAI/Moonlight",
        "commit": "c2ad5b20c605086526a179d36901bfc41b52b44b",
        "path": "examples/toy_train.py",
        "sha256": "8df3ec6e2f2cd5af8aee59ffb48b2219f394da6f049d95d881c66a6d13d00874",
        "license": "MIT",
    },
}


@dataclass(frozen=True)
class MatrixLayout:
    """Bind logical matrix geometry to its real owner; rebuild communication groups
    rather than serializing process-group objects."""

    entry: object
    parallel: object
    optimizer_sharded: bool = False

    @property
    def shape(self):
        if self.entry.global_shape is not None:
            return self.entry.global_shape
        shape = list(self.entry.shape)
        if self.entry.tp_dimension is not None:
            shape[self.entry.tp_dimension] *= self.entry.tp_group.size
        return tuple(shape)

    def gather(self, value):
        from .portable import gather_tensor

        return gather_tensor(
            value, self.entry, self.parallel, optimizer_sharded=self.optimizer_sharded, to_cpu=False
        )

    def local(self, value):
        from .portable import local_tensor

        return local_tensor(
            value, self.entry, self.parallel, optimizer_sharded=self.optimizer_sharded
        )


def rebind_matrix_layout(original, owner, *, optimizer_sharded=None):
    """Carry explicit geometry through sharding/offload instead of inferring it from
    a flat local tensor."""
    layout = getattr(original, "_aster_muon_layout", None)
    if layout is not None:
        owner._aster_muon_layout = replace(
            layout,
            optimizer_sharded=layout.optimizer_sharded
            if optimizer_sharded is None
            else optimizer_sharded,
        )


def newton_schulz(gradient, *, steps=5, epsilon=1e-7):
    """Apply the declared BF16 fifth-order Newton-Schulz iteration.
    This is not exact SVD orthogonalization and does not force all singular values to 1."""
    if gradient.ndim < 2 or not gradient.is_floating_point() or gradient.layout != torch.strided:
        raise ValueError("Muon NS requires a dense floating matrix or explicit matrix batch")
    if (
        type(steps) is not int
        or not 1 <= steps < 100
        or type(epsilon) not in (int, float)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Invalid Muon NS steps/epsilon")
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite Muon momentum")
    with torch.autocast(gradient.device.type, enabled=False):
        value = gradient.bfloat16()
        transpose = value.shape[-2] > value.shape[-1]
        if transpose:
            value = value.mT
        value = value / (value.norm(dim=(-2, -1), keepdim=True) + epsilon)
        for _ in range(steps):
            gram = value @ value.mT
            polynomial = -4.7750 * gram + 2.0315 * gram @ gram
            value = 3.4445 * value + polynomial @ value
        return value.mT if transpose else value


def _groups(param_groups):
    if not isinstance(param_groups, (list, tuple)) or not param_groups:
        raise ValueError("Muon requires explicit nonempty parameter groups")
    groups, seen = [], set()
    for raw in param_groups:
        if (
            not isinstance(raw, Mapping)
            or type(raw.get("use_muon")) is not bool
            or raw.get("profile") not in SOURCES
        ):
            raise ValueError(
                "Every optimizer group must explicitly declare use_muon and keller/moonlight profile"
            )
        group = dict(raw)
        group["params"] = list(group.get("params", ()))
        if not group["params"]:
            raise ValueError("Empty Muon optimizer groups are not accepted")
        if any(
            not isinstance(p, nn.Parameter) or not p.is_floating_point() for p in group["params"]
        ):
            raise ValueError("Muon optimizer needs floating Parameters")
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise ValueError("A parameter has exactly one Muon/Adam optimizer group owner")
            seen.add(id(parameter))
        common = {
            "params",
            "param_names",
            "initial_lr",
            "use_muon",
            "profile",
            "lr",
            "weight_decay",
            "missing_grad",
            "source_commit",
            "source_sha256",
        }
        algorithm = (
            {"momentum", "nesterov", "ns_steps", "normalization_epsilon", "matrix_kind"}
            if group["use_muon"]
            else {"betas", "eps"}
        )
        if set(group) - (common | algorithm):
            raise ValueError(
                f"Unknown Muon optimizer group fields: {set(group) - (common | algorithm)}"
            )

        source = SOURCES[group["profile"]]
        for name, field in (("source_commit", "commit"), ("source_sha256", "sha256")):
            group.setdefault(name, source[field])
            if group[name] != source[field]:
                raise ValueError("Muon profile source pin does not match the locked implementation")
        moon = group["profile"] == "moonlight"
        group.setdefault("lr", 0.001 if moon else (0.02 if group["use_muon"] else 0.0003))
        group.setdefault("weight_decay", 0.1 if moon else 0.0)
        group.setdefault("missing_grad", "skip")
        if any(
            type(group[key]) not in (int, float) or not math.isfinite(group[key]) or group[key] < 0
            for key in ("lr", "weight_decay")
        ) or group["missing_grad"] not in {"skip", "zero"}:
            raise ValueError("Invalid Muon learning rate/decay/missing-gradient policy")
        if "param_names" in group and (
            len(group["param_names"]) != len(group["params"])
            or any(not isinstance(n, str) or not n for n in group["param_names"])
        ):
            raise ValueError("Optimizer param_names must align with unique logical Parameters")
        if group["use_muon"]:
            for key, default in dict(
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                normalization_epsilon=1e-7,
                matrix_kind="matrix",
            ).items():
                group.setdefault(key, default)
            if (
                type(group["momentum"]) not in (int, float)
                or not math.isfinite(group["momentum"])
                or not 0 <= group["momentum"] < 1
                or type(group["nesterov"]) is not bool
                or type(group["ns_steps"]) is not int
                or not 1 <= group["ns_steps"] < 100
                or type(group["normalization_epsilon"]) not in (int, float)
                or not math.isfinite(group["normalization_epsilon"])
                or group["normalization_epsilon"] <= 0
                or group["matrix_kind"] not in {"matrix", "conv2d", "batched"}
            ):
                raise ValueError("Invalid explicit Muon matrix/NS controls")
            if moon and group["matrix_kind"] != "matrix":
                raise ValueError("Locked Moonlight profile defines only 2D parameter matrices")
            for parameter in group["params"]:
                required = {"matrix": 2, "conv2d": 4, "batched": 3}[group["matrix_kind"]]
                layout = getattr(parameter, "_aster_muon_layout", None)
                shape = layout.shape if isinstance(layout, MatrixLayout) else parameter.shape
                if len(shape) != required:
                    raise ValueError(
                        "Muon parameter geometry differs from explicit matrix_kind; never orthogonalize a flat shard"
                    )
                if getattr(parameter, "_aster_tp_sharded", False) and layout is None:
                    raise ValueError(
                        "Standalone Muon needs a full matrix, not a local tensor shard"
                    )
        else:
            group.setdefault("betas", (0.9, 0.95))
            group.setdefault("eps", 1e-8 if moon else 1e-10)
            if (
                not isinstance(group["betas"], (list, tuple))
                or len(group["betas"]) != 2
                or any(
                    type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta < 1
                    for beta in group["betas"]
                )
                or type(group["eps"]) not in (int, float)
                or not math.isfinite(group["eps"])
                or group["eps"] <= 0
            ):
                raise ValueError("Invalid auxiliary Adam moments/epsilon")
            group["betas"] = tuple(group["betas"])
        groups.append(group)
    return groups


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Share one optimizer interface while assigning each parameter to exactly one algorithm."""

    def __init__(self, param_groups):
        super().__init__(_groups(param_groups), {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    if group["missing_grad"] == "skip":
                        continue
                    gradient = torch.zeros_like(parameter)
                if gradient.layout != torch.strided or not torch.isfinite(gradient).all():
                    raise FloatingPointError("Muon/Adam requires finite dense gradients")
                state = self.state[parameter]
                if group["use_muon"]:
                    momentum = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                    if group["profile"] == "keller":
                        momentum.lerp_(gradient, 1 - group["momentum"])
                        direction = (
                            gradient.lerp(momentum, group["momentum"])
                            if group["nesterov"]
                            else momentum
                        )
                    else:
                        momentum.mul_(group["momentum"]).add_(gradient)
                        direction = (
                            gradient.add(momentum, alpha=group["momentum"])
                            if group["nesterov"]
                            else momentum
                        )
                    layout = getattr(parameter, "_aster_muon_layout", None)
                    if layout is not None:
                        direction = layout.gather(direction)
                    full_shape = direction.shape
                    if group["matrix_kind"] == "conv2d":
                        direction = direction.reshape(direction.shape[0], -1)
                    update = newton_schulz(
                        direction, steps=group["ns_steps"], epsilon=group["normalization_epsilon"]
                    )
                    if group["profile"] == "keller":
                        update.mul_(math.sqrt(max(1.0, update.shape[-2] / update.shape[-1])))
                        learning_rate = group["lr"]
                    else:
                        learning_rate = group["lr"] * 0.2 * math.sqrt(max(update.shape[-2:]))
                    update = update.reshape(full_shape)
                    if layout is not None:
                        update = layout.local(update)
                    update = update.to(parameter.device).reshape_as(parameter)
                else:
                    first = state.setdefault("exp_avg", torch.zeros_like(parameter))
                    second = state.setdefault("exp_avg_sq", torch.zeros_like(parameter))
                    state["step"] = state.get("step", 0) + 1
                    step = state["step"]
                    beta1, beta2 = group["betas"]
                    first.lerp_(gradient, 1 - beta1)
                    second.lerp_(gradient.square(), 1 - beta2)
                    if group["profile"] == "keller":
                        update = (first / (1 - beta1**step)) / (
                            (second / (1 - beta2**step)).sqrt() + group["eps"]
                        )
                        learning_rate = group["lr"]
                    else:
                        update = first / (group["eps"] + second.sqrt())
                        learning_rate = group["lr"] / (
                            (1 - beta1**step) / math.sqrt(1 - beta2**step)
                        )
                if not torch.isfinite(update).all():
                    raise FloatingPointError("Muon/Adam produced a nonfinite update")
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-learning_rate)
        return loss


class MuonFactory:
    """Bind logical parameter names to actual post-sharding owners."""

    def __init__(self, param_groups):
        if not isinstance(param_groups, (list, tuple)) or not param_groups:
            raise ValueError("MuonFactory needs explicit named parameter groups")
        self.groups = deepcopy(param_groups)
        for group in self.groups:
            if (
                not isinstance(group, dict)
                or "params" in group
                or "param_names" in group
                or "names" not in group
            ):
                raise ValueError("MuonFactory groups use logical names, never parameter objects")
            names = group["names"]
            if (
                not isinstance(names, (tuple, list))
                or not names
                or any(not isinstance(n, str) or not n for n in names)
            ):
                raise ValueError("MuonFactory needs explicit nonempty logical FQNs")

    def __call__(self, parameters):
        raise RuntimeError(
            "MuonFactory requires the Trainer role/layout protocol; use MuonWithAuxAdam for standalone full parameters"
        )

    def build(self, model, parallel, parameters):
        from .portable import logical_tensors

        entries = {
            entry.name: entry
            for entry in logical_tensors(model, parallel)
            if entry.parameter and entry.tensor.requires_grad
        }
        if any(entry.ep_dimension is not None for entry in entries.values()):
            raise ValueError(
                "Muon expert/packed matrix geometry requires a separate explicit provider"
            )
        if any(
            getattr(parallel.config, axis, 1) != 1
            for axis in (
                "pipeline_parallel",
                "context_parallel",
                "expert_parallel",
                "expert_tensor_parallel",
                "gtp_remat",
            )
        ):
            raise ValueError(
                "Current MuonFactory profile supports TP x DP x ZeRO only; PP/CP/EP/ETP/GTP remain explicit followups"
            )
        groups = []
        for specification in self.groups:
            group = deepcopy(specification)
            names = group.pop("names")
            if any(name not in entries for name in names):
                raise ValueError("MuonFactory names a nonexistent logical parameter")
            group["params"] = [entries[name].tensor for name in names]
            groups.append(group)
            for name in names:
                entries[name].tensor._aster_muon_layout = MatrixLayout(entries[name], parallel)
        actual = [p for group in groups for p in group["params"]]
        if {id(p) for p in actual} != {id(p) for p in parameters}:
            raise ValueError("MuonFactory groups must cover the entire trainable role")
        return MuonWithAuxAdam(groups)

    @staticmethod
    def declaration(optimizer, model, parallel, owners=None):

        from .portable import logical_tensors

        normalized = _groups(optimizer.param_groups)
        for actual, expected in zip(optimizer.param_groups, normalized):
            if set(actual) != set(expected):
                raise ValueError("Muon runtime group lost a required configuration field")
        options = {
            id(parameter): {
                key: value for key, value in group.items() if key not in {"params", "param_names"}
            }
            for group in normalized
            for parameter in group["params"]
        }
        entries = logical_tensors(model, parallel)
        owners = (
            owners
            if owners is not None
            else {id(entry.tensor): entry.tensor for entry in entries if entry.parameter}
        )
        declaration = {
            entry.name: options[id(owners[id(entry.tensor)])]
            for entry in entries
            if entry.parameter and entry.tensor.requires_grad
        }
        canonical = {}
        for entry in entries:
            if entry.parameter and id(entry.tensor) in owners:
                canonical.setdefault(id(owners[id(entry.tensor)]), entry.name)

        order = tuple(
            canonical[id(parameter)]
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        return declaration, order

    def validate_parallel(self, optimizer, model, parallel):
        copies = parallel.world.gather_objects(self.declaration(optimizer, model, parallel))
        by_name = {}
        if any(copy[1] != copies[0][1] for copy in copies):
            raise ValueError("Muon parameter collective order differs across replicas")
        for copy, _ in copies:
            for name, configuration in copy.items():
                if name in by_name and configuration != by_name[name]:
                    raise ValueError("Muon parameter profile/options differ across replicas")
                by_name[name] = configuration

    @classmethod
    def from_model(
        cls, model, *, auxiliary_modules, profile, muon_options=None, auxiliary_options=None
    ):

        if getattr(model, "_aster_training_owned", False):
            raise ValueError("Select Muon parameter ownership before creating the Trainer")
        if not isinstance(auxiliary_modules, tuple) or any(
            not isinstance(name, str) for name in auxiliary_modules
        ):
            raise ValueError(
                "Auxiliary module paths must be an explicit tuple (output heads included)"
            )
        excluded = set()
        for path in auxiliary_modules:
            try:
                child = model.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"Unknown auxiliary module path: {path}") from exc
            excluded.update(id(p) for p in child.parameters())
        for module in model.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                excluded.update(id(p) for p in module.parameters())
        names = {True: [], False: []}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                names[parameter.ndim == 2 and id(parameter) not in excluded].append(name)
        groups = []
        for use_muon, options in ((True, muon_options), (False, auxiliary_options)):
            if options is not None and not isinstance(options, Mapping):
                raise TypeError("Muon options must be explicit mappings")
            if options is not None and set(options) & {"names", "params", "use_muon", "profile"}:
                raise ValueError("Options cannot overwrite parameter ownership/profile")
            if names[use_muon]:
                groups.append(
                    dict(
                        names=names[use_muon], use_muon=use_muon, profile=profile, **(options or {})
                    )
                )
        return cls(groups)
