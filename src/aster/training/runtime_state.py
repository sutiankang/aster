"""Declared non-persistent model semantics, separate from weights and inference caches."""

from collections.abc import Mapping

import torch


def runtime_buffers(model, *, require_finite=False):
    """Return live semantic-buffer views; clone explicitly when creating snapshots."""
    from aster.models.serialization import semantic_buffers

    values = semantic_buffers(model)
    for name, value in values.items():
        if value.layout != torch.strided or value.is_meta or value.requires_grad:
            raise ValueError(f"Semantic runtime buffer must be real, dense and detached: {name}")
        if require_finite and not torch.isfinite(value).all():
            raise ValueError(f"Semantic runtime buffer must be finite: {name}")
    return values


def runtime_descriptor(model):
    values = runtime_buffers(model)
    aliases = {}
    for name, value in values.items():
        aliases.setdefault(id(value), []).append(name)
    return {
        "buffers": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
            for name, value in sorted(values.items())
        },
        "aliases": sorted(sorted(names) for names in aliases.values() if len(names) > 1),
    }


def snapshot_runtime_state(model):
    return {
        "schema_version": 1,
        "semantic_buffers": {
            name: value.detach().cpu().clone()
            for name, value in runtime_buffers(model, require_finite=True).items()
        },
    }


def validate_runtime_state(model, state, *, strict_dtype=True):

    if (
        not isinstance(state, Mapping)
        or set(state) != {"schema_version", "semantic_buffers"}
        or type(state["schema_version"]) is not int
        or state["schema_version"] != 1
    ):
        raise ValueError("Invalid semantic runtime state schema")
    expected, saved = runtime_buffers(model), state["semantic_buffers"]
    if not isinstance(saved, Mapping) or set(saved) != set(expected):
        raise ValueError("Semantic runtime buffer names differ")
    aliases = {}
    for name, target in expected.items():
        value = saved[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != target.layout
            or value.is_meta
            or value.shape != target.shape
            or value.requires_grad
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Semantic runtime buffer shape/layout/value differs: {name}")
        if strict_dtype and value.dtype != target.dtype:
            raise ValueError(f"Semantic runtime buffer dtype differs: {name}")
        if not strict_dtype and (
            value.is_floating_point() != target.is_floating_point()
            or value.is_complex() != target.is_complex()
            or (
                not value.is_floating_point()
                and not value.is_complex()
                and value.dtype != target.dtype
            )
        ):
            raise ValueError(f"Semantic runtime buffer numeric kind differs: {name}")
        previous = aliases.setdefault(id(target), name)
        if previous != name and (
            saved[previous].dtype != value.dtype or not torch.equal(saved[previous], value)
        ):
            raise ValueError("Aliased semantic buffers have contradictory saved values")
    return expected


@torch.no_grad()
def restore_runtime_state(model, state):
    """Restore only after full-rank preflight without replacing tensor or alias identities."""
    expected = validate_runtime_state(model, state)
    for name, value in expected.items():
        value.copy_(state["semantic_buffers"][name].to(value.device))


@torch.no_grad()
def apply_runtime_state(model, state):

    if any(getattr(module, "_aster_training_owned", False) for module in model.modules()):
        raise ValueError(
            "Install runtime state on an independent deployment model, not a Trainer-owned role"
        )
    if any(
        hasattr(parameter, "_aster_tp_dimension") or hasattr(parameter, "_aster_ep_dimension")
        for parameter in model.parameters()
    ):
        raise ValueError(
            "Public runtime-state installation requires a complete dense deployment model"
        )
    expected = validate_runtime_state(model, state, strict_dtype=False)
    replacements = {}
    for name, target in expected.items():
        if id(target) not in replacements:
            replacements[id(target)] = (
                state["semantic_buffers"][name].detach().to(device=target.device).clone()
            )
    for name, target in expected.items():
        owner, _, attribute = name.rpartition(".")
        model.get_submodule(owner)._buffers[attribute] = replacements[id(target)]
    return model
