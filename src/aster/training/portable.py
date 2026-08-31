"""Logical-parameter export and optimizer resharding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any
import math
from copy import deepcopy

import torch
from torch import nn
import torch.distributed as dist

from .parallel import Group, all_gather_flat
from .sharding import ShardOptimizer, Zero3Unit
from .offload import CPUOptimizer
from .muon import MuonWithAuxAdam


@dataclass
class LogicalTensor:
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    group: Group
    dp_sharded: bool
    tp_dimension: int | None
    parameter: bool
    storage_name: str
    persistent: bool = True
    global_shape: tuple[int, ...] | None = None
    ep_dimension: int | None = None
    ep_group: Group | None = None
    tp_group: Group | None = None
    tp_stripes: int = 1
    semantic: bool = False


def logical_tensors(model, context) -> list[LogicalTensor]:
    result = []
    renamed_dense = []

    def rename_parameters(module, prefix, start):

        owner = module.module if isinstance(module, Zero3Unit) else module
        mapping = getattr(owner, "_aster_parameter_key_map", None)
        if mapping is None:
            return
        if not isinstance(mapping, Mapping):
            raise TypeError("Parameter key map must be an explicit mapping")

        def valid_name(name):
            return (
                isinstance(name, str)
                and bool(name)
                and all(name.split("."))
                and "/" not in name
                and "\\" not in name
            )

        if any(not valid_name(key) or not valid_name(value) for key, value in mapping.items()):
            raise ValueError("Parameter key map requires nonempty relative dotted names")
        entries = result[start:]
        by_name = {entry.name: entry for entry in entries}
        if len(by_name) != len(entries):
            raise ValueError("Duplicate logical parameter names before mapping")
        missing = [key for key in mapping if prefix + key not in by_name]
        if missing:
            raise ValueError(f"Parameter key map names nonexistent logical tensors: {missing}")
        if any(not by_name[prefix + key].parameter for key in mapping):
            raise ValueError("Parameter key map cannot rename buffers")
        final_names = [
            prefix + mapping.get(entry.name[len(prefix) :], entry.name[len(prefix) :])
            for entry in entries
        ]
        if len(set(final_names)) != len(final_names):
            raise ValueError(
                "Parameter key map collision; aliases retain distinct public names and one storage owner"
            )
        for entry, final_name in zip(entries, final_names):
            if entry.name == final_name:
                continue
            entry.name = final_name
            if not entry.dp_sharded:
                entry.storage_name = final_name
                renamed_dense.append(entry)

    def visit(module, prefix):
        start = len(result)
        if isinstance(module, Zero3Unit):
            for index, (name, shape, shard) in enumerate(
                zip(module.names, module.shapes, module.shards)
            ):
                result.append(
                    LogicalTensor(
                        prefix + name,
                        shard,
                        shape,
                        module.group,
                        True,
                        getattr(shard, "_aster_tp_dimension", None),
                        True,
                        prefix + f"shards.{index}",
                    )
                )
            rename_parameters(module, prefix, start)
            return
        for name, parameter in module.named_parameters(recurse=False):
            result.append(
                LogicalTensor(
                    prefix + name,
                    parameter,
                    tuple(parameter.shape),
                    getattr(parameter, "_aster_gradient_group", context.dp_cp_gtp),
                    False,
                    getattr(parameter, "_aster_tp_dimension", None),
                    True,
                    prefix + name,
                )
            )
        semantic = getattr(module, "_aster_semantic_buffers", ())
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            result.append(
                LogicalTensor(
                    prefix + name,
                    buffer,
                    tuple(buffer.shape),
                    context.dp_cp_gtp,
                    False,
                    getattr(buffer, "_aster_tp_dimension", None),
                    False,
                    prefix + name,
                    name not in module._non_persistent_buffers_set,
                    semantic=name in semantic,
                )
            )
        for name, child in module.named_children():
            visit(child, prefix + name + ".")
        for alias, target in getattr(module, "_aster_zero3_parameter_aliases", {}).items():
            found = [entry for entry in result if entry.name == prefix + target]
            if len(found) != 1:
                raise ValueError(
                    "ZeRO3 container alias must identify one existing logical leaf parameter"
                )

            result.append(replace(found[0], name=prefix + alias))
        rename_parameters(module, prefix, start)

    visit(model, "")
    for entry in result:
        tensor_group = getattr(entry.tensor, "_aster_tp_group", context.tp)
        stripes = getattr(entry.tensor, "_aster_tp_stripes", 1)
        if (
            not isinstance(tensor_group, Group)
            or tensor_group.ranks not in {context.tp.ranks, context.etp.ranks}
            or type(stripes) is not int
            or stripes < 1
        ):
            raise ValueError(
                "Tensor layout requires a declared attention TP/ETP group and positive stripe count"
            )
        if entry.tp_dimension is not None:
            if (
                not 0 <= entry.tp_dimension < len(entry.shape)
                or entry.shape[entry.tp_dimension] % stripes
            ):
                raise ValueError("Tensor stripe axis must divide into equal local stripes")
            entry.tp_group, entry.tp_stripes = tensor_group, stripes
        elif stripes != 1 or hasattr(entry.tensor, "_aster_tp_group"):
            raise ValueError("Tensor group/stripes require an explicit tensor shard axis")
        expert_axis = getattr(entry.tensor, "_aster_ep_dimension", None)
        expert_group = getattr(entry.tensor, "_aster_ep_group", None)
        if expert_axis is not None or expert_group is not None:
            if (
                not entry.parameter
                or type(expert_axis) is not int
                or not 0 <= expert_axis < len(entry.shape)
                or not isinstance(expert_group, Group)
                or expert_group.ranks != context.ep.ranks
            ):
                raise ValueError(
                    "Expert storage requires an explicit valid axis and matching EP group"
                )
            if entry.tp_dimension == expert_axis:
                raise ValueError("Expert and tensor shards require distinct axes")
            entry.ep_dimension, entry.ep_group = expert_axis, expert_group
        shape = getattr(entry.tensor, "_aster_tp_global_shape", None)
        if shape is not None:
            if (
                entry.tp_dimension is None
                or not isinstance(shape, tuple)
                or len(shape) != len(entry.shape)
                or any(type(size) is not int or size < 1 for size in shape)
            ):
                raise ValueError(
                    "Explicit TP global shape requires positive dimensions and a shard axis"
                )
            axis = entry.tp_dimension
            if entry.ep_group is not None or entry.tp_stripes != 1:
                raise ValueError(
                    "Ceil-padded global shapes cannot silently combine with expert/striped layouts"
                )
            if (
                any(shape[i] != entry.shape[i] for i in range(len(shape)) if i != axis)
                or math.ceil(shape[axis] / entry.tp_group.size) != entry.shape[axis]
            ):
                raise ValueError("TP global shape must match its ceil-padded local storage")
            entry.global_shape = shape
    if renamed_dense:
        storage_keys = set(model.state_dict())
        if any(entry.storage_name not in storage_keys for entry in renamed_dense):
            raise ValueError(
                "Parameter key map requires a matching public state_dict codec for dense EMA/storage"
            )
    if context.pp.size > 1:
        names = getattr(model, "parameter_names", None)
        if names is None:
            raise ValueError("PP 导出必须提供原模型 parameter_names 映射，不能猜 stage 的全局名称")
        if set(names) != {entry.name for entry in result}:
            raise ValueError("PP parameter_names 必须完整覆盖参数与 buffer")
        for entry in result:
            entry.name = names[entry.name]
    if len({entry.name for entry in result}) != len(result):
        raise ValueError("Duplicate final logical parameter names")
    return result


def gather_tensor(tensor, entry, context, *, optimizer_sharded=False, to_cpu=True):
    value = tensor.detach()
    communication_device = getattr(entry.tensor, "_aster_compute_device", entry.tensor.device)
    if entry.dp_sharded or optimizer_sharded:
        value = all_gather_flat(value.to(communication_device).flatten(), entry.group)[
            : math.prod(entry.shape)
        ].reshape(entry.shape)
    if entry.tp_dimension is not None and entry.tp_group.size > 1:
        value = value.to(communication_device).contiguous()
        outputs = [torch.empty_like(value) for _ in entry.tp_group.ranks]
        dist.all_gather(outputs, value, group=entry.tp_group.handle)

        striped = [piece.chunk(entry.tp_stripes, dim=entry.tp_dimension) for piece in outputs]
        value = torch.cat(
            [
                torch.cat([pieces[index] for pieces in striped], dim=entry.tp_dimension)
                for index in range(entry.tp_stripes)
            ],
            dim=entry.tp_dimension,
        )
    if entry.ep_group is not None and entry.ep_group.size > 1:
        value = value.to(communication_device).contiguous()
        outputs = [torch.empty_like(value) for _ in entry.ep_group.ranks]
        dist.all_gather(outputs, value, group=entry.ep_group.handle)
        value = torch.cat(outputs, dim=entry.ep_dimension)
    if entry.global_shape is not None:
        value = value.narrow(entry.tp_dimension, 0, entry.global_shape[entry.tp_dimension])
    return value.cpu().clone() if to_cpu else value.clone()


def local_tensor(global_tensor, entry, context, *, optimizer_sharded=False):
    value = global_tensor
    if entry.ep_group is not None:
        if value.shape[entry.ep_dimension] != entry.shape[entry.ep_dimension] * entry.ep_group.size:
            raise ValueError(f"Expert global storage shape mismatch: {entry.name}")
        value = value.chunk(entry.ep_group.size, dim=entry.ep_dimension)[entry.ep_group.rank]
    if entry.tp_dimension is not None:
        if entry.global_shape is not None:
            if tuple(value.shape) != entry.global_shape:
                raise ValueError(f"Unpadded global parameter shape mismatch: {entry.name}")
            shape = list(entry.global_shape)
            shape[entry.tp_dimension] = entry.shape[entry.tp_dimension] * entry.tp_group.size
            padded = value.new_zeros(shape)
            padded.narrow(entry.tp_dimension, 0, value.shape[entry.tp_dimension]).copy_(value)
            value = padded
        if value.shape[entry.tp_dimension] % (entry.tp_group.size * entry.tp_stripes):
            raise ValueError("新 TP/ETP 无法整除全局参数stripe轴")
        value = torch.cat(
            [
                stripe.chunk(entry.tp_group.size, dim=entry.tp_dimension)[entry.tp_group.rank]
                for stripe in value.chunk(entry.tp_stripes, dim=entry.tp_dimension)
            ],
            dim=entry.tp_dimension,
        )
    if tuple(value.shape) != entry.shape:
        raise ValueError(f"逻辑参数形状不匹配: {entry.name}")
    if entry.dp_sharded or optimizer_sharded:
        width = math.ceil(value.numel() / entry.group.size)
        padded = torch.zeros(width * entry.group.size, dtype=value.dtype, device=value.device)
        padded[: value.numel()].copy_(value.flatten())
        value = padded[entry.group.rank * width : (entry.group.rank + 1) * width]
    return value


def optimizer_mapping(role):
    wrapper = role.optimizer
    if isinstance(wrapper, ShardOptimizer):
        mapping = {
            id(original): owner for original, owner in zip(wrapper.originals, wrapper.shards)
        }
        optimizer = wrapper.optimizer
        sharded = True
    else:
        mapping = {id(parameter): parameter for parameter in role.parameters}
        optimizer = wrapper
        sharded = False
    if isinstance(optimizer, CPUOptimizer):
        cpu_map = {
            id(original): master for original, master in zip(optimizer.originals, optimizer.masters)
        }
        mapping = {key: cpu_map[id(owner)] for key, owner in mapping.items()}
        optimizer = optimizer.optimizer
    return optimizer, mapping, sharded


def _owner(entry, context):
    replicas = getattr(entry.tensor, "_aster_gradient_group", entry.group)
    tensor_owner = entry.tp_group.rank if entry.tp_group is not None else context.tp.rank
    return (
        entry.group.rank == 0
        and replicas.rank == 0
        and tensor_owner == 0
        and (entry.ep_group is None or entry.ep_group.rank == 0)
    )


def gather_role(role, context):
    entries = logical_tensors(role.model, context)
    optimizer, owners, optimizer_sharded = (
        optimizer_mapping(role) if role.trainable else (None, {}, False)
    )
    if optimizer is not None and type(optimizer) not in {
        torch.optim.Adam,
        torch.optim.AdamW,
        torch.optim.RAdam,
        torch.optim.SGD,
        MuonWithAuxAdam,
    }:
        raise ValueError(
            "portable optimizer 重分片仅实现 Adam/AdamW/RAdam/SGD/显式MuonWithAuxAdam 状态语义"
        )
    options = {}
    if optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                options[id(parameter)] = {
                    key: value for key, value in group.items() if key != "params"
                }
    local = {}
    for entry in entries:
        value = gather_tensor(entry.tensor, entry, context)
        record = {
            "value": value,
            "parameter": entry.parameter,
            "persistent": entry.persistent,
            "semantic": entry.semantic,
            "optimizer": None,
        }

        if role.ema is not None:
            record["ema"] = (
                gather_tensor(role.ema.shadow[entry.storage_name], entry, context)
                if entry.storage_name in role.ema.shadow
                else value.clone()
            )
        if role.trainable and entry.parameter and id(entry.tensor) in owners:
            owner = owners[id(entry.tensor)]
            state = {}
            source_state = (
                optimizer._aster_state_loader(owner)
                if hasattr(optimizer, "_aster_state_loader")
                else optimizer.state.get(owner, {})
            )
            for name, state_value in source_state.items():
                if isinstance(state_value, torch.Tensor) and (
                    state_value.ndim > 0
                    or name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}
                ):
                    state[name] = gather_tensor(
                        state_value, entry, context, optimizer_sharded=optimizer_sharded
                    )
                elif isinstance(state_value, torch.Tensor):
                    state[name] = state_value.detach().cpu().clone()
                else:
                    state[name] = state_value
            record["optimizer"] = {"state": state, "options": options[id(owner)]}
        if _owner(entry, context):
            local[entry.name] = record
    merged = {}
    for part in context.world.gather_objects(local):
        for name, record in part.items():
            if name in merged:
                raise ValueError(f"portable 逻辑名称冲突: {name}，需修正 PP/expert 参数映射")
            merged[name] = record
    return {
        "optimizer_class": type(optimizer).__name__ if optimizer is not None else None,
        "tensors": merged,
        "updates": role.updates,
        "successful_update": deepcopy(role.successful_update),
        "ema": {"decay": role.ema.decay, "updates": role.ema.updates} if role.ema else None,
        "scheduler": role.scheduler.state_dict() if role.scheduler is not None else None,
    }


def validate_role_runtime(role, context, payload):

    from .runtime_state import runtime_buffers

    runtime_buffers(role.model)
    aliases = {}
    for entry in logical_tensors(role.model, context):
        saved = payload["tensors"][entry.name]
        if saved.get("semantic", False) != entry.semantic:
            raise ValueError(f"Portable semantic buffer identity differs: {entry.name}")
        if not entry.semantic:
            continue
        value = saved["value"]
        if (
            not isinstance(value, torch.Tensor)
            or value.is_meta
            or value.layout != entry.tensor.layout
            or value.dtype != entry.tensor.dtype
            or value.requires_grad
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Portable semantic buffer dtype/layout/value differs: {entry.name}")
        local = local_tensor(value, entry, context)
        previous = aliases.setdefault(id(entry.tensor), local)
        if not torch.equal(previous, local):
            raise ValueError("Portable semantic buffer aliases disagree")


@torch.no_grad()
def restore_role(role, context, payload):
    entries = logical_tensors(role.model, context)
    optimizer, owners, optimizer_sharded = (
        optimizer_mapping(role) if role.trainable else (None, {}, False)
    )
    if payload["optimizer_class"] != (type(optimizer).__name__ if optimizer is not None else None):
        raise ValueError("portable optimizer 算法不一致")
    if (role.ema is None) != (payload["ema"] is None):
        raise ValueError("portable EMA 契约不一致")
    if role.ema is not None and role.ema.decay != payload["ema"]["decay"]:
        raise ValueError("portable EMA decay 改变")
    names = set().union(
        *[set(value) for value in context.world.gather_objects([entry.name for entry in entries])]
    )
    if names != set(payload["tensors"]):
        raise ValueError("portable 全局参数名集合不一致")
    groups = {}
    if optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                groups[id(parameter)] = group
        optimizer.state.clear()
    for entry in entries:
        saved = payload["tensors"][entry.name]
        if saved["parameter"] != entry.parameter:
            raise ValueError("parameter/buffer 身份改变")
        local = local_tensor(saved["value"], entry, context)
        entry.tensor.copy_(local.to(entry.tensor.device, entry.tensor.dtype))
        if role.ema is not None and entry.storage_name in role.ema.shadow:
            local_ema = local_tensor(saved["ema"], entry, context)
            role.ema.shadow[entry.storage_name].copy_(
                local_ema.to(role.ema.shadow[entry.storage_name].device)
            )
        if role.trainable and entry.parameter and id(entry.tensor) in owners:
            owner = owners[id(entry.tensor)]
            if saved["optimizer"] is None:
                raise ValueError("缺少可训练参数 optimizer 状态")
            options = saved["optimizer"]["options"]
            group = groups[id(owner)]

            prior = group.get("_aster_restored_options")
            if prior is not None and prior != options:
                raise ValueError("新 optimizer 参数组混合了不同来源超参数")
            group.update(options)
            group["_aster_restored_options"] = options
            owner_value = local_tensor(
                saved["value"], entry, context, optimizer_sharded=optimizer_sharded
            )
            owner.copy_(owner_value.to(owner.device, owner.dtype))
            state = {}
            for key, state_value in saved["optimizer"]["state"].items():
                if isinstance(state_value, torch.Tensor) and (
                    state_value.ndim > 0
                    or key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}
                ):
                    state[key] = local_tensor(
                        state_value, entry, context, optimizer_sharded=optimizer_sharded
                    ).to(owner.device)
                elif isinstance(state_value, torch.Tensor):
                    state[key] = state_value.to(
                        owner.device if options.get("capturable") or options.get("fused") else "cpu"
                    )
                else:
                    state[key] = state_value
            optimizer.state[owner] = state
    if optimizer is not None:
        for group in optimizer.param_groups:
            group.pop("_aster_restored_options", None)

    if isinstance(role.optimizer, ShardOptimizer):
        from .offload import CPUOptimizer

        inner = role.optimizer.optimizer
        if isinstance(inner, CPUOptimizer):
            for original, master in zip(inner.originals, inner.masters):
                original.copy_(master.to(original.device))
    if (role.scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("portable scheduler 契约不一致")
    if role.scheduler is not None:
        role.scheduler.load_state_dict(payload["scheduler"])
    role.updates = payload["updates"]
    if role.ema is not None:
        role.ema.updates = payload["ema"]["updates"]
    from .offload import DiskOptimizer

    wrapper = (
        role.optimizer.optimizer if isinstance(role.optimizer, ShardOptimizer) else role.optimizer
    )
    if isinstance(wrapper, DiskOptimizer):
        wrapper.evict_all()
