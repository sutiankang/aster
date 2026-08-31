"""Native ZeRO optimizer/gradient sharding and leaf-level parameter rematerialization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from .parallel import Group, all_gather_flat, reduce_scatter_flat
from .muon import MuonWithAuxAdam, rebind_matrix_layout


def padded_shard(value: torch.Tensor, group: Group) -> tuple[torch.Tensor, int]:
    flat = value.detach().flatten()
    width = math.ceil(flat.numel() / group.size)
    padded = torch.zeros(width * group.size, dtype=flat.dtype, device=flat.device)
    padded[: flat.numel()].copy_(flat)
    return padded[group.rank * width : (group.rank + 1) * width].clone(), flat.numel()


def padded_gradient(value: torch.Tensor, total: int) -> torch.Tensor:
    result = torch.zeros(total, dtype=value.dtype, device=value.device)
    result[: value.numel()].copy_(value.flatten())
    return result


class ShardOptimizer:
    """Shard optimizer state per parameter while preserving grad=None for unused parameters."""

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, group: Group, stage: int
    ):
        if stage not in {1, 2}:
            raise ValueError("ShardOptimizer 仅负责 ZeRO1/2")
        if type(optimizer) not in {
            torch.optim.Adam,
            torch.optim.AdamW,
            torch.optim.RAdam,
            torch.optim.SGD,
            MuonWithAuxAdam,
        }:
            raise TypeError(
                "当前自主 ZeRO1/2 仅支持 Adam/AdamW/RAdam/SGD/显式MuonWithAuxAdam；不猜测自定义优化器状态分片"
            )
        if optimizer.state:
            raise ValueError("请在训练前初始化 ZeRO；已有 optimizer 状态需从运行时 checkpoint 加载")
        self.group, self.stage = group, stage
        self.originals, self.shards, self.sizes, self.parameter_groups = [], [], [], []
        groups = []
        for old_group in optimizer.param_groups:
            local_parameters = []
            for parameter in old_group["params"]:
                parameter_group = getattr(parameter, "_aster_gradient_group", group)
                shard, size = padded_shard(parameter, parameter_group)
                local = nn.Parameter(shard, requires_grad=parameter.requires_grad)
                local._aster_tp_sharded = getattr(parameter, "_aster_tp_sharded", False)
                local._aster_unique_norm_owner = getattr(
                    parameter, "_aster_unique_norm_owner", True
                )
                local._aster_gradient_group = parameter_group
                local._aster_valid_numel = max(
                    0, min(shard.numel(), size - parameter_group.rank * shard.numel())
                )
                rebind_matrix_layout(parameter, local, optimizer_sharded=True)
                self.originals.append(parameter)
                self.shards.append(local)
                self.sizes.append(size)
                local_parameters.append(local)
                self.parameter_groups.append(parameter_group)
            groups.append(
                {
                    **{k: v for k, v in old_group.items() if k != "params"},
                    "params": local_parameters,
                }
            )

        self.optimizer = type(optimizer)(groups)

    def accumulate_gradient(self, parameter: nn.Parameter, gradient: torch.Tensor) -> torch.Tensor:
        if self.stage == 1:
            return gradient.detach().float()
        group = getattr(parameter, "_aster_gradient_group", self.group)
        width = math.ceil(parameter.numel() / group.size)
        return reduce_scatter_flat(
            padded_gradient(gradient.detach().float(), width * group.size), group
        )

    def prepare(self, gradients: dict[int, torch.Tensor | None]) -> None:
        for original, shard, size, group in zip(
            self.originals, self.shards, self.sizes, self.parameter_groups
        ):
            grad = gradients[id(original)]
            original.grad = None
            if grad is None:
                shard.grad = None
                continue
            if self.stage == 1:
                group.all_reduce(grad)
                width = shard.numel()
                grad = padded_gradient(grad, width * group.size)[
                    group.rank * width : (group.rank + 1) * width
                ]
            shard.grad = grad.to(shard.dtype).reshape_as(shard)

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        for original, shard, size, group in zip(
            self.originals, self.shards, self.sizes, self.parameter_groups
        ):
            original.copy_(all_gather_flat(shard.detach(), group)[:size].reshape_as(original))

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "shards": [p.detach().clone() for p in self.shards],
        }

    def load_state_dict(self, state):
        if len(state["shards"]) != len(self.shards):
            raise ValueError("ZeRO optimizer 参数数不一致")
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for original, shard, size, saved, group in zip(
                self.originals, self.shards, self.sizes, state["shards"], self.parameter_groups
            ):
                shard.copy_(saved)
                original.copy_(all_gather_flat(shard, group)[:size].reshape_as(original))


class _RematerializedUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unit, template, tensor_count, *tensors):
        ctx.unit, ctx.template, ctx.tensor_count = unit, template, tensor_count
        ctx.save_for_backward(*tensors)
        ctx.cpu_rng = torch.random.get_rng_state()
        device = unit.compute_device
        ctx.cuda_device = device.index if device.type == "cuda" else None
        ctx.cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ctx.autocast_enabled = torch.is_autocast_enabled(device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(device.type)
        ctx.device_type = device.type
        return unit._call(template, tensors[:tensor_count], tensors[tensor_count:])

    @staticmethod
    def backward(ctx, output_gradient):
        unit = ctx.unit
        saved = ctx.saved_tensors
        inputs = [
            value.detach().requires_grad_(value.requires_grad)
            for value in saved[: ctx.tensor_count]
        ]
        shards = saved[ctx.tensor_count :]

        devices = [ctx.cuda_device] if ctx.cuda_device is not None else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.enable_grad(),
            torch.autocast(ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled),
        ):
            torch.random.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng, ctx.cuda_device)
            full_parameters = unit._gather(shards)
            differentiable_parameters = {
                name: value.detach().requires_grad_(shard.requires_grad)
                for (name, value), shard in zip(full_parameters.items(), shards)
            }
            args, kwargs = unit._arguments(ctx.template, inputs)
            output = torch.func.functional_call(
                unit.module, differentiable_parameters, args, kwargs, strict=False
            )
            train_inputs = [value for value in inputs if value.requires_grad]
            train_parameters = [
                value for value in differentiable_parameters.values() if value.requires_grad
            ]
            gradients = torch.autograd.grad(
                output, train_inputs + train_parameters, output_gradient, allow_unused=True
            )
        input_grads = iter(gradients[: len(train_inputs)])
        returned_inputs = [next(input_grads) if value.requires_grad else None for value in inputs]
        shard_grads = []
        parameter_grads = iter(gradients[len(train_inputs) :])
        for shard in shards:
            if not shard.requires_grad:
                shard_grads.append(None)
                continue
            gradient = next(parameter_grads)
            active = unit.group.all_reduce(
                torch.tensor(int(gradient is not None), device=unit.compute_device),
                torch.distributed.ReduceOp.MAX,
            )
            if not bool(active):
                shard_grads.append(None)
                continue
            if gradient is None:
                gradient = torch.zeros(
                    unit.shapes[len(shard_grads)], dtype=shard.dtype, device=unit.compute_device
                )
            full_grad = padded_gradient(gradient, shard.numel() * unit.group.size)
            shard_grads.append(reduce_scatter_flat(full_grad, unit.group).to(shard.device))
        unit.releases += 1
        return (None, None, None, *returned_inputs, *shard_grads)


@dataclass(frozen=True)
class ShardedParameterMetadata:
    """Read-only geometry, not a route around parameter gather/release boundaries."""

    dtype: torch.dtype
    device: torch.device
    shape: torch.Size
    logical_name: str

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        raise RuntimeError(
            "ZeRO3 parameter metadata is not a Tensor; call the owning module forward, not a raw-parameter functional operation"
        )

    def __getattr__(self, name):
        raise RuntimeError(
            f"ZeRO3 parameter {self.logical_name} only exposes dtype/device/shape metadata outside its forward; unsupported access {name}"
        )


class Zero3Unit(nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            names = self.__dict__.get("names", ())
            if name in names:
                index = names.index(name)
                shard = super().__getattr__("shards")[index]
                return ShardedParameterMetadata(
                    shard.dtype,
                    self.compute_device,
                    torch.Size(self.shapes[index]),
                    f"{self.logical_name}.{name}",
                )

            module = self.__dict__.get("_modules", {}).get("module")
            if module is not None and hasattr(module, name):
                value = getattr(module, name)
                if value is None or isinstance(
                    value, (str, int, float, bool, tuple, torch.dtype, torch.device)
                ):
                    return value
            raise

    def __init__(
        self,
        module: nn.Module,
        group: Group,
        name: str,
        shared=None,
        initializer=None,
        device=None,
        offload_parameters=False,
    ):
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.max_norm is not None:
            raise ValueError(
                "Embedding(max_norm) mutates weights in forward; rematerialization requires a persistent projection protocol"
            )
        super().__init__()
        self.module, self.group, self.logical_name = module, group, name
        self.compute_device = (
            torch.device(device) if device is not None else next(module.parameters()).device
        )
        self.offload_parameters = offload_parameters
        self.names, self.shapes, self.sizes = [], [], []
        self.shards = nn.ParameterList()
        self.gathers = self.releases = 0
        self.trace: list[tuple[str, tuple[int, ...]]] | None = None
        if list(module.buffers()):
            raise ValueError("当前 ZeRO3 叶子单元不接受有 buffer 的潜在有状态前向")
        if len({p.requires_grad for p in module.parameters(recurse=False)}) > 1:
            raise ValueError("ZeRO3 叶子不能混合冻结参数，请分离计算单元")
        shared = {} if shared is None else shared
        for parameter_name, parameter in list(
            module.named_parameters(recurse=False, remove_duplicate=False)
        ):
            self.names.append(parameter_name)
            self.shapes.append(tuple(parameter.shape))
            self.sizes.append(parameter.numel())
            previous = shared.get(id(parameter))

            parameter_layout = (
                tuple(parameter.shape),
                parameter.dtype,
                parameter.requires_grad,
                group.ranks,
                getattr(parameter, "_aster_tp_dimension", None),
                getattr(getattr(parameter, "_aster_tp_group", None), "ranks", None),
                getattr(parameter, "_aster_tp_stripes", 1),
                getattr(parameter, "_aster_ep_dimension", None),
                getattr(getattr(parameter, "_aster_ep_group", None), "ranks", None),
            )
            if previous is not None:
                local, layout = previous
                if layout != parameter_layout:
                    raise ValueError("共享参数的 ZeRO/TP 布局不一致，不能复制后假装保持 tying")
                shard = local.detach()
            else:
                if parameter.is_meta:
                    if initializer is None or device is None:
                        raise ValueError("meta 参数需要显式 shard initializer 与计算设备")
                    width = math.ceil(parameter.numel() / group.size)
                    offset = group.rank * width
                    valid = max(0, min(width, parameter.numel() - offset))
                    storage_device = (
                        torch.device("cpu") if offload_parameters else self.compute_device
                    )
                    value = initializer(
                        f"{name}.{parameter_name}",
                        tuple(parameter.shape),
                        parameter.dtype,
                        offset,
                        valid,
                        storage_device,
                    )
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.numel() != valid
                        or value.dtype != parameter.dtype
                        or value.device != storage_device
                    ):
                        raise ValueError(
                            "shard initializer 必须返回指定 device/dtype/局部有效元素数，不可返回完整参数"
                        )
                    shard = torch.zeros(width, device=storage_device, dtype=parameter.dtype)
                    shard[:valid].copy_(value.detach().reshape(-1))
                else:
                    shard, _ = padded_shard(parameter, group)
                if offload_parameters:
                    shard = shard.cpu()
                    if self.compute_device.type == "cuda":
                        shard = shard.pin_memory()
                local = nn.Parameter(shard, requires_grad=parameter.requires_grad)
                shared[id(parameter)] = (local, parameter_layout)
            local._aster_tp_sharded = getattr(parameter, "_aster_tp_sharded", False)
            local._aster_gradient_group = group
            local._aster_compute_device = self.compute_device
            for attribute in (
                "_aster_tp_dimension",
                "_aster_extra_gradient_group",
                "_aster_tp_global_shape",
                "_aster_pp_tied_key",
                "_aster_unique_norm_owner",
                "_aster_ep_dimension",
                "_aster_ep_group",
                "_aster_tp_group",
                "_aster_tp_stripes",
            ):
                if hasattr(parameter, attribute):
                    setattr(local, attribute, getattr(parameter, attribute))
            local._aster_valid_numel = max(
                0, min(shard.numel(), parameter.numel() - group.rank * shard.numel())
            )
            self.shards.append(local)

            module._parameters[parameter_name] = nn.Parameter(
                torch.empty(0, dtype=parameter.dtype, device=self.compute_device),
                requires_grad=False,
            )

    def _gather(self, shards):
        self.gathers += 1
        return {
            name: all_gather_flat(shard.to(self.compute_device), self.group)[:size].reshape(shape)
            for name, size, shape, shard in zip(self.names, self.sizes, self.shapes, shards)
        }

    def _arguments(self, template, tensors):
        leaves, positions, spec = template
        values = list(leaves)
        for position, tensor in zip(positions, tensors):
            values[position] = tensor
        return tree_unflatten(values, spec)

    def _call(self, template, inputs, shards):
        args, kwargs = self._arguments(template, inputs)
        parameters = self._gather(shards)
        output = torch.func.functional_call(self.module, parameters, args, kwargs, strict=False)
        if not isinstance(output, torch.Tensor):
            raise TypeError("ZeRO3 叶子单元当前要求单 Tensor 输出")
        self.releases += 1
        return output

    def forward(self, *args, **kwargs):
        if self.trace is not None:
            self.trace.append((self.logical_name, self.group.ranks))
        leaves, spec = tree_flatten((args, kwargs))
        positions = [i for i, value in enumerate(leaves) if isinstance(value, torch.Tensor)]
        tensors = [leaves[i] for i in positions]
        template = (
            [None if i in positions else value for i, value in enumerate(leaves)],
            positions,
            spec,
        )
        if not torch.is_grad_enabled():
            return self._call(template, tensors, list(self.shards))
        return _RematerializedUnit.apply(self, template, len(tensors), *tensors, *self.shards)


def shard_module(
    model: nn.Module,
    group: Group,
    *,
    initializer=None,
    device=None,
    prefix="model",
    offload_parameters=False,
) -> nn.Module:
    """Preserve tensor aliases with one physical shard and one optimizer owner."""

    container_aliases = {}
    for path, module in model.named_modules():
        own = list(module.parameters(recurse=False))
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.max_norm is not None:
            raise ValueError(
                f"Embedding(max_norm) has a persistent forward mutation incompatible with ZeRO3: {path}"
            )
        if own and isinstance(module, (nn.ParameterList, nn.ParameterDict)):
            raise ValueError(
                f"ZeRO3 不拦截裸 ParameterList/ParameterDict 索引: {path}；provider 需放入显式 forward 计算单元（如 Embedding），或声明父单元物化边界"
            )
        if own and list(module.children()):
            leaves = {}
            for child_path, child in module.named_modules():
                if (
                    child_path
                    and not list(child.children())
                    and not isinstance(child, (nn.ParameterList, nn.ParameterDict))
                ):
                    for name, parameter in child.named_parameters(recurse=False):
                        leaves.setdefault(id(parameter), f"{child_path}.{name}")
            aliases = {
                name: leaves[id(parameter)]
                for name, parameter in module.named_parameters(
                    recurse=False, remove_duplicate=False
                )
                if id(parameter) in leaves
            }
            if (
                type(module) is not nn.Module
                or len(aliases) != len(module._parameters)
                or any(parameter is None for parameter in module._parameters.values())
            ):
                raise ValueError(
                    f"ZeRO3 直接参数需位于显式叶子计算单元，纯容器只允许已注册叶子别名: {path}"
                )
            container_aliases[id(module)] = aliases
            continue
        if own and list(module.buffers()):
            raise ValueError(f"ZeRO3 叶子含 buffer，无法保证重算无状态: {path}")
        if own and len({parameter.requires_grad for parameter in own}) > 1:
            raise ValueError(f"ZeRO3 叶子不能混合冻结参数，请分离计算单元: {path}")
    shared = {}

    def convert(module: nn.Module, prefix: str):
        aliases = container_aliases.get(id(module))
        if aliases:
            for name in aliases:
                del module._parameters[name]
            for name, child in list(module.named_children()):
                module.add_module(name, convert(child, f"{prefix}.{name}"))
            module._aster_zero3_parameter_aliases = dict(aliases)
            for name, target in aliases.items():
                owner, _, parameter_name = target.rpartition(".")

                setattr(module, name, getattr(module.get_submodule(owner), parameter_name))
            return module
        own = list(module.parameters(recurse=False))
        if own:
            if list(module.children()):
                raise ValueError("ZeRO3 当前要求直接参数位于叶子模块")
            groups = [getattr(parameter, "_aster_gradient_group", group) for parameter in own]
            if any(value.ranks != groups[0].ranks for value in groups):
                raise ValueError("ZeRO3 单元直接参数必须属于同一分片组")
            return Zero3Unit(
                module, groups[0], prefix, shared, initializer, device, offload_parameters
            )
        for name, child in list(module.named_children()):
            module.add_module(name, convert(child, f"{prefix}.{name}"))
        return module

    return convert(model, prefix)


def zero3_units(model: nn.Module) -> list[Zero3Unit]:
    return [module for module in model.modules() if isinstance(module, Zero3Unit)]
