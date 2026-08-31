"""One owner for loss normalization, backward, communication, updates, and checkpoints.

Accumulate gradients separately for each loss term, then combine them only after
collecting global denominators for the full window. This avoids retaining all
microbatch graphs, at the cost of one gradient buffer per loss term."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import random
import weakref
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import numpy as np
from torch import nn
import torch.distributed as dist

from aster.core.contracts import LossBundle, LossTerm
from aster.core.serialization import read_json
from .parallel import ParallelContext
from .sharding import ShardOptimizer, Zero3Unit, shard_module, zero3_units
from .pipeline import (
    PipelineStage,
    VirtualPipelineStage,
    PipelineObjective,
    pipeline_events,
    interleaved_events,
)
from .offload import CPUOptimizer, DiskOptimizer
from .activation import activation_storage
from .communication import GradientBucketReducer
from .portable import (
    gather_role,
    restore_role,
    logical_tensors,
    gather_tensor,
    local_tensor,
    optimizer_mapping,
    validate_role_runtime,
)
from .runtime_state import (
    runtime_descriptor,
    snapshot_runtime_state,
    validate_runtime_state,
    restore_runtime_state,
)
from .state import EMA, atomic_json, read_payload, restore_rng, rng_state, write_payload
from .provenance import validate_update_record
from .gradient_ratio import GradientRatioRegistry
from .muon import MuonFactory, MuonWithAuxAdam


@dataclass(frozen=True)
class StepResult:
    step: int
    phase: str
    role: str
    loss: float | None
    terms: Mapping[str, Mapping[str, Any]]
    grad_norm: float | None
    updated: bool
    overflow: bool


@dataclass
class _Role:
    model: nn.Module
    optimizer: Any
    parameters: list[nn.Parameter]
    trainable: bool
    ema: EMA | None = None
    scheduler: Any = None
    updates: int = 0
    optimizer_identity: Any = None
    successful_update: dict | None = None


def _model_configuration(model):

    original = model.module if isinstance(model, Zero3Unit) else model
    config = getattr(original, "config", None)
    if config is None:
        return None
    codec = getattr(config, "to_dict", None)
    if not callable(codec):
        raise ValueError("Model config must explicitly implement to_dict for checkpoint identity")
    value = codec()
    if not isinstance(value, dict):
        raise ValueError("Model config.to_dict must return a JSON object")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _objective_configuration(objective):

    if objective is None:
        return None
    codec_name = next(
        (name for name in ("config_dict", "to_dict") if hasattr(objective, name)), None
    )
    if codec_name is None:
        return None
    codec = getattr(objective, codec_name)
    if not callable(codec):
        raise ValueError(f"Objective {codec_name} must be callable")
    configuration = codec()
    if not isinstance(configuration, dict):
        raise ValueError(f"Objective {codec_name} must return a JSON object")

    def normalize(value):
        if value is None or type(value) in {str, bool, int}:
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, dict) and all(type(key) is str for key in value):
            return {key: normalize(item) for key, item in value.items()}
        raise ValueError(
            "Objective configuration must contain only finite JSON values and string keys"
        )

    return {
        "class": f"{type(objective).__module__}.{type(objective).__qualname__}",
        "codec": codec_name,
        "configuration": normalize(configuration),
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        objective: Callable | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 3e-4,
        optimizer_factory: Callable[[list[nn.Parameter]], torch.optim.Optimizer] | None = None,
        device: str | torch.device = "cpu",
        accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        max_grad_value: float | None = None,
        precision: str = "fp32",
        parallel: ParallelContext | None = None,
        zero_stage: int = 0,
        ema_decay: float | None = None,
        offload_optimizer: str = "none",
        activation_offload: str = "none",
        offload_directory: str | Path | None = None,
        communication_overlap: bool = False,
        bucket_bytes: int = 25 * 1024 * 1024,
        sharded_initializer: Callable | None = None,
        offload_parameters: str = "none",
    ):
        if type(accumulation_steps) is not int or accumulation_steps < 1:
            raise ValueError("accumulation_steps 必须为正整数")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr 必须为正数")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm 必须为正数或 None")
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(
                "全局 autocast 仅 fp32/bf16/fp16；FP8 需 provider 显式 FP8Linear，不能只改 flag 或静默回退"
            )
        if type(zero_stage) is not int or zero_stage not in {0, 1, 2, 3}:
            raise ValueError("zero_stage 必须为 0/1/2/3")
        if offload_optimizer not in {"none", "cpu", "nvme"}:
            raise ValueError("optimizer offload 支持 none/cpu/nvme（显式磁盘目录）")
        if (offload_optimizer == "nvme") != (offload_directory is not None):
            raise ValueError("nvme offload 必须且仅能显式提供 offload_directory")
        self.offload_directory = offload_directory
        if activation_offload not in {"none", "cpu"}:
            raise ValueError("activation_offload 仅 none/cpu")
        self.activation_offload = activation_offload
        if (
            type(communication_overlap) is not bool
            or type(bucket_bytes) is not int
            or bucket_bytes < 1
        ):
            raise ValueError("非法通信 overlap/bucket 配置")
        if communication_overlap and zero_stage != 0:
            raise ValueError(
                "当前 async bucket 实现仅 ZeRO0；ZeRO RS/prefetch overlap 未实现，不能静默回退"
            )
        self.communication_overlap, self.bucket_bytes = communication_overlap, bucket_bytes
        if sharded_initializer is not None and (
            not callable(sharded_initializer) or zero_stage != 3
        ):
            raise ValueError("sharded_initializer 仅支持 ZeRO3，并且必须可调用")
        self.sharded_initializer = sharded_initializer
        if offload_parameters not in {"none", "cpu"} or (
            offload_parameters != "none" and zero_stage != 3
        ):
            raise ValueError(
                "parameter offload 当前仅 ZeRO3 CPU storage，NVMe parameter paging 未实现"
            )
        self.offload_parameters = offload_parameters
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 训练当前要求 CUDA；CPU 不能替代硬件精度验收")
        self.parallel = parallel or ParallelContext()
        if self.parallel.pp.size > 1 and (
            not isinstance(model, PipelineStage) or model.group.ranks != self.parallel.pp.ranks
        ):
            raise ValueError("PP 需要显式匹配拓扑的 PipelineStage，不自动猜模型切分")
        if self.parallel.gtp_remat.size > 1 and zero_stage != 0:
            raise ValueError(
                "GTP reference 已拥有 remat shard；与额外 ZeRO DP shard 的嵌套布局尚未实现"
            )
        self.replica_group = self.parallel.dp_cp_gtp
        self.loss_groups = {}
        self.objective, self.lr = objective, lr
        self.accumulation_steps, self.max_grad_norm, self.precision, self.zero_stage = (
            accumulation_steps,
            max_grad_norm,
            precision,
            zero_stage,
        )
        self.max_grad_value = max_grad_value
        self.offload_optimizer = offload_optimizer
        self.roles: dict[str, _Role] = {}
        self._owners: dict[int, tuple[str, Any]] = {}
        self.states: dict[str, Any] = {}
        self._embedding_projection = None
        self._gradient_ratio = GradientRatioRegistry(self)
        self._pending_gradient_ratios = {}
        self.target_links = {}
        self.steps = 0
        self._busy = self._failed = False
        self.loss_scale = 65536.0 if precision == "fp16" else 1.0
        self.scale_successes = 0

        objective_error, objective_configuration = None, None
        try:
            objective_configuration = _objective_configuration(self.objective)

            validate_context = getattr(self.objective, "validate_training_context", None)
            if validate_context is not None:
                if not callable(validate_context):
                    raise TypeError("Objective context validator must be callable")
                validate_context(model, self.parallel)
            if self.max_grad_value is not None and (
                type(self.max_grad_value) not in {int, float}
                or not math.isfinite(self.max_grad_value)
                or self.max_grad_value <= 0
            ):
                raise ValueError("max_grad_value must be finite and positive or None")
        except Exception as exc:
            objective_error = f"{type(exc).__name__}: {exc}"
        declarations = self.parallel.world.gather_objects(
            (objective_error, objective_configuration, self.max_grad_value)
        )
        self._collective_error(
            str([item[0] for item in declarations])
            if any(item[0] for item in declarations)
            else None
        )
        self._collective_error(
            "Objective configuration differs across ranks"
            if any(item[1] != objective_configuration for item in declarations)
            else None
        )
        self._collective_error(
            "Gradient value clipping differs across ranks"
            if any(item[2] != self.max_grad_value for item in declarations)
            else None
        )
        self.add_role(
            "model",
            model,
            optimizer=optimizer,
            optimizer_factory=optimizer_factory,
            ema_decay=ema_decay,
        )
        self.model = self.roles["model"].model

    def add_role(
        self,
        name: str,
        module: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_factory: Callable[[list[nn.Parameter]], torch.optim.Optimizer] | None = None,
        trainable: bool = True,
        ema_decay: float | None = None,
        scheduler: Any = None,
    ) -> nn.Module:
        if self._busy or not name or name in self.roles:
            raise ValueError("角色重名、为空或训练中修改角色图")
        if type(trainable) is not bool:
            raise TypeError("trainable 必须为 bool")
        self._collective_error(
            "optimizer and optimizer_factory are mutually exclusive"
            if optimizer is not None and optimizer_factory is not None
            else None
        )
        self._collective_error(
            "optimizer_factory must be callable and belong to a trainable role"
            if optimizer_factory is not None and (not callable(optimizer_factory) or not trainable)
            else None
        )

        protocols = self.parallel.world.gather_objects(isinstance(optimizer_factory, MuonFactory))
        if any(protocol != protocols[0] for protocol in protocols):
            raise ValueError("Optimizer factory protocol differs across ranks")

        projected = [
            path or "<root>"
            for path, child in module.named_modules()
            if isinstance(child, (nn.Embedding, nn.EmbeddingBag)) and child.max_norm is not None
        ]
        unsafe_projection = self.parallel.world.size > 1 or (
            trainable and (self.zero_stage != 0 or self.offload_optimizer != "none")
        )
        self._collective_error(
            f"Embedding(max_norm) forward会原地修改参数；分布式/ZeRO/offload须显式max_norm=None并登记/调用持久投影: {projected}"
            if projected and unsafe_projection
            else None
        )
        configuration_error = None
        try:
            _model_configuration(module)
        except Exception as exc:
            configuration_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(configuration_error)
        original_parameters = list(module.parameters())
        if trainable and (self.parallel.ep.size > 1 or self.parallel.etp.size > 1):
            expert_parameters = [
                p for p in original_parameters if hasattr(p, "_aster_ep_dimension")
            ]
            self._collective_error(
                "EP/ETP training requires an explicit expert-layout provider"
                if not expert_parameters
                else None
            )
            self._collective_error(
                "Expert parameters require matching EP and EDP ownership"
                if any(
                    getattr(p, "_aster_ep_group", None) is not self.parallel.ep
                    or getattr(p, "_aster_gradient_group", None) is not self.parallel.edp
                    for p in expert_parameters
                )
                else None
            )
            self._collective_error(
                "ETP expert parameters require matching tensor shard group/axis"
                if self.parallel.etp.size > 1
                and any(
                    getattr(p, "_aster_tp_group", None) is not self.parallel.etp
                    or getattr(p, "_aster_tp_dimension", None) is None
                    for p in expert_parameters
                )
                else None
            )

        overlap = {
            self._owners[id(p)][0]
            for p in original_parameters
            if id(p) in self._owners and self._owners[id(p)][1]() is p
        }
        if overlap:
            raise ValueError(f"同一个 tensor 不能属于多个角色/optimizer，已属 {sorted(overlap)}")
        meta = any(parameter.is_meta for parameter in original_parameters)
        if meta:
            if (
                not trainable
                or self.zero_stage != 3
                or self.sharded_initializer is None
                or not all(parameter.is_meta for parameter in original_parameters)
            ):
                raise ValueError(
                    "meta 构建要求完整meta参数、ZeRO3训练角色和显式 sharded_initializer"
                )
            if any(buffer.is_meta for buffer in module.buffers()):
                raise ValueError("meta buffer 需由provider显式初始化，不能猜测非参数状态")
            for child in module.modules():
                for key, value in child._buffers.items():
                    if value is not None:
                        child._buffers[key] = value.to(self.device)
        else:
            module.to(self.device)
        if not trainable:
            if optimizer is not None or ema_decay is not None or scheduler is not None:
                raise ValueError("冻结角色不能拥有 optimizer/EMA/scheduler")
            module.requires_grad_(False)
        elif self.zero_stage == 3:
            if optimizer is not None:
                raise ValueError(
                    "ZeRO3 会替换参数为 shards；请让运行时创建 optimizer，再从 checkpoint 恢复"
                )
            for parameter in module.parameters():
                group = getattr(parameter, "_aster_gradient_group", self.replica_group)
                if group.size > 1 and not meta:
                    dist.broadcast(parameter.data, src=group.ranks[0], group=group.handle)
            module = shard_module(
                module,
                self.replica_group,
                initializer=self.sharded_initializer,
                device=self.device,
                prefix=name,
                offload_parameters=self.offload_parameters == "cpu",
            )
        parameters = [p for p in module.parameters() if p.requires_grad]
        if trainable and not parameters:
            raise ValueError("训练角色没有可更新参数")
        if trainable:
            error = None
            try:
                if optimizer_factory is not None:
                    optimizer = (
                        optimizer_factory.build(module, self.parallel, list(parameters))
                        if isinstance(optimizer_factory, MuonFactory)
                        else optimizer_factory(list(parameters))
                    )
                    if not isinstance(optimizer, torch.optim.Optimizer):
                        raise TypeError(
                            "optimizer_factory must return a native torch.optim.Optimizer"
                        )
                    if optimizer.state:
                        raise ValueError(
                            "optimizer_factory must create fresh state; restore through checkpoint"
                        )
                if optimizer is None:
                    optimizer = torch.optim.AdamW(parameters, lr=self.lr)
                actual = [p for group in optimizer.param_groups for p in group["params"]]
                if len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {
                    id(p) for p in parameters
                }:
                    raise ValueError("optimizer 必须恰好且无重复地拥有角色的全部可训练参数")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._collective_error(error)
            if isinstance(optimizer_factory, MuonFactory):
                optimizer_factory.validate_parallel(optimizer, module, self.parallel)

            for parameter in module.parameters():
                group = getattr(parameter, "_aster_gradient_group", self.replica_group)
                if group.size > 1 and self.zero_stage != 3:
                    dist.broadcast(parameter.data, src=group.ranks[0], group=group.handle)
            if self.zero_stage in {1, 2}:
                optimizer = ShardOptimizer(module, optimizer, self.replica_group, self.zero_stage)
            if self.offload_optimizer != "none":
                wrap = (
                    (lambda base: CPUOptimizer(base))
                    if self.offload_optimizer == "cpu"
                    else (lambda base: DiskOptimizer(base, self.offload_directory))
                )
                if isinstance(optimizer, ShardOptimizer):
                    optimizer.optimizer = wrap(optimizer.optimizer)
                else:
                    optimizer = wrap(optimizer)
            if scheduler is not None:
                native = optimizer
                while hasattr(native, "optimizer"):
                    native = native.optimizer
                if getattr(scheduler, "optimizer", None) is not native:
                    raise ValueError(
                        "scheduler 必须绑定实际状态所有者；分片/offload 后请用 set_scheduler 工厂"
                    )
        self.roles[name] = _Role(
            module,
            optimizer,
            parameters,
            trainable,
            EMA(module, ema_decay) if ema_decay is not None else None,
            scheduler,
        )
        if trainable:
            native = optimizer
            while hasattr(native, "optimizer"):
                native = native.optimizer
            _, actual_owners, _ = optimizer_mapping(self.roles[name])
            names = {
                id(parameter): parameter_name
                for parameter_name, parameter in module.named_parameters()
            }
            owner_names = {id(owner): names[original] for original, owner in actual_owners.items()}
            self.roles[name].optimizer_identity = {
                "class": f"{type(native).__module__}.{type(native).__qualname__}",
                "initial_groups": deepcopy(
                    [
                        {key: value for key, value in group.items() if key != "params"}
                        for group in native.param_groups
                    ]
                ),
                "parameter_groups": [
                    [owner_names[id(parameter)] for parameter in group["params"]]
                    for group in native.param_groups
                ],
            }
        for parameter in original_parameters + list(module.parameters()):
            self._owners[id(parameter)] = (name, weakref.ref(parameter))

        for child in module.modules():
            child._aster_training_owned = True
        return module

    def set_scheduler(self, factory: Callable, *, role: str = "model") -> Any:
        if self._busy or role not in self.roles or not self.roles[role].trainable:
            raise ValueError("非法 scheduler role/更新时机")
        optimizer = self.roles[role].optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        scheduler = factory(optimizer)
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise ValueError("scheduler factory 返回错误 owner")
        self.roles[role].scheduler = scheduler
        return scheduler

    def clone_target(
        self,
        source_role: str,
        target_role: str,
        *,
        factory: Callable[[], nn.Module],
        source_path: str = "",
    ) -> nn.Module:
        """Rebuild a frozen target from its provider and copy logical weights rather than
        deep-copying sharded placeholders."""
        if self._busy or self._failed or source_role not in self.roles:
            raise RuntimeError("当前边界不能创建 target")
        if not isinstance(source_path, str):
            raise TypeError("source_path 必须为明确子模块路径")
        if source_path:
            if self.parallel.pp.size > 1:
                raise ValueError("PP 子树 target 需要额外全局名称映射，当前不隐式剥离跨stage前缀")
            self.roles[source_role].model.get_submodule(source_path)
        module = factory()
        if not isinstance(module, nn.Module):
            raise TypeError("target factory 必须返回 nn.Module")
        target = self.add_role(target_role, module, trainable=False)
        try:
            self.update_target(source_role, target_role, 0.0, source_path=source_path)
        except Exception:
            self._failed = True
            raise
        self.target_links[target_role] = {"source_role": source_role, "source_path": source_path}
        target.eval()
        return target

    @torch.no_grad()
    def update_target(
        self,
        source_role: str,
        target_role: str,
        decay: float,
        *,
        buffers: str = "copy",
        source_path: str | None = None,
    ) -> None:
        """Apply target = decay * target + (1 - decay) * source."""
        error = None
        if self._busy or self._failed:
            error = "仅成功 phase 边界允许更新 target"
        elif (
            source_role == target_role
            or source_role not in self.roles
            or target_role not in self.roles
        ):
            error = "非法 source/target 角色"
        elif self.roles[target_role].trainable:
            error = "target 必须是无 optimizer 的冻结角色"
        elif not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay <= 1:
            error = "decay 必须属于 [0,1]"
        elif buffers not in {"copy", "ema"}:
            error = "buffer 更新语义只能 copy/ema"
        link = self.target_links.get(target_role)
        if source_path is None:
            source_path = link["source_path"] if link is not None else ""
        if not isinstance(source_path, str):
            error = "source_path 必须为子模块路径"
        elif link is not None and link != {"source_role": source_role, "source_path": source_path}:
            error = "target source/path 与已注册链接不一致"
        elif source_path and self.parallel.pp.size > 1:
            error = "PP 子树 target 尚需显式全局映射"
        self._collective_error(error)
        declaration = (source_role, target_role, source_path, decay, buffers)
        if any(value != declaration for value in self.parallel.world.gather_objects(declaration)):
            raise ValueError("各 rank 的 target source/path/decay/buffer 策略不一致")

        source = target = None
        error = None
        try:
            source_module = self.roles[source_role].model
            if source_path:
                source_module = source_module.get_submodule(source_path)
            source = {entry.name: entry for entry in logical_tensors(source_module, self.parallel)}
            target = {
                entry.name: entry
                for entry in logical_tensors(self.roles[target_role].model, self.parallel)
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        error = None
        if source.keys() != target.keys():
            error = "target 需要同一 PP/EP 分区的完整逻辑名称集合"
        else:
            aliases = {}
            for key in source:
                a, b = source[key], target[key]
                ashape, bshape = list(a.shape), list(b.shape)
                if a.tp_dimension is not None:
                    ashape[a.tp_dimension] *= a.tp_group.size
                if b.tp_dimension is not None:
                    bshape[b.tp_dimension] *= b.tp_group.size
                if a.ep_group is not None:
                    ashape[a.ep_dimension] *= a.ep_group.size
                if b.ep_group is not None:
                    bshape[b.ep_dimension] *= b.ep_group.size
                if a.global_shape is not None:
                    ashape = list(a.global_shape)
                if b.global_shape is not None:
                    bshape = list(b.global_shape)
                if (
                    a.parameter != b.parameter
                    or ashape != bshape
                    or a.tensor.dtype != b.tensor.dtype
                ):
                    error = f"target 逻辑形状/类型/dtype 不匹配: {key}"
                    break
                prior = aliases.setdefault(id(b.tensor), id(a.tensor))
                if prior != id(a.tensor):
                    error = "target 把两个独立 source 参数错误地 tying"
                    break
        self._collective_error(error)
        try:
            updated = set()
            for key in sorted(source):
                entry = target[key]
                if id(entry.tensor) in updated:
                    continue
                updated.add(id(entry.tensor))
                value = local_tensor(
                    gather_tensor(source[key].tensor, source[key], self.parallel),
                    entry,
                    self.parallel,
                )
                value = value.to(entry.tensor.device, entry.tensor.dtype)
                if entry.tensor.is_floating_point() and (entry.parameter or buffers == "ema"):
                    entry.tensor.lerp_(value, 1 - float(decay))
                else:
                    entry.tensor.copy_(value)
        except Exception:
            self._failed = True
            raise

    def register_state(self, name: str, stateful: Any) -> None:
        """Attach replay, environment, sampler, or generator state to the shared checkpoint."""
        if self._busy or not name or name in self.states:
            raise ValueError("重复状态名或运行中注册状态")
        if not callable(getattr(stateful, "state_dict", None)) or not callable(
            getattr(stateful, "load_state_dict", None)
        ):
            raise TypeError("注册状态需要 state_dict/load_state_dict")
        self.states[name] = stateful

    def register_embedding_projection(
        self, role: str, path: str, *, max_norm: float = 1.0, norm_type: float = 2.0
    ) -> None:

        from .projection import EmbeddingProjectionRegistry

        registry = self._embedding_projection or EmbeddingProjectionRegistry(self)
        registry.register(role, path, max_norm=max_norm, norm_type=norm_type)
        if self._embedding_projection is None:
            self.register_state("_embedding_projection", registry)
            self._embedding_projection = registry

    def project_embedding(self, role: str, path: str, indices: torch.Tensor):

        self._collective_error(
            "No embedding projection policy is registered"
            if self._embedding_projection is None
            else None
        )
        return self._embedding_projection.project(role, path, indices)

    def register_loss_group(self, term_name: str, group) -> None:
        """Declare the sample-reduction group; SP-local tokens and TP-replicated losses differ."""
        if self._busy or not term_name or term_name in self.loss_groups:
            raise ValueError("重复目标通信组或训练中修改")
        declarations = self.parallel.world.gather_objects((term_name, group.ranks))
        if any(name != term_name for name, _ in declarations):
            raise ValueError("各 rank 注册的 loss name 不一致")
        for rank, (_, ranks) in enumerate(declarations):
            if (
                rank not in ranks
                or any(member >= len(declarations) for member in ranks)
                or any(declarations[member][1] != ranks for member in ranks)
            ):
                raise ValueError("loss group 成员的通信域声明不一致")
        self.loss_groups[term_name] = group

    def register_gradient_ratio(
        self,
        name: str,
        *,
        role: str = "model",
        reference_term: str,
        target_term: str,
        parameter: str,
        eps: float = 1e-4,
        min_ratio: float = 0.0,
        max_ratio: float = 1e4,
        multiplier: float = 1.0,
    ) -> None:
        """Bind two separately normalized objectives to explicit logical parameter names."""
        self._gradient_ratio.register(
            name,
            role=role,
            reference_term=reference_term,
            target_term=target_term,
            parameter=parameter,
            eps=eps,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            multiplier=multiplier,
        )

    def last_gradient_ratio(self, name: str):
        """Return committed weighting state; overflow or failed updates do not replace it."""
        if self._busy or self._failed:
            raise RuntimeError("Gradient ratio provenance requires an idle valid Trainer")
        if name not in self._gradient_ratio.policies:
            raise ValueError("Unknown gradient ratio policy")
        records = self._gradient_ratio.validate_records(
            self._gradient_ratio.records, {key: role.updates for key, role in self.roles.items()}
        )
        return records[name]

    def _agree_gradient_ratios(self, records):
        copies = self.parallel.world.gather_objects(records)
        if any(value != copies[0] for value in copies):
            raise ValueError("Gradient ratio successful records differ across WORLD ranks")

    def _ratio_snapshot(self):
        error, records = None, None
        try:
            records = self._gradient_ratio.validate_records(
                self._gradient_ratio.records,
                {name: role.updates for name, role in self.roles.items()},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self._agree_gradient_ratios(records)
        return records

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
            enabled=self.precision != "fp32",
        )

    def _collective_error(self, error: str | None, group=None) -> None:
        messages = (group or self.parallel.world).gather_objects(error)
        if any(messages):
            raise ValueError(
                "跨 rank 契约失败: " + "; ".join(str(message) for message in messages if message)
            )

    def _terms(self, result: Any) -> tuple[LossTerm, ...]:
        if isinstance(result, LossTerm):
            return (result,)
        if isinstance(result, LossBundle):
            return result.terms
        raise TypeError(
            "objective 必须返回 LossTerm 或 LossBundle；裸 mean 无法正确累积/分布式归一化"
        )

    def _forward(self, model, objective, batch):
        function = objective.forward if hasattr(objective, "forward") else objective
        return function(model, batch)

    def step(self, microbatches: Iterable[Any]) -> StepResult:
        return self.phase("train", microbatches=microbatches)

    def last_successful_update(self, *, role: str = "model"):
        """Return a copy of the last committed phase declaration, not inferred training history."""
        if self._busy or self._failed:
            raise RuntimeError("Successful update provenance requires an idle valid Trainer")
        if role not in self.roles:
            raise ValueError("Unknown role for successful update provenance")
        current = self.roles[role]
        return validate_update_record(current.successful_update, role=role, updates=current.updates)

    def _agree_update_records(self, records):
        copies = self.parallel.world.gather_objects(records)
        if any(record != copies[0] for record in copies):
            raise ValueError("Successful update provenance differs across WORLD ranks")

    def phase(
        self,
        name: str,
        *,
        role: str = "model",
        objective: Callable | None = None,
        microbatches: Iterable[Any],
        freeze_roles: tuple[str, ...] = (),
    ) -> StepResult:
        objective, declared_objective = self._validate_phase_declaration(
            name, role, objective, freeze_roles
        )
        current = self.roles[role]
        if isinstance(current.model, PipelineStage) and current.model.schedule in {
            "1f1b",
            "interleaved1f1b",
        }:
            if not isinstance(objective, PipelineObjective) or objective.specs is None:
                raise ValueError(
                    "1F1B 需要 PipelineObjective 的显式 specs，避免跨 stage 元数据 barrier"
                )
        batches = list(microbatches)
        self._collective_error(
            None
            if len(batches) == self.accumulation_steps
            else "microbatches 数量必须等于 accumulation_steps"
        )
        self._validate_tied_optimizer_owners(current)
        batches = self._prepare_microbatches(current.model, objective, batches)
        self._pending_gradient_ratios = {}
        self._busy = True
        failure = None
        result = None
        frozen = [
            (parameter, parameter.requires_grad)
            for key in freeze_roles
            for parameter in self.roles[key].model.parameters()
        ]
        for parameter, _ in frozen:
            parameter.requires_grad_(False)
        current = self.roles[role]
        was_training = current.model.training
        current.model.train()
        try:
            result = self._run_phase(name, role, current, objective, batches)
        except Exception as exc:
            failure = exc
        finally:
            try:
                for parameter, requires_grad in frozen:
                    parameter.requires_grad_(requires_grad)
                current.model.train(was_training)
            except Exception as exc:
                if failure is None:
                    failure = exc
            finally:
                self._busy = False
                self._failed = failure is not None or result is None

        errors = self.parallel.world.gather_objects(
            None if failure is None else f"{type(failure).__name__}: {failure}"
        )
        if any(errors):
            self._failed = True
            if failure is not None:
                raise failure
            raise RuntimeError(f"Another rank failed before successful phase commit: {errors}")
        if result.updated:
            current.successful_update = {
                "role": role,
                "role_updates": current.updates,
                "phase": name,
                "objective_configuration": deepcopy(declared_objective),
            }
            self._gradient_ratio.records.update(deepcopy(self._pending_gradient_ratios))
        self._pending_gradient_ratios = {}
        return result

    def _validate_phase_declaration(self, name, role, objective, freeze_roles):

        actual = self.objective if objective is None else objective
        error, declaration = None, None
        try:
            if self._busy or self._failed:
                raise RuntimeError("运行时正忙或已失败；失败后必须从完整 checkpoint 恢复")
            if not isinstance(name, str) or not name:
                raise ValueError("phase name 必须为非空字符串")
            if (
                not isinstance(role, str)
                or role not in self.roles
                or not self.roles[role].trainable
            ):
                raise ValueError("phase 必须指向可训练角色")
            if (
                not isinstance(freeze_roles, tuple)
                or any(not isinstance(key, str) for key in freeze_roles)
                or len(set(freeze_roles)) != len(freeze_roles)
                or role in freeze_roles
                or any(key not in self.roles for key in freeze_roles)
            ):
                raise ValueError("非法冻结角色")
            if actual is None:
                raise ValueError("phase 缺少 objective")
            configuration = _objective_configuration(actual)
            for policy in self._gradient_ratio.policies.values():
                if policy["role"] == role:
                    self._gradient_ratio._domains(policy)
                    self._gradient_ratio._entry(policy)
            if self.max_grad_value is not None and (
                type(self.max_grad_value) not in {int, float}
                or not math.isfinite(self.max_grad_value)
                or self.max_grad_value <= 0
            ):
                raise ValueError("max_grad_value must be finite and positive or None")
            declaration = (
                name,
                role,
                freeze_roles,
                configuration,
                self.max_grad_norm,
                self.max_grad_value,
                self.roles[role].updates,
                self._gradient_ratio.describe(),
            )
        except Exception as exc:
            error = (isinstance(exc, RuntimeError), f"{type(exc).__name__}: {exc}")
        records = self.parallel.world.gather_objects((error, declaration))
        errors = [(rank, item[0][1]) for rank, item in enumerate(records) if item[0] is not None]
        if errors:
            exception = (
                RuntimeError
                if any(item[0] is not None and item[0][0] for item in records)
                else ValueError
            )
            raise exception(f"Phase declaration preflight failed: {errors}")
        if any(item[1] != records[0][1] for item in records):
            raise ValueError(
                "Phase declaration differs across ranks: name/role/freeze_roles/objective configuration/gradient clipping"
            )
        return actual, deepcopy(configuration)

    def _validate_tied_optimizer_owners(self, role):

        declarations, error, muon = {}, None, None
        try:
            optimizer, owners, _ = optimizer_mapping(role)
            options = {
                id(parameter): {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if type(optimizer) is MuonWithAuxAdam:
                muon = MuonFactory.declaration(optimizer, role.model, self.parallel, owners)
            for parameter in role.parameters:
                key = getattr(parameter, "_aster_pp_tied_key", None)
                if key is None:
                    continue
                group = getattr(parameter, "_aster_extra_gradient_group", None)
                if (
                    not isinstance(key, str)
                    or not key
                    or group is None
                    or group.size != 2
                    or key in declarations
                ):
                    raise ValueError(
                        "PP tied owner requires one unique parameter and a two-member shared-gradient group"
                    )
                declarations[key] = {
                    "ranks": group.ranks,
                    "shape": tuple(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                    "options": options[id(owners[id(parameter)])],
                    "norm_owner": getattr(parameter, "_aster_unique_norm_owner", True),
                }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records = self.parallel.world.gather_objects((error, declarations, muon))
        if any(item[0] for item in records):
            raise ValueError(
                f"Invalid optimizer owner declaration: {[item[0] for item in records if item[0]]}"
            )
        if any(item[2] != records[0][2] for item in records):
            raise ValueError("Muon runtime parameter order/options differ across ranks")
        for _, items, _ in records:
            for key, item in items.items():
                partners = [records[rank][1].get(key) for rank in item["ranks"]]
                comparable = lambda record: {
                    key: value for key, value in record.items() if key != "norm_owner"
                }
                if any(
                    partner is None or comparable(partner) != comparable(item)
                    for partner in partners
                ):
                    raise ValueError(
                        "PP tied optimizer algorithm/options/layout differ between shared-weight owners"
                    )
                if sum(partner["norm_owner"] is True for partner in partners) != 1:
                    raise ValueError(
                        "PP tied weight requires exactly one global gradient-norm owner"
                    )

    def _prepare_microbatches(self, model, objective, batches):
        preflight = getattr(objective, "preflight_microbatches", None)
        if preflight is None:
            return batches
        batches = list(batches)
        counts = self.parallel.world.gather_objects(len(batches))
        if not counts[0] or any(count != counts[0] for count in counts):
            raise ValueError(
                "Preflight requires the same nonempty microbatch count on every model rank"
            )

        error, prepared = None, None
        try:
            if not callable(preflight):
                raise TypeError("objective preflight_microbatches must be callable")
            prepared = list(preflight(model, batches))
            if len(prepared) != len(batches):
                raise ValueError("Preflight cannot change microbatch count")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        return prepared

    def _run_phase(self, name, role_name, role, objective, batches):
        parameters = role.parameters
        schema = None
        buffers: dict[str, list[torch.Tensor | None]] = {}
        counts = {}
        sums = {}
        used = {}
        overflow = False
        units = zero3_units(role.model)
        reducer = (
            GradientBucketReducer(parameters, self.replica_group, bucket_bytes=self.bucket_bytes)
            if self.communication_overlap
            else None
        )
        is_pipeline = isinstance(role.model, PipelineStage)
        contract_group = self.parallel.stage if is_pipeline else self.parallel.world

        def forward_batch(batch):
            trace = []
            for unit in units:
                unit.trace = trace if unit.group.size > 1 else None
            error = None
            terms = ()
            try:
                with self._autocast(), activation_storage(self.activation_offload, self.device):
                    terms = self._terms(self._forward(role.model, objective, batch))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                for unit in units:
                    unit.trace = None
            self._collective_error(error, contract_group)
            pipeline_input = (
                role.model.input_leaf if isinstance(role.model, PipelineStage) else None
            )
            return terms, trace, pipeline_input

        def backward_batch(prepared):
            nonlocal schema, overflow
            terms, trace, pipeline_input = prepared
            signature = tuple((term.name, term.unit, term.weight) for term in terms)
            signatures = contract_group.gather_objects(signature)
            if any(value != signature for value in signatures) or (
                schema is not None and signature != schema
            ):
                raise ValueError("每个 rank/microbatch 的 loss name/unit/weight 必须一致")
            traces = dict(zip(contract_group.ranks, contract_group.gather_objects(trace)))

            domains = {group for events in traces.values() for _, group in events}
            for group in domains:
                if any(rank not in traces for rank in group):
                    raise ValueError("ZeRO3 group crosses the current pipeline execution domain")
                sequences = [
                    [key for key, owner in traces[rank] if owner == group] for rank in group
                ]
                if any(value != sequences[0] for value in sequences):
                    raise ValueError("ZeRO3 单元执行顺序必须在实际 DP/EDP owner group 一致")
            schema = signature
            ratio_error, probe_terms = None, set()
            try:
                probe_terms = self._gradient_ratio.probe_terms(role_name, signature)
            except Exception as exc:
                ratio_error = f"{type(exc).__name__}: {exc}"
            self._collective_error(ratio_error, contract_group)

            active_indices = [
                i
                for i, term in enumerate(terms)
                if term.numerator.requires_grad and (term.weight > 0 or term.name in probe_terms)
            ]
            for index, term in enumerate(terms):
                key = term.name
                if key not in buffers:
                    buffers[key] = [None] * len(parameters)
                    used[key] = [False] * len(parameters)
                    counts[key] = 0.0
                    sums[key] = 0.0
                count = float(term.denominator.detach())
                numerator = float(term.numerator.detach())
                self._collective_error(
                    "零分母目标的 numerator 必须为零" if count == 0 and numerator != 0 else None,
                    contract_group,
                )
                if not isinstance(role.model, VirtualPipelineStage) or role.model.is_last:
                    counts[key] += count
                    sums[key] += numerator
                overflow |= not math.isfinite(numerator)
                if index in active_indices:
                    differentiated = parameters + (
                        [pipeline_input] if pipeline_input is not None else []
                    )
                    gradients = torch.autograd.grad(
                        term.numerator * self.loss_scale,
                        differentiated,
                        retain_graph=index != active_indices[-1],
                        allow_unused=True,
                    )
                    if pipeline_input is not None:
                        role.model.send_input_gradient(gradients[-1])
                        gradients = gradients[:-1]
                else:
                    gradients = (None,) * len(parameters)
                ready = [None] * len(parameters)
                for i, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    group = getattr(parameter, "_aster_gradient_group", self.replica_group)
                    flag = group.all_reduce(
                        torch.tensor(int(gradient is not None), device=self.device),
                        dist.ReduceOp.MAX,
                    )
                    extra = getattr(parameter, "_aster_extra_gradient_group", None)
                    if extra is not None:
                        extra.all_reduce(flag, dist.ReduceOp.MAX)
                    if not bool(flag):
                        continue
                    used[key][i] = True
                    gradient = (
                        torch.zeros_like(parameter, dtype=torch.float32)
                        if gradient is None
                        else gradient.detach().float() / self.loss_scale
                    )
                    overflow |= not bool(torch.isfinite(gradient).all())
                    if isinstance(role.optimizer, ShardOptimizer):
                        gradient = role.optimizer.accumulate_gradient(parameter, gradient)
                    if extra is not None:
                        communication_value = gradient.to(self.device)
                        extra.all_reduce(communication_value)
                        gradient = communication_value.to(gradient.device)
                    if reducer is not None:
                        ready[i] = gradient
                    elif buffers[key][i] is None:
                        buffers[key][i] = gradient.clone()
                    else:
                        buffers[key][i].add_(gradient)
                if reducer is not None:
                    reducer.submit(key, ready)

        if is_pipeline:
            pending = {}
            events = (
                interleaved_events(
                    len(batches),
                    role.model.group.size,
                    role.model.group.rank,
                    role.model.local_chunks,
                )
                if isinstance(role.model, VirtualPipelineStage)
                else [
                    (operation, 0, index)
                    for operation, index in pipeline_events(
                        len(batches),
                        role.model.group.size,
                        role.model.group.rank,
                        role.model.schedule,
                    )
                ]
            )
            for operation, chunk, index in events:
                if isinstance(role.model, VirtualPipelineStage):
                    role.model.set_chunk(chunk)
                if operation == "forward":
                    pending[chunk, index] = forward_batch(batches[index])
                    role.model.peak_live_graphs = max(role.model.peak_live_graphs, len(pending))
                else:
                    backward_batch(pending.pop((chunk, index)))
            role.model.flush_pending()
            if isinstance(objective, PipelineObjective):
                objective.synchronize_totals(role.model, sums, counts)
        else:
            for batch in batches:
                backward_batch(forward_batch(batch))
        if reducer is not None:
            reducer.finish(buffers)
            self.last_communication_buckets = reducer.launched_buckets
        signatures = self.parallel.world.gather_objects(schema)
        if any(value != schema for value in signatures):
            raise ValueError("各 PP stage 目标 schema 不一致")

        records = {}
        combined: dict[int, torch.Tensor | None] = {id(parameter): None for parameter in parameters}
        for key, unit, weight in schema:
            totals = torch.tensor([sums[key], counts[key]], device=self.device, dtype=torch.float64)
            self.loss_groups.get(key, self.replica_group).all_reduce(totals)
            numerator, denominator = totals.tolist()
            mean = numerator / denominator if denominator else 0.0
            records[key] = {
                "numerator": numerator if math.isfinite(numerator) else None,
                "denominator": denominator,
                "unit": unit,
                "weight": weight,
                "mean": mean if math.isfinite(mean) else None,
            }
        bad = torch.tensor(int(overflow), device=self.device)
        self.parallel.world.all_reduce(bad, dist.ReduceOp.MAX)
        overflow = bool(bad)
        weights = {key: value["weight"] for key, value in records.items()}
        if not overflow:
            weights, pending, ratio_overflow = self._gradient_ratio.resolve(
                role_name, parameters, buffers, records, already_reduced=reducer is not None
            )
            overflow |= ratio_overflow
            if not overflow:
                self._pending_gradient_ratios = pending
        loss = 0.0
        for key, unit, _ in schema:
            weight, denominator = weights[key], records[key]["denominator"]
            if any(
                policy["role"] == role_name and policy["target_term"] == key
                for policy in self._gradient_ratio.policies.values()
            ):
                records[key]["effective_weight"] = weight
            loss += weight * (
                records[key]["mean"] if records[key]["mean"] is not None else float("nan")
            )
            if denominator == 0 or weight == 0:
                continue
            for parameter, gradient in zip(parameters, buffers[key]):
                if gradient is not None:
                    contribution = gradient * (weight / denominator)
                    if combined[id(parameter)] is None:
                        combined[id(parameter)] = contribution
                    else:
                        combined[id(parameter)].add_(contribution)
        bad = torch.tensor(int(overflow), device=self.device)
        self.parallel.world.all_reduce(bad, dist.ReduceOp.MAX)
        overflow = bool(bad)
        active = bool(
            self.parallel.world.all_reduce(
                torch.tensor(
                    int(any(gradient is not None for gradient in combined.values())),
                    device=self.device,
                ),
                dist.ReduceOp.MAX,
            )
        )
        norm = None
        if active and not overflow:
            if isinstance(role.optimizer, ShardOptimizer):
                role.optimizer.prepare(combined)
                optimized = role.optimizer.shards
            else:
                for parameter in parameters:
                    gradient = combined[id(parameter)]

                    if gradient is not None and self.zero_stage == 0 and reducer is None:
                        getattr(parameter, "_aster_gradient_group", self.replica_group).all_reduce(
                            gradient
                        )
                    parameter.grad = gradient.to(parameter.dtype) if gradient is not None else None
                optimized = parameters
            squared = torch.zeros((), device=self.device, dtype=torch.float64)
            for parameter in optimized:
                if parameter.grad is None:
                    continue
                if not getattr(parameter, "_aster_unique_norm_owner", True):
                    continue
                group = getattr(parameter, "_aster_gradient_group", self.replica_group)
                if self.zero_stage == 0 and group.rank != 0:
                    continue
                if self.parallel.tp.rank and not getattr(parameter, "_aster_tp_sharded", False):
                    continue
                valid = getattr(parameter, "_aster_valid_numel", parameter.numel())
                squared += (
                    parameter.grad.detach()
                    .flatten()[:valid]
                    .double()
                    .square()
                    .sum()
                    .to(squared.device)
                )
            self.parallel.world.all_reduce(squared)
            norm = float(squared.sqrt())
            if not math.isfinite(norm):
                overflow = True
                norm = None
        if active and not overflow:
            coefficient = (
                min(1.0, self.max_grad_norm / (norm + 1e-12))
                if self.max_grad_norm is not None
                else 1.0
            )
            for parameter in optimized:
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient)

                    if self.max_grad_value is not None:
                        parameter.grad.clamp_(-self.max_grad_value, self.max_grad_value)
            role.optimizer.step()
            role.updates += 1
            self.steps += 1
            if role.scheduler is not None:
                role.scheduler.step()
            if role.ema is not None:
                role.ema.update(role.model)
        if self.precision == "fp16":
            self.scale_successes = 0 if overflow else self.scale_successes + int(active)
            if overflow:
                self.loss_scale = max(self.loss_scale / 2, 1.0)
            elif self.scale_successes >= 2000:
                self.loss_scale *= 2
                self.scale_successes = 0
        return StepResult(
            self.steps,
            name,
            role_name,
            loss if math.isfinite(loss) else None,
            records,
            norm,
            active and not overflow,
            overflow,
        )

    def fit(self, batches: Iterable[Any], steps: int) -> list[StepResult]:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps 必须为正整数")
        iterator = iter(batches)
        results = []
        for _ in range(steps):
            window = []
            for _ in range(self.accumulation_steps):
                try:
                    window.append(next(iterator))
                except StopIteration as error:
                    raise ValueError("数据不足以完成指定更新窗口；不静默丢弃尾部") from error
            results.append(self.step(window))
        return results

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Any]) -> Mapping[str, Mapping[str, Any]]:
        if self._busy or self.objective is None:
            raise RuntimeError("运行中或缺少 objective")
        was_training = self.model.training
        self.model.eval()
        totals = {}
        schema = None
        try:
            batches = self._prepare_microbatches(self.model, self.objective, batches)
            if isinstance(self.model, VirtualPipelineStage):
                materialized = list(batches)
                events = interleaved_events(
                    len(materialized),
                    self.model.group.size,
                    self.model.group.rank,
                    self.model.local_chunks,
                )
                iterator = (
                    (chunk, materialized[index])
                    for operation, chunk, index in events
                    if operation == "forward"
                )
            else:
                iterator = ((0, batch) for batch in batches)
            for chunk, batch in iterator:
                if isinstance(self.model, VirtualPipelineStage):
                    self.model.set_chunk(chunk)
                with self._autocast():
                    terms = self._terms(self._forward(self.model, self.objective, batch))
                current = tuple((term.name, term.unit, term.weight) for term in terms)
                if schema is not None and schema != current:
                    raise ValueError("evaluate 目标 schema 改变")
                schema = current
                for term in terms:
                    value = totals.setdefault(term.name, [0.0, 0.0])
                    if not isinstance(self.model, VirtualPipelineStage) or self.model.is_last:
                        value[0] += float(term.numerator)
                        value[1] += float(term.denominator)
            if schema is None:
                raise ValueError("不能评估空数据")
            if isinstance(self.model, PipelineStage):
                self.model.flush_pending()
                if isinstance(self.objective, PipelineObjective):
                    sums, counts = (
                        {key: value[0] for key, value in totals.items()},
                        {key: value[1] for key, value in totals.items()},
                    )
                    self.objective.synchronize_totals(self.model, sums, counts)
                    totals = {key: [sums[key], counts[key]] for key in totals}
            signatures = self.parallel.world.gather_objects(schema)
            if any(signature != schema for signature in signatures):
                raise ValueError("evaluate 跨 rank schema 不一致")
            records = {}
            for name, unit, weight in schema:
                total = torch.tensor(totals[name], device=self.device, dtype=torch.float64)
                self.loss_groups.get(name, self.replica_group).all_reduce(total)
                numerator, denominator = total.tolist()
                if not math.isfinite(numerator):
                    raise FloatingPointError("评估出现非有限目标")
                records[name] = {
                    "mean": numerator / denominator if denominator else None,
                    "numerator": numerator,
                    "denominator": denominator,
                    "unit": unit,
                    "weight": weight,
                }
            return records
        finally:
            self.model.train(was_training)

    def _identity(self):
        return {
            "precision": self.precision,
            "zero_stage": self.zero_stage,
            "parallel": self.parallel.to_dict(),
            "initial_lr": self.lr,
            "offload_optimizer": self.offload_optimizer,
            "offload_parameters": self.offload_parameters,
            "activation_offload": self.activation_offload,
            "communication_overlap": self.communication_overlap,
            "bucket_bytes": self.bucket_bytes,
            "target_links": self.target_links,
            "objective_configuration": _objective_configuration(self.objective),
            "embedding_projection_policies": self._embedding_projection.describe()
            if self._embedding_projection is not None
            else [],
            "gradient_ratio_policies": self._gradient_ratio.describe(),
            "loss_groups": {name: list(group.ranks) for name, group in self.loss_groups.items()},
            "accumulation_steps": self.accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "max_grad_value": self.max_grad_value,
            "torch_version": str(torch.__version__),
            "device": str(self.device),
            "roles": {
                name: {
                    "trainable": role.trainable,
                    "model_class": f"{type(role.model).__module__}.{type(role.model).__qualname__}",
                    "model_configuration": _model_configuration(role.model),
                    "semantic_runtime": runtime_descriptor(role.model),
                    "ema_decay": role.ema.decay if role.ema else None,
                    "scheduler_class": type(role.scheduler).__qualname__
                    if role.scheduler is not None
                    else None,
                    "optimizer_class": type(role.optimizer).__qualname__,
                    "optimizer_identity": role.optimizer_identity,
                    "pipeline_schedule": role.model.schedule
                    if isinstance(role.model, PipelineStage)
                    else None,
                    "pipeline_chunks": role.model.local_chunks
                    if isinstance(role.model, PipelineStage)
                    else None,
                    "pipeline_names": role.model.parameter_names
                    if isinstance(role.model, PipelineStage)
                    else None,
                    "precision_contracts": {
                        key: module.precision_contract()
                        for key, module in role.model.named_modules()
                        if callable(getattr(module, "precision_contract", None))
                    },
                    "zero3_parameter_aliases": {
                        key: dict(module._aster_zero3_parameter_aliases)
                        for key, module in role.model.named_modules()
                        if getattr(module, "_aster_zero3_parameter_aliases", None)
                    },
                    "parameter_key_maps": {
                        key: dict(module._aster_parameter_key_map)
                        for key, module in role.model.named_modules()
                        if getattr(module, "_aster_parameter_key_map", None)
                    },
                    "gradient_layout": [
                        (
                            key,
                            list(getattr(value, "_aster_gradient_group", self.replica_group).ranks),
                            list(value._aster_extra_gradient_group.ranks)
                            if hasattr(value, "_aster_extra_gradient_group")
                            else None,
                            getattr(value, "_aster_tp_dimension", None),
                            list(value._aster_tp_group.ranks)
                            if hasattr(value, "_aster_tp_group")
                            else None,
                            getattr(value, "_aster_tp_stripes", 1),
                            getattr(value, "_aster_tp_global_shape", None),
                            getattr(value, "_aster_ep_dimension", None),
                            list(value._aster_ep_group.ranks)
                            if hasattr(value, "_aster_ep_group")
                            else None,
                            getattr(value, "_aster_pp_tied_key", None),
                            getattr(value, "_aster_unique_norm_owner", True),
                        )
                        for key, value in role.model.named_parameters()
                    ],
                    "parameters": [
                        (key, list(value.shape), str(value.dtype))
                        for key, value in role.model.state_dict().items()
                    ],
                }
                for name, role in self.roles.items()
            },
            "state_names": sorted(self.states),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        self._collective_error(
            "仅成功 phase 边界允许保存；失败内存不具有事务回滚保证"
            if self._busy or self._failed
            else None
        )
        records, error = None, None
        try:
            records = {name: self.last_successful_update(role=name) for name in self.roles}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self._agree_update_records(records)
        ratios = self._ratio_snapshot()
        path = Path(path).absolute()
        entry = None
        error = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            identity = self._identity()
            payload = {
                "identity": identity,
                "gradient_ratio_records": ratios,
                "steps": self.steps,
                "loss_scale": self.loss_scale,
                "scale_successes": self.scale_successes,
                "rng": rng_state(),
                "states": {name: obj.state_dict() for name, obj in self.states.items()},
                "roles": {
                    name: {
                        "model": role.model.state_dict(),
                        "runtime_state": snapshot_runtime_state(role.model),
                        "optimizer": role.optimizer.state_dict() if role.trainable else None,
                        "scheduler": role.scheduler.state_dict()
                        if role.scheduler is not None
                        else None,
                        "ema": role.ema.state_dict() if role.ema else None,
                        "updates": role.updates,
                        "successful_update": records[name],
                    }
                    for name, role in self.roles.items()
                },
            }
            entry = write_payload(path.parent, path.name, payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        entries = self.parallel.world.gather_objects(entry)
        error = None
        if self.parallel.rank == 0:
            try:
                atomic_json(
                    path, {"schema": 1, "world_size": self.parallel.world.size, "entries": entries}
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self.parallel.world.barrier()
        return path

    def export_state_dict(
        self, *, role: str = "model", ema: bool = False, only_rank_zero: bool = True
    ):
        """Collectively merge supported ZeRO/TP/EP layouts into logical-name CPU tensors.
        Every participating rank must enter this operation."""
        if self._busy or self._failed or role not in self.roles:
            raise RuntimeError("不能从当前运行状态导出")
        current = self.roles[role]
        if ema and current.ema is None:
            raise ValueError("角色未启用 EMA")
        portable = gather_role(replace(current, trainable=False, optimizer=None), self.parallel)
        if only_rank_zero and self.parallel.rank != 0:
            return None

        return {
            name: record["ema" if ema else "value"]
            for name, record in portable["tensors"].items()
            if record.get("persistent", True)
        }

    def export_runtime_state(self, *, role: str = "model", only_rank_zero: bool = True):
        """Collectively export declared non-persistent semantic buffers without changing weight keys."""
        error, entries = None, None
        try:
            if self._busy or self._failed or role not in self.roles:
                raise RuntimeError("Cannot export runtime state from current Trainer state")
            snapshot_runtime_state(self.roles[role].model)
            entries = [
                entry
                for entry in logical_tensors(self.roles[role].model, self.parallel)
                if entry.semantic
            ]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        declarations = self.parallel.world.gather_objects((role, only_rank_zero))
        if any(value != declarations[0] for value in declarations):
            raise ValueError("Runtime export role/options differ across ranks")
        local = {entry.name: gather_tensor(entry.tensor, entry, self.parallel) for entry in entries}
        merged = {}
        for part in self.parallel.world.gather_objects(local):
            for name, value in part.items():
                if name in merged and (
                    merged[name].dtype != value.dtype or not torch.equal(merged[name], value)
                ):
                    raise ValueError(f"Semantic runtime replicas disagree: {name}")
                merged[name] = value
        if only_rank_zero and self.parallel.rank != 0:
            return None
        return {"schema_version": 1, "semantic_buffers": merged}

    def save_portable_checkpoint(self, path: str | Path) -> Path:
        """Save reshardable model/optimizer/EMA/scheduler state, excluding rank-local RNG
        and data cursors."""
        if self._busy or self._failed:
            raise RuntimeError("只有成功边界允许迁移快照")
        if self.states:
            raise ValueError(
                "注册了数据/replay/环境状态；当前 portable 格式不能静默丢弃它们，请用原生精确 checkpoint"
            )
        records, error = None, None
        try:
            records = {name: self.last_successful_update(role=name) for name in self.roles}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self._agree_update_records(records)
        ratios = self._ratio_snapshot()
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "optimizer_reshard_v1",
            "source_layout": self.parallel.to_dict(),
            "precision": self.precision,
            "gradient_ratio_policies": self._gradient_ratio.describe(),
            "gradient_ratio_records": ratios,
            "steps": self.steps,
            "loss_scale": self.loss_scale,
            "scale_successes": self.scale_successes,
            "max_grad_norm": self.max_grad_norm,
            "max_grad_value": self.max_grad_value,
            "roles": {name: gather_role(role, self.parallel) for name, role in self.roles.items()},
        }
        entry = None
        error = None
        if self.parallel.rank == 0:
            try:
                entry = write_payload(path.parent, path.name, payload)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        entries = self.parallel.world.gather_objects(entry)
        if self.parallel.rank == 0:
            try:
                atomic_json(
                    path, {"schema": 1, "kind": "optimizer_reshard_v1", "entry": entries[0]}
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self.parallel.world.barrier()
        return path

    def load_portable_checkpoint(self, path: str | Path, *, seed: int) -> None:
        if self._busy:
            raise RuntimeError("不能在 phase 中迁移")
        if type(seed) is not int or seed < 0:
            raise ValueError("跨拓扑迁移需要显式指定新的非负 seed")
        if self.states:
            raise ValueError("当前 portable 迁移不能恢复 rank 局部 data/replay 状态")
        path = Path(path).absolute()
        manifest = read_json(path)
        if manifest.get("kind") != "optimizer_reshard_v1":
            raise ValueError("不是 optimizer 重分片 checkpoint")
        payload = read_payload(path.parent, manifest["entry"], trusted=False)
        if (
            payload["precision"] != self.precision
            or payload["max_grad_norm"] != self.max_grad_norm
            or payload.get("max_grad_value") != self.max_grad_value
            or set(payload["roles"]) != set(self.roles)
        ):
            raise ValueError("迁移必须保持角色/精度/梯度裁剪语义")

        records, error = {}, None
        try:
            records = {
                name: validate_update_record(
                    saved.get("successful_update"), role=name, updates=saved["updates"]
                )
                for name, saved in payload["roles"].items()
            }
            for name, role in self.roles.items():
                validate_role_runtime(role, self.parallel, payload["roles"][name])
            if payload.get("gradient_ratio_policies", []) != self._gradient_ratio.describe():
                raise ValueError("Portable gradient ratio policy identity differs")
            ratios = self._gradient_ratio.validate_records(
                payload.get("gradient_ratio_records", {}),
                {name: saved["updates"] for name, saved in payload["roles"].items()},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self._agree_update_records(records)
        self._agree_gradient_ratios(ratios)
        self._failed = True
        for name, role in self.roles.items():
            restore_role(role, self.parallel, payload["roles"][name])
        for name, role in self.roles.items():
            role.successful_update = records[name]
        self._gradient_ratio.records = ratios
        self.steps, self.loss_scale, self.scale_successes = (
            payload["steps"],
            payload["loss_scale"],
            payload["scale_successes"],
        )
        random.seed(seed + self.parallel.rank)
        np.random.seed((seed + self.parallel.rank) % 2**32)
        torch.manual_seed(seed + self.parallel.rank)
        self.migration_record = {
            "mode": "optimizer_reshard",
            "source_layout": payload["source_layout"],
            "target_layout": self.parallel.to_dict(),
            "new_seed": seed,
            "exact_rank_rng_or_data_resume": False,
        }
        self.parallel.world.barrier()
        self._failed = False

    def load_checkpoint(self, path: str | Path, trusted: bool = False) -> None:
        if self._busy:
            raise RuntimeError("不能在 phase 中恢复")
        path = Path(path).absolute()
        error = None
        payload = None
        try:
            manifest = read_json(path)
            if (
                manifest.get("schema") != 1
                or manifest.get("world_size") != self.parallel.world.size
            ):
                raise ValueError("checkpoint schema/拓扑不一致；跨拓扑恢复需要 portable 模式")
            payload = read_payload(
                path.parent, manifest["entries"][self.parallel.rank], trusted=trusted
            )
            if payload["identity"] != self._identity():
                raise ValueError("checkpoint 模型/角色/精度/累积/布局配置不一致")
            if set(payload["roles"]) != set(self.roles):
                raise ValueError("Checkpoint role set differs")
            records = {
                name: validate_update_record(
                    saved.get("successful_update"), role=name, updates=saved["updates"]
                )
                for name, saved in payload["roles"].items()
            }
            for name, role in self.roles.items():
                state = payload["roles"][name].get(
                    "runtime_state", {"schema_version": 1, "semantic_buffers": {}}
                )
                validate_runtime_state(role.model, state)
            ratios = self._gradient_ratio.validate_records(
                payload.get("gradient_ratio_records", {}),
                {name: saved["updates"] for name, saved in payload["roles"].items()},
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self._agree_update_records(records)
        self._agree_gradient_ratios(ratios)
        self._failed = True
        error = None
        try:
            for name, saved in payload["roles"].items():
                role = self.roles[name]
                role.model.load_state_dict(saved["model"], strict=True)
                restore_runtime_state(
                    role.model,
                    saved.get("runtime_state", {"schema_version": 1, "semantic_buffers": {}}),
                )
                if role.trainable:
                    role.optimizer.load_state_dict(saved["optimizer"])
                if role.scheduler is not None:
                    role.scheduler.load_state_dict(saved["scheduler"])
                if role.ema is not None:
                    role.ema.load_state_dict(saved["ema"])
                role.updates = saved["updates"]
                role.successful_update = records[name]
            self._gradient_ratio.records = ratios
            for name, state in payload["states"].items():
                self.states[name].load_state_dict(state)
            self.steps, self.loss_scale, self.scale_successes = (
                payload["steps"],
                payload["loss_scale"],
                payload["scale_successes"],
            )
            restore_rng(payload["rng"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error)
        self.parallel.world.barrier()
        self._failed = False
