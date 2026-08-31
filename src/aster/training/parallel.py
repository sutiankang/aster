"""Explicit process groups and native differentiable tensor-parallel operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F


@dataclass(frozen=True)
class Group:
    ranks: tuple[int, ...] = (0,)
    handle: Any = None
    rank: int = 0

    def __post_init__(self):
        if (
            not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
        ):
            raise ValueError("Group ranks 必须是非空、唯一的非负整数")
        if type(self.rank) is not int or not 0 <= self.rank < len(self.ranks):
            raise ValueError("Group local rank 越界")
        if len(self.ranks) > 1 and self.handle is None:
            raise ValueError("多成员 Group 必须持显式 handle；None 不能冒充 local 或 WORLD")

    @property
    def size(self) -> int:
        return len(self.ranks)

    def all_reduce(self, tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM) -> torch.Tensor:
        if self.size > 1:
            dist.all_reduce(tensor, op=op, group=self.handle)
        return tensor

    def gather_objects(self, value: Any) -> list[Any]:
        if self.size == 1:
            return [value]
        outputs = [None] * self.size
        dist.all_gather_object(outputs, value, group=self.handle)
        return outputs

    def barrier(self) -> None:
        if self.size > 1:
            dist.barrier(group=self.handle)


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    data_parallel: int = 1
    gtp_remat: int = 1
    expert_parallel: int = 1
    expert_tensor_parallel: int = 1

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数")

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel
            * self.pipeline_parallel
            * self.context_parallel
            * self.data_parallel
            * self.gtp_remat
        )


class ParallelContext:
    """Use rank order [DP, PP, CP, GTP_remat, TP]; a unit rematerialization axis
    preserves the four-axis layout."""

    def __init__(self, config: ParallelConfig | None = None):
        if config is None:
            config = ParallelConfig(
                data_parallel=dist.get_world_size() if dist.is_initialized() else 1
            )
        self.config = config
        actual = dist.get_world_size() if dist.is_initialized() else 1
        if actual > 1:
            configurations = [None] * actual
            dist.all_gather_object(configurations, vars(config))
            if any(value != vars(config) for value in configurations):
                raise ValueError("所有 WORLD rank 的并行网格配置必须一致")
        if actual != config.world_size:
            raise ValueError("分布式 WORLD 与 ParallelConfig 乘积不一致")
        expert_active = config.expert_parallel > 1 or config.expert_tensor_parallel > 1
        if expert_active:
            if any(
                value != 1
                for value in (config.pipeline_parallel, config.context_parallel, config.gtp_remat)
            ):
                raise ValueError("Native expert folding currently rejects PP/CP/GTP")
            if config.tensor_parallel not in {1, config.expert_tensor_parallel}:
                raise ValueError(
                    "Native expert folding requires attention TP=1 or attention TP=ETP"
                )
            if actual % (config.expert_parallel * config.expert_tensor_parallel):
                raise ValueError(
                    "WORLD must divide into EP x ETP x EDP; these are not extra WORLD multipliers"
                )
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = Group(
            tuple(range(actual)), dist.group.WORLD if actual > 1 else None, self.rank
        )
        dimensions = (
            config.data_parallel,
            config.pipeline_parallel,
            config.context_parallel,
            config.gtp_remat,
            config.tensor_parallel,
        )
        grid = torch.arange(actual).reshape(dimensions)
        for axis, name in enumerate(("dp", "pp", "cp", "gtp_remat", "tp")):
            rows = grid.movedim(axis, -1).reshape(-1, dimensions[axis]).tolist()
            for row in rows:
                handle = dist.new_group(row) if actual > 1 and len(row) > 1 else None
                if self.rank in row:
                    setattr(self, name, Group(tuple(row), handle, row.index(self.rank)))

        expert_rows = (
            [list(range(actual))]
            if expert_active
            else grid.permute(1, 2, 3, 4, 0).reshape(-1, dimensions[0]).tolist()
        )
        for row in expert_rows:
            expert_grid = torch.tensor(row).reshape(
                -1, config.expert_parallel, config.expert_tensor_parallel
            )
            for axis, name in enumerate(("edp", "ep", "etp")):
                rows = expert_grid.movedim(axis, -1).reshape(-1, expert_grid.shape[axis]).tolist()
                for members in rows:
                    handle = dist.new_group(members) if actual > 1 and len(members) > 1 else None
                    if self.rank in members:
                        setattr(self, name, Group(tuple(members), handle, members.index(self.rank)))
        for name, order, width in (
            ("dp_cp", (1, 3, 4, 0, 2), dimensions[0] * dimensions[2]),
            ("dp_tp", (1, 2, 3, 0, 4), dimensions[0] * dimensions[4]),
            ("tp_pp", (0, 2, 3, 1, 4), dimensions[1] * dimensions[4]),
            (
                "stage",
                (1, 0, 2, 3, 4),
                dimensions[0] * dimensions[2] * dimensions[3] * dimensions[4],
            ),
            ("dp_gtp", (1, 2, 4, 0, 3), dimensions[0] * dimensions[3]),
            ("dp_cp_gtp", (1, 4, 0, 2, 3), dimensions[0] * dimensions[2] * dimensions[3]),
            ("dp_tp_gtp", (1, 2, 0, 3, 4), dimensions[0] * dimensions[3] * dimensions[4]),
        ):
            for row in grid.permute(order).reshape(-1, width).tolist():
                handle = dist.new_group(row) if actual > 1 and len(row) > 1 else None
                if self.rank in row:
                    setattr(self, name, Group(tuple(row), handle, row.index(self.rank)))

    def to_dict(self):
        return dict(vars(self.config))


def all_gather_flat(shard: torch.Tensor, group: Group) -> torch.Tensor:
    if group.size == 1:
        return shard.clone()
    outputs = [torch.empty_like(shard) for _ in group.ranks]
    dist.all_gather(outputs, shard.contiguous(), group=group.handle)
    return torch.cat(outputs)


def reduce_scatter_flat(full: torch.Tensor, group: Group) -> torch.Tensor:
    if full.numel() % group.size:
        raise ValueError("RS buffer 必须按组大小 padding")
    if group.size == 1:
        return full.clone()
    output = torch.empty(full.numel() // group.size, dtype=full.dtype, device=full.device)

    dist.reduce_scatter_tensor(output, full.contiguous(), group=group.handle)
    return output


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, grad):
        return ctx.group.all_reduce(grad.contiguous().clone()), None


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        return group.all_reduce(value.clone())

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dimension):
        ctx.group, ctx.dimension = group, dimension
        if group.size == 1:
            return value
        outputs = [torch.empty_like(value) for _ in group.ranks]
        dist.all_gather(outputs, value.contiguous(), group=group.handle)
        return torch.cat(outputs, dim=dimension)

    @staticmethod
    def backward(ctx, grad):
        return (
            grad.chunk(ctx.group.size, dim=ctx.dimension)[ctx.group.rank].contiguous(),
            None,
            None,
        )


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, dimension):
        if value.shape[dimension] % group.size:
            raise ValueError("分片轴必须整除并行大小")
        ctx.group, ctx.dimension = group, dimension
        return value.chunk(group.size, dim=dimension)[group.rank].contiguous()

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        if group.size == 1:
            return grad, None, None
        outputs = [torch.empty_like(grad) for _ in group.ranks]
        dist.all_gather(outputs, grad.contiguous(), group=group.handle)
        return torch.cat(outputs, dim=ctx.dimension), None, None


class ColumnParallelLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, group: Group, *, bias=True, gather_output=False
    ):
        super().__init__()
        if out_features % group.size:
            raise ValueError("输出维度不能被 TP 整除")
        self.group, self.gather_output = group, gather_output
        self.weight = nn.Parameter(torch.empty(out_features // group.size, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features // group.size)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.weight._aster_tp_sharded = True
        self.weight._aster_tp_dimension = 0
        if self.bias is not None:
            self.bias._aster_tp_sharded = True
        if self.bias is not None:
            self.bias._aster_tp_dimension = 0

    def forward(self, inputs):
        outputs = F.linear(_Copy.apply(inputs, self.group), self.weight, self.bias)
        return _Gather.apply(outputs, self.group, -1) if self.gather_output else outputs


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: Group,
        *,
        bias=True,
        input_is_parallel=True,
    ):
        super().__init__()
        if in_features % group.size:
            raise ValueError("输入维度不能被 TP 整除")
        self.group, self.input_is_parallel = group, input_is_parallel
        self.weight = nn.Parameter(torch.empty(out_features, in_features // group.size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.weight._aster_tp_sharded = True
        self.weight._aster_tp_dimension = 1

    def forward(self, inputs):
        local = inputs if self.input_is_parallel else _Scatter.apply(inputs, self.group, -1)
        output = _Reduce.apply(F.linear(local, self.weight), self.group)
        return output + self.bias if self.bias is not None else output


class _VocabCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, group, ignore_index):

        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        size = logits.shape[-1]
        start = group.rank * size
        maximum = group.all_reduce(logits.max(-1).values.clone(), dist.ReduceOp.MAX)
        exp = (logits - maximum.unsqueeze(-1)).exp()
        denominator = group.all_reduce(exp.sum(-1))
        valid = targets != ignore_index
        if ((targets[valid] < 0) | (targets[valid] >= size * group.size)).any():
            raise ValueError("target 超出全局词表")
        own = valid & (targets >= start) & (targets < start + size)
        local_targets = (targets - start).clamp(0, size - 1)

        selected = torch.where(own, logits.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1), 0.0)
        selected = group.all_reduce(selected)
        probability = exp / denominator.unsqueeze(-1)
        ctx.save_for_backward(probability, local_targets, own, valid)
        return (maximum + denominator.log() - selected) * valid

    @staticmethod
    def backward(ctx, grad):
        probability, targets, own, valid = ctx.saved_tensors
        gradient = probability.clone()
        gradient.scatter_add_(-1, targets.unsqueeze(-1), -own.to(gradient.dtype).unsqueeze(-1))
        return gradient * (grad * valid).unsqueeze(-1), None, None, None


def vocab_parallel_cross_entropy(logits, targets, group: Group, *, ignore_index=-100):
    return _VocabCE.apply(logits, targets, group, ignore_index)
