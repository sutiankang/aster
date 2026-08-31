"""GPipe, 1F1B, and virtual pipeline scheduling with shared loss contracts."""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.distributed as dist

from aster.core.contracts import LossTerm, LossBundle
from .parallel import Group

_DTYPES = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.bool,
)


@dataclass(frozen=True)
class PipelineLossSpec:
    name: str
    unit: str
    weight: float = 1.0
    differentiable: bool = True

    def __post_init__(self):
        if not self.name or not self.unit or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("非法 pipeline loss schema")
        if type(self.differentiable) is not bool:
            raise TypeError("differentiable 必须为 bool")


def pipeline_events(microbatches: int, size: int, rank: int, schedule: str):
    """Generate warmup/steady/drain events; after warmup, reclaim the oldest backward
    graph for each forward step."""
    if schedule == "gpipe":
        return [("forward", i) for i in range(microbatches)] + [
            ("backward", i) for i in range(microbatches)
        ]
    if schedule == "serial":
        return [(kind, i) for i in range(microbatches) for kind in ("forward", "backward")]
    if schedule != "1f1b":
        raise ValueError("未知流水线 schedule")
    warmup = min(size - rank - 1, microbatches)
    events = [("forward", i) for i in range(warmup)]
    events += [
        (kind, index)
        for i in range(microbatches - warmup)
        for kind, index in (("forward", i + warmup), ("backward", i))
    ]
    return events + [("backward", i) for i in range(microbatches - warmup, microbatches)]


class _ReceiveGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, stage):
        ctx.stage = stage
        ctx.stage_index = stage.stage_index
        return output.new_zeros(())

    @staticmethod
    def backward(ctx, ignored_scalar_gradient):

        return ctx.stage._receive_gradient(ctx.stage_index), None


class PipelineStage(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Group,
        *,
        schedule: str = "gpipe",
        parameter_names: dict[str, str] | None = None,
    ):
        super().__init__()
        if schedule not in {"gpipe", "serial", "1f1b"}:
            raise ValueError("当前 PipelineStage 仅实现 gpipe/serial/1f1b，不伪装 VPP")
        self.module, self.group, self.schedule = module, group, schedule
        self.parameter_names = dict(parameter_names) if parameter_names is not None else None
        self.input_leaf = None
        self._pending = []
        self.peak_live_graphs = 0
        self.active_chunk = 0

    @property
    def local_chunks(self):
        return 1

    @property
    def stage_index(self):
        return self.active_chunk * self.group.size + self.group.rank

    @property
    def is_last(self):
        return self.stage_index == self.local_chunks * self.group.size - 1

    @property
    def device(self):
        return next(self.parameters()).device

    def _send_tensor(self, value, destination, tag):
        if value.dtype not in _DTYPES:
            raise ValueError("流水线 Tensor dtype 未声明")
        header = torch.tensor(
            [_DTYPES.index(value.dtype), *value.shape], device=self.device, dtype=torch.int64
        )
        length = torch.tensor(header.numel(), device=self.device, dtype=torch.int64)
        self._send(length, destination, tag)
        self._send(header, destination, tag + 1)
        self._send(value.detach().contiguous(), destination, tag + 2)

    def _send(self, tensor, destination, tag):

        self._pending = [(work, value) for work, value in self._pending if not work.is_completed()]
        self._pending.append(
            (dist.isend(tensor, dst=destination, group=self.group.handle, tag=tag), tensor)
        )

    def flush_pending(self):
        for work, _ in self._pending:
            work.wait()
        self._pending.clear()

    def _receive_tensor(self, source, tag):
        length = torch.empty((), device=self.device, dtype=torch.int64)
        dist.recv(length, src=source, group=self.group.handle, tag=tag)
        if not 1 <= int(length) <= 17:
            raise ValueError("非法流水线 Tensor rank 元数据")
        header = torch.empty(int(length), device=self.device, dtype=torch.int64)
        dist.recv(header, src=source, group=self.group.handle, tag=tag + 1)
        dtype_index, *shape = header.tolist()
        if not 0 <= dtype_index < len(_DTYPES) or any(size < 0 for size in shape):
            raise ValueError("非法流水线 Tensor shape/dtype")
        output = torch.empty(shape, dtype=_DTYPES[dtype_index], device=self.device)
        dist.recv(output, src=source, group=self.group.handle, tag=tag + 2)
        return output

    def forward(self, inputs):
        if self.stage_index:
            inputs = self._receive_tensor(
                self.group.ranks[(self.group.rank - 1) % self.group.size],
                100 + (self.stage_index - 1) * 8,
            )
            if not inputs.is_floating_point():
                raise ValueError("中间 activation 必须可微浮点 Tensor")
            inputs.requires_grad_(torch.is_grad_enabled())
            self.input_leaf = inputs
        else:
            self.input_leaf = None
        output = self._run_module(inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("当前流水线 stage 需要单 Tensor 输出")
        if self.is_last:
            return output
        self._send_tensor(
            output,
            self.group.ranks[(self.group.rank + 1) % self.group.size],
            100 + self.stage_index * 8,
        )
        return _ReceiveGradient.apply(output, self)

    def _run_module(self, inputs):
        return self.module(inputs)

    def _receive_gradient(self, stage_index):
        source = self.group.ranks[(self.group.rank + 1) % self.group.size]
        tag = 100 + stage_index * 8 + 4
        present = torch.empty((), dtype=torch.int64, device=self.device)
        dist.recv(present, src=source, group=self.group.handle, tag=tag)
        return self._receive_tensor(source, tag + 1) if bool(present) else None

    def send_input_gradient(self, gradient):
        if not self.stage_index:
            return
        destination = self.group.ranks[(self.group.rank - 1) % self.group.size]
        tag = 100 + (self.stage_index - 1) * 8 + 4
        self._send(
            torch.tensor(int(gradient is not None), device=self.device, dtype=torch.int64),
            destination,
            tag,
        )
        if gradient is not None:
            self._send_tensor(gradient, destination, tag + 1)


def interleaved_events(microbatches, size, rank, chunks):
    """Schedule non-stale-weight interleaved 1F1B; virtual stage v belongs to v % PP."""
    if size < 2 or chunks < 2 or microbatches % size:
        raise ValueError("interleaved 需要 PP>=2/chunks>=2/M%PP==0")
    total = microbatches * chunks
    warmup = min((chunks - 1) * size + 2 * (size - rank - 1), total)
    forward_counts, backward_counts = [0] * chunks, [0] * chunks
    result = []
    for operation in range(total + warmup):
        if operation < total:
            chunk = (operation // size) % chunks
            result.append(("forward", chunk, forward_counts[chunk]))
            forward_counts[chunk] += 1
        if operation >= warmup:
            chunk = chunks - 1 - ((operation - warmup) // size) % chunks
            result.append(("backward", chunk, backward_counts[chunk]))
            backward_counts[chunk] += 1
    if forward_counts != [microbatches] * chunks or backward_counts != [microbatches] * chunks:
        raise RuntimeError("interleaved 事件数量不一致")
    return result


class VirtualPipelineStage(PipelineStage):
    """Own explicit local chunks mapped to global parameter names."""

    def __init__(self, chunks, group, *, parameter_names=None):
        if len(chunks) < 2 or group.size < 2:
            raise ValueError("VirtualPipelineStage 需要至少两个 chunks 与 ranks")
        if dist.get_backend(group.handle) == "nccl":
            raise NotImplementedError(
                "当前 interleaved 依赖按 edge 的 P2P tag；NCCL 不支持 tag，需独立 channel/有序batched调度后才允许使用"
            )
        super().__init__(
            nn.ModuleList(chunks), group, schedule="1f1b", parameter_names=parameter_names
        )
        self.schedule = "interleaved1f1b"

    @property
    def local_chunks(self):
        return len(self.module)

    def set_chunk(self, chunk):
        if not 0 <= chunk < self.local_chunks:
            raise ValueError("virtual chunk 越界")
        self.active_chunk = chunk

    def _run_module(self, inputs):
        return self.module[self.active_chunk](inputs)


class PipelineObjective:
    """Accept (inputs, *criterion_inputs); the criterion returns the shared loss contract."""

    def __init__(self, criterion: Callable, *, specs: tuple[PipelineLossSpec, ...] | None = None):
        self.criterion, self.specs = criterion, tuple(specs) if specs is not None else None
        if self.specs is not None and (
            not self.specs
            or any(not isinstance(spec, PipelineLossSpec) for spec in self.specs)
            or len({spec.name for spec in self.specs}) != len(self.specs)
        ):
            raise ValueError("pipeline specs 必须非空且无重复名称")

    def synchronize_totals(self, model, sums, counts):
        """Communicate final loss statistics once after the schedule, not as a microbatch barrier."""
        if self.specs is None:
            return
        value = torch.tensor(
            [[sums[spec.name], counts[spec.name]] for spec in self.specs],
            dtype=torch.float64,
            device=model.device,
        )
        if model.group.size > 1:
            dist.broadcast(value, src=model.group.ranks[-1], group=model.group.handle)
        for spec, (numerator, denominator) in zip(self.specs, value.tolist()):
            sums[spec.name], counts[spec.name] = numerator, denominator

    def forward(self, model: PipelineStage, batch: tuple[Any, ...]):
        if (
            not isinstance(model, PipelineStage)
            or not isinstance(batch, (list, tuple))
            or not batch
        ):
            raise TypeError("PipelineObjective 需要 PipelineStage 和 tuple batch")
        output = model(batch[0])
        result = self.criterion(output, *batch[1:]) if model.is_last else None
        if self.specs is not None:
            if model.is_last:
                terms = result.terms if isinstance(result, LossBundle) else (result,)
                actual = tuple(
                    (term.name, term.unit, term.weight, term.numerator.requires_grad)
                    for term in terms
                )
                expected = tuple(
                    (
                        spec.name,
                        spec.unit,
                        spec.weight,
                        spec.differentiable and torch.is_grad_enabled(),
                    )
                    for spec in self.specs
                )
                if actual != expected:
                    raise ValueError("criterion 与预声明 pipeline schema 不一致")
                return result
            return LossBundle(
                tuple(
                    LossTerm(
                        output if spec.differentiable else output.detach(),
                        torch.ones((), device=output.device),
                        spec.unit,
                        spec.name,
                        spec.weight,
                    )
                    for spec in self.specs
                )
            )
        if model.is_last:
            terms = result.terms if isinstance(result, LossBundle) else (result,)
            if any(not isinstance(term, LossTerm) for term in terms):
                raise TypeError("pipeline criterion 必须返回 LossTerm/Bundle")
            metadata = [
                (
                    term.name,
                    term.unit,
                    term.weight,
                    float(term.numerator.detach()),
                    float(term.denominator),
                    term.numerator.requires_grad,
                )
                for term in terms
            ]
        else:
            metadata = None
        metadata = model.group.gather_objects(metadata)[-1]
        if model.is_last:
            return result
        terms = tuple(
            LossTerm(
                output.double() + numerator
                if differentiable
                else output.detach().double() + numerator,
                torch.tensor(denominator, device=output.device, dtype=torch.float64),
                unit,
                name,
                weight,
            )
            for name, unit, weight, numerator, denominator, differentiable in metadata
        )
        return LossBundle(terms)
