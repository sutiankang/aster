"""Explicit FP8 HYBRID training with separate reference and CUDA scaled-matmul paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.distributed as dist
from .parallel import Group


@dataclass(frozen=True)
class FP8Recipe:
    scaling: str = "delayed"
    history_length: int = 16
    margin: int = 0
    power_of_two: bool = True

    def __post_init__(self):
        if self.scaling not in {"current", "delayed"}:
            raise ValueError("FP8 scaling 必须 current/delayed")
        if type(self.history_length) is not int or self.history_length < 1:
            raise ValueError("amax history 长度必须为正整数")
        if type(self.margin) is not int or self.margin < 0 or type(self.power_of_two) is not bool:
            raise ValueError("非法 FP8 margin/power_of_two")


class FP8Quantizer(nn.Module):
    def __init__(self, dtype, recipe=FP8Recipe(), *, group=None):
        super().__init__()
        if dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
            raise ValueError("仅支持 E4M3FN/E5M2")
        self.dtype, self.recipe, self.group = dtype, recipe, group or Group()
        self.register_buffer("history", torch.zeros(recipe.history_length, dtype=torch.float32))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("last_inverse_scale", torch.ones((), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, value):
        if not value.is_floating_point() or value.numel() == 0:
            raise ValueError("FP8 量化需要非空浮点张量")
        amax = self.group.all_reduce(value.detach().float().abs().max(), dist.ReduceOp.MAX)
        if not bool(torch.isfinite(amax)):
            raise FloatingPointError("FP8 amax 非有限，不能污染缩放历史")
        previous = self.history.max()
        chosen = amax if self.recipe.scaling == "current" or not bool(self.updates) else previous
        maximum = torch.finfo(self.dtype).max
        raw_scale = maximum / chosen.clamp_min(torch.finfo(torch.float32).tiny)
        if self.recipe.power_of_two:
            scale = torch.exp2(torch.floor(torch.log2(raw_scale)) - self.recipe.margin)
        else:
            scale = raw_scale / (2**self.recipe.margin)

        scale = torch.where(chosen == 0, torch.ones_like(scale), scale).clamp_max(
            torch.finfo(torch.float32).max
        )
        inverse = scale.reciprocal().clone()
        quantized = (value.detach().float() * scale).clamp(-maximum, maximum).to(self.dtype)
        if self.training:
            self.history.copy_(self.history.roll(1))
            self.history[0].copy_(amax)
            self.updates.add_(1)
            self.last_inverse_scale.copy_(inverse)
        return quantized, inverse


def fp8_matmul(left, right, left_inverse, right_inverse, *, implementation):
    if implementation == "reference":
        return (left.float() * left_inverse) @ (right.float() * right_inverse)
    if implementation != "scaled_mm":
        raise ValueError("未知 FP8 GEMM implementation")
    if left.device.type != "cuda" or right.device.type != "cuda":
        raise RuntimeError("scaled_mm 需要 CUDA，禁止回退为 CPU reference")
    if left.ndim != 2 or right.ndim != 2 or any(size % 16 for size in (*left.shape, *right.shape)):
        raise ValueError(
            "当前 scaled_mm 路径要求所有矩阵维度整除16，provider 需显式padding并计mask"
        )
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("本 PyTorch 构建没有 scaled_mm")

    right = right.T.contiguous().T
    output = torch._scaled_mm(
        left.contiguous(),
        right,
        scale_a=left_inverse,
        scale_b=right_inverse,
        out_dtype=torch.float32,
    )
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("scaled_mm 返回签名改变，需要版本适配，不能猜测")
    return output


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, module):
        flat = inputs.reshape(-1, inputs.shape[-1])
        qx, sx = module.inputs(flat)
        qw, sw = module.weights(weight)
        ctx.save_for_backward(qx, qw, sx, sw)
        ctx.module, ctx.input_shape, ctx.input_dtype, ctx.weight_dtype = (
            module,
            inputs.shape,
            inputs.dtype,
            weight.dtype,
        )
        ctx.has_bias = bias is not None
        output = fp8_matmul(qx, qw.T, sx, sw, implementation=module.implementation)
        if bias is not None:
            output += bias.float()
        return output.reshape(*inputs.shape[:-1], weight.shape[0]).to(inputs.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        qx, qw, sx, sw = ctx.saved_tensors
        flat = gradient.reshape(-1, gradient.shape[-1])
        qg, sg = ctx.module.gradients(flat)
        dx = fp8_matmul(qg, qw, sg, sw, implementation=ctx.module.implementation)
        dw = fp8_matmul(qg.T, qx, sg, sx, implementation=ctx.module.implementation)
        db = flat.sum(0) if ctx.has_bias else None
        return dx.reshape(ctx.input_shape).to(ctx.input_dtype), dw.to(ctx.weight_dtype), db, None


class FP8Linear(nn.Module):
    """Keep quantizer histories in checkpoint buffers and master weights at higher precision."""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=True,
        recipe=FP8Recipe(),
        implementation="reference",
        amax_group=None,
    ):
        super().__init__()
        if implementation not in {"reference", "scaled_mm"}:
            raise ValueError("FP8 implementation 必须显式 reference/scaled_mm")
        self.implementation, self.recipe = implementation, recipe
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
        self.inputs = FP8Quantizer(torch.float8_e4m3fn, recipe, group=amax_group)
        self.weights = FP8Quantizer(torch.float8_e4m3fn, recipe, group=amax_group)
        self.gradients = FP8Quantizer(torch.float8_e5m2, recipe, group=amax_group)

    @classmethod
    def from_linear(cls, layer, **kwargs):
        if type(layer) is not nn.Linear:
            raise TypeError("from_linear 仅接受原生 nn.Linear，不猜其他并行层语义")
        result = cls(
            layer.in_features, layer.out_features, bias=layer.bias is not None, **kwargs
        ).to(layer.weight.device)
        result.weight, result.bias = layer.weight, layer.bias
        return result

    def forward(self, inputs):
        with torch.autocast(inputs.device.type, enabled=False):
            return _FP8Linear.apply(inputs, self.weight, self.bias, self)

    def precision_contract(self):
        return {
            "implementation": self.implementation,
            "recipe": dict(vars(self.recipe)),
            "format": "hybrid_e4m3fn_e5m2",
            "amax_group": list(self.inputs.group.ranks),
            "history_clock": "each_quantizer_call",
        }
