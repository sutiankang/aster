"""Native momentum and Newton-Schulz matrix updates."""

from __future__ import annotations

import math
import torch


def orthogonalize(gradient, *, steps=5, eps=1e-7, coefficients=(3.4445, -4.7750, 2.0315)):
    if gradient.ndim != 2 or not gradient.is_floating_point():
        raise ValueError("Muon 只接收浮点二维矩阵")
    if type(steps) is not int or not 1 <= steps < 100 or eps <= 0:
        raise ValueError("非法 Newton–Schulz 配置")
    value = gradient.to(torch.bfloat16).clone()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    value.div_(value.norm().clamp(min=eps))
    a, b, c = coefficients
    for _ in range(steps):
        gram = value @ value.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        value = torch.addmm(value, polynomial, value, beta=a)
    return value.T if transposed else value


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
        eps=1e-7,
        adjust_lr="original",
    ):
        if not math.isfinite(lr) or lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("非法 Muon 超参数")
        if adjust_lr not in {"original", "match_rms_adamw", "none"}:
            raise ValueError("未知 Muon LR 校正")
        if (
            type(ns_steps) is not int
            or not 1 <= ns_steps < 100
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("非法 Muon 迭代参数")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_steps=ns_steps,
                eps=eps,
                adjust_lr=adjust_lr,
            ),
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2 or not parameter.is_floating_point():
                    raise ValueError("Muon 参数必须为二维矩阵")
                if getattr(parameter, "_aster_tp_sharded", False):
                    raise ValueError("本地矩阵 Muon 不支持 TP 分片，需全矩阵正交化再重分片")

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
                    continue
                if gradient.is_sparse:
                    raise ValueError("Muon 不支持稀疏梯度")
                state = self.state[parameter]
                momentum = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                momentum.lerp_(gradient, 1 - group["momentum"])
                direction = (
                    gradient.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum
                )
                direction = orthogonalize(direction, steps=group["ns_steps"], eps=group["eps"])
                rows, columns = parameter.shape
                ratio = (
                    math.sqrt(max(1.0, rows / columns))
                    if group["adjust_lr"] == "original"
                    else 0.2 * math.sqrt(max(rows, columns))
                    if group["adjust_lr"] == "match_rms_adamw"
                    else 1.0
                )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"] * ratio)
        return loss
