"""Native normalization formulas with explicit precision and parameter ownership."""

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Normalize by root mean square without subtracting the mean.
    FP16/BF16 statistics are computed in FP32 to avoid overflow when squaring."""

    def __init__(self, size: int, eps: float = 1e-6, *, zero_centered=False):
        super().__init__()
        if size < 1 or eps <= 0:
            raise ValueError("Invalid normalization dimensions/epsilon")
        self.weight = nn.Parameter(torch.zeros(size) if zero_centered else torch.ones(size))
        self.variance_epsilon, self.zero_centered = eps, zero_centered

    def forward(self, hidden):

        values = hidden if hidden.dtype == torch.float64 else hidden.float()
        values = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        if self.zero_centered:
            return (values * (self.weight.float() + 1)).to(hidden.dtype)
        return values.to(hidden.dtype) * self.weight


class LayerNorm(nn.Module):
    """Mean-centered LayerNorm with optional bias; not interchangeable with RMSNorm."""

    def __init__(self, size, eps=1e-5, *, bias=True):
        super().__init__()
        if size < 1 or eps <= 0:
            raise ValueError("Invalid normalization dimensions/epsilon")
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None
        self.eps = eps

    def forward(self, hidden):
        return F.layer_norm(hidden, self.weight.shape, self.weight, self.bias, self.eps)


class FloatRMSNorm(nn.Module):
    """Compute RMS statistics and scale multiplication in FP32 before casting back."""

    def __init__(self, size, eps=1e-6, *, with_scale=True):
        super().__init__()
        if size < 1 or eps <= 0:
            raise ValueError("Invalid RMS dimensions/epsilon")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size)) if with_scale else None

    def forward(self, hidden):
        values = hidden.float()
        values = values * torch.pow(values.square().mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            values = values * self.weight.float()
        return values.to(hidden.dtype)


class _BatchNormAffine(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(size)), nn.Parameter(torch.zeros(size))

    def forward(self, normalized):
        dtype = torch.float64 if normalized.dtype == torch.float64 else torch.float32
        return (normalized.to(dtype) * self.weight.to(dtype) + self.bias.to(dtype)).to(
            normalized.dtype
        )


class BatchNorm1d(nn.Module):
    """Two-dimensional [N,C] normalization with separate statistics and affine leaves,
    preventing repeated running-stat updates during ZeRO-3 recomputation."""

    def __init__(self, size, eps=1e-5, momentum=0.1):
        super().__init__()
        from .parameter_codec import register_parameter_codec

        if type(size) is not int or size < 1 or not 0 < eps or not 0 <= momentum <= 1:
            raise ValueError("Invalid BatchNorm configuration")
        self.size, self.eps, self.momentum = size, eps, momentum
        self.affine = _BatchNormAffine(size)
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_var", torch.ones(size))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))
        register_parameter_codec(self, {"affine.weight": "weight", "affine.bias": "bias"})

    def forward(self, value):
        if value.ndim != 2 or value.shape[-1] != self.size:
            raise ValueError("BatchNorm1d expects explicit [N,C] rows")
        if self.training:
            if len(value) < 2:
                raise ValueError("Training BatchNorm requires at least two rows")
            self.num_batches_tracked.add_(1)
        normalized = F.batch_norm(
            value,
            self.running_mean,
            self.running_var,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )
        return self.affine(normalized)
