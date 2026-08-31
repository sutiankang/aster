"""Interval-conditioned DiT with distinct MeanFlow duration and Shortcut step semantics."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import FieldOutput
from .serialization import LocalModelMixin
from .generative import timestep_embedding
from .drifting import _sincos_2d


def _dense(incoming, outgoing, *, zero=False, normal=False):
    module = nn.Linear(incoming, outgoing)
    if zero:
        nn.init.zeros_(module.weight)
    elif normal:
        nn.init.normal_(module.weight, std=0.02)
    else:
        nn.init.xavier_uniform_(module.weight)
    nn.init.zeros_(module.bias)
    return module


class IntervalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, hidden = config.hidden_size, int(config.hidden_size * config.mlp_ratio)
        self.heads, self.variant = config.num_heads, config.variant
        self.norm1, self.norm2 = (
            nn.LayerNorm(d, eps=1e-6, elementwise_affine=False),
            nn.LayerNorm(d, eps=1e-6, elementwise_affine=False),
        )
        self.qkv, self.projection = _dense(d, 3 * d), _dense(d, d)
        if config.variant == "shortcut":
            for section in self.qkv.weight.chunk(3, 0):
                nn.init.xavier_uniform_(section)
        self.mlp = nn.Sequential(_dense(d, hidden), nn.GELU(approximate="tanh"), _dense(hidden, d))
        self.modulation = nn.Sequential(
            nn.SiLU(), _dense(d, 6 * d, zero=config.variant == "meanflow")
        )

    def forward(self, x, conditioning):
        s1, a1, g1, s2, a2, g2 = (
            value[:, None] for value in self.modulation(conditioning).chunk(6, -1)
        )
        normalized = self.norm1(x) * (1 + a1) + s1
        b, length, width = x.shape
        dimension = width // self.heads
        q, k, v = (
            self.qkv(normalized)
            .reshape(b, length, 3, self.heads, dimension)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = q * (dimension ** (-0.5 if self.variant == "meanflow" else -1.0))

        attention = (q @ k.transpose(-1, -2)).float().softmax(-1)
        with torch.autocast(x.device.type, enabled=False):
            y = (attention @ v.float()).transpose(1, 2).reshape(b, length, width)
        x = x + g1 * self.projection(y)
        return x + g2 * self.mlp(self.norm2(x) * (1 + a2) + s2)


@dataclass(frozen=True)
class IntervalDiTConfig:
    variant: str = "meanflow"
    input_size: int = 32
    in_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 1000

    def __post_init__(self):
        if (
            self.variant not in {"meanflow", "shortcut"}
            or any(
                type(x) is not int or x < 1
                for x in (
                    self.input_size,
                    self.in_channels,
                    self.patch_size,
                    self.hidden_size,
                    self.num_layers,
                    self.num_heads,
                    self.num_classes,
                )
            )
            or self.input_size % self.patch_size
            or self.hidden_size % self.num_heads
            or self.hidden_size % 4
            or not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
        ):
            raise ValueError("Invalid interval DiT variant or geometry")

    @property
    def interval_semantics(self):
        return "duration" if self.variant == "meanflow" else "negative_log2_step"

    def to_dict(self):
        return {"architecture": "interval_dit", **asdict(self)}


class IntervalDiT(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, p = config.hidden_size, config.patch_size
        self.patch = nn.Conv2d(config.in_channels, d, p, stride=p)

        nn.init.xavier_uniform_(self.patch.weight.flatten(1))
        nn.init.zeros_(self.patch.bias)
        self.time = nn.Sequential(_dense(256, d, normal=True), nn.SiLU(), _dense(d, d, normal=True))
        self.interval = nn.Sequential(
            _dense(256, d, normal=True), nn.SiLU(), _dense(d, d, normal=True)
        )
        self.classes = nn.Embedding(config.num_classes + 1, d)
        nn.init.normal_(self.classes.weight, std=0.02)
        self.register_buffer("position", _sincos_2d(d, config.input_size // p), persistent=True)
        self.blocks = nn.ModuleList(IntervalBlock(config) for _ in range(config.num_layers))
        self.norm = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), _dense(d, 2 * d, zero=True))
        self.output = _dense(d, p * p * config.in_channels, zero=True)

    def forward(self, sample, time, interval, condition=None):
        c = self.config
        if (
            sample.ndim != 4
            or tuple(sample.shape[1:]) != (c.in_channels, c.input_size, c.input_size)
            or not sample.is_floating_point()
            or not torch.isfinite(sample).all()
        ):
            raise ValueError("Interval DiT expects finite BCHW samples matching its fixed geometry")
        b = len(sample)
        for name, value in [("time", time), ("interval", interval)]:
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (b,)
                or value.device != sample.device
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
                or (value < 0).any()
            ):
                raise ValueError(f"{name} must be finite floating B values on the sample device")
        if (time > 1).any() or c.variant == "meanflow" and (interval > time).any():
            raise ValueError("MeanFlow requires 0 <= duration <= time <= 1")
        labels = (
            torch.full((b,), c.num_classes, dtype=torch.int64, device=sample.device)
            if condition is None
            else condition
        )
        if (
            labels.shape != (b,)
            or labels.dtype != torch.int64
            or labels.device != sample.device
            or (labels < 0).any()
            or (labels > c.num_classes).any()
        ):
            raise ValueError(
                "Interval DiT requires class indices; final class is the explicit null label"
            )
        x = self.patch(sample).flatten(2).transpose(1, 2)
        x = (x.float() + self.position.float()).to(x.dtype)
        conditioning = (
            self.time(timestep_embedding(time, 256))
            + self.interval(timestep_embedding(interval, 256))
            + self.classes(labels).to(x.dtype)
        )
        for block in self.blocks:
            x = block(x, conditioning)
        shift, scale = self.modulation(conditioning).chunk(2, -1)
        x = self.output(self.norm(x) * (1 + scale[:, None]) + shift[:, None])
        p, grid = c.patch_size, c.input_size // c.patch_size
        x = (
            x.reshape(b, grid, grid, p, p, c.in_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape_as(sample)
        )
        return FieldOutput(x, "average_velocity")
