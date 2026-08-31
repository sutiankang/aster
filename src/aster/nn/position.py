"""Rotary frequency scaling and coordinate layouts compatible with declared weights."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class RopeConfig:
    kind: str = "default"
    theta: float = 10000.0
    factor: float = 1.0
    original_max_position_embeddings: int = 8192
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    interleaved: bool = False

    def __post_init__(self):
        if self.kind not in {"default", "linear", "llama3", "yarn"}:
            raise ValueError(
                "Unsupported RoPE scaling; never silently replace it with default RoPE"
            )
        if self.theta <= 1 or self.factor < 1 or self.original_max_position_embeddings < 1:
            raise ValueError("Invalid RoPE frequency/scale/context")
        if (
            not 0 < self.low_freq_factor < self.high_freq_factor
            or not 0 < self.beta_slow <= self.beta_fast
        ):
            raise ValueError("Invalid RoPE interpolation boundaries")
        if self.attention_factor is not None and self.attention_factor <= 0:
            raise ValueError("attention_factor must be positive")

    def to_dict(self):
        return asdict(self)


class RotaryEmbedding(nn.Module):
    _aster_semantic_buffers = ("inv_freq",)

    def __init__(self, dim: int, config: RopeConfig):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("Rotary dimensions must be positive/even")
        self.config, self.dim = config, dim
        inv = 1 / (config.theta ** (torch.arange(0, dim, 2).float() / dim))
        amplitude = 1.0
        if config.kind == "linear":
            inv = inv / config.factor
        elif config.kind == "llama3":
            wavelength = 2 * math.pi / inv
            low = config.original_max_position_embeddings / config.low_freq_factor
            high = config.original_max_position_embeddings / config.high_freq_factor
            smooth = (
                config.original_max_position_embeddings / wavelength - config.low_freq_factor
            ) / (config.high_freq_factor - config.low_freq_factor)
            scaled = (1 - smooth) * inv / config.factor + smooth * inv
            inv = torch.where(
                wavelength > low, inv / config.factor, torch.where(wavelength < high, inv, scaled)
            )
        elif config.kind == "yarn":

            def correction(rotations):
                return (
                    dim
                    * math.log(config.original_max_position_embeddings / (rotations * 2 * math.pi))
                    / (2 * math.log(config.theta))
                )

            low, high = (
                max(math.floor(correction(config.beta_fast)), 0),
                min(math.ceil(correction(config.beta_slow)), dim - 1),
            )
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
            inv = inv * (1 - ramp) + (inv / config.factor) * ramp

            def mscale(value):
                return 1.0 if config.factor <= 1 else 0.1 * value * math.log(config.factor) + 1

            amplitude = (
                mscale(config.mscale) / mscale(config.mscale_all_dim)
                if config.mscale is not None and config.mscale_all_dim is not None
                else mscale(1.0)
            )
        self.attention_factor = (
            config.attention_factor if config.attention_factor is not None else amplitude
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, states, positions):
        if states.shape[-1] != self.dim or positions.shape != (states.shape[0], states.shape[-2]):
            raise ValueError("RoPE expects states[B,H,S,D] and positions[B,S]")

        with torch.autocast(device_type=states.device.type, enabled=False):
            angle = positions.float()[..., None] * self.inv_freq.float()[None, None]
            cos, sin = (
                angle.cos()[:, None] * self.attention_factor,
                angle.sin()[:, None] * self.attention_factor,
            )
        cos, sin = cos.to(states), sin.to(states)
        if self.config.interleaved:
            left, right = states[..., 0::2], states[..., 1::2]
        else:
            left, right = states.chunk(2, dim=-1)

        return torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
