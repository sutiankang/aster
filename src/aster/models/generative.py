"""Conditional fields and latent codecs; schedules and solvers remain in the methods layer."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import FieldOutput
from .serialization import LocalModelMixin


def timestep_embedding(time, dim, max_period=10000):

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=time.device).float() / max(half, 1)
    )
    angles = time.float().reshape(-1, 1) * frequencies
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


def _norm(channels):
    return nn.GroupNorm(math.gcd(channels, 32), channels, eps=1e-5)


def _zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_dim=None, dropout=0.0):
        super().__init__()
        self.norm1, self.conv1 = _norm(incoming), nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2, self.conv2 = _norm(outgoing), _zero(nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.time = nn.Linear(time_dim, outgoing * 2) if time_dim else None
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.time is not None:
            if time is None:
                raise ValueError("Time conditioning is required")
            scale, shift = self.time(F.silu(time)).to(h.dtype).chunk(2, -1)

            h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        return self.skip(x) + self.conv2(self.dropout(F.silu(h)))


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must divide into heads")
        self.norm, self.qkv = _norm(channels), nn.Conv1d(channels, 3 * channels, 1)
        self.proj, self.heads = _zero(nn.Conv1d(channels, channels, 1)), heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = (
            self.qkv(self.norm(x).flatten(2))
            .reshape(b, 3, self.heads, c // self.heads, h * w)
            .unbind(1)
        )

        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h * w)).reshape(b, c, h, w)


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 3
    model_channels: int = 32
    out_channels: int | None = None
    channel_mult: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = (1, 2)
    num_heads: int = 4
    dropout: float = 0.0
    condition_dim: int = 0
    num_classes: int = 0
    prediction_type: str = "epsilon"

    def __post_init__(self):
        object.__setattr__(self, "channel_mult", tuple(self.channel_mult))
        object.__setattr__(self, "attention_levels", tuple(self.attention_levels))
        if (
            min(self.in_channels, self.model_channels, self.num_res_blocks, self.num_heads) < 1
            or not self.channel_mult
        ):
            raise ValueError("Invalid UNet dimensions")
        if any(m < 1 for m in self.channel_mult) or any(
            i not in range(len(self.channel_mult)) for i in self.attention_levels
        ):
            raise ValueError("Invalid channel multipliers or attention levels")
        if not 0 <= self.dropout < 1 or self.condition_dim < 0 or self.num_classes < 0:
            raise ValueError("Invalid dropout or conditioning dimensions")
        if self.prediction_type not in {
            "epsilon",
            "x0",
            "v",
            "score",
            "velocity",
            "edm_residual",
            "consistency_residual",
        }:
            raise ValueError("Invalid prediction type")

    def to_dict(self):
        return {"architecture": "unet2d", **asdict(self)}


class UNet2D(LocalModelMixin, nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        c, time_dim = config.model_channels, 4 * config.model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.condition = nn.Linear(config.condition_dim, time_dim) if config.condition_dim else None
        self.classes = (
            nn.Embedding(config.num_classes + 1, time_dim) if config.num_classes else None
        )
        self.input = nn.Conv2d(config.in_channels, c, 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        channels, current = [c], c
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                outgoing = c * mult
                block = nn.ModuleDict(
                    {
                        "res": ResBlock(current, outgoing, time_dim, config.dropout),
                        "attn": SpatialAttention(outgoing, config.num_heads)
                        if level in config.attention_levels
                        else nn.Identity(),
                    }
                )
                self.down.append(block)
                current = outgoing
                channels.append(current)
            if level + 1 < len(config.channel_mult):
                self.down.append(
                    nn.ModuleDict(
                        {"downsample": nn.Conv2d(current, current, 3, stride=2, padding=1)}
                    )
                )
                channels.append(current)
        self.middle = nn.ModuleList(
            [
                ResBlock(current, current, time_dim),
                SpatialAttention(current, config.num_heads),
                ResBlock(current, current, time_dim),
            ]
        )
        for level in reversed(range(len(config.channel_mult))):
            for i in range(config.num_res_blocks + 1):
                outgoing = c * config.channel_mult[level]
                modules = {
                    "res": ResBlock(current + channels.pop(), outgoing, time_dim, config.dropout),
                    "attn": SpatialAttention(outgoing, config.num_heads)
                    if level in config.attention_levels
                    else nn.Identity(),
                }
                if level and i == config.num_res_blocks:
                    modules["upsample"] = nn.Conv2d(outgoing, outgoing, 3, padding=1)
                self.up.append(nn.ModuleDict(modules))
                current = outgoing
        assert not channels
        self.output = nn.Sequential(
            _norm(current),
            nn.SiLU(),
            _zero(nn.Conv2d(current, config.out_channels or config.in_channels, 3, padding=1)),
        )

    def forward(self, sample, time, condition=None):
        if sample.ndim != 4 or sample.shape[1] != self.config.in_channels:
            raise ValueError("UNet expects BCHW input")
        divisor = 2 ** (len(self.config.channel_mult) - 1)
        if sample.shape[-1] % divisor or sample.shape[-2] % divisor:
            raise ValueError("Image dimensions must divide by the downsample factor")
        if time.ndim == 0:
            time = time.expand(sample.shape[0])
        if time.shape != (sample.shape[0],):
            raise ValueError("One time per sample is required")
        t = self.time_embed(
            timestep_embedding(time, self.config.model_channels).to(self.input.weight.dtype)
        )
        if self.condition is not None:
            vector = (
                sample.new_zeros(sample.shape[0], self.config.condition_dim)
                if condition is None
                else condition
            )
            t = t + self.condition(vector)
        elif self.classes is not None:
            labels = (
                torch.full(
                    (sample.shape[0],),
                    self.config.num_classes,
                    device=sample.device,
                    dtype=torch.long,
                )
                if condition is None
                else condition
            )
            t = t + self.classes(labels)
        elif condition is not None:
            raise ValueError("Unconditional UNet does not accept condition")
        h = self.input(sample)
        skips = [h]
        for block in self.down:
            h = (
                block["downsample"](h)
                if "downsample" in block
                else block["attn"](block["res"](h, t))
            )
            skips.append(h)
        h = self.middle[2](self.middle[1](self.middle[0](h, t)), t)
        for block in self.up:
            h = block["attn"](block["res"](torch.cat((h, skips.pop()), dim=1), t))
            if "upsample" in block:
                h = block["upsample"](F.interpolate(h, scale_factor=2, mode="nearest"))
        return FieldOutput(self.output(h), self.config.prediction_type)


class AdaLNBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(width * mlp_ratio), width),
        )
        self.ada = nn.Sequential(nn.SiLU(), _zero(nn.Linear(width, 6 * width)))
        self.heads = heads

    def forward(self, x, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(condition).chunk(6, -1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        b, n, c = h.shape
        q, k, v = (
            self.qkv(h)
            .reshape(b, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + gate1[:, None] * self.proj(h)
        return x + gate2[:, None] * self.mlp(
            self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        )


@dataclass(frozen=True)
class DiTConfig:
    in_channels: int = 4
    out_channels: int | None = None
    patch_size: int = 2
    hidden_size: int = 64
    num_layers: int = 3
    num_heads: int = 4
    mlp_ratio: float = 4.0
    condition_dim: int = 0
    num_classes: int = 0
    prediction_type: str = "velocity"

    def __post_init__(self):
        if (
            min(
                self.in_channels, self.patch_size, self.hidden_size, self.num_layers, self.num_heads
            )
            < 1
            or self.hidden_size % self.num_heads
            or self.hidden_size % 4
        ):
            raise ValueError(
                "Invalid DiT dimensions; width must divide into heads and four sin/cos parts"
            )
        if self.condition_dim < 0 or self.num_classes < 0 or self.mlp_ratio <= 0:
            raise ValueError("Invalid DiT configuration")

    def to_dict(self):
        return {"architecture": "dit", **asdict(self)}


class DiT(LocalModelMixin, nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        d, p = config.hidden_size, config.patch_size
        self.patch = nn.Conv2d(config.in_channels, d, p, stride=p)
        self.time = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.condition = nn.Linear(config.condition_dim, d) if config.condition_dim else None
        self.classes = nn.Embedding(config.num_classes + 1, d) if config.num_classes else None
        self.blocks = nn.ModuleList(
            [AdaLNBlock(d, config.num_heads, config.mlp_ratio) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), _zero(nn.Linear(d, 2 * d)))
        self.output = _zero(nn.Linear(d, p * p * (config.out_channels or config.in_channels)))

    def forward(self, sample, time, condition=None):
        b, c, h, w = sample.shape
        p, d = self.config.patch_size, self.config.hidden_size
        if c != self.config.in_channels or h % p or w % p:
            raise ValueError("DiT expects channels and patch-divisible BCHW dimensions")
        x = self.patch(sample).flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.arange(h // p, device=x.device),
            torch.arange(w // p, device=x.device),
            indexing="ij",
        )
        x = (
            x
            + torch.cat(
                (
                    timestep_embedding(xx.flatten(), d // 2),
                    timestep_embedding(yy.flatten(), d // 2),
                ),
                -1,
            ).to(x.dtype)[None]
        )
        if time.ndim == 0:
            time = time.expand(b)
        if time.shape != (b,):
            raise ValueError("One time per sample is required")
        t = self.time(timestep_embedding(time, 256).to(x.dtype))
        if self.condition is not None:
            t = t + self.condition(
                x.new_zeros(b, self.config.condition_dim) if condition is None else condition
            )
        elif self.classes is not None:
            t = t + self.classes(
                torch.full((b,), self.config.num_classes, device=x.device, dtype=torch.long)
                if condition is None
                else condition
            )
        elif condition is not None:
            raise ValueError("Unconditional DiT does not accept condition")
        for block in self.blocks:
            x = block(x, t)
        shift, scale = self.ada(t).chunk(2, -1)
        x = self.output(self.norm(x) * (1 + scale[:, None]) + shift[:, None])
        channels = self.config.out_channels or c
        x = (
            x.reshape(b, h // p, w // p, p, p, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, channels, h, w)
        )
        return FieldOutput(x, self.config.prediction_type)


@dataclass
class DiagonalGaussian:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self, generator=None):
        return self.mean + (0.5 * self.logvar).exp() * torch.randn(
            self.mean.shape, dtype=self.mean.dtype, device=self.mean.device, generator=generator
        )

    def mode(self):
        return self.mean

    def kl(self):
        """Sum KL(q(z|x) || N(0,I)) over latent coordinates and preserve the sample dimension."""
        dtype = torch.promote_types(self.mean.dtype, self.logvar.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        mean, logvar = self.mean.to(dtype), self.logvar.to(dtype)
        return 0.5 * (mean.square() + torch.expm1(logvar) - logvar).flatten(1).sum(1)


@dataclass(frozen=True)
class AutoencoderConfig:
    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 32
    channel_mult: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    scaling_factor: float = 1.0
    shift_factor: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "channel_mult", tuple(self.channel_mult))
        if (
            min(self.in_channels, self.latent_channels, self.base_channels, self.num_res_blocks) < 1
            or not self.channel_mult
            or min(self.channel_mult) < 1
        ):
            raise ValueError("Invalid autoencoder dimensions")
        if (
            not math.isfinite(self.scaling_factor)
            or self.scaling_factor <= 0
            or not math.isfinite(self.shift_factor)
        ):
            raise ValueError("Invalid latent transform")

    def to_dict(self):
        return {"architecture": "autoencoder_kl", **asdict(self)}


class AutoencoderKL(LocalModelMixin, nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        c = config.base_channels
        encoder = [nn.Conv2d(config.in_channels, c, 3, padding=1)]
        current = c
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                encoder.append(ResBlock(current, c * mult))
                current = c * mult
            if level + 1 < len(config.channel_mult):
                encoder.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
        encoder += [
            ResBlock(current, current),
            SpatialAttention(current, 1),
            ResBlock(current, current),
            _norm(current),
            nn.SiLU(),
            nn.Conv2d(current, 2 * config.latent_channels, 3, padding=1),
        ]
        self.encoder = nn.Sequential(*encoder)
        self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)
        decoder = [
            nn.Conv2d(config.latent_channels, current, 3, padding=1),
            ResBlock(current, current),
            SpatialAttention(current, 1),
            ResBlock(current, current),
        ]
        for level in reversed(range(len(config.channel_mult))):
            for _ in range(config.num_res_blocks + 1):
                decoder.append(ResBlock(current, c * config.channel_mult[level]))
                current = c * config.channel_mult[level]
            if level:
                decoder += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current, current, 3, padding=1),
                ]
        decoder += [_norm(current), nn.SiLU(), nn.Conv2d(current, config.in_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, images):
        divisor = 2 ** (len(self.config.channel_mult) - 1)
        if (
            images.ndim != 4
            or images.shape[1] != self.config.in_channels
            or any(s % divisor for s in images.shape[-2:])
        ):
            raise ValueError("Invalid VAE image dimensions")
        mean, logvar = self.quant_conv(self.encoder(images)).chunk(2, dim=1)
        return DiagonalGaussian(mean, logvar.clamp(-30, 20))

    def decode(self, latent, *, scaled=False):
        if scaled:
            latent = latent / self.config.scaling_factor + self.config.shift_factor
        return self.decoder(self.post_quant_conv(latent))

    def latent(self, images, *, sample=True, generator=None):
        posterior = self.encode(images)
        raw = posterior.sample(generator) if sample else posterior.mode()
        return (raw - self.config.shift_factor) * self.config.scaling_factor

    def forward(self, images, *, sample_posterior=True, generator=None):
        posterior = self.encode(images)
        z = posterior.sample(generator) if sample_posterior else posterior.mode()
        return self.decode(z), posterior
