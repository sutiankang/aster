"""Wan2.1 video fields with 3D patches, rotary positions, and image cross-attention."""

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import FieldOutput
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class WanVideoConfig:
    latent_channels: int = 16
    condition_channels: int = 0
    hidden_size: int = 96
    intermediate_size: int = 384
    num_heads: int = 4
    num_layers: int = 4
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 4096
    text_length: int = 512
    image_dim: int = 1280
    frequency_dim: int = 256
    image_conditioned: bool = False
    first_last_frames: bool = False
    image_tokens_per_frame: int = 257
    qk_norm: bool = True
    cross_attention_norm: bool = True
    norm_eps: float = 1e-6
    time_scale: float = 1000.0
    window: tuple[int, int] = (-1, -1)

    def __post_init__(self):
        object.__setattr__(self, "patch_size", tuple(self.patch_size))
        object.__setattr__(self, "window", tuple(self.window))
        dimensions = (
            self.latent_channels,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.num_layers,
            self.text_dim,
            self.text_length,
            self.image_dim,
            self.frequency_dim,
            self.image_tokens_per_frame,
            *self.patch_size,
        )
        if any(type(v) is not int or v < 1 for v in dimensions) or len(self.patch_size) != 3:
            raise ValueError("Wan requires positive integer dimensions and a 3D patch")
        if (
            self.hidden_size % self.num_heads
            or (self.hidden_size // self.num_heads) % 2
            or self.frequency_dim % 2
        ):
            raise ValueError("Wan head and frequency dimensions must be even")
        if type(self.condition_channels) is not int or self.condition_channels < 0:
            raise ValueError("Invalid video conditioning channels")
        if self.image_conditioned != bool(self.condition_channels) or (
            self.first_last_frames and not self.image_conditioned
        ):
            raise ValueError(
                "Image-conditioned Wan requires both image features and conditional video channels"
            )
        if len(self.window) != 2 or any(type(v) is not int or v < -1 for v in self.window):
            raise ValueError("Attention window is (left,right), -1 means unlimited")
        if any(not math.isfinite(v) or v <= 0 for v in (self.norm_eps, self.time_scale)):
            raise ValueError("Invalid Wan normalization/time scale")

    def to_dict(self):
        return {"architecture": "wan21_video", **asdict(self)}


class WanRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value):

        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def forward(self, value):
        return F.layer_norm(
            value.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(value.dtype)


class WanModulation(nn.Module):
    """Own modulation parameters in a computational leaf compatible with ZeRO gather/release."""

    def __init__(self, branches, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, branches, width) / math.sqrt(width))

    def forward(self, condition):
        if condition.ndim == 2:
            condition = condition[:, None]
        return self.weight.float() + condition.float()


class WanImagePosition(nn.Module):
    def __init__(self, tokens, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, tokens, width))

    def forward(self, features):
        return features + self.weight


def wan_time_embedding(time, width):
    frequency = 10000 ** (
        -torch.arange(width // 2, device=time.device, dtype=torch.float64) / (width // 2)
    )
    angle = time.double()[:, None] * frequency
    return torch.cat((angle.cos(), angle.sin()), -1).float()


def wan_rope(value, grid):

    frames, height, width = grid
    sequence = frames * height * width
    half = value.shape[-1] // 2
    sizes = (half - 2 * (half // 3), half // 3, half // 3)
    coordinates = torch.meshgrid(
        *(torch.arange(n, device=value.device, dtype=torch.float64) for n in grid), indexing="ij"
    )
    phases = []
    for coordinate, size in zip(coordinates, sizes):
        frequency = 10000 ** (
            -torch.arange(size, device=value.device, dtype=torch.float64) / max(size, 1)
        )
        phases.append(coordinate.reshape(-1, 1) * frequency)
    angle = torch.cat(phases, -1)[None, :, None]
    pairs = value[:, :sequence].double().reshape(*value[:, :sequence].shape[:-1], half, 2)
    real, imaginary = pairs.unbind(-1)
    rotated = torch.stack(
        (
            real * angle.cos() - imaginary * angle.sin(),
            real * angle.sin() + imaginary * angle.cos(),
        ),
        -1,
    ).flatten(-2)
    return torch.cat((rotated.to(value.dtype), value[:, sequence:]), 1)


class WanAttention(nn.Module):
    def __init__(self, config, *, cross=False):
        super().__init__()
        width = config.hidden_size
        self.config, self.cross = config, cross
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q = WanRMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()
        if cross and config.image_conditioned:
            self.k_img, self.v_img = nn.Linear(width, width), nn.Linear(width, width)
            self.norm_k_img = (
                WanRMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()
            )

    def forward(self, value, *, grid=None, text=None, image=None):
        b, length, width = value.shape
        heads = self.config.num_heads
        split = lambda x: x.reshape(b, -1, heads, width // heads)
        source = text if self.cross else value
        q, k, v = (
            split(self.norm_q(self.q(value))),
            split(self.norm_k(self.k(source))),
            split(self.v(source)),
        )
        mask = None
        if not self.cross:
            q, k = wan_rope(q, grid), wan_rope(k, grid)
            positions = torch.arange(length, device=value.device)
            mask = (positions[None, :] < math.prod(grid)).expand(length, -1)
            left, right = self.config.window
            if left >= 0:
                mask = mask & (positions[None, :] >= positions[:, None] - left)
            if right >= 0:
                mask = mask & (positions[None, :] <= positions[:, None] + right)
        attend = lambda query, key, val, attention_mask=None: (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                val.transpose(1, 2),
                attn_mask=attention_mask,
            )
            .transpose(1, 2)
            .reshape(b, length, width)
        )
        out = attend(q, k, v, mask)
        if self.cross and self.config.image_conditioned:
            out = out + attend(
                q, split(self.norm_k_img(self.k_img(image))), split(self.v_img(image))
            )
        return self.o(out)


class WanVideoBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.modulation = WanModulation(6, width)
        self.norm1 = WanLayerNorm(width, eps=config.norm_eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(width, eps=config.norm_eps, elementwise_affine=False)
        self.norm3 = (
            WanLayerNorm(width, eps=config.norm_eps)
            if config.cross_attention_norm
            else nn.Identity()
        )
        self.self_attn, self.cross_attn = WanAttention(config), WanAttention(config, cross=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, width),
        )

    def forward(self, value, modulation, grid, text, image):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(modulation).unbind(1)
        affine = lambda x, shift, scale: (x.float() * (1 + scale[:, None]) + shift[:, None]).to(
            value.dtype
        )
        residual = self.self_attn(affine(self.norm1(value), shift1, scale1), grid=grid)
        value = (value.float() + residual.float() * gate1[:, None]).to(value.dtype)
        value = value + self.cross_attn(self.norm3(value), text=text, image=image)
        residual = self.ffn(affine(self.norm2(value), shift2, scale2))
        return (value.float() + residual.float() * gate2[:, None]).to(value.dtype)


class WanVideoHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modulation = WanModulation(2, config.hidden_size)
        self.norm = WanLayerNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=False)
        self.head = nn.Linear(
            config.hidden_size, math.prod(config.patch_size) * config.latent_channels
        )

    def forward(self, value, time_embedding):
        shift, scale = self.modulation(time_embedding).unbind(1)
        return self.head(
            (self.norm(value).float() * (1 + scale[:, None]) + shift[:, None]).to(value.dtype)
        )


@dataclass(frozen=True)
class WanPreparedInput:
    hidden: torch.Tensor
    embedding: torch.Tensor
    modulation: torch.Tensor
    grid: tuple[int, int, int]
    text: torch.Tensor
    image: torch.Tensor | None
    tokens: int


class WanVideoDiT(LocalModelMixin, nn.Module):
    def __init__(self, config: WanVideoConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.patch_embedding = nn.Conv3d(
            config.latent_channels + config.condition_channels,
            width,
            config.patch_size,
            stride=config.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, width), nn.GELU(approximate="tanh"), nn.Linear(width, width)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.frequency_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        self.blocks = nn.ModuleList(WanVideoBlock(config) for _ in range(config.num_layers))
        self.head = WanVideoHead(config)
        if config.image_conditioned:
            self.image_projection = nn.Sequential(
                nn.LayerNorm(config.image_dim),
                nn.Linear(config.image_dim, config.image_dim),
                nn.GELU(),
                nn.Linear(config.image_dim, width),
                nn.LayerNorm(width),
            )
            if config.first_last_frames:
                self.image_position = WanImagePosition(
                    2 * config.image_tokens_per_frame, config.image_dim
                )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for branch in (self.text_embedding, self.time_embedding):
            for module in branch.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    def forward(self, sample, time, condition=None, *, sequence_length=None):
        prepared = self.prepare(sample, time, condition, sequence_length=sequence_length)
        return self.finish(self.run_blocks(prepared), prepared)

    def prepare(self, sample, time, condition=None, *, sequence_length=None):

        c = self.config
        if sample.ndim != 5 or min(sample.shape) < 1 or sample.shape[1] != c.latent_channels:
            raise ValueError("Wan expects nonempty B,C,T,H,W latent videos")
        if not sample.is_floating_point() or any(
            n % p for n, p in zip(sample.shape[2:], c.patch_size)
        ):
            raise ValueError("Video dimensions must divide exactly into 3D patches")
        if not isinstance(condition, dict) or set(condition) - {
            "text",
            "text_lengths",
            "image_features",
            "video_condition",
        }:
            raise ValueError("Wan requires an explicit typed text/image/video condition mapping")
        b = sample.shape[0]
        text = condition.get("text")
        if (
            not isinstance(text, torch.Tensor)
            or text.ndim != 3
            or text.shape[0] != b
            or text.shape[2] != c.text_dim
            or not 0 < text.shape[1] <= c.text_length
        ):
            raise ValueError("Text features must be B,L,text_dim with L <= configured text_length")
        lengths = condition.get("text_lengths")
        if lengths is not None:
            if (
                lengths.shape != (b,)
                or lengths.dtype not in (torch.int32, torch.int64)
                or bool(((lengths < 0) | (lengths > text.shape[1])).any())
            ):
                raise ValueError("Invalid original text lengths")
            text = text.masked_fill(
                (torch.arange(text.shape[1], device=text.device)[None] >= lengths[:, None])[
                    ..., None
                ],
                0,
            )

        text = self.text_embedding(F.pad(text, (0, 0, 0, c.text_length - text.shape[1])))
        image = None
        if c.image_conditioned:
            image, video = condition.get("image_features"), condition.get("video_condition")
            if not isinstance(video, torch.Tensor) or video.shape != (
                b,
                c.condition_channels,
                *sample.shape[2:],
            ):
                raise ValueError("Conditional video shape/channel mismatch")
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 3
                or image.shape[0] != b
                or image.shape[2] != c.image_dim
                or image.shape[1] < 1
            ):
                raise ValueError("Image features must be B,L,image_dim")
            if c.first_last_frames:
                if image.shape[1] != 2 * c.image_tokens_per_frame:
                    raise ValueError("First/last frame image token count mismatch")
                image = self.image_position(image)
            image = self.image_projection(image)
            sample = torch.cat((sample, video), 1)
        elif any(key in condition for key in ("image_features", "video_condition")):
            raise ValueError("Text-only Wan cannot silently ignore image conditions")
        if time.ndim == 0:
            time = time.expand(b)
        if (
            time.shape != (b,)
            or not torch.isfinite(time).all()
            or bool(((time < 0) | (time > 1)).any())
        ):
            raise ValueError("Wan time is one normalized noise fraction sigma in [0,1] per video")
        hidden = self.patch_embedding(sample)
        grid = hidden.shape[2:]
        hidden = hidden.flatten(2).transpose(1, 2)
        tokens = hidden.shape[1]
        if sequence_length is not None:
            if type(sequence_length) is not int or sequence_length < tokens:
                raise ValueError("Sequence padding cannot drop real video tokens")
            hidden = F.pad(hidden, (0, 0, 0, sequence_length - tokens))
        embedding = self.time_embedding(
            wan_time_embedding(time * c.time_scale, c.frequency_dim).to(hidden.dtype)
        )
        modulation = self.time_projection(embedding).reshape(b, 6, c.hidden_size)
        return WanPreparedInput(hidden, embedding, modulation, tuple(grid), text, image, tokens)

    def run_blocks(self, prepared):

        if not isinstance(prepared, WanPreparedInput):
            raise TypeError("Wan backbone requires its typed prepared input")
        hidden = prepared.hidden
        for block in self.blocks:
            hidden = block(
                hidden, prepared.modulation, prepared.grid, prepared.text, prepared.image
            )
        return hidden

    def finish(self, hidden, prepared):

        if not isinstance(prepared, WanPreparedInput) or hidden.shape != prepared.hidden.shape:
            raise ValueError("Wan output head requires the matching prepared geometry")
        c, b, grid = self.config, hidden.shape[0], prepared.grid
        patches = self.head(hidden, prepared.embedding)[:, : prepared.tokens]
        patches = patches.reshape(b, *grid, *c.patch_size, c.latent_channels)

        video = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            b, c.latent_channels, *(n * p for n, p in zip(grid, c.patch_size))
        )
        return FieldOutput(video.float(), "velocity")
