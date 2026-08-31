"""Native deterministic LightningDiT/DitGen forward computation for Drifting.

References:
https://github.com/lambertae/drifting/blob/main/models/generator.py"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import FieldOutput
from .serialization import LocalModelMixin


def _linear(incoming, outgoing, *, normal=False, zero=False):
    layer = nn.Linear(incoming, outgoing)
    if zero:
        nn.init.zeros_(layer.weight)
    elif normal:
        nn.init.normal_(layer.weight, std=0.02)
    else:
        nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DriftRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return (normalized * self.weight).to(value.dtype)


class LearnedOffset(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.weight = nn.Parameter(initial)

    def forward(self, value):

        return (value.float() + self.weight.float()).to(value.dtype)


def _sincos_2d(width, grid):
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    frequency = 1 / 10000 ** (torch.arange(width // 4, dtype=torch.float64) / (width // 4))

    components = []
    for axis in (xx, yy):
        phase = axis.flatten()[:, None] * frequency[None]
        components += [phase.sin(), phase.cos()]
    return torch.cat(components, -1).float()[None]


def _rope(value):
    length, width = value.shape[-2:]
    frequency = 1 / 10000 ** (
        torch.arange(width // 2, device=value.device, dtype=value.dtype) / (width // 2)
    )
    phase = torch.arange(length, device=value.device, dtype=value.dtype)[:, None] * frequency[None]
    phase = torch.cat((phase, phase), -1)
    first, second = value.chunk(2, -1)
    return value * phase.cos() + torch.cat((-second, first), -1) * phase.sin()


class DriftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, dim = config.hidden_size, config.hidden_size // config.num_heads
        self.qkv, self.proj = _linear(width, 3 * width), _linear(width, width)
        norm = DriftRMSNorm if config.use_rmsnorm else lambda d: nn.LayerNorm(d, eps=1e-6)
        self.q_norm = norm(dim) if config.use_qknorm else nn.Identity()
        self.k_norm = norm(dim) if config.use_qknorm else nn.Identity()

    def forward(self, value):
        b, n, width = value.shape
        heads, dim = self.config.num_heads, width // self.config.num_heads
        q, k, v = self.qkv(value).reshape(b, n, 3, heads, dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.config.attn_fp32:
            q, k, v = q.float(), k.float(), v.float()
        if self.config.use_rope:
            q, k = _rope(q), _rope(k)

        with torch.autocast(value.device.type, enabled=False):
            score = (q * dim**-0.5) @ k.transpose(-1, -2)
            result = score.softmax(-1) @ v
        return self.proj(result.transpose(1, 2).reshape(b, n, width).to(value.dtype))


class DriftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = config.use_swiglu
        hidden = int(config.hidden_size * config.mlp_ratio)
        if self.swiglu:
            hidden = ((int(2 * hidden / 3) + 31) // 32) * 32
        self.up = _linear(config.hidden_size, hidden)
        self.gate = _linear(config.hidden_size, hidden) if self.swiglu else None
        self.down = _linear(hidden, config.hidden_size)

    def forward(self, value):
        hidden = (
            F.silu(self.up(value)) * self.gate(value)
            if self.swiglu
            else F.gelu(self.up(value), approximate="none")
        )
        return self.down(hidden)


class DriftBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm = (
            DriftRMSNorm
            if config.use_rmsnorm
            else lambda d: nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        )
        self.norm1, self.norm2 = norm(config.hidden_size), norm(config.hidden_size)
        self.attention, self.mlp = DriftAttention(config), DriftMLP(config)
        self.modulation = _linear(config.cond_dim, 6 * config.hidden_size, zero=True)

    def forward(self, value, condition):
        with torch.autocast(value.device.type, enabled=False):
            chunks = self.modulation(F.silu(condition.float())).to(value.dtype).chunk(6, -1)
        s1, a1, g1, s2, a2, g2 = (x[:, None] for x in chunks)
        value = value + g1 * self.attention(self.norm1(value) * (1 + a1) + s1)
        return value + g2 * self.mlp(self.norm2(value) * (1 + a2) + s2)


@dataclass(frozen=True)
class DriftingConfig:
    input_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 64
    cond_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 1001
    n_cls_tokens: int = 0
    noise_classes: int = 0
    noise_coords: int = 1
    use_qknorm: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    attn_fp32: bool = True
    prediction_type: str = "x0"

    def __post_init__(self):
        sizes = (
            self.input_size,
            self.in_channels,
            self.out_channels,
            self.patch_size,
            self.hidden_size,
            self.cond_dim,
            self.num_layers,
            self.num_heads,
            self.num_classes,
            self.noise_coords,
        )
        if (
            any(type(x) is not int or x < 1 for x in sizes)
            or self.input_size % self.patch_size
            or self.hidden_size % 4
            or self.hidden_size % self.num_heads
            or self.use_rope
            and (self.hidden_size // self.num_heads) % 2
            or type(self.n_cls_tokens) is not int
            or self.n_cls_tokens < 0
            or type(self.noise_classes) is not int
            or self.noise_classes < 0
            or not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
            or self.prediction_type != "x0"
        ):
            raise ValueError("Invalid native Drifting/LightningDiT configuration")

    def to_dict(self):
        return {"architecture": "drifting_generator", **asdict(self)}


class DriftingGenerator(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_embed = nn.Embedding(config.num_classes, config.cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        self.noise_embeds = nn.ModuleList(
            nn.Embedding(config.noise_classes, config.cond_dim)
            for _ in range(config.noise_coords)
            if config.noise_classes
        )
        for embedding in self.noise_embeds:
            nn.init.normal_(embedding.weight, std=0.02)
        self.cfg_embedder = nn.Sequential(
            _linear(256, config.cond_dim, normal=True),
            nn.SiLU(),
            _linear(config.cond_dim, config.cond_dim, normal=True),
        )
        self.cfg_norm = DriftRMSNorm(config.cond_dim)
        self.patch = _linear(config.patch_size**2 * config.in_channels, config.hidden_size)
        self.position = LearnedOffset(
            _sincos_2d(config.hidden_size, config.input_size // config.patch_size)
        )
        if config.n_cls_tokens:
            self.class_projection = _linear(config.cond_dim, config.hidden_size)
            self.class_position = LearnedOffset(
                torch.randn(1, config.n_cls_tokens, config.hidden_size) * 0.02
            )
        else:
            self.class_projection, self.class_position = None, None
        self.blocks = nn.ModuleList(DriftBlock(config) for _ in range(config.num_layers))
        self.norm = (
            DriftRMSNorm(config.hidden_size)
            if config.use_rmsnorm
            else nn.LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False)
        )
        self.modulation = _linear(config.cond_dim, 2 * config.hidden_size, zero=True)
        self.output = _linear(
            config.hidden_size, config.patch_size**2 * config.out_channels, zero=True
        )

    def forward(self, noise, cfg_scale, condition):
        config = self.config
        labels = condition.get("labels") if isinstance(condition, dict) else condition
        discrete_noise = condition.get("noise_labels") if isinstance(condition, dict) else None
        if (
            noise.ndim != 4
            or tuple(noise.shape[1:]) != (config.in_channels, config.input_size, config.input_size)
            or noise.dtype not in (torch.float32, torch.bfloat16)
            or not torch.isfinite(noise).all()
        ):
            raise ValueError(
                "Drifting noise must be finite FP32/BF16 BCHW matching its fixed patch geometry"
            )
        batch = len(noise)
        if (
            not isinstance(labels, torch.Tensor)
            or labels.shape != (batch,)
            or labels.dtype != torch.int64
            or labels.device != noise.device
            or (labels < 0).any()
            or (labels >= config.num_classes).any()
        ):
            raise ValueError(
                "Drifting needs aligned explicit class labels, not null/unconditional interpolation"
            )
        scale = torch.as_tensor(cfg_scale, device=noise.device, dtype=torch.float32)
        if scale.ndim == 0:
            scale = scale.expand(batch)
        if scale.shape != (batch,) or not torch.isfinite(scale).all() or (scale < 1).any():
            raise ValueError("Drifting guidance embedding requires finite cfg_scale >= 1")
        activation_dtype = (
            torch.get_autocast_dtype(noise.device.type)
            if torch.is_autocast_enabled(noise.device.type)
            else torch.float32
        )
        vector = self.class_embed(labels).to(activation_dtype)
        if self.noise_embeds:
            if (
                not isinstance(discrete_noise, torch.Tensor)
                or discrete_noise.shape != (batch, config.noise_coords)
                or discrete_noise.dtype != torch.int64
                or discrete_noise.device != noise.device
                or (discrete_noise < 0).any()
                or (discrete_noise >= config.noise_classes).any()
            ):
                raise ValueError(
                    "Discrete generator noise must be supplied explicitly as noise_labels"
                )
            for i, embedding in enumerate(self.noise_embeds):
                vector = vector + embedding(discrete_noise[:, i]).to(activation_dtype)
        elif discrete_noise is not None:
            raise ValueError("noise_labels provided to a model without discrete noise embeddings")
        frequency = torch.exp(-math.log(10000) * torch.arange(128, device=noise.device) / 128)
        phase = scale[:, None] * frequency[None]
        vector = vector + 0.02 * self.cfg_norm(
            self.cfg_embedder(torch.cat((phase.cos(), phase.sin()), -1))
        )
        p, grid = config.patch_size, config.input_size // config.patch_size

        patches = (
            noise.reshape(batch, config.in_channels, grid, p, grid, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, grid * grid, -1)
        )
        value = self.position(self.patch(patches))
        vector = vector.to(value.dtype)
        if self.class_projection is not None:
            tokens = self.class_projection(vector)[:, None].expand(-1, config.n_cls_tokens, -1)
            value = torch.cat((self.class_position(tokens), value), 1)
        for block in self.blocks:
            value = block(value, vector)
        with torch.autocast(value.device.type, enabled=False):
            shift, scale = self.modulation(F.silu(vector.float())).to(value.dtype).chunk(2, -1)
        value = self.output(self.norm(value) * (1 + scale[:, None]) + shift[:, None])
        value = value[:, config.n_cls_tokens :]
        sample = (
            value.reshape(batch, grid, grid, p, p, config.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, config.out_channels, config.input_size, config.input_size)
        )
        return FieldOutput(sample, "x0")

    @torch.no_grad()
    def generate(self, labels, *, cfg_scale=1.0, temperature=1.0, generator=None):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Noise temperature must be finite and positive")
        c = self.config
        noise = (
            torch.randn(
                len(labels),
                c.in_channels,
                c.input_size,
                c.input_size,
                device=labels.device,
                generator=generator,
            )
            * temperature
        )
        condition = labels
        if c.noise_classes:
            condition = dict(
                labels=labels,
                noise_labels=torch.randint(
                    c.noise_classes,
                    (len(labels), c.noise_coords),
                    device=labels.device,
                    generator=generator,
                ),
            )
        return self(noise, cfg_scale, condition).prediction
