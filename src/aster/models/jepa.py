"""Native image/video JEPA encoding and masked-token prediction."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from .serialization import LocalModelMixin


def _sincos(coordinate, width):
    if width % 2:
        raise ValueError("Sinusoidal axis width must be even")
    frequency = 10000.0 ** (-torch.arange(width // 2, dtype=torch.float64) / (width // 2))
    phase = coordinate.reshape(-1).double()[:, None] * frequency[None]
    return torch.cat((phase.sin(), phase.cos()), -1)


def jepa_positions(grid, width, *, uniform_power=False):

    if len(grid) == 2:
        height, horizontal = torch.meshgrid(
            torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij"
        )
        return torch.cat(
            (_sincos(height, width // 2), _sincos(horizontal, width // 2)), -1
        ).float()[None]
    time, height, horizontal = torch.meshgrid(*(torch.arange(size) for size in grid), indexing="ij")
    if uniform_power:
        widths = (math.ceil(width / 6) * 2,) * 3
    else:
        widths = (width // 2, width // 4, width // 4)
    return torch.cat(
        [_sincos(axis, part) for axis, part in zip((time, height, horizontal), widths)], -1
    )[..., :width].float()[None]


def select_patches(tokens, indices):
    if (
        indices.ndim != 2
        or len(indices) != len(tokens)
        or indices.dtype != torch.long
        or indices.shape[1] < 1
        or (indices < 0).any()
        or (indices >= tokens.shape[1]).any()
    ):
        raise ValueError("Patch indices must be nonempty BxK long tensors in grid range")
    if any(row.unique().numel() != row.numel() for row in indices):
        raise ValueError("Patch selection cannot contain duplicates")
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


@dataclass(frozen=True)
class JEPAEncoderConfig:
    image_size: int = 32
    num_frames: int = 4
    patch_size: int = 8
    tubelet_size: int = 2
    in_channels: int = 3
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    uniform_power: bool = False
    dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if (
            min(
                self.image_size,
                self.num_frames,
                self.patch_size,
                self.tubelet_size,
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
            )
            < 1
        ):
            raise ValueError("JEPA dimensions must be positive")
        if (
            self.image_size % self.patch_size
            or self.num_frames > 1
            and self.num_frames % self.tubelet_size
            or self.hidden_size % self.num_attention_heads
        ):
            raise ValueError("Patch/tubelet/head dimensions must divide exactly")
        if not self.uniform_power and self.hidden_size % (8 if self.num_frames > 1 else 4):
            raise ValueError("JEPA sincos dimension is incompatible with axis split")
        if not 0 <= self.dropout < 1 or self.initializer_range <= 0:
            raise ValueError("Invalid dropout/initializer")

    @property
    def grid(self):
        spatial = (self.image_size // self.patch_size,) * 2
        return (self.num_frames // self.tubelet_size, *spatial) if self.num_frames > 1 else spatial

    def to_dict(self):
        return {"architecture": "jepa_encoder", **asdict(self)}


class JEPABlock(nn.Module):
    def __init__(self, width, hidden, heads, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width, eps=1e-6), nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.fc1, self.fc2 = nn.Linear(width, hidden), nn.Linear(hidden, width)

    def forward(self, tokens):
        batch, length, width = tokens.shape
        q, k, v = (
            self.qkv(self.norm1(tokens))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attention.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + F.dropout(self.proj(attended), self.dropout, self.training)
        hidden = F.dropout(F.gelu(self.fc1(self.norm2(tokens))), self.dropout, self.training)
        return tokens + F.dropout(self.fc2(hidden), self.dropout, self.training)


def _initialize(module, standard_deviation):
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(child.weight, std=standard_deviation)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)

    blocks = [child for child in module.modules() if isinstance(child, JEPABlock)]
    with torch.no_grad():
        for depth, block in enumerate(blocks, 1):
            block.proj.weight.div_(math.sqrt(2 * depth))
            block.fc2.weight.div_(math.sqrt(2 * depth))


class JEPAEncoder(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        if c.num_frames > 1:
            self.patch_embed = nn.Conv3d(
                c.in_channels,
                c.hidden_size,
                (c.tubelet_size, c.patch_size, c.patch_size),
                stride=(c.tubelet_size, c.patch_size, c.patch_size),
            )
        else:
            self.patch_embed = nn.Conv2d(
                c.in_channels, c.hidden_size, c.patch_size, stride=c.patch_size
            )
        self.register_buffer(
            "pos_embed", jepa_positions(c.grid, c.hidden_size, uniform_power=c.uniform_power)
        )
        self.blocks = nn.ModuleList(
            [
                JEPABlock(c.hidden_size, c.intermediate_size, c.num_attention_heads, c.dropout)
                for _ in range(c.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(c.hidden_size, eps=1e-6)
        _initialize(self, c.initializer_range)

    def forward(self, pixel_values, indices=None):
        c = self.config
        expected = (
            (c.in_channels, c.num_frames, c.image_size, c.image_size)
            if c.num_frames > 1
            else (c.in_channels, c.image_size, c.image_size)
        )
        if tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(
                "JEPA configured image/video grid differs; preprocess explicitly, no silent temporal resizing"
            )
        tokens = self.patch_embed(pixel_values).flatten(2).transpose(1, 2) + self.pos_embed.to(
            pixel_values
        )
        if indices is not None:
            tokens = select_patches(tokens, indices)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


@dataclass(frozen=True)
class JEPAConfig:
    encoder: JEPAEncoderConfig | dict | None = None
    predictor_hidden_size: int = 64
    predictor_intermediate_size: int = 256
    predictor_layers: int = 2
    predictor_heads: int = 4
    num_mask_tokens: int = 2

    def __post_init__(self):
        encoder = self.encoder
        if encoder is None:
            encoder = JEPAEncoderConfig()
        if isinstance(encoder, dict):
            payload = dict(encoder)
            architecture = payload.pop("architecture", "jepa_encoder")
            if architecture != "jepa_encoder":
                raise ValueError("JEPA requires an explicit JEPAEncoder config")
            encoder = JEPAEncoderConfig(**payload)
        if not isinstance(encoder, JEPAEncoderConfig):
            raise TypeError("Invalid JEPA encoder config")
        object.__setattr__(self, "encoder", encoder)
        if (
            min(
                self.predictor_hidden_size,
                self.predictor_intermediate_size,
                self.predictor_layers,
                self.predictor_heads,
                self.num_mask_tokens,
            )
            < 1
            or self.predictor_hidden_size % self.predictor_heads
        ):
            raise ValueError("Invalid JEPA predictor dimensions")
        if not encoder.uniform_power and self.predictor_hidden_size % (
            8 if encoder.num_frames > 1 else 4
        ):
            raise ValueError("Predictor sincos axis dimensions differ")

    def to_dict(self):
        return {"architecture": "jepa", **asdict(self)}


class JEPAPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        width = c.predictor_hidden_size
        self.input_proj = nn.Linear(c.encoder.hidden_size, width)

        self.mask_tokens = nn.Embedding(c.num_mask_tokens, width)
        nn.init.zeros_(self.mask_tokens.weight)
        self.register_buffer(
            "pos_embed",
            jepa_positions(c.encoder.grid, width, uniform_power=c.encoder.uniform_power),
        )
        self.blocks = nn.ModuleList(
            [
                JEPABlock(
                    width, c.predictor_intermediate_size, c.predictor_heads, c.encoder.dropout
                )
                for _ in range(c.predictor_layers)
            ]
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.output_proj = nn.Linear(width, c.encoder.hidden_size)
        _initialize(self, c.encoder.initializer_range)

    def forward(self, context, context_indices, target_indices, *, mask_index=0):
        if (
            context.shape[:2] != context_indices.shape
            or not 0 <= mask_index < self.config.num_mask_tokens
        ):
            raise ValueError("Context embedding and patch selection must align")
        positions = self.pos_embed.to(context).expand(len(context), -1, -1)
        prefix = self.input_proj(context) + select_patches(positions, context_indices)
        target = self.mask_tokens(torch.tensor(mask_index, device=context.device))[
            None, None
        ] + select_patches(positions, target_indices)
        tokens = torch.cat((prefix, target), 1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_proj(self.norm(tokens)[:, prefix.shape[1] :])


class JEPAModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = JEPAEncoder(config.encoder)
        self.predictor = JEPAPredictor(config)

    def forward(self, pixel_values, context_indices, target_indices, *, mask_index=0):
        context = self.encoder(pixel_values, context_indices)
        return self.predictor(context, context_indices, target_indices, mask_index=mask_index)


def tube_masks(batch_size, grid, *, keep_fraction=0.25, device="cpu", generator=None):

    if len(grid) != 3 or min(grid) < 1 or batch_size < 1 or not 0 < keep_fraction < 1:
        raise ValueError("Tube masking needs a valid THW grid")
    temporal, height, width = grid
    spatial = height * width
    keep = int(spatial * keep_fraction)
    if not 0 < keep < spatial:
        raise ValueError("Both context and prediction regions must be nonempty")
    order = torch.rand(batch_size, spatial, device=device, generator=generator).argsort(-1)
    offsets = torch.arange(temporal, device=device)[None, :, None] * spatial
    context = (order[:, None, :keep] + offsets).reshape(batch_size, -1).sort(-1).values
    target = (order[:, None, keep:] + offsets).reshape(batch_size, -1).sort(-1).values
    return context, target
