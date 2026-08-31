"""Trainable CLIP vision encoding, normalization, and spatial position interpolation."""

from dataclasses import asdict, dataclass
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import LayerNorm
from aster.nn.attention import scaled_attention
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class CLIPVisionConfig:
    architecture: ClassVar[str] = "clip_vision"
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_channels: int = 3
    image_size: int = 16
    patch_size: int = 4
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_act: str = "quick_gelu"
    initializer_range: float = 0.02
    initializer_factor: float = 1.0

    def __post_init__(self):
        if (
            min(
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_channels,
                self.image_size,
                self.patch_size,
            )
            < 1
        ):
            raise ValueError("Invalid CLIP vision dimensions")
        if self.hidden_size % self.num_attention_heads or self.image_size % self.patch_size:
            raise ValueError("Incompatible CLIP heads/patch grid")
        if (
            self.hidden_act not in {"quick_gelu", "gelu"}
            or not 0 <= self.attention_dropout < 1
            or min(self.layer_norm_eps, self.initializer_range, self.initializer_factor) <= 0
        ):
            raise ValueError("Invalid CLIP formula/numerics")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass
class VisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def normalize_clip_pixels(images):

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("CLIP normalization requires RGB BCHW")
    values = images.float() / 255 if images.dtype == torch.uint8 else images
    if (
        not values.is_floating_point()
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Float RGB pixels must be finite and in [0,1]")
    mean = values.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = values.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (values - mean) / std


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.class_embedding = nn.Parameter(torch.empty(c.hidden_size))
        self.patch_embedding = nn.Conv2d(
            c.num_channels, c.hidden_size, c.patch_size, c.patch_size, bias=False
        )
        self.position_embedding = nn.Embedding(
            (c.image_size // c.patch_size) ** 2 + 1, c.hidden_size
        )

    def forward(self, pixels, interpolate=False):
        c = self.config
        if pixels.ndim != 4 or pixels.shape[1] != c.num_channels:
            raise ValueError("Vision encoder expects BCHW pixels")
        height, width = pixels.shape[-2:]
        if height % c.patch_size or width % c.patch_size:
            raise ValueError("Image grid must be divisible by patch size")
        if not interpolate and (height, width) != (c.image_size, c.image_size):
            raise ValueError("Non-native image size requires explicit position interpolation")
        patches = (
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        hidden = torch.cat((self.class_embedding.expand(pixels.shape[0], 1, -1), patches), 1)
        positions = self.position_embedding.weight[None]
        if interpolate and (height, width) != (c.image_size, c.image_size):
            grid = c.image_size // c.patch_size
            patch_positions = (
                positions[:, 1:].reshape(1, grid, grid, c.hidden_size).permute(0, 3, 1, 2)
            )
            patch_positions = F.interpolate(
                patch_positions,
                size=(height // c.patch_size, width // c.patch_size),
                mode="bicubic",
                align_corners=False,
            )
            positions = torch.cat((positions[:, :1], patch_positions.flatten(2).transpose(1, 2)), 1)
        return hidden + positions


class CLIPAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.out_proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, hidden, attention_mask=None):
        b, s, d = hidden.shape
        c = self.config

        def split(proj):
            return proj(hidden).reshape(b, s, c.num_attention_heads, -1).transpose(1, 2)

        visible = torch.ones(b, 1, s, s, dtype=torch.bool, device=hidden.device)
        if attention_mask is not None:
            if (
                attention_mask.shape != (b, s)
                or not ((attention_mask == 0) | (attention_mask == 1)).all()
            ):
                raise ValueError("Encoder padding mask must be binary [B,S]")
            visible = visible & attention_mask[:, None, None, :].bool()
        value = scaled_attention(
            split(self.q_proj),
            split(self.k_proj),
            split(self.v_proj),
            visible,
            dropout=c.attention_dropout,
            training=self.training,
        )
        return self.out_proj(value.transpose(1, 2).reshape(b, s, d))


class CLIPMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)
        self.activation = c.hidden_act

    def forward(self, hidden):
        hidden = self.fc1(hidden)
        if self.activation == "quick_gelu":
            hidden = hidden * torch.sigmoid(1.702 * hidden)
        else:
            hidden = F.gelu(
                hidden, approximate="tanh" if self.activation == "gelu_pytorch_tanh" else "none"
            )
        return self.fc2(hidden)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = CLIPAttention(c)
        self.layer_norm1 = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.layer_norm2 = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.mlp = CLIPMLP(c)

    def forward(self, hidden, attention_mask=None):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden), attention_mask)
        return hidden + self.mlp(self.layer_norm2(hidden))


class CLIPVisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        c, factor = config, config.initializer_factor
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
                nn.init.normal_(module.weight, std=c.initializer_range * factor)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, CLIPAttention):
                for projection in (module.q_proj, module.k_proj, module.v_proj):
                    nn.init.normal_(
                        projection.weight,
                        std=c.hidden_size**-0.5 * (2 * c.num_hidden_layers) ** -0.5 * factor,
                    )
                nn.init.normal_(module.out_proj.weight, std=c.hidden_size**-0.5 * factor)
            if isinstance(module, CLIPMLP):
                nn.init.normal_(module.fc1.weight, std=(2 * c.hidden_size) ** -0.5 * factor)
                nn.init.normal_(
                    module.fc2.weight,
                    std=c.hidden_size**-0.5 * (2 * c.num_hidden_layers) ** -0.5 * factor,
                )
        nn.init.normal_(self.embeddings.class_embedding, std=c.hidden_size**-0.5 * factor)

    def forward(self, pixel_values, *, output_hidden_states=False, interpolate_pos_encoding=False):
        vision = self
        hidden = vision.pre_layrnorm(vision.embeddings(pixel_values, interpolate_pos_encoding))
        states = [hidden] if output_hidden_states else None
        for layer in vision.encoder.layers:
            hidden = layer(hidden)
            if states is not None:
                states.append(hidden)

        return VisionOutput(
            hidden,
            vision.post_layernorm(hidden[:, 0]),
            tuple(states) if states is not None else None,
        )
