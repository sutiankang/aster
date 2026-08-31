"""DINOv2 register-token vision encoders with explicit OpenVLA feature selection."""

from dataclasses import dataclass, asdict
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn.normalization import LayerNorm
from aster.nn.attention import scaled_attention
from .serialization import LocalModelMixin
from .vision import VisionOutput


@dataclass(frozen=True)
class DinoVisionConfig:
    architecture: ClassVar[str] = "dinov2_register_vision"
    hidden_size: int = 32
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    intermediate_size: int = 64
    num_register_tokens: int = 4
    image_size: int = 8
    patch_size: int = 2
    layer_norm_eps: float = 1e-6
    layerscale_value: float = 1e-5
    qkv_bias: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if (
            min(
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.image_size,
                self.patch_size,
            )
            < 1
            or self.num_register_tokens < 0
        ):
            raise ValueError("Invalid DINO dimensions")
        if (
            self.hidden_size % self.num_attention_heads
            or self.image_size % self.patch_size
            or min(self.layer_norm_eps, self.layerscale_value, self.initializer_range) <= 0
        ):
            raise ValueError("Invalid DINO grid/normalization")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class LayerScale(nn.Module):
    def __init__(self, width, initial):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.full((width,), initial))

    def forward(self, hidden):
        return hidden * self.scale_factor


class DinoAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=c.qkv_bias)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, hidden):
        b, s, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(b, s, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        visible = torch.ones(b, 1, s, s, dtype=torch.bool, device=hidden.device)
        return self.proj(scaled_attention(q, k, v, visible).transpose(1, 2).reshape(b, s, width))


class DinoMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = (
            LayerNorm(c.hidden_size, c.layer_norm_eps),
            LayerNorm(c.hidden_size, c.layer_norm_eps),
        )
        self.attn, self.mlp = DinoAttention(c), DinoMLP(c)
        self.ls1, self.ls2 = (
            LayerScale(c.hidden_size, c.layerscale_value),
            LayerScale(c.hidden_size, c.layerscale_value),
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(
            3, config.hidden_size, config.patch_size, stride=config.patch_size
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.reg_token = nn.Parameter(
            torch.zeros(1, config.num_register_tokens, config.hidden_size)
        )
        self.pos_embed = nn.Parameter(
            torch.empty(1, (config.image_size // config.patch_size) ** 2, config.hidden_size)
        )
        self.blocks = nn.ModuleList(DinoBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embed, std=config.initializer_range)
        nn.init.normal_(self.cls_token, std=1e-6)

    def embed(self, pixels):
        c = self.config
        if (
            pixels.ndim != 4
            or pixels.shape[1:] != (3, c.image_size, c.image_size)
            or not pixels.is_floating_point()
        ):
            raise ValueError(
                "DINO expects explicitly preprocessed floating BCHW on its configured fixed grid"
            )
        patches = (
            self.patch_embed.proj(pixels.to(self.pos_embed.dtype)).flatten(2).transpose(1, 2)
            + self.pos_embed
        )
        return torch.cat(
            (
                self.cls_token.expand(len(pixels), -1, -1),
                self.reg_token.expand(len(pixels), -1, -1),
                patches,
            ),
            1,
        )

    def patch_features(self, pixels, layer=-2):

        index = layer if layer >= 0 else len(self.blocks) + layer
        if not 0 <= index < len(self.blocks):
            raise ValueError("DINO feature layer outside backbone")
        hidden = self.embed(pixels)
        for block in self.blocks[: index + 1]:
            hidden = block(hidden)
        return hidden[:, 1 + self.config.num_register_tokens :]

    def forward(self, pixel_values, *, output_hidden_states=False):
        hidden = self.embed(pixel_values)
        states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            if output_hidden_states:
                states.append(hidden)
        hidden = self.norm(hidden)
        return VisionOutput(hidden, hidden[:, 0], tuple(states) if output_hidden_states else None)
