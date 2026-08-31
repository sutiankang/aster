"""Standard ViT with class tokens, interpolated absolute positions, and pre-normalization."""

from dataclasses import asdict, dataclass
import math
import re
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from ..nn.normalization import LayerNorm
from ..nn.attention import scaled_attention
from ..nn.parameter_codec import register_parameter_codec
from .serialization import LocalModelMixin
from .vision import VisionOutput


@dataclass(frozen=True)
class ViTConfig:
    architecture: ClassVar[str] = "vit"
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 64
    image_size: int = 16
    patch_size: int = 4
    num_channels: int = 3
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    qkv_bias: bool = True

    def __post_init__(self):
        if any(
            type(x) is not int or x < 1
            for x in (
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.intermediate_size,
                self.image_size,
                self.patch_size,
                self.num_channels,
            )
        ):
            raise ValueError("Invalid ViT dimensions")
        if self.hidden_size % self.num_attention_heads or self.image_size % self.patch_size:
            raise ValueError("Incompatible ViT heads/patch grid")
        if any(
            not math.isfinite(x) or not 0 <= x < 1
            for x in (self.hidden_dropout_prob, self.attention_probs_dropout_prob)
        ):
            raise ValueError("Invalid ViT dropout")
        if (
            any(
                not math.isfinite(x) or x <= 0
                for x in (self.layer_norm_eps, self.initializer_range)
            )
            or type(self.qkv_bias) is not bool
        ):
            raise ValueError("Invalid ViT numerical configuration")

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))


class _ViTPositions(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.cls_token = nn.Parameter(torch.empty(1, 1, c.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.empty(1, (c.image_size // c.patch_size) ** 2 + 1, c.hidden_size)
        )
        nn.init.trunc_normal_(self.cls_token, std=c.initializer_range)
        nn.init.trunc_normal_(self.position_embeddings, std=c.initializer_range)

    def forward(self, patches, height, width, interpolate):
        c = self.config
        tokens = torch.cat((self.cls_token.expand(len(patches), -1, -1), patches), 1)
        positions = self.position_embeddings
        if (height, width) != (c.image_size, c.image_size):
            if not interpolate:
                raise ValueError("Non-native ViT resolution requires explicit interpolation")
            side = c.image_size // c.patch_size
            patches = positions[:, 1:].reshape(1, side, side, c.hidden_size).permute(0, 3, 1, 2)
            patches = F.interpolate(
                patches,
                size=(height // c.patch_size, width // c.patch_size),
                mode="bicubic",
                align_corners=False,
            )
            positions = torch.cat(
                (positions[:, :1], patches.permute(0, 2, 3, 1).reshape(1, -1, c.hidden_size)), 1
            )
        return tokens + positions


class ViTEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = nn.Conv2d(
            c.num_channels, c.hidden_size, c.patch_size, c.patch_size
        )
        self.positions = _ViTPositions(c)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)
        register_parameter_codec(
            self,
            {
                "positions.cls_token": "cls_token",
                "positions.position_embeddings": "position_embeddings",
            },
        )

    def forward(self, pixels, interpolate):
        c = self.config
        if (
            pixels.ndim != 4
            or min(pixels.shape) < 1
            or pixels.shape[1] != c.num_channels
            or not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
        ):
            raise ValueError("ViT requires finite normalized BCHW pixels")
        if min(pixels.shape[-2:]) < c.patch_size:
            raise ValueError("ViT image is smaller than its patch")
        patches = (
            self.patch_embeddings.projection(
                pixels.to(self.patch_embeddings.projection.weight.dtype)
            )
            .flatten(2)
            .transpose(1, 2)
        )
        return self.dropout(self.positions(patches, *pixels.shape[-2:], interpolate))


class ViTAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.qkv_bias)
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x):
        c = self.config
        b, s, d = x.shape
        q, k, v = [
            layer(x)
            .reshape(b, s, c.num_attention_heads, d // c.num_attention_heads)
            .transpose(1, 2)
            for layer in (self.q_proj, self.k_proj, self.v_proj)
        ]
        out = scaled_attention(
            q,
            k,
            v,
            torch.ones(b, 1, s, s, dtype=torch.bool, device=x.device),
            dropout=c.attention_probs_dropout_prob,
            training=self.training,
        )
        return self.o_proj(out.transpose(1, 2).reshape_as(x))


class ViTMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1, self.fc2 = (
            nn.Linear(c.hidden_size, c.intermediate_size),
            nn.Linear(c.intermediate_size, c.hidden_size),
        )

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ViTLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention, self.mlp = ViTAttention(c), ViTMLP(c)
        self.layernorm_before, self.layernorm_after = (
            LayerNorm(c.hidden_size, c.layer_norm_eps),
            LayerNorm(c.hidden_size, c.layer_norm_eps),
        )
        self.dropout = nn.Dropout(c.hidden_dropout_prob)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.layernorm_before(x)))
        return x + self.dropout(self.mlp(self.layernorm_after(x)))


class ViTModel(LocalModelMixin, nn.Module):
    def __init__(self, config=ViTConfig()):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.layers = nn.ModuleList(ViTLayer(config) for _ in range(config.num_hidden_layers))
        self.layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values, *, interpolate_pos_encoding=False, output_hidden_states=False):
        x = self.embeddings(pixel_values, interpolate_pos_encoding)
        history = []
        for layer in self.layers:
            if output_hidden_states:
                history.append(x)
            x = layer(x)
        if output_hidden_states:
            history.append(x)
        return VisionOutput(
            self.layernorm(x), None, tuple(history) if output_hidden_states else None
        )


def convert_vit_state_dict(tensors, *, layout):

    if layout not in {"transformers_4.57", "transformers_5.16"}:
        raise ValueError("Declare the exact supported ViT weight layout")
    if not isinstance(tensors, dict) or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in tensors.items()
    ):
        raise ValueError("ViT checkpoint must contain only named tensors")
    converted = {}
    substitutions = {
        "attention.attention.query": "attention.q_proj",
        "attention.attention.key": "attention.k_proj",
        "attention.attention.value": "attention.v_proj",
        "attention.output.dense": "attention.o_proj",
        "intermediate.dense": "mlp.fc1",
        "output.dense": "mlp.fc2",
    }
    for name, value in tensors.items():
        target = name
        if layout == "transformers_4.57":
            match = re.fullmatch(r"encoder\.layer\.(\d+)\.(.+)", name)
            if match:
                suffix = match[2]
                for old, new in substitutions.items():
                    if suffix.startswith(old + "."):
                        suffix = new + suffix[len(old) :]
                        break
                target = "layers." + match[1] + "." + suffix
            elif name.startswith("layers."):
                raise ValueError("Mixed old/new ViT weight layouts")
        elif name.startswith("encoder.layer."):
            raise ValueError("Mixed old/new ViT weight layouts")
        if target in converted:
            raise ValueError("ViT weight key mapping collision")
        converted[target] = value
    return converted
