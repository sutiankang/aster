"""Native SigLIP dual encoders with architecture-specific normalization and pooling."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import LayerNorm
from .serialization import LocalModelMixin
from .vision import CLIPAttention, CLIPMLP, CLIPEncoderLayer, VisionOutput


def _check_encoder(c):
    if (
        min(c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads) < 1
        or c.hidden_size % c.num_attention_heads
    ):
        raise ValueError("Invalid SigLIP encoder dimensions")
    if (
        c.hidden_act not in {"gelu_pytorch_tanh", "gelu"}
        or c.layer_norm_eps <= 0
        or not 0 <= c.attention_dropout < 1
    ):
        raise ValueError("Invalid SigLIP formula/numerics")


@dataclass(frozen=True)
class SigLIPVisionConfig:
    architecture: ClassVar[str] = "siglip_vision"
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_channels: int = 3
    image_size: int = 16
    patch_size: int = 4
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    vision_use_head: bool = True

    def __post_init__(self):
        _check_encoder(self)
        if (
            min(self.num_channels, self.image_size, self.patch_size) < 1
            or self.image_size % self.patch_size
        ):
            raise ValueError("Invalid SigLIP image/patch grid")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class SigLIPTextConfig:
    architecture: ClassVar[str] = "siglip_text"
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_position_embeddings: int = 16
    projection_size: int = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    def __post_init__(self):
        _check_encoder(self)
        if min(self.vocab_size, self.max_position_embeddings, self.projection_size) < 1:
            raise ValueError("Invalid SigLIP text dimensions")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class SigLIPConfig:
    architecture: ClassVar[str] = "siglip"
    text_config: SigLIPTextConfig = field(default_factory=SigLIPTextConfig)
    vision_config: SigLIPVisionConfig = field(default_factory=SigLIPVisionConfig)

    def __post_init__(self):
        if not isinstance(self.text_config, SigLIPTextConfig) or not isinstance(
            self.vision_config, SigLIPVisionConfig
        ):
            raise ValueError("SigLIP needs its real text and vision configurations")
        if (
            self.text_config.projection_size != self.vision_config.hidden_size
            or not self.vision_config.vision_use_head
        ):
            raise ValueError(
                "Contrastive tower outputs must have the same width and a vision pooling head"
            )

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
        }


@dataclass
class ContrastiveOutput:
    logits_per_text: torch.Tensor
    logits_per_image: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    text_model_output: VisionOutput
    vision_model_output: VisionOutput


def normalize_siglip_pixels(images):

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("SigLIP normalization expects RGB BCHW")
    values = images.float() / 255 if images.dtype == torch.uint8 else images
    if (
        not values.is_floating_point()
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Float pixels must be finite in [0,1]")
    return (values - 0.5) / 0.5


class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c

        self.patch_embedding = nn.Conv2d(
            c.num_channels, c.hidden_size, c.patch_size, c.patch_size, bias=True
        )
        self.position_embedding = nn.Embedding((c.image_size // c.patch_size) ** 2, c.hidden_size)

    def forward(self, pixels, interpolate=False):
        c = self.config
        if pixels.ndim != 4 or pixels.shape[1] != c.num_channels:
            raise ValueError("SigLIP expects BCHW pixels")
        h, w = pixels.shape[-2:]
        if min(h, w) < c.patch_size or h % c.patch_size or w % c.patch_size:
            raise ValueError("Image dimensions must be positive patch multiples")
        if not interpolate and (h, w) != (c.image_size, c.image_size):
            raise ValueError("Non-native grid requires explicit position interpolation")
        hidden = (
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        positions = self.position_embedding.weight[None]
        if (h, w) != (c.image_size, c.image_size):
            side = c.image_size // c.patch_size
            grid = positions.reshape(1, side, side, c.hidden_size).permute(0, 3, 1, 2)
            positions = (
                F.interpolate(
                    grid,
                    size=(h // c.patch_size, w // c.patch_size),
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2)
            )
        return hidden + positions


class SigLIPTextEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.token_embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        self.position_embedding = nn.Embedding(c.max_position_embeddings, c.hidden_size)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids/inputs_embeds")
        hidden = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.config.hidden_size
            or not 0 < hidden.shape[1] <= self.config.max_position_embeddings
        ):
            raise ValueError("Invalid SigLIP text shape/context")
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
        if position_ids.shape not in {(1, hidden.shape[1]), hidden.shape[:2]}:
            raise ValueError("Invalid SigLIP absolute positions")
        return hidden + self.position_embedding(position_ids)


class SigLIPPoolingHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.probe = nn.Parameter(torch.empty(1, 1, c.hidden_size))

        self.attention = nn.MultiheadAttention(
            c.hidden_size, c.num_attention_heads, batch_first=True
        )
        self.layernorm = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.mlp = CLIPMLP(c)

    def forward(self, hidden):
        probe = self.probe.expand(hidden.shape[0], -1, -1)
        pooled = self.attention(probe, hidden, hidden)[0]
        return (pooled + self.mlp(self.layernorm(pooled)))[:, 0]


def _initialize_siglip(model):

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            fan_in = module.weight.shape[1] * math.prod(module.weight.shape[2:])
            nn.init.trunc_normal_(module.weight, std=fan_in**-0.5 / 0.87962566103423978)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=module.weight.shape[1] ** -0.5)
    for module in model.modules():
        if isinstance(module, CLIPAttention):
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        elif isinstance(module, CLIPMLP):
            for proj in (module.fc1, module.fc2):
                nn.init.xavier_uniform_(proj.weight)
                nn.init.normal_(proj.bias, std=1e-6)
        elif isinstance(module, SigLIPPoolingHead):
            nn.init.xavier_uniform_(module.probe)
            nn.init.xavier_uniform_(module.attention.in_proj_weight)
            nn.init.zeros_(module.attention.in_proj_bias)


def _encoder(c):
    module = nn.Module()
    module.layers = nn.ModuleList(CLIPEncoderLayer(c) for _ in range(c.num_hidden_layers))
    return module


def _run_encoder(encoder, hidden, padding, collect):
    states = [hidden] if collect else None
    for layer in encoder.layers:
        hidden = layer(hidden, padding)
        if states is not None:
            states.append(hidden)
    return hidden, tuple(states) if states is not None else None


class SigLIPVisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = _encoder(config)
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        if config.vision_use_head:
            self.head = SigLIPPoolingHead(config)
        _initialize_siglip(self)

    def forward(self, pixel_values, *, interpolate_pos_encoding=False, output_hidden_states=False):
        hidden = self.embeddings(pixel_values, interpolate_pos_encoding)
        hidden, states = _run_encoder(self.encoder, hidden, None, output_hidden_states)
        hidden = self.post_layernorm(hidden)
        return VisionOutput(
            hidden, self.head(hidden) if self.config.vision_use_head else None, states
        )


class SigLIPTextModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SigLIPTextEmbeddings(config)
        self.encoder = _encoder(config)
        self.final_layer_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.projection_size)
        _initialize_siglip(self)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        hidden = self.embeddings(input_ids, position_ids, inputs_embeds)
        hidden, states = _run_encoder(self.encoder, hidden, attention_mask, output_hidden_states)
        hidden = self.final_layer_norm(hidden)

        return VisionOutput(hidden, self.head(hidden[:, -1]), states)


class SigLIPModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_model = SigLIPTextModel(config.text_config)
        self.vision_model = SigLIPVisionModel(config.vision_config)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids,
        pixel_values,
        *,
        attention_mask=None,
        position_ids=None,
        interpolate_pos_encoding=False,
        output_hidden_states=False,
    ):
        text = self.text_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        vision = self.vision_model(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            output_hidden_states=output_hidden_states,
        )

        t = text.pooler_output / text.pooler_output.norm(dim=-1, keepdim=True)
        v = vision.pooler_output / vision.pooler_output.norm(dim=-1, keepdim=True)
        logits = (t @ v.T) * self.logit_scale.exp() + self.logit_bias
        return ContrastiveOutput(logits, logits.T, t, v, text, vision)
