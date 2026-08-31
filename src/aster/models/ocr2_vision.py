"""SAM grids and Qwen2 visual causal queries for DeepSeek-OCR2."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import RMSNorm, RopeConfig
from aster.nn.sam import SAMImageEncoder
from .config import Qwen2Config
from .decoder import DecoderLayer
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class OCR2SAMConfig:
    architecture: ClassVar[str] = "ocr2_sam"
    image_size: int = 32
    patch_size: int = 2
    in_channels: int = 3
    hidden_size: int = 24
    intermediate_size: int = 96
    depth: int = 2
    num_heads: int = 4
    neck_channels: int = 8
    downsample_channels: int = 16
    output_channels: int = 32
    window_size: int = 7
    global_attn_indexes: tuple[int, ...] = (1,)
    norm_eps: float = 1e-6

    def __post_init__(self):
        object.__setattr__(self, "global_attn_indexes", tuple(self.global_attn_indexes))
        if (
            min(
                self.image_size,
                self.patch_size,
                self.in_channels,
                self.hidden_size,
                self.intermediate_size,
                self.depth,
                self.num_heads,
                self.neck_channels,
                self.downsample_channels,
                self.output_channels,
                self.window_size,
            )
            < 1
            or self.norm_eps <= 0
        ):
            raise ValueError("Invalid OCR2 SAM dimensions")
        if self.image_size % (4 * self.patch_size) or self.hidden_size % self.num_heads:
            raise ValueError("SAM image grid must survive two stride-2 stages and valid heads")
        if tuple(sorted(set(self.global_attn_indexes))) != self.global_attn_indexes or any(
            x not in range(self.depth) for x in self.global_attn_indexes
        ):
            raise ValueError("SAM global-attention layer indices must be explicit")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class OCR2SAMEncoder(LocalModelMixin, SAMImageEncoder):
    pass


@dataclass(frozen=True)
class OCR2VisualConfig:
    architecture: ClassVar[str] = "ocr2_visual"
    sam_config: OCR2SAMConfig = field(default_factory=OCR2SAMConfig)
    decoder_config: Qwen2Config = field(
        default_factory=lambda: Qwen2Config(
            rope=RopeConfig(theta=1000000), num_hidden_layers=2, max_position_embeddings=128
        )
    )
    local_image_size: int = 24

    def __post_init__(self):
        if (
            not isinstance(self.sam_config, OCR2SAMConfig)
            or type(self.decoder_config) is not Qwen2Config
        ):
            raise ValueError("OCR2 visual requires actual SAM + Qwen2 configurations")
        if self.sam_config.output_channels != self.decoder_config.hidden_size:
            raise ValueError("SAM output and visual Qwen2 width must match")
        if not 0 < self.local_image_size < self.sam_config.image_size or self.local_image_size % (
            4 * self.sam_config.patch_size
        ):
            raise ValueError("Local OCR view must be smaller and divisible by patch*4")
        if self.decoder_config.use_sliding_window or self.decoder_config.rope.kind != "default":
            raise ValueError("OCR2 visual Qwen2 uses explicit mixed visibility/default RoPE")
        if 2 * self.global_queries > self.decoder_config.max_position_embeddings:
            raise ValueError("Visual decoder context must fit image tokens plus queries")

    @property
    def local_queries(self):
        return (self.local_image_size // (4 * self.sam_config.patch_size)) ** 2

    @property
    def global_queries(self):
        return (self.sam_config.image_size // (4 * self.sam_config.patch_size)) ** 2

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "sam_config": self.sam_config.to_dict(),
            "decoder_config": self.decoder_config.to_dict(),
            "local_image_size": self.local_image_size,
        }


def visual_causal_mask(image_tokens, device=None):

    count = 2 * image_tokens
    mask = torch.zeros(count, count, dtype=torch.bool, device=device)
    mask[:image_tokens, :image_tokens] = True
    mask[image_tokens:, :image_tokens] = True
    mask[image_tokens:, image_tokens:] = torch.ones(
        image_tokens, image_tokens, dtype=torch.bool, device=device
    ).tril()
    return mask[None, None]


class OCR2QueryLayer(DecoderLayer):
    def forward(self, hidden, positions, mask):
        normal = self.input_layernorm(hidden)
        attention = self.self_attn
        b, length, _ = normal.shape
        heads, kv, dim = attention.num_heads, attention.num_kv_heads, attention.head_dim
        q = attention.q_proj(normal).reshape(b, length, heads, dim).transpose(1, 2)
        k = attention.k_proj(normal).reshape(b, length, kv, dim).transpose(1, 2)
        v = attention.v_proj(normal).reshape(b, length, kv, dim).transpose(1, 2)
        q, k = attention.rope(q, positions), attention.rope(k, positions)

        value = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(heads // kv, 1),
            v.repeat_interleave(heads // kv, 1),
            attn_mask=mask,
            dropout_p=attention.dropout if self.training else 0.0,
        )
        hidden = hidden + attention.o_proj(value.transpose(1, 2).reshape(b, length, heads * dim))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class OCR2QueryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.decoder_config

        self.model = nn.Module()
        self.model.model = nn.Module()
        self.model.model.layers = nn.ModuleList(
            OCR2QueryLayer(c, i) for i in range(c.num_hidden_layers)
        )
        self.model.model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.query_768 = nn.Embedding(config.local_queries, c.hidden_size)
        self.query_1024 = nn.Embedding(config.global_queries, c.hidden_size)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=c.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image_features):
        c = self.config
        tokens = image_features.flatten(2).transpose(1, 2)
        if tokens.shape[1] == c.local_queries:
            embedding = self.query_768
        elif tokens.shape[1] == c.global_queries:
            embedding = self.query_1024
        else:
            raise ValueError("Image feature count does not match either learned OCR query table")
        queries = embedding(torch.arange(tokens.shape[1], device=tokens.device))[None].expand(
            len(tokens), -1, -1
        )
        hidden = torch.cat((tokens, queries), 1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(
            len(hidden), -1
        )
        mask = visual_causal_mask(tokens.shape[1], hidden.device)
        for layer in self.model.model.layers:
            hidden = layer(hidden, positions, mask)
        return self.model.model.norm(hidden)[:, tokens.shape[1] :]


class OCR2VisualEncoder(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sam_model = SAMImageEncoder(config.sam_config)
        self.qwen2_model = OCR2QueryEncoder(config)

    def forward(self, pixel_values):
        if pixel_values.shape[-1] not in (
            self.config.local_image_size,
            self.config.sam_config.image_size,
        ):
            raise ValueError("OCR view needs the configured local/global resolution")
        return self.qwen2_model(self.sam_model(pixel_values))
