"""Gemma4 visual tokens and image/video conditioning over the native text backbone."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn.normalization import FloatRMSNorm
from aster.nn.position import RopeConfig, RotaryEmbedding
from aster.nn.attention import scaled_attention
from .serialization import LocalModelMixin, configuration_key
from .gemma4 import Gemma4TextConfig, Gemma4ForCausalLM, activation


@dataclass(frozen=True)
class Gemma4VisionConfig:
    architecture: ClassVar[str] = "gemma4_vision"
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 128
    attention_dropout: float = 0.0
    rope_theta: float = 100.0
    pooling_kernel_size: int = 2
    patch_size: int = 2
    position_embedding_size: int = 32
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        sizes = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.pooling_kernel_size,
            self.patch_size,
            self.position_embedding_size,
        )
        if (
            any(type(v) is not int or v < 1 for v in sizes)
            or self.head_dim % 4
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError(
                "Gemma4 vision needs positive dimensions, GQA divisibility and head_dim multiple of four"
            )
        if self.pooling_kernel_size == 1:
            raise ValueError(
                "Pool=1 is not admitted: the pinned official path returns a padding mask as valid mask"
            )
        if (
            self.hidden_activation not in {"silu", "gelu", "gelu_pytorch_tanh"}
            or not 0 <= self.attention_dropout < 1
        ):
            raise ValueError("Invalid Gemma4 vision activation/dropout")
        if self.rope_theta <= 1 or any(
            not math.isfinite(v) or v <= 0
            for v in (self.rms_norm_eps, self.rope_theta, self.initializer_range)
        ):
            raise ValueError("Invalid Gemma4 vision numeric configuration")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass
class PackedVisionOutput:
    last_hidden_state: torch.Tensor
    counts: tuple[int, ...]
    hidden_states: tuple[torch.Tensor, ...] | None = None


class ClippableLinear(nn.Module):
    def __init__(self, c, incoming, outgoing):
        super().__init__()
        self.use_clipped_linears = c.use_clipped_linears
        self.linear = nn.Linear(incoming, outgoing, bias=False)
        if self.use_clipped_linears:
            for name, value in (
                ("input_min", -math.inf),
                ("input_max", math.inf),
                ("output_min", -math.inf),
                ("output_max", math.inf),
            ):
                self.register_buffer(name, torch.tensor(value))

    def forward(self, x):
        if self.use_clipped_linears:
            x = x.clamp(self.input_min, self.input_max)
        x = self.linear(x)
        return x.clamp(self.output_min, self.output_max) if self.use_clipped_linears else x


class PatchEmbedder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_proj = nn.Linear(3 * c.patch_size**2, c.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, c.position_embedding_size, c.hidden_size)
        )

    def forward(self, pixels, positions, padding):

        x = self.input_proj((2 * (pixels - 0.5)).to(self.input_proj.weight.dtype))
        xy = positions.clamp_min(0)
        positional = F.embedding(xy[..., 0], self.position_embedding_table[0]) + F.embedding(
            xy[..., 1], self.position_embedding_table[1]
        )
        return x + positional.masked_fill(padding[..., None], 0)


class VisionAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        for name, width in (
            ("q_proj", c.num_attention_heads),
            ("k_proj", c.num_key_value_heads),
            ("v_proj", c.num_key_value_heads),
        ):
            self.add_module(name, ClippableLinear(c, c.hidden_size, width * c.head_dim))
        self.o_proj = ClippableLinear(c, c.num_attention_heads * c.head_dim, c.hidden_size)
        self.q_norm, self.k_norm = (
            FloatRMSNorm(c.head_dim, c.rms_norm_eps),
            FloatRMSNorm(c.head_dim, c.rms_norm_eps),
        )
        self.v_norm = FloatRMSNorm(c.head_dim, c.rms_norm_eps, with_scale=False)

    def forward(self, x, positions, visible, rope):
        c, b, s = self.config, x.shape[0], x.shape[1]

        def split(value, heads):
            return value.view(b, s, heads, c.head_dim).transpose(1, 2)

        def rotate(value):

            pieces = value.chunk(2, -1)
            return torch.cat([rope(pieces[i], positions[..., i]) for i in range(2)], -1)

        q = rotate(self.q_norm(split(self.q_proj(x), c.num_attention_heads)))
        k = rotate(self.k_norm(split(self.k_proj(x), c.num_key_value_heads)))
        v = self.v_norm(split(self.v_proj(x), c.num_key_value_heads))
        value = scaled_attention(
            q, k, v, visible, scale=1.0, dropout=c.attention_dropout, training=self.training
        )
        return self.o_proj(value.transpose(1, 2).reshape(b, s, -1))


class VisionMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.activation = c.hidden_activation
        self.gate_proj, self.up_proj = (
            ClippableLinear(c, c.hidden_size, c.intermediate_size),
            ClippableLinear(c, c.hidden_size, c.intermediate_size),
        )
        self.down_proj = ClippableLinear(c, c.intermediate_size, c.hidden_size)

    def forward(self, x):
        return self.down_proj(activation(self.gate_proj(x), self.activation) * self.up_proj(x))


class VisionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = VisionAttention(c), VisionMLP(c)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            self.add_module(name, FloatRMSNorm(c.hidden_size, c.rms_norm_eps))

    def forward(self, x, positions, visible, rope):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), positions, visible, rope)
        )
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Gemma4VisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.patch_embedder = PatchEmbedder(c)
        self.encoder = nn.Module()
        self.encoder.rotary_emb = RotaryEmbedding(c.head_dim // 2, RopeConfig(theta=c.rope_theta))
        self.encoder.layers = nn.ModuleList(VisionLayer(c) for _ in range(c.num_hidden_layers))
        if c.standardize:
            self.register_buffer("std_bias", torch.zeros(c.hidden_size))
            self.register_buffer("std_scale", torch.ones(c.hidden_size))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=c.initializer_range)

    def forward(self, pixel_values, pixel_position_ids, *, output_hidden_states=False):
        c, p = self.config, pixel_position_ids
        if (
            pixel_values.ndim != 3
            or pixel_values.shape[-1] != 3 * c.patch_size**2
            or p.shape != (*pixel_values.shape[:2], 2)
            or p.dtype != torch.long
        ):
            raise ValueError(
                "Gemma4 vision expects patches[B,N,3P²] and xy integer positions[B,N,2]"
            )
        if (
            not pixel_values.is_floating_point()
            or not torch.isfinite(pixel_values).all()
            or (pixel_values < 0).any()
            or (pixel_values > 1).any()
        ):
            raise ValueError("Gemma4 patch pixels must be finite floats in [0,1]")
        k, b, n = c.pooling_kernel_size, p.shape[0], p.shape[1]
        if not b or not n or n % (k * k):
            raise ValueError("Padded patch budget must be a positive multiple of pooling area")
        pad = (p == -1).all(-1)
        if (((p < 0).any(-1)) != pad).any() or (p >= c.position_embedding_size).any():
            raise ValueError("Invalid xy positions: padding is exactly (-1,-1)")
        counts = []
        for row in p:
            valid = row[(row >= 0).all(-1)]
            if not len(valid):
                raise ValueError("Empty image is not a valid visual sample")
            width, height = (valid.max(0).values + 1).tolist()

            if (
                width % k
                or height % k
                or len(valid) != width * height
                or len(torch.unique(valid[:, 0] + width * valid[:, 1])) != len(valid)
            ):
                raise ValueError(
                    "Visual coordinates must form a complete rectangle divisible by the pooling kernel"
                )
            counts.append(len(valid) // (k * k))
        x = self.patch_embedder(pixel_values, p, pad)
        visible = (~pad)[:, None, None].expand(b, 1, n, n)
        history = []
        for layer in self.encoder.layers:
            if output_hidden_states:
                history.append(x)
            x = layer(x, p, visible, self.encoder.rotary_emb)
        if output_hidden_states:
            history.append(x)

        x = x.masked_fill(pad[..., None], 0)
        xy = p.clamp_min(0)
        width = xy[..., 0].max(-1, keepdim=True).values + 1
        group = xy[..., 0] // k + (width // k) * (xy[..., 1] // k)
        weights = F.one_hot(group, n // (k * k)).float() / (k * k)
        pooled = (weights.transpose(1, 2) @ x.float()).to(x.dtype)
        valid_groups = weights.any(1)
        pooled = (pooled.float() * c.hidden_size**0.5)[valid_groups]
        if c.standardize:
            pooled = (pooled - self.std_bias.float()) * self.std_scale.float()
        return PackedVisionOutput(
            pooled.to(x.dtype), tuple(counts), tuple(history) if output_hidden_states else None
        )


def pack_gemma4_images(images, config, *, max_patches=None):

    images = list(images)
    if not images:
        raise ValueError("No images")
    patches, positions = [], []
    p, k = config.patch_size, config.pooling_kernel_size
    for image in images:
        if (
            image.ndim != 3
            or image.shape[0] != 3
            or image.shape[1] % (p * k)
            or image.shape[2] % (p * k)
        ):
            raise ValueError(
                "RGB image dimensions must be divisible by patch_size*pooling_kernel_size"
            )
        h, w = image.shape[1] // p, image.shape[2] // p
        patches.append(image.reshape(3, h, p, w, p).permute(1, 3, 2, 4, 0).reshape(h * w, -1))
        y, x = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )
        positions.append(torch.stack((x, y), -1).reshape(h * w, 2))
    budget = max(v.shape[0] for v in patches) if max_patches is None else max_patches
    if type(budget) is not int or budget % (k * k) or budget < max(v.shape[0] for v in patches):
        raise ValueError("Invalid patch budget")
    return {
        "pixel_values": torch.stack([F.pad(v, (0, 0, 0, budget - len(v))) for v in patches]),
        "pixel_position_ids": torch.stack(
            [F.pad(v, (0, 0, 0, budget - len(v)), value=-1) for v in positions]
        ),
    }


@dataclass(frozen=True)
class Gemma4Config:
    architecture: ClassVar[str] = "gemma4"
    text_config: Gemma4TextConfig = field(default_factory=Gemma4TextConfig)
    vision_config: Gemma4VisionConfig = field(default_factory=Gemma4VisionConfig)
    image_token_id: int = 60
    video_token_id: int = 61

    def __post_init__(self):
        if not isinstance(self.text_config, Gemma4TextConfig) or not isinstance(
            self.vision_config, Gemma4VisionConfig
        ):
            raise ValueError(
                "Gemma4 multimodal requires actual Gemma4 text and vision configurations"
            )
        if self.text_config.pad_token_id is None:
            raise ValueError("Visual token identity PLE requires an explicit padding token")
        if (
            any(type(v) is not int or v < 0 for v in (self.image_token_id, self.video_token_id))
            or self.image_token_id == self.video_token_id
        ):
            raise ValueError("Image/video token IDs must be distinct nonnegative integers")
        if self.text_config.pad_token_id in {self.image_token_id, self.video_token_id}:
            raise ValueError("Visual placeholder cannot be PAD")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
        }


class VisionConnector(nn.Module):
    def __init__(self, vision, text):
        super().__init__()
        self.embedding_pre_projection_norm = FloatRMSNorm(
            vision.hidden_size, vision.rms_norm_eps, with_scale=False
        )
        self.embedding_projection = nn.Linear(vision.hidden_size, text.hidden_size, bias=False)
        nn.init.normal_(self.embedding_projection.weight, std=text.initializer_range)

    def forward(self, x):
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config, self.model_key = config, configuration_key(config)
        text = Gemma4ForCausalLM(config.text_config)
        self.model = nn.Module()
        self.model.language_model = text.model
        self.model.vision_tower = Gemma4VisionModel(config.vision_config)
        self.model.embed_vision = VisionConnector(config.vision_config, config.text_config)
        self.lm_head = text.lm_head

    @property
    def text_config(self):
        return self.config.text_config

    def get_decoder(self):
        return self.model.language_model

    def get_image_features(self, pixel_values, image_position_ids):
        visual = self.model.vision_tower(pixel_values, image_position_ids)
        return tuple(self.model.embed_vision(visual.last_hidden_state).split(visual.counts))

    def get_video_features(self, pixel_values_videos, video_position_ids):

        if pixel_values_videos.ndim != 4 or video_position_ids.shape != (
            *pixel_values_videos.shape[:3],
            2,
        ):
            raise ValueError("Gemma4 video needs packed frames[V,F,N,3P²] and positions[V,F,N,2]")
        v, frames = pixel_values_videos.shape[:2]
        if not frames:
            raise ValueError("Video has no frames")
        visual = self.model.vision_tower(
            pixel_values_videos.flatten(0, 1), video_position_ids.flatten(0, 1)
        )
        counts = tuple(
            sum(visual.counts[index * frames : (index + 1) * frames]) for index in range(v)
        )
        return tuple(self.model.embed_vision(visual.last_hidden_state).split(counts))

    @staticmethod
    def _merge(inputs, mask, features, owners, name):
        b = inputs.shape[0]
        if owners is None:
            if len(features) != b:
                raise ValueError(
                    f"{name}_batch_indices required unless there is exactly one {name} per sample"
                )
            owners = torch.arange(b, device=inputs.device)
        if (
            owners.shape != (len(features),)
            or owners.dtype != torch.long
            or (owners < 0).any()
            or (owners >= b).any()
        ):
            raise ValueError(f"Invalid {name} ownership indices")
        pieces = []
        for row in range(b):
            owned = [feature for index, feature in enumerate(features) if int(owners[index]) == row]
            count = sum(len(feature) for feature in owned)
            if count != int(mask[row].sum()):
                raise ValueError(f"{name} features/placeholders mismatch in sample {row}")
            pieces.extend(owned)
        if not pieces:
            raise ValueError(f"No {name} features")
        values = torch.cat(pieces).to(inputs)
        return inputs.masked_scatter(mask[..., None].expand_as(inputs), values)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
        per_layer_inputs=None,
        pixel_values=None,
        image_position_ids=None,
        image_batch_indices=None,
        pixel_values_videos=None,
        video_position_ids=None,
        video_batch_indices=None,
        mm_token_type_ids=None,
    ):
        if input_ids is None:
            if any(
                v is not None
                for v in (
                    pixel_values,
                    image_position_ids,
                    pixel_values_videos,
                    video_position_ids,
                    mm_token_type_ids,
                    image_batch_indices,
                    video_batch_indices,
                )
            ):
                raise ValueError(
                    "Raw visual inputs require token IDs; embedding reverse-lookup is intentionally unsupported"
                )
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                state=state,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                per_layer_inputs=per_layer_inputs,
            )
        if inputs_embeds is not None or per_layer_inputs is not None:
            raise ValueError("Token IDs already determine embeddings and token-identity PLE")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("Text tokens must be integer [B,S]")
        image_mask, video_mask = (
            input_ids == self.config.image_token_id,
            input_ids == self.config.video_token_id,
        )
        safe_ids = input_ids.masked_fill(image_mask | video_mask, self.text_config.pad_token_id)
        inputs = self.get_input_embeddings()(safe_ids)

        ple = (
            self.get_per_layer_inputs(safe_ids)
            if self.text_config.hidden_size_per_layer_input
            else None
        )
        if (pixel_values is None) != (image_position_ids is None) or (
            pixel_values_videos is None
        ) != (video_position_ids is None):
            raise ValueError("Visual pixels and their position IDs must be supplied together")
        if pixel_values is not None:
            inputs = self._merge(
                inputs,
                image_mask,
                self.get_image_features(pixel_values, image_position_ids),
                image_batch_indices,
                "image",
            )
        elif image_mask.any() or image_batch_indices is not None:
            raise ValueError("Image placeholders need complete image features")
        if pixel_values_videos is not None:
            inputs = self._merge(
                inputs,
                video_mask,
                self.get_video_features(pixel_values_videos, video_position_ids),
                video_batch_indices,
                "video",
            )
        elif video_mask.any() or video_batch_indices is not None:
            raise ValueError("Video placeholders need complete video features")
        blocks = None
        if mm_token_type_ids is not None:
            if (
                mm_token_type_ids.shape != input_ids.shape
                or mm_token_type_ids.dtype != torch.long
                or not ((mm_token_type_ids >= 0) & (mm_token_type_ids <= 2)).all()
            ):
                raise ValueError(
                    "Multimodal types must be 0=text, 1=image, 2=video with shape [B,S]"
                )
            if not torch.equal(mm_token_type_ids == 1, image_mask) or not torch.equal(
                mm_token_type_ids == 2, video_mask
            ):
                raise ValueError("Multimodal types disagree with visual token placeholders")
            if self.text_config.use_bidirectional_attention == "vision":
                vision = mm_token_type_ids > 0
                prior = F.pad(vision[:, :-1], (1, 0), value=False)
                blocks = (vision & ~prior).long().cumsum(1) - 1
                blocks = blocks.masked_fill(~vision, -1)
        return super().forward(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            per_layer_inputs=ple,
            vision_block_ids=blocks,
        )
