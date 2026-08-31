"""Kimi K2.5/K2.6 MoonViT, temporal pooling, and MLA language components."""

from dataclasses import asdict, dataclass, field
import itertools
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import LayerNorm, RopeConfig
from aster.nn.vision import packed_vision_attention
from .config import DeepSeekV3Config
from .decoder import CausalLM
from .moe import DeepSeekV3ForCausalLM
from .serialization import LocalModelMixin, configuration_key
from .vision import VisionOutput


@dataclass(frozen=True)
class KimiK25VisionConfig:
    architecture: ClassVar[str] = "kimi_k25_vision"
    patch_size: int = 2
    pos_emb_height: int = 4
    pos_emb_width: int = 4
    pos_emb_time: int = 4
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    hidden_size: int = 32
    intermediate_size: int = 64
    merge_kernel_size: tuple[int, int] = (2, 2)
    rope: RopeConfig = field(default_factory=RopeConfig)
    max_position_embeddings: int = 128
    initializer_range: float = 0.02

    def __post_init__(self):
        object.__setattr__(self, "merge_kernel_size", tuple(self.merge_kernel_size))
        if (
            min(
                self.patch_size,
                self.pos_emb_height,
                self.pos_emb_width,
                self.pos_emb_time,
                self.num_attention_heads,
                self.num_hidden_layers,
                self.hidden_size,
                self.intermediate_size,
            )
            < 1
        ):
            raise ValueError("Invalid MoonViT dimensions")
        if (
            self.pos_emb_height != self.pos_emb_width
            or self.hidden_size % self.num_attention_heads
            or (self.hidden_size // self.num_attention_heads) % 4
        ):
            raise ValueError(
                "MoonViT requires a square learned grid and head dimensions divisible by four"
            )
        if len(self.merge_kernel_size) != 2 or min(self.merge_kernel_size) < 1:
            raise ValueError("Merge kernel must declare positive height and width")
        if self.rope.kind != "default" or self.rope.interleaved:
            raise ValueError(
                "MoonViT spatial frequencies currently implement its native default scheme"
            )

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class KimiK25Config:
    architecture: ClassVar[str] = "kimi_k25"
    text_config: DeepSeekV3Config = field(default_factory=DeepSeekV3Config)
    vision_config: KimiK25VisionConfig = field(default_factory=KimiK25VisionConfig)
    projection_hidden_size: int = 32
    projection_layer_norm_eps: float = 1e-5
    image_token_id: int = 28
    video_token_id: int = 29
    vision_start_token_id: int = 26
    vision_end_token_id: int = 27

    def __post_init__(self):
        if (
            type(self.text_config) is not DeepSeekV3Config
            or type(self.vision_config) is not KimiK25VisionConfig
        ):
            raise TypeError("KimiK25 uses explicit MLA text and MoonViT vision configs")
        if (
            self.projection_hidden_size != self.vision_config.hidden_size
            or self.projection_layer_norm_eps <= 0
        ):
            raise ValueError("Projector pre-norm must match each unmerged visual patch width")
        ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if min(ids) < 0 or len(set(ids)) != len(ids):
            raise ValueError("Special IDs must be nonnegative and distinct")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "projection_hidden_size": self.projection_hidden_size,
            "projection_layer_norm_eps": self.projection_layer_norm_eps,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        }


def normalize_kimi_pixels(images):

    values = images.float() / 255 if images.dtype == torch.uint8 else images
    if (
        values.ndim != 4
        or values.shape[1] != 3
        or not values.is_floating_point()
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Kimi normalization requires finite RGB TCHW uint8 or float [0,1]")
    return (values - 0.5) / 0.5


def pack_kimi_patches(frames, config):

    p = config.patch_size
    if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point():
        raise ValueError("MoonViT pixels must be floating RGB TCHW")
    t, _, height, width = frames.shape
    if t < 1 or t > config.pos_emb_time or height % p or width % p:
        raise ValueError("Invalid temporal range or patch alignment")
    h, w = height // p, width // p
    kh, kw = config.merge_kernel_size
    if h % kh or w % kw:
        raise ValueError("Visual grid cannot be merged with this kernel")
    patches = frames.reshape(t, 3, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, p, p)
    return patches, torch.tensor([[t, h, w]], dtype=torch.long, device=frames.device)


def _grids(grid, c):
    if (
        grid is None
        or grid.ndim != 2
        or grid.shape[1] != 3
        or grid.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("MoonViT grid must be integer [items,3]")
    rows = grid.tolist()
    kh, kw = c.merge_kernel_size
    if not rows or any(
        min(row) < 1 or row[0] > c.pos_emb_time or row[1] % kh or row[2] % kw for row in rows
    ):
        raise ValueError("Invalid MoonViT temporal/spatial grid")
    return rows


class KimiPositionEmbedding(nn.Module):
    _aster_semantic_buffers = ("time_position_embeddings",)

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.position_embeddings = nn.Parameter(
            torch.empty(c.pos_emb_height, c.pos_emb_width, c.hidden_size)
        )
        nn.init.trunc_normal_(self.position_embeddings)
        frequencies = 10000 ** (-torch.arange(0, c.hidden_size, 2).float() / c.hidden_size)
        angles = torch.arange(c.pos_emb_time).float()[:, None] * frequencies
        temporal = torch.cat((angles.sin(), angles.cos()), -1)
        self.register_buffer(
            "time_position_embeddings",
            torch.cat((torch.zeros(1, c.hidden_size), temporal)),
            persistent=False,
        )

    def forward(self, hidden, grids):
        table = self.position_embeddings.permute(2, 0, 1)[None]
        positions = []
        for t, h, w in grids:
            spatial = (
                F.interpolate(table, size=(h, w), mode="bicubic", align_corners=False)[0]
                .permute(1, 2, 0)
                .reshape(h * w, -1)
            )
            frame_ids = torch.arange(t, device=hidden.device) + int(t > 1)
            positions.append(
                (spatial[None] + self.time_position_embeddings[frame_ids, None]).reshape(
                    t * h * w, -1
                )
            )
        return hidden + torch.cat(positions).to(hidden.dtype)


class KimiPatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Conv2d(3, c.hidden_size, c.patch_size, stride=c.patch_size)
        self.pos_emb = KimiPositionEmbedding(c)

    def forward(self, patches, grids):
        hidden = self.proj(patches.to(self.proj.weight.dtype)).flatten(1)
        return self.pos_emb(hidden, grids)


class KimiAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, hidden, angles, lengths):
        q, k, v = (
            projection(hidden).reshape(hidden.shape[0], self.heads, -1)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        return self.proj(packed_vision_attention(q, k, v, angles, lengths))


class KimiMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, hidden):
        return self.fc2(F.gelu(self.fc1(hidden), approximate="tanh"))


class KimiVisionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(c.hidden_size, 1e-5), LayerNorm(c.hidden_size, 1e-5)
        self.attn, self.mlp = KimiAttention(c), KimiMLP(c)

    def forward(self, hidden, angles, lengths):
        hidden = hidden + self.attn(self.norm1(hidden), angles, lengths)
        return hidden + self.mlp(self.norm2(hidden))


class KimiK25VisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = KimiPatchEmbed(config)
        self.layers = nn.ModuleList(
            KimiVisionLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = LayerNorm(config.hidden_size, 1e-5)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values, grid_thw):
        c = self.config
        grids = _grids(grid_thw, c)
        if pixel_values.shape != (
            sum(t * h * w for t, h, w in grids),
            3,
            c.patch_size,
            c.patch_size,
        ):
            raise ValueError("Packed patch count/layout disagrees with MoonViT grid")
        hidden = self.patch_embed(pixel_values, grids)
        coordinates = []
        for t, h, w in grids:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=hidden.device),
                torch.arange(w, device=hidden.device),
                indexing="ij",
            )
            coordinates.append(torch.stack((xx, yy), -1).reshape(-1, 2).repeat(t, 1))
        spatial_dim = (c.hidden_size // c.num_attention_heads) // 2
        frequency = c.rope.theta ** (
            -torch.arange(0, spatial_dim, 2, device=hidden.device).float() / spatial_dim
        )

        angles = (torch.cat(coordinates).float()[:, None, :] * frequency[None, :, None]).flatten(1)
        lengths = [t * h * w for t, h, w in grids]
        for layer in self.layers:
            hidden = layer(hidden, angles, lengths)
        hidden = self.final_layernorm(hidden)
        kh, kw = c.merge_kernel_size
        outputs, offset = [], 0
        for t, h, w in grids:
            clip = hidden[offset : offset + t * h * w].reshape(
                t, h // kh, kh, w // kw, kw, c.hidden_size
            )

            outputs.append(
                clip.permute(0, 1, 3, 2, 4, 5)
                .mean(0)
                .reshape((h // kh) * (w // kw), kh * kw, c.hidden_size)
            )
            offset += t * h * w
        return VisionOutput(hidden, torch.cat(outputs))


class KimiProjector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.width = (
            c.vision_config.hidden_size
            * c.vision_config.merge_kernel_size[0]
            * c.vision_config.merge_kernel_size[1]
        )
        self.pre_norm = LayerNorm(c.projection_hidden_size, c.projection_layer_norm_eps)
        self.in_proj = nn.Linear(self.width, self.width)
        self.out_proj = nn.Linear(self.width, c.text_config.hidden_size)

    def forward(self, hidden):
        hidden = self.pre_norm(hidden).reshape(-1, self.width)
        return self.out_proj(F.gelu(self.in_proj(hidden)))


class KimiK25ForConditionalGeneration(CausalLM):
    state_kind = "mla_latent"

    def __init__(self, config):
        nn.Module.__init__(self)
        text = DeepSeekV3ForCausalLM(config.text_config)
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.language_model, self.lm_head = text.model, text.lm_head
        self.model.vision_tower = KimiK25VisionModel(config.vision_config)
        self.model.mm_projector = KimiProjector(config)
        for module in self.model.mm_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.text_config.initializer_range)
                nn.init.zeros_(module.bias)

    def get_decoder(self):
        return self.model.language_model

    @property
    def decoder_config(self):
        return self.config.text_config

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify token IDs or explicit embeddings")
        any_visual = any(
            x is not None
            for x in (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        )
        if any_visual and (state is not None or input_ids is None):
            raise ValueError("Kimi visual prefill requires input IDs and a fresh state")
        if input_ids is not None:
            placeholders = (input_ids == self.config.image_token_id) | (
                input_ids == self.config.video_token_id
            )

            embeddings = self.get_input_embeddings()(input_ids.masked_fill(placeholders, 0))
        else:
            embeddings = inputs_embeds
        for token_id, pixels, grid in (
            (self.config.image_token_id, pixel_values, image_grid_thw),
            (self.config.video_token_id, pixel_values_videos, video_grid_thw),
        ):
            mask = None if input_ids is None else input_ids == token_id
            if pixels is None:
                if grid is not None or mask is not None and mask.any():
                    raise ValueError("Visual placeholders/grid require their pixel patches")
                continue
            rows = _grids(grid, self.config.vision_config)
            kh, kw = self.config.vision_config.merge_kernel_size
            expected_lengths = [h * w // (kh * kw) for _, h, w in rows]
            actual_lengths = [
                sum(1 for _ in group)
                for row in mask.tolist()
                for key, group in itertools.groupby(row)
                if key
            ]
            if expected_lengths != actual_lengths:
                raise ValueError(
                    "Each Kimi visual span must match its temporally pooled spatial grid"
                )
            features = self.model.mm_projector(self.model.vision_tower(pixels, grid).pooler_output)
            embeddings = embeddings.masked_scatter(mask[..., None], features.to(embeddings))
        return super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
