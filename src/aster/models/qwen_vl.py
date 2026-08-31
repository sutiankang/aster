"""Qwen3-VL packed visual grids, DeepStack, and interleaved three-axis MRoPE."""

from dataclasses import asdict, dataclass, field
import itertools
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import StateCapabilities
from aster.nn import LayerNorm, RopeConfig, RotaryEmbedding
from aster.nn.attention import scaled_attention
from aster.nn.vision import packed_vision_attention
from .config import Qwen3Config
from .decoder import CausalLM
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    architecture: ClassVar[str] = "qwen3_vl_vision"
    depth: int = 2
    hidden_size: int = 32
    intermediate_size: int = 64
    num_heads: int = 4
    in_channels: int = 3
    patch_size: int = 2
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 32
    num_position_embeddings: int = 16
    deepstack_visual_indexes: tuple[int, ...] = (0, 1)
    initializer_range: float = 0.02

    def __post_init__(self):
        object.__setattr__(self, "deepstack_visual_indexes", tuple(self.deepstack_visual_indexes))
        if (
            min(
                self.depth,
                self.hidden_size,
                self.intermediate_size,
                self.num_heads,
                self.in_channels,
                self.patch_size,
                self.spatial_merge_size,
                self.temporal_patch_size,
                self.out_hidden_size,
            )
            < 1
        ):
            raise ValueError("Invalid Qwen3-VL vision dimensions")
        if self.hidden_size % self.num_heads or (self.hidden_size // self.num_heads) % 4:
            raise ValueError(
                "Vision heads need dimensions divisible by four for H/W rotary coordinates"
            )
        if (
            math.isqrt(self.num_position_embeddings) ** 2 != self.num_position_embeddings
            or self.num_position_embeddings < 1
        ):
            raise ValueError("Learned vision position table must form a square grid")
        if tuple(
            sorted(set(self.deepstack_visual_indexes))
        ) != self.deepstack_visual_indexes or any(
            x not in range(self.depth) for x in self.deepstack_visual_indexes
        ):
            raise ValueError(
                "DeepStack visual layers must be distinct and in increasing depth order"
            )

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class Qwen3VLTextConfig(Qwen3Config):
    architecture: ClassVar[str] = "qwen3_vl_text"
    head_dim: int = 12
    mrope_section: tuple[int, int, int] = (2, 2, 2)
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(theta=500000.0))

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if (
            len(self.mrope_section) != 3
            or min(self.mrope_section) < 0
            or sum(self.mrope_section) != self.attention_head_dim // 2
        ):
            raise ValueError("mRoPE T/H/W sections must cover the rotary half dimension")
        if self.rope.interleaved or self.layer_types is not None or self.sliding_window is not None:
            raise ValueError(
                "Qwen3-VL has its own interleaved three-axis rotary and full causal text attention"
            )


@dataclass(frozen=True)
class Qwen3VLConfig:
    architecture: ClassVar[str] = "qwen3_vl"
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    image_token_id: int = 28
    video_token_id: int = 29
    vision_start_token_id: int = 26
    vision_end_token_id: int = 27

    def __post_init__(self):
        if not isinstance(self.text_config, Qwen3VLTextConfig) or not isinstance(
            self.vision_config, Qwen3VLVisionConfig
        ):
            raise TypeError("Qwen3-VL needs its actual text and vision configurations")
        if (
            self.vision_config.out_hidden_size != self.text_config.hidden_size
            or len(self.vision_config.deepstack_visual_indexes) > self.text_config.num_hidden_layers
        ):
            raise ValueError("Vision merger width/DeepStack depth do not match the text model")
        tokens = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if len(set(tokens)) != 4 or any(not 0 <= x < self.text_config.vocab_size for x in tokens):
            raise ValueError("Multimodal special IDs must be distinct vocabulary entries")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        }


def pack_qwen_pixels(frames, config):

    c = config
    if (
        frames.ndim != 4
        or frames.shape[0] < 1
        or frames.shape[1] != c.in_channels
        or not frames.is_floating_point()
    ):
        raise ValueError("Frames must be nonempty normalized floating TCHW")
    t, channels, height, width = frames.shape
    p, m, tp = c.patch_size, c.spatial_merge_size, c.temporal_patch_size
    if height % (p * m) or width % (p * m):
        raise ValueError("Image size must be divisible by patch*merge size")
    extra = (-t) % tp
    if extra:
        frames = torch.cat((frames, frames[-1:].expand(extra, -1, -1, -1)))
    gt, gh, gw = frames.shape[0] // tp, height // p, width // p
    values = frames.reshape(gt, tp, channels, gh // m, m, p, gw // m, m, p)
    values = values.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gt * gh * gw, channels * tp * p * p)
    return values, torch.tensor([[gt, gh, gw]], dtype=torch.long, device=frames.device)


def _validated_grids(grid, config):
    if (
        grid is None
        or grid.ndim != 2
        or grid.shape[1] != 3
        or grid.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Vision grid must be integer [items,T/H/W]")
    rows = grid.tolist()
    m = config.spatial_merge_size
    if not rows or any(min(row) < 1 or row[1] % m or row[2] % m for row in rows):
        raise ValueError("Vision grid is empty or cannot be spatially merged")
    return rows


class QwenPatchProjection(nn.Conv3d):
    def forward(self, pixels):

        return super().forward(pixels.to(self.weight.dtype))


class QwenVisionPatchEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        size = (c.temporal_patch_size, c.patch_size, c.patch_size)
        self.proj = QwenPatchProjection(c.in_channels, c.hidden_size, size, stride=size, bias=True)

    def forward(self, packed):
        c = self.config
        return self.proj(
            packed.reshape(-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        ).flatten(1)


class QwenVisionMerger(nn.Module):
    def __init__(self, c, postshuffle=False):
        super().__init__()
        self.width, self.postshuffle = c.hidden_size * c.spatial_merge_size**2, postshuffle
        self.norm = LayerNorm(self.width if postshuffle else c.hidden_size, 1e-6)
        self.linear_fc1 = nn.Linear(self.width, self.width)
        self.linear_fc2 = nn.Linear(self.width, c.out_hidden_size)

    def forward(self, hidden):
        hidden = self.norm(hidden.reshape(-1, self.width) if self.postshuffle else hidden).reshape(
            -1, self.width
        )
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden)))


class QwenVisionMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.linear_fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, hidden):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden), approximate="tanh"))


class QwenVisionAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_heads
        self.qkv = nn.Linear(c.hidden_size, c.hidden_size * 3)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, hidden, angles, lengths):
        n, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(n, 3, self.heads, -1).unbind(1)
        return self.proj(packed_vision_attention(q, k, v, angles, lengths))


class QwenVisionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(c.hidden_size, 1e-6), LayerNorm(c.hidden_size, 1e-6)
        self.attn, self.mlp = QwenVisionAttention(c), QwenVisionMLP(c)

    def forward(self, hidden, angles, lengths):
        hidden = hidden + self.attn(self.norm1(hidden), angles, lengths)
        return hidden + self.mlp(self.norm2(hidden))


@dataclass
class QwenVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]


class Qwen3VLVisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = QwenVisionPatchEmbedding(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList(QwenVisionBlock(config) for _ in range(config.depth))
        self.merger = QwenVisionMerger(config)
        self.deepstack_merger_list = nn.ModuleList(
            QwenVisionMerger(config, True) for _ in config.deepstack_visual_indexes
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv3d)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values, grid_thw):
        c = self.config
        grids = _validated_grids(grid_thw, c)
        if pixel_values.ndim != 2 or pixel_values.shape != (
            sum(t * h * w for t, h, w in grids),
            c.in_channels * c.temporal_patch_size * c.patch_size**2,
        ):
            raise ValueError("Packed pixel shape does not match declared grids")
        hidden = self.patch_embed(pixel_values)
        m, learned_side = c.spatial_merge_size, math.isqrt(c.num_position_embeddings)
        coordinates, positions, lengths = [], [], []

        table = self.pos_embed(torch.arange(c.num_position_embeddings, device=hidden.device))
        table = table.reshape(learned_side, learned_side, c.hidden_size).permute(2, 0, 1)[None]
        for t, h, w in grids:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=hidden.device),
                torch.arange(w, device=hidden.device),
                indexing="ij",
            )
            coords = (
                torch.stack((yy, xx), -1)
                .reshape(h // m, m, w // m, m, 2)
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, 2)
            )
            coordinates.append(coords.repeat(t, 1))
            interpolated = F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)[
                0
            ].permute(1, 2, 0)
            positions.append(
                interpolated.reshape(h // m, m, w // m, m, c.hidden_size)
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, c.hidden_size)
                .repeat(t, 1)
            )
            lengths.extend([h * w] * t)
        hidden = hidden + torch.cat(positions).to(hidden.dtype)
        head = c.hidden_size // c.num_heads
        frequencies = 10000 ** (
            -torch.arange(0, head // 2, 2, device=hidden.device).float() / (head // 2)
        )
        angles = (torch.cat(coordinates).float()[..., None] * frequencies).flatten(1)
        deep = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, angles, lengths)
            if index in c.deepstack_visual_indexes:
                deep.append(
                    self.deepstack_merger_list[c.deepstack_visual_indexes.index(index)](hidden)
                )
        return QwenVisionOutput(hidden, self.merger(hidden), tuple(deep))


class InterleavedMRope(RotaryEmbedding):
    def __init__(self, c, dimension=None):
        super().__init__(c.attention_head_dim if dimension is None else dimension, c.rope)
        self.sections = c.mrope_section

    def forward(self, values, positions):
        if positions.ndim == 2:
            positions = positions[None].expand(3, -1, -1)
        if positions.shape[0] != 3:
            raise ValueError("mRoPE needs temporal/height/width coordinates")
        angles = positions.float()[..., None] * self.inv_freq.to(values.device)

        mixed = angles[0].clone()
        for axis in (1, 2):
            mixed[..., axis : self.sections[axis] * 3 : 3] = angles[
                axis, ..., axis : self.sections[axis] * 3 : 3
            ]
        cosine = torch.cat((mixed, mixed), -1).cos().to(values.dtype) * self.attention_factor
        sine = torch.cat((mixed, mixed), -1).sin().to(values.dtype) * self.attention_factor
        a, b = values.chunk(2, -1)
        return values * cosine[:, None] + torch.cat((-b, a), -1) * sine[:, None]


class Qwen3VLTextForCausalLM(CausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.self_attn.rope = InterleavedMRope(config)

    def validate_positions(self, positions, hidden):
        shape = (3, hidden.shape[0], hidden.shape[1])
        if positions.shape not in (hidden.shape[:2], shape) or (positions < 0).any():
            raise ValueError(
                "Qwen3-VL positions must be [B,S] or explicit [3,B,S] T/H/W coordinates"
            )


@dataclass(frozen=True)
class VisionLanguageState:
    token_state: object
    rope_delta: torch.Tensor
    kind: str = "qwen3_vl_kv"

    @property
    def model_key(self):
        return self.token_state.model_key

    @property
    def seen_tokens(self):
        return self.token_state.seen_tokens

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(self.token_state.fork(), self.rope_delta.clone())

    def reorder(self, indices):
        return type(self)(
            self.token_state.reorder(indices), self.rope_delta.index_select(0, indices)
        )

    def truncate(self, length):
        raise ValueError(
            "Truncating visual spans also changes mRoPE delta; checkpoint+replay is required"
        )


def multimodal_positions(
    input_ids, modality_ids, merge_size, image_grid=None, video_grid=None, padding=None
):

    if (
        modality_ids.shape != input_ids.shape
        or not ((modality_ids >= 0) & (modality_ids <= 2)).all()
    ):
        raise ValueError("modality_ids must match input_ids and use text=0/image=1/video=2")
    rows = {1: [] if image_grid is None else image_grid.tolist(), 2: []}
    if video_grid is not None:
        for t, h, w in video_grid.tolist():
            rows[2].extend([[1, h, w]] * t)
    offsets = {1: 0, 2: 0}
    result = torch.zeros(3, *input_ids.shape, dtype=torch.long, device=input_ids.device)
    deltas = []
    valid = torch.ones_like(input_ids, dtype=torch.bool) if padding is None else padding.bool()
    if valid.shape != input_ids.shape or not valid.any(-1).all():
        raise ValueError("Every multimodal sequence needs at least one valid token")
    for b in range(input_ids.shape[0]):
        current, segments = 0, []
        for kind, tokens in itertools.groupby(modality_ids[b, valid[b]].tolist()):
            count = sum(1 for _ in tokens)
            if kind == 0:
                segment = torch.arange(count, device=input_ids.device)[None].expand(3, -1) + current
                current += count
            else:
                if offsets[kind] >= len(rows[kind]):
                    raise ValueError("Missing grid for a visual token span")
                t, h, w = rows[kind][offsets[kind]]
                offsets[kind] += 1
                h, w = h // merge_size, w // merge_size
                if t * h * w != count:
                    raise ValueError("Visual token span length differs from its declared grid")
                coords = torch.meshgrid(
                    torch.arange(t, device=input_ids.device),
                    torch.arange(h, device=input_ids.device),
                    torch.arange(w, device=input_ids.device),
                    indexing="ij",
                )
                segment = torch.stack(coords).reshape(3, -1) + current
                current += max(h, w)
            segments.append(segment)
        positions = torch.cat(segments, -1)
        result[:, b, valid[b]] = positions
        deltas.append(positions.max() + 1 - valid[b].sum())
    if any(offsets[kind] != len(rows[kind]) for kind in (1, 2)):
        raise ValueError("Unused visual grids have no matching token span")
    return result, torch.stack(deltas)[:, None]


class Qwen3VLForConditionalGeneration(Qwen3VLTextForCausalLM):
    vision_state_type = VisionLanguageState

    def __init__(self, config):
        nn.Module.__init__(self)
        text = Qwen3VLTextForCausalLM(config.text_config)
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.language_model = text.model
        self.lm_head = text.lm_head
        self.model.visual = Qwen3VLVisionModel(config.vision_config)

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
        mm_token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one token or embedding input")
        embeddings = (
            self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        )
        b, s = embeddings.shape[:2]
        if state is not None:
            if (
                not isinstance(state, self.vision_state_type)
                or state.model_key != self.model_key
                or state.rope_delta.shape != (b, 1)
            ):
                raise ValueError("Vision-language state/model/batch mismatch")
            if any(
                x is not None
                for x in (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
            ):
                raise ValueError("New visual context requires a new prefill state")
            token_state, delta = state.token_state, state.rope_delta
        else:
            token_state, delta = None, torch.zeros(b, 1, dtype=torch.long, device=embeddings.device)
        has_visual = any(
            x is not None
            for x in (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        )
        if has_visual and (input_ids is None or mm_token_type_ids is None):
            raise ValueError("Visual prefill needs input_ids and explicit mm_token_type_ids")
        modalities = (
            torch.zeros(b, s, dtype=torch.long, device=embeddings.device)
            if mm_token_type_ids is None
            else mm_token_type_ids
        )
        if modalities.shape != (b, s):
            raise ValueError("Modality IDs must align with current input tokens")
        if input_ids is not None:
            expected = torch.zeros_like(input_ids)
            expected = torch.where(input_ids == self.config.image_token_id, 1, expected)
            expected = torch.where(input_ids == self.config.video_token_id, 2, expected)
            if not torch.equal(expected, modalities):
                raise ValueError("Modality IDs disagree with image/video placeholder IDs")
        additions = None
        if has_visual:
            additions = [
                torch.zeros_like(embeddings)
                for _ in self.config.vision_config.deepstack_visual_indexes
            ]
            for kind, pixels, grid in (
                (1, pixel_values, image_grid_thw),
                (2, pixel_values_videos, video_grid_thw),
            ):
                mask = modalities == kind
                if pixels is None:
                    if grid is not None or mask.any():
                        raise ValueError("Missing pixels for declared visual tokens/grid")
                    continue
                grids = _validated_grids(grid, self.config.vision_config)
                if kind == 1 and any(t != 1 for t, _, _ in grids):
                    raise ValueError("Image grids must have temporal size one")
                visual = self.model.visual(pixels, grid)
                if int(mask.sum()) != visual.pooler_output.shape[0]:
                    raise ValueError("Visual feature and placeholder counts differ")
                embeddings = embeddings.masked_scatter(
                    mask[..., None], visual.pooler_output.to(embeddings)
                )
                for index, features in enumerate(visual.deepstack_features):
                    additions[index] = additions[index] + torch.zeros_like(
                        embeddings
                    ).masked_scatter(mask[..., None], features.to(embeddings))
            computed, delta = multimodal_positions(
                input_ids,
                modalities,
                self.config.vision_config.spatial_merge_size,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
            )
            if position_ids is None:
                position_ids = computed
        elif modalities.any():
            raise ValueError("Visual placeholders require visual prefill data")
        if position_ids is None:
            seen = 0 if token_state is None else token_state.seen_tokens
            if attention_mask is None:
                positions = torch.arange(seen, seen + s, device=embeddings.device)[None].expand(
                    b, -1
                )
            else:
                if attention_mask.shape != (b, seen + s):
                    raise ValueError("Padding mask must cover complete history")
                positions = (attention_mask.long().cumsum(-1) - 1).masked_fill(
                    attention_mask == 0, 0
                )[:, -s:]
            position_ids = (positions + delta)[None].expand(3, -1, -1)
        output = super().forward(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            state=token_state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            layer_additions=additions,
        )
        if use_cache:
            output.state = self.vision_state_type(output.state, delta)
        output.auxiliary = {**(output.auxiliary or {}), "rope_delta": delta}
        return output
