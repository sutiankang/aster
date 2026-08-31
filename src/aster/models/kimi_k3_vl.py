"""Kimi K3 MoonViT and PatchMergerMLPV2 visual conditioning."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn.vision import packed_vision_attention
from .kimi import pack_kimi_patches, _grids
from .kimi_k3 import KimiK3TextConfig, KimiK3ForCausalLM
from .serialization import LocalModelMixin, configuration_key
from .vision import VisionOutput


@dataclass(frozen=True)
class KimiK3VisionConfig:
    architecture: ClassVar[str] = "kimi_k3_vision"
    patch_size: int = 2
    pos_emb_height: int = 4
    pos_emb_width: int = 4
    pos_emb_time: int = 4
    hidden_size: int = 32
    qkv_hidden_size: int = 48
    intermediate_size: int = 64
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    merge_kernel_size: tuple[int, int] = (2, 2)
    initializer_range: float = 0.02

    def __post_init__(self):
        object.__setattr__(self, "merge_kernel_size", tuple(self.merge_kernel_size))
        dims = (
            self.patch_size,
            self.pos_emb_height,
            self.pos_emb_width,
            self.pos_emb_time,
            self.hidden_size,
            self.qkv_hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_hidden_layers,
        )
        if any(type(x) is not int or x < 1 for x in dims) or self.hidden_size % 2:
            raise ValueError("Invalid K3 vision/time-embedding dimensions")
        if (
            self.qkv_hidden_size % self.num_attention_heads
            or (self.qkv_hidden_size // self.num_attention_heads) % 4
        ):
            raise ValueError("K3 spatial RoPE requires attention head dimensions divisible by four")
        if len(self.merge_kernel_size) != 2 or any(
            type(x) is not int or x < 1 for x in self.merge_kernel_size
        ):
            raise ValueError("K3 vision requires a positive 2D merge kernel")
        if self.initializer_range <= 0:
            raise ValueError("Vision initializer must be positive")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class K3Position(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.weight = nn.Parameter(torch.empty(c.pos_emb_height, c.pos_emb_width, c.hidden_size))
        nn.init.normal_(self.weight, std=c.initializer_range)

    def forward(self, hidden, grids):
        positions = []
        for t, h, w in grids:
            if (h, w) == self.weight.shape[:2]:
                spatial = self.weight.flatten(0, 1)
            else:
                spatial = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1)[None],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    .permute(1, 2, 0)
                    .reshape(h * w, -1)
                )
            if t == 1:
                positions.append(spatial)
            else:
                omega = 10000 ** (
                    -torch.arange(
                        self.config.hidden_size // 2, dtype=torch.float32, device=hidden.device
                    )
                    / (self.config.hidden_size / 2)
                )
                angles = torch.arange(t, dtype=torch.float32, device=hidden.device)[:, None] * omega
                time = torch.cat((angles.sin(), angles.cos()), -1).to(spatial.dtype)
                positions.append((spatial[None] + time[:, None]).reshape(t * h * w, -1))
        return hidden + torch.cat(positions).to(hidden.dtype)


class K3PatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Conv2d(3, c.hidden_size, c.patch_size, stride=c.patch_size, bias=False)
        self.pos_emb = K3Position(c)

    def forward(self, pixels, grids):
        return self.pos_emb(self.proj(pixels.to(self.proj.weight.dtype)).flatten(1), grids)


class K3VisionMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc0 = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.fc1 = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, value):
        return self.fc1(F.gelu(self.fc0(value), approximate="tanh"))


class K3VisionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_attention_heads

        self.norm0, self.norm1 = nn.RMSNorm(c.hidden_size), nn.RMSNorm(c.hidden_size)
        self.wqkv = nn.Linear(c.hidden_size, 3 * c.qkv_hidden_size, bias=False)
        self.wo = nn.Linear(c.qkv_hidden_size, c.hidden_size, bias=False)
        self.mlp = K3VisionMLP(c)

    def forward(self, value, angles, lengths):
        q, k, v = self.wqkv(self.norm0(value)).reshape(len(value), 3, self.heads, -1).unbind(1)
        value = value + self.wo(
            packed_vision_attention(
                q, k, v, angles, lengths, interleaved=True, implementation="sdpa"
            )
        )
        return value + self.mlp(self.norm1(value))


class KimiK3VisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = K3PatchEmbed(config)
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList(
            K3VisionBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.encoder.final_layernorm = nn.RMSNorm(config.hidden_size)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=config.initializer_range)

    def forward(self, pixel_values, grid_thw):
        c = self.config
        grids = _grids(grid_thw, c)
        if any(h > 512 or w > 512 for _, h, w in grids):
            raise ValueError("K3 vision rotary grid exceeds published 512x512 support")
        if (
            pixel_values.shape
            != (sum(t * h * w for t, h, w in grids), 3, c.patch_size, c.patch_size)
            or not pixel_values.is_floating_point()
            or not torch.isfinite(pixel_values).all()
        ):
            raise ValueError("K3 finite normalized packed pixels must match declared grids")
        hidden = self.patch_embed(pixel_values, grids)
        coordinates = []
        for t, h, w in grids:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=hidden.device),
                torch.arange(w, device=hidden.device),
                indexing="ij",
            )
            coordinates.append(torch.stack((xx, yy), -1).reshape(-1, 2).repeat(t, 1))
        d = c.qkv_hidden_size // c.num_attention_heads
        freq = 10000 ** (-torch.arange(0, d, 4, device=hidden.device).float() / d)
        angles = (torch.cat(coordinates).float()[:, None, :] * freq[None, :, None]).flatten(1)
        lengths = [t * h * w for t, h, w in grids]
        for block in self.encoder.blocks:
            hidden = block(hidden, angles, lengths)
        hidden = self.encoder.final_layernorm(hidden)
        outputs, offset = [], 0
        kh, kw = c.merge_kernel_size
        for t, h, w in grids:
            clip = (
                hidden[offset : offset + t * h * w]
                .reshape(t, h // kh, kh, w // kw, kw, c.hidden_size)
                .permute(0, 1, 3, 2, 4, 5)
            )
            pooled = clip[0] if t == 1 else clip.mean(0)
            outputs.append(pooled.reshape((h // kh) * (w // kw), kh * kw, c.hidden_size))
            offset += t * h * w
        return VisionOutput(hidden, torch.cat(outputs))


@dataclass(frozen=True)
class KimiK3Config:
    architecture: ClassVar[str] = "kimi_k3"
    text_config: KimiK3TextConfig = field(default_factory=KimiK3TextConfig)
    vision_config: KimiK3VisionConfig = field(default_factory=KimiK3VisionConfig)
    media_token_id: int = 31
    projector_ln_eps: float = 1e-5

    def __post_init__(self):
        if (
            type(self.text_config) is not KimiK3TextConfig
            or type(self.vision_config) is not KimiK3VisionConfig
        ):
            raise ValueError("K3 requires dedicated text and MoonViT-V2 configurations")
        if (
            type(self.media_token_id) is not int
            or not 0 <= self.media_token_id < self.text_config.vocab_size
            or self.projector_ln_eps <= 0
        ):
            raise ValueError("Invalid K3 media-token/projector configuration")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "media_token_id": self.media_token_id,
            "projector_ln_eps": self.projector_ln_eps,
        }


class K3Projector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.width = (
            c.vision_config.hidden_size
            * c.vision_config.merge_kernel_size[0]
            * c.vision_config.merge_kernel_size[1]
        )
        self.proj = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.GELU(),
            nn.Linear(self.width, c.text_config.hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(c.text_config.hidden_size, eps=c.projector_ln_eps)

    def forward(self, features):
        return self.post_norm(self.proj(features.reshape(-1, self.width)))


@dataclass(frozen=True)
class KimiK3VisionState:
    language_state: object
    model_key: str
    kind: str = "kimi_k3_multimodal"

    @property
    def seen_tokens(self):
        return self.language_state.seen_tokens

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(self.language_state.fork(), self.model_key)

    def reorder(self, indices):
        return type(self)(self.language_state.reorder(indices), self.model_key)

    def truncate(self, length):
        raise ValueError("K3 multimodal recurrent cache requires snapshot+replay")


class KimiK3ForConditionalGeneration(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.vision_tower = KimiK3VisionModel(config.vision_config)
        self.mm_projector = K3Projector(config)
        self.language_model = KimiK3ForCausalLM(config.text_config)
        for module in self.mm_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.text_config.initializer_range)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(self, pixel_values, grid_thw):
        return self.mm_projector(self.vision_tower(pixel_values, grid_thw).pooler_output)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        grid_thw=None,
        media_batch_indices=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("K3 VLM needs exactly one input-ID/embedding sequence")
        if state is not None and (
            not isinstance(state, KimiK3VisionState) or state.model_key != self.model_key
        ):
            raise ValueError("K3 VLM state must belong to this complete multimodal configuration")
        has_media = pixel_values is not None
        if not has_media and (grid_thw is not None or media_batch_indices is not None):
            raise ValueError("Media grids/owners require actual pixels")
        if has_media and (state is not None or input_ids is None):
            raise ValueError("Raw media requires an uncached token-ID prefill")
        hidden = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        if has_media:
            c = self.config
            rows = _grids(grid_thw, c.vision_config)
            b = hidden.shape[0]
            if media_batch_indices is None:
                if len(rows) != b:
                    raise ValueError("Multiple media items need explicit per-item batch ownership")
                owners = list(range(b))
            else:
                if media_batch_indices.shape != (len(rows),) or media_batch_indices.dtype not in {
                    torch.int32,
                    torch.int64,
                }:
                    raise ValueError("Media owners must be an integer index per declared grid")
                owners = media_batch_indices.tolist()
                if any(x < 0 or x >= b for x in owners):
                    raise ValueError("Media owner outside token batch")
            features = self.get_image_features(pixel_values, grid_thw)
            kh, kw = c.vision_config.merge_kernel_size
            counts = [(h // kh) * (w // kw) for _, h, w in rows]
            chunks = features.split(counts)
            replacement = []
            slots = input_ids.eq(c.media_token_id)
            if (
                attention_mask is not None
                and (slots & ~attention_mask[:, -input_ids.shape[1] :].bool()).any()
            ):
                raise ValueError("Media placeholders cannot be attention padding")
            for index in range(b):
                owned = [chunks[j] for j, owner in enumerate(owners) if owner == index]
                count = sum(len(x) for x in owned)
                if int(slots[index].sum()) != count:
                    raise ValueError("Per-example media placeholders do not match owned features")
                if owned:
                    replacement.append(torch.cat(owned))
            hidden = hidden.masked_scatter(
                slots[..., None].expand_as(hidden), torch.cat(replacement).to(hidden.dtype)
            )
        elif input_ids is not None and input_ids.eq(self.config.media_token_id).any():
            raise ValueError("Reserved media token requires actual visual features in prefill")
        result = self.language_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=None if state is None else state.language_state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        updated = KimiK3VisionState(result.state, self.model_key) if use_cache else None
        return TokenOutput(result.logits, updated, result.hidden_states, result.auxiliary)
