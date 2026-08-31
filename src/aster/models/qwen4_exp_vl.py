"""Flash-Next visual conditioning with per-layer embeddings bound to original token IDs."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
from aster.core import StateCapabilities
from .qwen35 import Qwen35VisionConfig
from .qwen_vl import Qwen3VLVisionModel, multimodal_positions, _validated_grids
from .qwen4_exp import Qwen4ExpTextConfig, Qwen4ExpForCausalLM
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Qwen4ExpVisionConfig(Qwen35VisionConfig):
    architecture: ClassVar[str] = "qwen4_exp_vision"


@dataclass(frozen=True)
class Qwen4ExpConfig:
    architecture: ClassVar[str] = "qwen4_exp"
    text_config: Qwen4ExpTextConfig = field(default_factory=Qwen4ExpTextConfig)
    vision_config: Qwen4ExpVisionConfig = field(default_factory=Qwen4ExpVisionConfig)
    image_token_id: int = 28
    video_token_id: int = 29
    vision_start_token_id: int = 26
    vision_end_token_id: int = 27

    def __post_init__(self):
        if not isinstance(self.text_config, Qwen4ExpTextConfig) or not isinstance(
            self.vision_config, Qwen4ExpVisionConfig
        ):
            raise ValueError("Flash-Next needs Qwen4Exp text and verified no-DeepStack vision")
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("Vision/text width mismatch")
        tokens = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if len(set(tokens)) != 4 or min(tokens) < 0 or max(tokens) >= self.text_config.vocab_size:
            raise ValueError("Invalid visual special IDs")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                )
            },
        }


@dataclass(frozen=True)
class Qwen4ExpVisionState:
    token_state: object
    rope_delta: torch.Tensor
    model_key: str
    kind: str = "qwen4_exp_multimodal"

    @property
    def seen_tokens(self):
        return self.token_state.seen_tokens

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(self.token_state.fork(), self.rope_delta.clone(), self.model_key)

    def reorder(self, indices):
        return type(self)(
            self.token_state.reorder(indices),
            self.rope_delta.index_select(0, indices),
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError("Visual MRoPE/QSA/PLE/recurrent state needs snapshot+replay")


class Qwen4ExpForConditionalGeneration(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.language_model = Qwen4ExpForCausalLM(config.text_config)
        self.vision_tower = Qwen3VLVisionModel(config.vision_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @staticmethod
    def official_weight_name(name):

        if name.startswith("language_model.model."):
            return "model.language_model." + name[len("language_model.model.") :]
        if name.startswith("language_model.lm_head."):
            return "lm_head." + name[len("language_model.lm_head.") :]
        if name.startswith("vision_tower."):
            return "model.visual." + name[len("vision_tower.") :]
        raise ValueError("Unknown Flash-Next composition parameter path")

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        ple_input_ids=None,
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
            raise ValueError("Supply exactly one token or embedding input")
        embedded = (
            self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        )
        batch, length = embedded.shape[:2]
        has_visual = any(
            x is not None
            for x in (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
        )
        if state is not None:
            if (
                not isinstance(state, Qwen4ExpVisionState)
                or state.model_key != self.model_key
                or state.kind != "qwen4_exp_multimodal"
                or state.rope_delta.shape != (batch, 1)
            ):
                raise ValueError("Flash-Next multimodal state/config mismatch")
            if has_visual:
                raise ValueError("New visual data requires a fresh prefill")
            previous, delta = state.token_state, state.rope_delta
        else:
            previous, delta = None, torch.zeros(batch, 1, dtype=torch.long, device=embedded.device)
        seen = 0 if previous is None else previous.seen_tokens
        if attention_mask is not None and (
            attention_mask.shape != (batch, seen + length)
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("Mask must cover the complete physical token history")
        if has_visual and (input_ids is None or mm_token_type_ids is None):
            raise ValueError("Visual prefill needs token IDs and explicit modality IDs")
        modalities = (
            torch.zeros(batch, length, dtype=torch.long, device=embedded.device)
            if mm_token_type_ids is None
            else mm_token_type_ids
        )
        if modalities.shape != (batch, length) or not ((modalities >= 0) & (modalities <= 2)).all():
            raise ValueError("Invalid modality IDs")
        if input_ids is not None:
            expected = torch.where(
                input_ids == self.config.image_token_id,
                1,
                torch.where(input_ids == self.config.video_token_id, 2, 0),
            )
            if not torch.equal(modalities, expected):
                raise ValueError("Modality IDs disagree with visual placeholders")
        if has_visual:
            for kind, pixels, grid in (
                (1, pixel_values, image_grid_thw),
                (2, pixel_values_videos, video_grid_thw),
            ):
                selected = modalities == kind
                if pixels is None:
                    if selected.any() or grid is not None:
                        raise ValueError("Missing visual pixels")
                    continue
                rows = _validated_grids(grid, self.config.vision_config)
                if kind == 1 and any(t != 1 for t, _, _ in rows):
                    raise ValueError("Static image has temporal grid one")
                if (
                    attention_mask is not None
                    and (selected & ~attention_mask[:, -length:].bool()).any()
                ):
                    raise ValueError("Visual placeholders cannot be padding")
                visual = self.vision_tower(pixels, grid).pooler_output
                if int(selected.sum()) != len(visual):
                    raise ValueError("Visual feature/placeholder count mismatch")
                embedded = embedded.masked_scatter(selected[..., None], visual.to(embedded))
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
            raise ValueError("Visual placeholders require pixels in a fresh prefill")
        if position_ids is None:
            positions = (
                torch.arange(seen, seen + length, device=embedded.device)[None].expand(batch, -1)
                if attention_mask is None
                else (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)[
                    :, -length:
                ]
            )
            position_ids = (positions + delta)[None].expand(3, -1, -1)
        tokens = input_ids if ple_input_ids is None else ple_input_ids
        result = self.language_model(
            inputs_embeds=embedded,
            ple_input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=previous,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        if use_cache:
            result.state = Qwen4ExpVisionState(result.state, delta, self.model_key)
        result.auxiliary = {**(result.auxiliary or {}), "rope_delta": delta}
        return result
