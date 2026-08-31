"""Connect native Qwen visual features to the shared Cosmos3 understanding/generation model.

Reference: NVIDIA/cosmos-framework at 0e034bc98ffa3c3dfa19f037871f3a8bbc1c4d05,
reasoner/qwen3_vl/utils.py and mot/unified_mot.py.
Copyright2026 NVIDIA. OpenMDW-1.1; see NOTICE.md for licensing.
Inputs are already packed/preprocessed; image geometry is not changed here."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
from aster.core import StateCapabilities
from .cosmos3 import Cosmos3Config, Cosmos3MoT, Cosmos3State
from .qwen_vl import Qwen3VLVisionConfig, Qwen3VLVisionModel, multimodal_positions, _validated_grids
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Cosmos3VLMConfig:
    architecture: ClassVar[str] = "cosmos3_vlm"
    mot: Cosmos3Config = field(default_factory=Cosmos3Config)
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    image_token_id: int = 28
    video_token_id: int = 29
    vision_start_token_id: int = 26
    vision_end_token_id: int = 27

    def __post_init__(self):
        for name, constructor in (("mot", Cosmos3Config), ("vision_config", Qwen3VLVisionConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = dict(value)
                architecture = value.pop("architecture", constructor.architecture)
                if architecture != constructor.architecture:
                    raise ValueError("Cosmos3 VLM nested architecture mismatch")
                object.__setattr__(self, name, constructor(**value))
            if type(getattr(self, name)) is not constructor:
                raise ValueError("Cosmos3 VLM needs explicit native sub-configs")
        if self.mot.hidden_act != "silu" or not self.mot.qk_norm_for_text:
            raise ValueError("Cosmos3 Qwen visual wrapper is not the Nemotron/SigLIP2 Edge branch")
        if (
            self.vision_config.out_hidden_size != self.mot.hidden_size
            or len(self.vision_config.deepstack_visual_indexes) > self.mot.num_hidden_layers
        ):
            raise ValueError(
                "Cosmos3 visual merger/DeepStack dimensions differ from MoT understanding"
            )
        tokens = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if len(set(tokens)) != 4 or any(
            type(x) is not int or not 0 <= x < self.mot.vocab_size for x in tokens
        ):
            raise ValueError("Cosmos3 VLM special token IDs must be distinct vocabulary entries")

    def to_dict(self):
        return dict(
            architecture=self.architecture,
            mot=self.mot.to_dict(),
            vision_config=self.vision_config.to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
        )


@dataclass(frozen=True)
class Cosmos3VLMState:
    token_state: Cosmos3State
    rope_delta: torch.Tensor
    model_key: str
    kind: str = "cosmos3_vlm_understanding"

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
        raise ValueError(
            "Cosmos3 visual coordinates require snapshot/replay, not generic KV truncation"
        )


class Cosmos3VLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.transformer = Cosmos3MoT(config.mot)
        self.visual = Qwen3VLVisionModel(config.vision_config)

    def forward(
        self,
        input_ids=None,
        *,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        vision=None,
        sound=None,
        action=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        c = self.config
        has_image, has_video = pixel_values is not None, pixel_values_videos is not None
        if has_image and has_video:
            raise ValueError(
                "This Cosmos3 reasoner recipe accepts image or video prefill, not both"
            )
        if state is not None and (
            not isinstance(state, Cosmos3VLMState)
            or state.kind != "cosmos3_vlm_understanding"
            or state.model_key != self.model_key
        ):
            raise ValueError("Cosmos3 VLM state/config mismatch")
        if state is not None and (
            has_image or has_video or image_grid_thw is not None or video_grid_thw is not None
        ):
            raise ValueError("Cosmos3 VLM media is only consumed in the uncached prefill")
        token_state = None if state is None else state.token_state
        if input_ids is None:
            if token_state is None:
                raise ValueError("Cosmos3 VLM needs token IDs or a completed understanding cache")
            input_ids = torch.empty(
                len(token_state.attention_mask),
                0,
                dtype=torch.int64,
                device=token_state.attention_mask.device,
            )
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or input_ids.shape[0] < 1:
            raise ValueError("Cosmos3 VLM IDs must be int64[B,S]")
        b, length = input_ids.shape
        seen = 0 if token_state is None else token_state.seen_tokens
        if attention_mask is None:
            valid = torch.ones(b, length, dtype=torch.bool, device=input_ids.device)
            attention_mask = (
                valid if token_state is None else torch.cat((token_state.attention_mask, valid), 1)
            )
        if (
            attention_mask.shape != (b, seen + length)
            or attention_mask.dtype != torch.bool
            or attention_mask.device != input_ids.device
        ):
            raise ValueError(
                "Cosmos3 VLM requires a bool mask covering the entire physical history"
            )
        delta = (
            torch.zeros(b, 1, dtype=torch.long, device=input_ids.device)
            if state is None
            else state.rope_delta
        )
        if delta.shape != (b, 1) or delta.dtype != torch.long or delta.device != input_ids.device:
            raise ValueError("Cosmos3 VLM invalid cached mRoPE delta")
        modalities = torch.where(
            input_ids == c.image_token_id, 1, torch.where(input_ids == c.video_token_id, 2, 0)
        )
        embeddings = self.transformer.embed_tokens(input_ids)
        additions = None
        if has_image or has_video:
            pixels, grid, kind = (
                (pixel_values, image_grid_thw, 1)
                if has_image
                else (pixel_values_videos, video_grid_thw, 2)
            )
            if (
                grid is None
                or (has_image and video_grid_thw is not None)
                or (has_video and image_grid_thw is not None)
            ):
                raise ValueError(
                    "Cosmos3 VLM pixels must have exactly their corresponding visual grid"
                )
            grids = _validated_grids(grid, c.vision_config)
            if has_image and any(t != 1 for t, _, _ in grids):
                raise ValueError("Cosmos3 image grids must have T=1")
            if ((modalities != 0) & (modalities != kind)).any() or (
                (modalities != 0) & ~attention_mask
            ).any():
                raise ValueError("Cosmos3 visual placeholders disagree with media kind or padding")
            for row in input_ids:
                visual_id = c.image_token_id if has_image else c.video_token_id
                for index in torch.nonzero(row == visual_id).flatten().tolist():
                    if (index == 0 or row[index - 1] != visual_id) and (
                        index == 0 or row[index - 1] != c.vision_start_token_id
                    ):
                        raise ValueError("Cosmos3 visual spans need an explicit vision_start token")
                    if (index + 1 == len(row) or row[index + 1] != visual_id) and (
                        index + 1 == len(row) or row[index + 1] != c.vision_end_token_id
                    ):
                        raise ValueError("Cosmos3 visual spans need an explicit vision_end token")

            visual = self.visual(
                pixels.to(device=embeddings.device, dtype=next(self.visual.parameters()).dtype),
                grid,
            )
            mask = modalities == kind
            if int(mask.sum()) != len(visual.pooler_output):
                raise ValueError("Cosmos3 visual features and placeholder counts differ")
            embeddings = embeddings.masked_scatter(
                mask[..., None], visual.pooler_output.to(embeddings)
            )
            additions = tuple(
                torch.zeros_like(embeddings).masked_scatter(mask[..., None], value.to(embeddings))
                for value in visual.deepstack_features
            )
            positions, delta = multimodal_positions(
                input_ids,
                modalities,
                c.vision_config.spatial_merge_size,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
            )
        else:
            if image_grid_thw is not None or video_grid_thw is not None or modalities.any():
                raise ValueError("Cosmos3 visual placeholders/grids require actual prefill pixels")

            positions = (
                (attention_mask.long().cumsum(-1) - 1).masked_fill(~attention_mask, 0)[:, seen:]
                + delta
            )[None].expand(3, -1, -1)
        output = self.transformer(
            inputs_embeds=embeddings,
            understanding_positions=positions,
            understanding_additions=additions,
            attention_mask=attention_mask,
            vision=vision,
            sound=sound,
            action=action,
            state=token_state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        if use_cache:
            output.text.state = Cosmos3VLMState(output.text.state, delta.clone(), self.model_key)
        return output

    def forward_text(self, *args, **kwargs):
        if any(kwargs.get(name) is not None for name in ("vision", "sound", "action")):
            raise ValueError("Cosmos3 forward_text is the understanding role only")
        return self.forward(*args, **kwargs).text
