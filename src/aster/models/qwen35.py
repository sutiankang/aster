"""Qwen3.5 dense/MoE and visual conditioning with Delta state and partial MRoPE."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
from aster.nn.delta import GatedDeltaNet, HybridState
from aster.nn.position import RopeConfig
from .hybrid import Qwen3NextConfig, Qwen3NextForCausalLM, HybridLayer
from .qwen_vl import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
    Qwen3VLForConditionalGeneration,
    VisionLanguageState,
    InterleavedMRope,
)
from .serialization import configuration_key


@dataclass(frozen=True)
class Qwen35TextConfig(Qwen3NextConfig):
    architecture: ClassVar[str] = "qwen3_5_text"
    num_experts: int = 0
    head_dim: int = 12
    partial_rotary_factor: float = 0.5
    mrope_section: tuple[int, int, int] = (1, 1, 1)
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(theta=10000000))

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        width = int(self.attention_head_dim * self.partial_rotary_factor)
        if (
            len(self.mrope_section) != 3
            or min(self.mrope_section) < 0
            or sum(self.mrope_section) != width // 2
        ):
            raise ValueError("Qwen3.5 MRoPE sections cover the partial rotary half-width")
        if (
            self.rope.interleaved
            or self.mlp_only_layers
            or self.decoder_sparse_step != 1
            or not self.norm_topk_prob
        ):
            raise ValueError(
                "Qwen3.5 uses explicit uniform dense/MoE schedule and split-half MRoPE"
            )
        if type(self) is Qwen35TextConfig and self.num_experts != 0:
            raise ValueError("Use Qwen35MoETextConfig for the MoE branch")


@dataclass(frozen=True)
class Qwen35MoETextConfig(Qwen35TextConfig):
    architecture: ClassVar[str] = "qwen3_5_moe_text"
    num_experts: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.num_experts < 1:
            raise ValueError("MoE architecture requires actual routed experts")


class Qwen35Layer(HybridLayer):
    def __init__(self, config, index):
        super().__init__(config, index)
        if self.kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, projection_layout="separate")
        else:
            self.self_attn.rope = InterleavedMRope(config, self.self_attn.rotary_dim)


class Qwen35ForCausalLM(Qwen3NextForCausalLM):
    layer_type = Qwen35Layer

    def validate_positions(self, positions, hidden):
        if (
            positions.shape not in (hidden.shape[:2], (3, *hidden.shape[:2]))
            or (positions < 0).any()
        ):
            raise ValueError("Qwen3.5 positions must be [B,S] or [3,B,S] T/H/W")


@dataclass(frozen=True)
class Qwen35VisionConfig(Qwen3VLVisionConfig):
    architecture: ClassVar[str] = "qwen3_5_vision"
    deepstack_visual_indexes: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.deepstack_visual_indexes:
            raise ValueError("Qwen3.5 vision does not have Qwen3-VL DeepStack heads")


@dataclass(frozen=True)
class Qwen35Config:
    architecture: ClassVar[str] = "qwen3_5"
    text_config: Qwen35TextConfig = field(default_factory=Qwen35TextConfig)
    vision_config: Qwen35VisionConfig = field(default_factory=Qwen35VisionConfig)
    image_token_id: int = 28
    video_token_id: int = 29
    vision_start_token_id: int = 26
    vision_end_token_id: int = 27

    def __post_init__(self):
        if not isinstance(self.text_config, Qwen35TextConfig) or not isinstance(
            self.vision_config, Qwen35VisionConfig
        ):
            raise ValueError(
                "Qwen3.5 requires its own hybrid text / no-DeepStack vision configurations"
            )
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("Vision/text width mismatch")
        ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if len(set(ids)) != 4 or min(ids) < 0 or max(ids) >= self.text_config.vocab_size:
            raise ValueError("Invalid multimodal special IDs")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            **{
                name: getattr(self, name)
                for name in (
                    "image_token_id",
                    "video_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                )
            },
        }


@dataclass(frozen=True)
class Qwen35VisionState(VisionLanguageState):
    kind: str = "qwen3_5_vl_hybrid"


class Qwen35ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    state_type, state_kind, vision_state_type = HybridState, "hybrid_delta", Qwen35VisionState

    def __init__(self, config):
        nn.Module.__init__(self)
        text = Qwen35ForCausalLM(config.text_config)
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.language_model, self.lm_head = text.model, text.lm_head
        self.model.visual = Qwen3VLVisionModel(config.vision_config)

    def create_state(self, layers, seen, kind):
        return HybridState(layers, seen, self.model_key, self.decoder_config.layer_types)
