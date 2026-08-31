"""Native multi-camera, language-conditioned pi0/pi0.5 action models."""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from .serialization import LocalModelMixin
from .siglip import SigLIPVisionConfig, SigLIPVisionModel
from .policies import PiConfig, PiActionExpert
from .actions import ActionOutput


@dataclass(frozen=True)
class PiVLAConfig:
    vision: SigLIPVisionConfig | dict | None = None
    expert: PiConfig | dict | None = None
    vocab_size: int = 259
    camera_names: tuple[str, ...] = ("front", "wrist")
    max_prompt_length: int = 128
    prompt_contains_state: bool = False

    def __post_init__(self):
        vision = (
            self.vision if self.vision is not None else SigLIPVisionConfig(vision_use_head=False)
        )
        expert = self.expert if self.expert is not None else PiConfig()
        if isinstance(vision, dict):
            values = dict(vision)
            architecture = values.pop("architecture", "siglip_vision")
            if architecture != "siglip_vision":
                raise ValueError("Pi vision must identify its SigLIP configuration")
            vision = SigLIPVisionConfig(**values)
        if isinstance(expert, dict):
            values = dict(expert)
            architecture = values.pop("architecture", "pi_action_expert")
            if architecture != "pi_action_expert":
                raise ValueError("Pi expert config differs")
            expert = PiConfig(**values)
        if (
            not isinstance(vision, SigLIPVisionConfig)
            or not isinstance(expert, PiConfig)
            or vision.vision_use_head
        ):
            raise ValueError("Pi needs a no-pooling SigLIP tower and its two-branch action expert")
        names = tuple(self.camera_names)
        if (
            not names
            or len(set(names)) != len(names)
            or not all(isinstance(name, str) and name for name in names)
            or min(self.vocab_size, self.max_prompt_length) < 1
        ):
            raise ValueError("Invalid camera order/vocabulary/prompt length")
        if expert.pi05 and not self.prompt_contains_state:
            raise ValueError(
                "pi0.5 omits continuous state token: declare state-bearing prompt preprocessing"
            )
        object.__setattr__(self, "vision", vision)
        object.__setattr__(self, "expert", expert)
        object.__setattr__(self, "camera_names", names)

    def to_dict(self):
        return {"architecture": "pi_vla", **asdict(self)}

    @property
    def action_dim(self):
        return self.expert.action_dim

    @property
    def action_horizon(self):
        return self.expert.action_horizon


class PiVLA(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = SigLIPVisionModel(config.vision)
        self.vision_projection = nn.Linear(config.vision.hidden_size, config.expert.prefix_width)

        nn.init.zeros_(self.vision_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)
        self.token_embedding = nn.Embedding(config.vocab_size, config.expert.prefix_width)
        self.action_expert = PiActionExpert(config.expert)

    def encode_observation(self, observation):
        images, masks = observation["images"], observation["image_masks"]
        if set(images) != set(self.config.camera_names) or set(masks) != set(images):
            raise ValueError("Camera names/masks must exactly match artifact camera order")
        tokens, valid = [], []
        batch_size = None
        self.vision.eval()
        for name in self.config.camera_names:
            pixels = images[name]
            if (
                pixels.ndim != 4
                or not pixels.is_floating_point()
                or not torch.isfinite(pixels).all()
                or (pixels.abs() > 1).any()
            ):
                raise ValueError(
                    "Pi images must be explicitly normalized floating RGB BCHW in [-1,1]"
                )
            if batch_size is None:
                batch_size = len(pixels)
            if (
                len(pixels) != batch_size
                or masks[name].shape != (batch_size,)
                or masks[name].dtype != torch.bool
            ):
                raise ValueError("Per-camera valid flags must be B booleans")
            features = self.vision(pixel_values=pixels).last_hidden_state
            tokens.append(self.vision_projection(features))
            valid.append(masks[name][:, None].expand(-1, features.shape[1]))
        prompt = observation.get("input_ids")
        if prompt is not None:
            if (
                prompt.ndim != 2
                or len(prompt) != batch_size
                or not 1 <= prompt.shape[1] <= self.config.max_prompt_length
                or prompt.dtype != torch.long
            ):
                raise ValueError("Pi tokenized prompt dimensions/dtype differ")
            padding = observation["attention_mask"]
            if padding.shape != prompt.shape or padding.dtype != torch.bool:
                raise ValueError("Prompt attention_mask must align as booleans")
            tokens.append(self.token_embedding(prompt) * math.sqrt(self.config.expert.prefix_width))
            valid.append(padding)
        if self.config.expert.pi05 and prompt is None:
            raise ValueError("pi0.5 requires state-bearing tokenized prompts")
        prefix_mask = torch.cat(valid, 1)
        if not prefix_mask.any(-1).all():
            raise ValueError("Every observation needs at least one valid image/language token")
        return {
            "prefix_embeds": torch.cat(tokens, 1),
            "prefix_mask": prefix_mask,
            "proprio": observation["proprio"],
        }

    def forward(self, sample, time, condition=None):
        if condition is None:
            raise ValueError("PiVLA needs the raw observation mapping")
        return self.action_expert(sample, time, self.encode_observation(condition))

    @torch.no_grad()
    def sample_actions(self, observation, *, noise=None, steps=10, cache_prefix=True):
        return self.action_expert.sample_actions(
            self.encode_observation(observation),
            noise=noise,
            steps=steps,
            cache_prefix=cache_prefix,
        )

    @torch.no_grad()
    def predict_chunk(self, observation, state=None):
        if state is not None:
            raise ValueError(
                "Pi observation cache is request-local; stale cross-observation state is rejected"
            )
        actions = self.sample_actions(observation)
        return ActionOutput(
            actions, torch.full(actions.shape[:2], -torch.inf, device=actions.device)
        )
