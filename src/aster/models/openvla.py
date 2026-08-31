"""Native OpenVLA/Prismatic fused vision and language action-token prediction."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.data.actions import ActionSpec, ActionNormalizer, UniformActionTokenizer
from .config import LlamaConfig
from .decoder import CausalLM
from .dinov2 import DinoVisionConfig, DinoVisionModel
from .siglip import SigLIPVisionConfig, SigLIPVisionModel
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class OpenVLAConfig:
    architecture: ClassVar[str] = "openvla"
    text_config: LlamaConfig = field(default_factory=lambda: LlamaConfig(vocab_size=128))
    dino_config: DinoVisionConfig = field(default_factory=DinoVisionConfig)
    siglip_config: SigLIPVisionConfig = field(
        default_factory=lambda: SigLIPVisionConfig(
            image_size=8,
            patch_size=2,
            num_hidden_layers=3,
            hidden_act="gelu",
            vision_use_head=False,
        )
    )
    pad_to_multiple_of: int = 64
    n_action_bins: int = 16
    empty_token_id: int = 2
    action_spec: ActionSpec | None = None
    norm_stats: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            type(self.text_config) is not LlamaConfig
            or not isinstance(self.dino_config, DinoVisionConfig)
            or not isinstance(self.siglip_config, SigLIPVisionConfig)
        ):
            raise ValueError(
                "OpenVLA supports explicit Llama + DINO-register + SigLIP configurations"
            )
        d, s = self.dino_config, self.siglip_config
        if (
            d.image_size != s.image_size
            or d.patch_size != s.patch_size
            or min(d.num_hidden_layers, s.num_hidden_layers) < 2
        ):
            raise ValueError(
                "Fused visual features require identical patch grids and a penultimate block"
            )
        if s.hidden_act != "gelu" or s.vision_use_head or d.num_register_tokens != 4:
            raise ValueError(
                "Original OpenVLA uses timm GELU SigLIP patches and DINOv2 reg4; no contrastive pooling head"
            )
        if (
            self.pad_to_multiple_of < 0
            or self.action_vocab_size <= self.n_action_bins + 1
            or self.n_action_bins < 2
            or not 0 <= self.empty_token_id < self.action_vocab_size
        ):
            raise ValueError(
                "Action vocabulary excludes padded LM rows and needs explicit bins/empty-token ID"
            )
        if self.norm_stats and self.action_spec is None:
            raise ValueError("Physical action statistics require an ActionSpec")
        for key, value in self.norm_stats.items():
            if not isinstance(key, str) or not key or set(value) != {"action"}:
                raise ValueError("Stats schema is {dataset:{action:{q01,q99,mask?}}}")
            stats = value["action"]
            if set(stats) - {"q01", "q99", "mask"} or not {"q01", "q99"} <= set(stats):
                raise ValueError("Unknown/missing action statistics")
            width = len(self.action_spec.names)
            low, high = torch.as_tensor(stats["q01"]), torch.as_tensor(stats["q99"])
            mask = torch.as_tensor(stats.get("mask", [True] * width))
            if (
                low.shape != (width,)
                or high.shape != (width,)
                or mask.shape != (width,)
                or mask.dtype != torch.bool
                or not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or ((high <= low) & mask).any()
            ):
                raise ValueError("Action quantiles/mask must match physical ActionSpec dimensions")

    @property
    def action_vocab_size(self):
        return self.text_config.vocab_size - self.pad_to_multiple_of

    @property
    def visual_tokens(self):
        return (self.dino_config.image_size // self.dino_config.patch_size) ** 2

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "dino_config": self.dino_config.to_dict(),
            "siglip_config": self.siglip_config.to_dict(),
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "n_action_bins": self.n_action_bins,
            "empty_token_id": self.empty_token_id,
            "action_spec": None if self.action_spec is None else self.action_spec.to_dict(),
            "norm_stats": self.norm_stats,
        }


@dataclass(frozen=True)
class DecodedActions:
    actions: torch.Tensor
    normalized_actions: torch.Tensor
    spec: ActionSpec
    statistics_key: str


def normalize_openvla_pixels(rgb):

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB BCHW")
    values = rgb.float() / 255 if rgb.dtype == torch.uint8 else rgb
    if (
        not values.is_floating_point()
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("RGB floating inputs must be finite [0,1]")
    mean = values.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = values.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return torch.cat(((values - mean) / std, (values - 0.5) / 0.5), 1)


class FusedVision(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.featurizer = DinoVisionModel(c.dino_config)
        self.fused_featurizer = SigLIPVisionModel(c.siglip_config)

    def forward(self, pixels):
        size = self.config.dino_config.image_size
        if pixels.ndim != 4 or pixels.shape[1:] != (6, size, size):
            raise ValueError("OpenVLA needs DINO/SigLIP normalized channel-stacked pixels")
        dino = self.featurizer.patch_features(pixels[:, :3], layer=-2)
        siglip = self.fused_featurizer.embeddings(pixels[:, 3:])
        for block in self.fused_featurizer.encoder.layers[:-1]:
            siglip = block(siglip)
        return torch.cat((dino, siglip), -1)


class PrismaticProjector(nn.Module):
    def __init__(self, vision_width, language_width):
        super().__init__()
        self.fc1 = nn.Linear(vision_width, 4 * vision_width)
        self.fc2 = nn.Linear(4 * vision_width, language_width)
        self.fc3 = nn.Linear(language_width, language_width)

    def forward(self, x):
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


class OpenVLAForActionPrediction(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.vision_backbone = FusedVision(config)
        self.projector = PrismaticProjector(
            config.dino_config.hidden_size + config.siglip_config.hidden_size,
            config.text_config.hidden_size,
        )
        self.language_model = CausalLM(config.text_config)

        self.language_model.model_key = self.model_key
        self.action_tokenizer = UniformActionTokenizer(
            config.action_vocab_size, bins=config.n_action_bins
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def align_labels(self, labels):
        if labels.ndim != 2 or labels.shape[1] < 1:
            raise ValueError("Expected text labels [B,S]")
        visual = labels.new_full((len(labels), self.config.visual_tokens), -100)
        return torch.cat((labels[:, :1], visual, labels[:, 1:]), 1)

    def prepare_action_prompt(self, input_ids):
        if input_ids.ndim != 2 or not input_ids.shape[1]:
            raise ValueError("Expected nonempty tokenized action prompt")
        ends = input_ids[:, -1] == self.config.empty_token_id
        if ends.all():
            return input_ids
        if ends.any():
            raise ValueError(
                "Mixed prompt endings need explicit collation before adding empty token"
            )
        return torch.cat(
            (input_ids, input_ids.new_full((len(input_ids), 1), self.config.empty_token_id)), 1
        )

    def _normalizer(self, key):
        keys = self.config.norm_stats
        if key is None:
            if len(keys) != 1:
                raise ValueError("Choose an explicit dataset statistics key")
            key = next(iter(keys))
        if key not in keys:
            raise ValueError("Unknown action statistics key")
        stats = keys[key]["action"]
        low, high = torch.tensor(stats["q01"]), torch.tensor(stats["q99"])
        mask = torch.tensor(stats.get("mask", [True] * len(low)), dtype=torch.bool)
        center = torch.where(mask, (low + high) / 2, 0)
        scale = torch.where(mask, (high - low) / 2, 1)
        return key, ActionNormalizer(
            center, scale, spec=self.config.action_spec, mode="quantile", clip=True
        )

    def action_tokens(self, actions, statistics_key=None):
        _, normalizer = self._normalizer(statistics_key)
        if actions.shape[-1] != len(normalizer.spec.names):
            raise ValueError("Action width differs from ActionSpec")
        return self.action_tokenizer.encode(normalizer.normalize(actions))

    def decode_actions(self, ids, statistics_key=None, *, strict=True):
        key, normalizer = self._normalizer(statistics_key)
        if ids.ndim < 1 or ids.shape[-1] != len(normalizer.spec.names):
            raise ValueError("Action token count differs from ActionSpec")
        if strict:
            normalized = self.action_tokenizer.decode(ids)
        else:
            if ids.dtype != torch.long:
                raise ValueError("Action IDs must be integer tensors")
            index = (self.config.action_vocab_size - ids - 1).clamp(
                0, self.config.n_action_bins - 2
            )
            normalized = self.action_tokenizer.centers.to(ids.device)[index].float()
        return DecodedActions(normalizer.denormalize(normalized), normalized, normalizer.spec, key)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if position_ids is not None:
            raise ValueError(
                "Prismatic inserts visual positions: pass position_ids=None and use its physical cache cursor"
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one token/embedding representation")
        hidden = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[1] < 1:
            raise ValueError("Invalid Prismatic language input")
        projected = None
        if pixel_values is not None:
            if state is not None:
                raise ValueError("New visual context requires a fresh prefill")
            if len(pixel_values) != len(hidden):
                raise ValueError("Image/text batch mismatch")
            projected = self.projector(self.vision_backbone(pixel_values))
            hidden = torch.cat((hidden[:, :1], projected.to(hidden), hidden[:, 1:]), 1)
            if attention_mask is not None:
                expected = hidden.shape[1] - projected.shape[1]
                if attention_mask.shape != (len(hidden), expected):
                    raise ValueError("Prefill mask must align with original text")
                attention_mask = torch.cat(
                    (
                        attention_mask[:, :1],
                        attention_mask.new_ones(len(hidden), projected.shape[1]),
                        attention_mask[:, 1:],
                    ),
                    1,
                )

        output = self.language_model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        output.auxiliary = {
            "projector_features": projected,
            "visual_tokens": 0 if projected is None else projected.shape[1],
        }
        return output


def convert_prismatic_state_dict(state_dict, model):

    mapped, ignored = {}, []
    prefix = "vision_backbone.fused_featurizer."
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            mapped[key] = tensor
            continue
        tail = key[len(prefix) :]
        if tail.startswith("attn_pool."):
            ignored.append(key)
            continue
        if tail == "pos_embed":
            mapped[prefix + "embeddings.position_embedding.weight"] = tensor.squeeze(0)
        elif tail.startswith("patch_embed.proj."):
            mapped[prefix + "embeddings.patch_embedding." + tail.split(".")[-1]] = tensor
        elif tail.startswith("norm."):
            mapped[prefix + "post_layernorm." + tail.split(".")[-1]] = tensor
        elif tail.startswith("blocks."):
            _, layer, *parts = tail.split(".")
            name = ".".join(parts)
            base = prefix + f"encoder.layers.{layer}."
            if name.startswith("attn.qkv."):
                for label, chunk in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, 0)):
                    mapped[base + "self_attn." + label + "." + parts[-1]] = chunk
            else:
                for before, after in (
                    ("norm1.", "layer_norm1."),
                    ("norm2.", "layer_norm2."),
                    ("attn.proj.", "self_attn.out_proj."),
                ):
                    if name.startswith(before):
                        name = after + name[len(before) :]
                        break
                mapped[base + name] = tensor
        else:
            raise ValueError(f"Unsupported Prismatic vision weight: {key}")
    expected = model.state_dict()
    if set(mapped) != set(expected):
        raise ValueError(
            f"Prismatic weight coverage mismatch: missing={set(expected) - set(mapped)}, unexpected={set(mapped) - set(expected)}"
        )
    for key in expected:
        if mapped[key].shape != expected[key].shape:
            raise ValueError(f"Prismatic weight shape mismatch: {key}")
    return mapped, tuple(ignored)
