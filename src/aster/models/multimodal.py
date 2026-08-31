"""Fixed-grid LLaVA visual feature selection, trainable projection, and placeholder replacement."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from .config import LlamaConfig
from .decoder import CausalLM
from .serialization import configuration_key
from .vision import CLIPVisionConfig, CLIPVisionModel


@dataclass(frozen=True)
class LlavaConfig:
    architecture: ClassVar[str] = "llava"
    text_config: LlamaConfig = field(default_factory=LlamaConfig)
    vision_config: CLIPVisionConfig = field(default_factory=CLIPVisionConfig)
    image_token_id: int = 31
    vision_feature_layer: int | tuple[int, ...] = -2
    vision_feature_select_strategy: str = "default"
    multimodal_projector_bias: bool = True

    def __post_init__(self):
        if not isinstance(self.text_config, LlamaConfig) or not isinstance(
            self.vision_config, CLIPVisionConfig
        ):
            raise TypeError("LLaVA requires explicit native text and CLIP vision configurations")
        if self.text_config.architecture not in {
            "llama",
            "qwen2",
            "qwen3",
            "mistral",
            "mixtral",
            "deepseek_v3",
        }:
            raise ValueError(
                "This LLaVA connector currently supports only dense/window/MLA decoder state"
            )
        if not 0 <= self.image_token_id < self.text_config.vocab_size:
            raise ValueError("Image placeholder must belong to the text vocabulary")
        if self.vision_feature_select_strategy not in {"default", "full"}:
            raise ValueError("Unsupported image feature selection")
        layers = (
            (self.vision_feature_layer,)
            if isinstance(self.vision_feature_layer, int)
            else self.vision_feature_layer
        )
        if not layers or any(
            not -self.vision_config.num_hidden_layers - 1
            <= x
            <= self.vision_config.num_hidden_layers
            for x in layers
        ):
            raise ValueError("Selected visual hidden layer does not exist")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "image_token_id": self.image_token_id,
            "vision_feature_layer": self.vision_feature_layer,
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "multimodal_projector_bias": self.multimodal_projector_bias,
        }


class MultiModalProjector(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, output_size, bias=bias)
        self.linear_2 = nn.Linear(output_size, output_size, bias=bias)

    def forward(self, features):
        return self.linear_2(F.gelu(self.linear_1(features)))


def replace_image_tokens(embeddings, image_features, image_mask):

    if image_mask.shape != embeddings.shape[:2] or image_mask.dtype != torch.bool:
        raise ValueError("Image placeholder mask must be boolean [batch,sequence]")
    if (
        image_features.ndim != 3
        or image_features.shape[0] != embeddings.shape[0]
        or image_features.shape[-1] != embeddings.shape[-1]
    ):
        raise ValueError("Image features must be [batch,image_tokens,text_hidden]")
    if not torch.all(image_mask.sum(-1) == image_features.shape[1]):
        raise ValueError(
            "Every sample's image placeholders must exactly match its visual feature count"
        )
    return embeddings.masked_scatter(image_mask[..., None], image_features.to(embeddings))


class LlavaForConditionalGeneration(CausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        from . import build_model

        text = build_model(config.text_config)
        self.config, self.model_key = config, configuration_key(config)
        self.state_kind = text.state_kind
        self.model = nn.Module()
        self.model.language_model = text.model
        self.lm_head = text.lm_head
        self.model.vision_tower = CLIPVisionModel(config.vision_config)
        count = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.model.multi_modal_projector = MultiModalProjector(
            count * config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.multimodal_projector_bias,
        )
        for module in self.model.multi_modal_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.text_config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_decoder(self):
        return self.model.language_model

    @property
    def decoder_config(self):
        return self.config.text_config

    def get_image_features(self, pixels):

        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.ndim != 5 or pixels.shape[1] < 1:
            raise ValueError("Pixels must be BCHW or BNCHW with a nonempty image count")
        batch, images = pixels.shape[:2]
        result = self.model.vision_tower(pixels.flatten(0, 1), output_hidden_states=True)
        indices = self.config.vision_feature_layer
        indices = (indices,) if isinstance(indices, int) else indices
        selected = [result.hidden_states[index] for index in indices]
        if self.config.vision_feature_select_strategy == "default":
            selected = [x[:, 1:] for x in selected]
        projected = self.model.multi_modal_projector(torch.cat(selected, -1))
        return projected.reshape(batch, images * projected.shape[1], projected.shape[2])

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        pixel_values=None,
        image_token_mask=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one token/embedding input")
        embeddings = (
            self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        )
        image_mask = (
            (input_ids == self.config.image_token_id) if input_ids is not None else image_token_mask
        )
        if (
            image_token_mask is not None
            and input_ids is not None
            and not torch.equal(image_mask, image_token_mask)
        ):
            raise ValueError("Explicit image mask disagrees with placeholder IDs")
        image_features = None
        if pixel_values is not None:
            if state is not None:
                raise ValueError(
                    "Image conditions belong to prefill; new image context requires a fresh state"
                )
            if image_mask is None:
                raise ValueError("Embedding-only image input requires explicit image_token_mask")
            image_features = self.get_image_features(pixel_values)
            embeddings = replace_image_tokens(embeddings, image_features, image_mask)
        elif image_mask is not None and image_mask.any():
            raise ValueError("Image placeholders require their pixel_values")
        output = super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        if image_features is not None:
            output.auxiliary = {**(output.auxiliary or {}), "image_features": image_features}
        return output
