"""DeepSeek-OCR2 visual reading queries and non-MLA language experts."""

from dataclasses import dataclass, field
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import LossTerm, StateCapabilities
from aster.nn.parameter_codec import register_parameter_codec
from .config import LlamaConfig
from .decoder import CausalLM, DecoderLayer, GatedMLP
from .ocr2_vision import OCR2VisualConfig, OCR2VisualEncoder
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class OCR2TextConfig(LlamaConfig):
    architecture: ClassVar[str] = "ocr2_text"
    num_key_value_heads: int = 4
    n_routed_experts: int = 4
    n_shared_experts: int = 2
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 16
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if (
            min(
                self.n_routed_experts,
                self.n_shared_experts,
                self.moe_intermediate_size,
                self.moe_layer_freq,
            )
            < 1
            or not 0 <= self.first_k_dense_replace < self.num_hidden_layers
        ):
            raise ValueError("Invalid OCR2 shared/routed expert schedule")
        if (
            not 1 <= self.num_experts_per_tok <= self.n_routed_experts
            or not math.isfinite(self.routed_scaling_factor)
            or self.routed_scaling_factor <= 0
        ):
            raise ValueError("Invalid OCR2 expert top-k/scale")


class OCR2Router(nn.Linear):
    def forward(self, hidden):

        return F.linear(hidden.float(), self.weight.float())


class OCR2MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.gate = OCR2Router(c.hidden_size, c.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            GatedMLP(c.hidden_size, c.moe_intermediate_size) for _ in range(c.n_routed_experts)
        )
        self.shared_experts = GatedMLP(c.hidden_size, c.moe_intermediate_size * c.n_shared_experts)

    def forward(self, hidden, padding=None):
        c = self.config
        flat = hidden.flatten(0, 1)
        logits = self.gate(flat)
        probabilities = logits.softmax(-1)
        weights, indices = probabilities.topk(c.num_experts_per_tok, -1, sorted=False)
        if c.num_experts_per_tok > 1 and c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * c.routed_scaling_factor
        result = flat.new_zeros(flat.shape, dtype=torch.float32)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)

            result.index_add_(0, rows, expert(flat[rows]).float() * weights[rows, slots, None])
        result = result.to(hidden.dtype).reshape_as(hidden) + self.shared_experts(hidden)
        valid = (
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if padding is None
            else padding[:, -hidden.shape[1] :].bool()
        )
        p = probabilities.reshape(*hidden.shape[:2], c.n_routed_experts)
        choices = (
            F.one_hot(indices, c.n_routed_experts)
            .float()
            .reshape(*hidden.shape[:2], c.num_experts_per_tok, c.n_routed_experts)
        )
        counts = valid.sum(-1)
        frequency = (choices * valid[..., None, None]).sum((1, 2)) / (
            counts[:, None].clamp_min(1) * c.num_experts_per_tok
        )
        mean = (p * valid[..., None]).sum(1) / counts[:, None].clamp_min(1)
        auxiliary = (frequency * mean).sum(-1) * c.n_routed_experts
        loss = LossTerm(
            auxiliary.masked_select(counts > 0).sum(),
            (counts > 0).sum(dtype=torch.int64),
            "sequence",
            "router_aux",
        )
        return result, {
            "logits": logits,
            "weights": weights,
            "indices": indices,
            "router_aux": loss,
        }


class OCR2DecoderLayer(DecoderLayer):
    def __init__(self, c, index):
        super().__init__(c, index)
        self.sparse = index >= c.first_k_dense_replace and index % c.moe_layer_freq == 0
        if self.sparse:
            self.mlp = OCR2MoE(c)

    def forward(self, hidden, positions, padding, previous, seen_tokens, use_cache):
        value, present = self.self_attn(
            self.input_layernorm(hidden),
            positions,
            padding,
            previous,
            seen_tokens=seen_tokens,
            use_cache=use_cache,
        )
        hidden = hidden + value
        normal = self.post_attention_layernorm(hidden)
        value, extra = self.mlp(normal, padding) if self.sparse else (self.mlp(normal), None)
        return hidden + value, present, extra


class OCR2ForCausalLM(CausalLM):
    layer_type = OCR2DecoderLayer

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        records = (output.auxiliary or {}).get("router", ())
        if records:
            terms = [x["router_aux"] for x in records]

            output.auxiliary["router_aux"] = LossTerm(
                sum(x.numerator for x in terms), terms[0].denominator, "sequence", "router_aux"
            )
        return output


@dataclass(frozen=True)
class OCR2Config:
    architecture: ClassVar[str] = "deepseek_ocr2"
    text_config: OCR2TextConfig = field(default_factory=OCR2TextConfig)
    vision_config: OCR2VisualConfig = field(default_factory=OCR2VisualConfig)
    image_token_id: int = 28
    freeze_vision: bool = False
    freeze_projector: bool = False

    def __post_init__(self):
        if (
            type(self.text_config) is not OCR2TextConfig
            or type(self.vision_config) is not OCR2VisualConfig
        ):
            raise ValueError(
                "OCR2 needs its actual non-MLA language and SAM/Qwen visual configurations"
            )
        if not 0 <= self.image_token_id < self.text_config.vocab_size:
            raise ValueError("Invalid document image placeholder ID")

    def to_dict(self):
        return dict(
            architecture=self.architecture,
            text_config=self.text_config.to_dict(),
            vision_config=self.vision_config.to_dict(),
            image_token_id=self.image_token_id,
            freeze_vision=self.freeze_vision,
            freeze_projector=self.freeze_projector,
        )


class OCR2Separator(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width) / math.sqrt(width))

    def forward(self, features):

        if isinstance(features, tuple):
            return torch.cat(
                tuple(torch.cat((row, self.weight[None].to(row.dtype)), 0) for row in features), 0
            )
        return torch.cat((features, self.weight[None].to(features.dtype)), 0)


@dataclass(frozen=True)
class OCR2State:
    language_state: object
    model_key: str
    kind: str = "ocr2_multimodal"

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
        raise ValueError("OCR visual spans need snapshot+replay for safe rollback")


class OCR2ForConditionalGeneration(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.language_model = OCR2ForCausalLM(config.text_config)
        self.vision_encoder = OCR2VisualEncoder(config.vision_config)
        self.projector = nn.Module()
        self.projector.layers = nn.Linear(
            config.vision_config.decoder_config.hidden_size, config.text_config.hidden_size
        )
        self.separator = OCR2Separator(config.text_config.hidden_size)
        register_parameter_codec(self, {"separator.weight": "view_seperator"})
        if config.freeze_vision:
            self.vision_encoder.requires_grad_(False)
        if config.freeze_projector:
            self.projector.requires_grad_(False)
            self.separator.requires_grad_(False)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @staticmethod
    def official_weight_name(name):
        for old, new in (
            ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."),
            ("vision_encoder.", "model."),
            ("projector.", "model.projector."),
        ):
            if name.startswith(old):
                return new + name[len(old) :]
        if name == "view_seperator":
            return "model.view_seperator"
        raise ValueError("Unknown OCR2 composition checkpoint path")

    def encode_document_views(
        self, pixel_values, pixel_values_local=None, images_spatial_crop=None
    ):
        c = self.config.vision_config
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (
            c.sam_config.in_channels,
            c.sam_config.image_size,
            c.sam_config.image_size,
        ):
            raise ValueError("Global document views must have configured BCHW size")
        batch = len(pixel_values)
        if pixel_values_local is None:
            pixel_values_local = (None,) * batch
        if len(pixel_values_local) != batch:
            raise ValueError("Local crop lists must align one-to-one with document rows")
        if images_spatial_crop is not None and (
            images_spatial_crop.shape != (batch, 2)
            or images_spatial_crop.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Local crop geometry is integer [documents,columns/rows]")
        if batch < 1:
            raise ValueError("Document batch must be nonempty")
        counts, crops = [], []

        for row, local in enumerate(pixel_values_local):
            count = 0 if local is None else len(local)
            counts.append(count)
            if images_spatial_crop is not None:
                geometry = tuple(images_spatial_crop[row].tolist())
                if (
                    min(geometry) < 0
                    or (count == 0 and geometry not in ((0, 0), (1, 1)))
                    or (count > 0 and geometry[0] * geometry[1] != count)
                ):
                    raise ValueError("Local crop count differs from declared geometry")
            if count:
                if local.ndim != 4 or local.shape[1:] != (
                    c.sam_config.in_channels,
                    c.local_image_size,
                    c.local_image_size,
                ):
                    raise ValueError("Local document crops must have configured PCHW size")
                crops.append(local)

        local_pixels = (
            torch.cat(crops, 0)
            if crops
            else pixel_values.new_empty(
                (0, c.sam_config.in_channels, c.local_image_size, c.local_image_size)
            )
        )
        local_features = self.vision_encoder(local_pixels).flatten(0, 1)
        global_features = self.vision_encoder(pixel_values).flatten(0, 1)
        projected = self.projector.layers(torch.cat((local_features, global_features), 0))
        local_count = len(local_features)
        all_local, all_global = projected[:local_count], projected[local_count:]
        documents, start = [], 0
        for row, count in enumerate(counts):
            end = start + count * c.local_queries

            documents.append(
                torch.cat(
                    (
                        all_local[start:end],
                        all_global[row * c.global_queries : (row + 1) * c.global_queries],
                    ),
                    0,
                )
            )
            start = end
        return self.separator(tuple(documents)).split(
            tuple(len(row) + 1 for row in documents), dim=0
        )

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_local=None,
        images_spatial_crop=None,
        images_seq_mask=None,
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
        if embedded.ndim != 3:
            raise ValueError("OCR language embeddings must be BSH")
        batch, length = embedded.shape[:2]
        previous = None
        if state is not None:
            if (
                not isinstance(state, OCR2State)
                or state.kind != "ocr2_multimodal"
                or state.model_key != self.model_key
            ):
                raise ValueError("OCR state/config mismatch")
            previous = state.language_state
            if any(
                x is not None
                for x in (pixel_values, pixel_values_local, images_spatial_crop, images_seq_mask)
            ):
                raise ValueError("Visual input requires a new OCR prefill")
        if pixel_values is not None:
            if input_ids is None or previous is not None or len(pixel_values) != batch:
                raise ValueError("OCR visual prefill requires aligned token IDs and fresh state")
            selected = (
                input_ids == self.config.image_token_id
                if images_seq_mask is None
                else images_seq_mask
            )
            if (
                selected.shape != (batch, length)
                or selected.dtype != torch.bool
                or not torch.equal(selected, input_ids == self.config.image_token_id)
            ):
                raise ValueError("OCR image mask must match explicit visual placeholder IDs")
            if attention_mask is not None and (
                attention_mask.shape != selected.shape or (selected & ~attention_mask.bool()).any()
            ):
                raise ValueError("Visual placeholders cannot be padding")
            documents = self.encode_document_views(
                pixel_values, pixel_values_local, images_spatial_crop
            )
            rows = []
            for row, features in enumerate(documents):
                if int(selected[row].sum()) != len(features):
                    raise ValueError("Per-document visual feature/placeholder count mismatch")
                rows.append(
                    embedded[row].masked_scatter(selected[row, :, None], features.to(embedded))
                )
            embedded = torch.stack(rows)
        elif (
            any(x is not None for x in (pixel_values_local, images_spatial_crop, images_seq_mask))
            or input_ids is not None
            and (input_ids == self.config.image_token_id).any()
        ):
            raise ValueError("Document visual placeholders/crops need a global image")
        result = self.language_model(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=previous,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        if use_cache:
            result.state = OCR2State(result.state, self.model_key)
        return result
