"""BLIP-2 vision, Q-Former compression, and causal or encoder-decoder language conditioning."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn import LayerNorm
from aster.nn.attention import scaled_attention
from .bert import BertResidual
from .vision import VisionOutput, CLIPMLP
from .config import T5Config, LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig
from .serialization import LocalModelMixin, configuration_key


def _validate_transformer(c, dropout, eps):
    if (
        any(
            type(v) is not int or v < 1
            for v in (
                c.hidden_size,
                c.intermediate_size,
                c.num_hidden_layers,
                c.num_attention_heads,
            )
        )
        or c.hidden_size % c.num_attention_heads
    ):
        raise ValueError("Invalid BLIP-2 Transformer dimensions")
    if (
        any(not math.isfinite(v) or not 0 <= v < 1 for v in dropout)
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("Invalid BLIP-2 normalization/dropout")
    if (
        c.hidden_act not in {"gelu", "gelu_pytorch_tanh"}
        or not math.isfinite(c.initializer_range)
        or c.initializer_range <= 0
    ):
        raise ValueError("Unsupported BLIP-2 activation/initialization")


@dataclass(frozen=True)
class Blip2QFormerConfig:
    architecture: ClassVar[str] = "blip2_qformer"
    hidden_size: int = 24
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    intermediate_size: int = 48
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    cross_attention_frequency: int = 2
    encoder_hidden_size: int = 32
    use_qformer_text_input: bool = False

    def __post_init__(self):
        _validate_transformer(
            self, (self.hidden_dropout_prob, self.attention_probs_dropout_prob), self.layer_norm_eps
        )
        if any(
            type(v) is not int or v < 1
            for v in (self.cross_attention_frequency, self.encoder_hidden_size)
        ):
            raise ValueError("Q-Former cross-attention frequency/source width must be positive")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class QFormerAttention(nn.Module):
    def __init__(self, c, cross=False):
        super().__init__()
        self.config = c
        self.attention = nn.Module()
        self.attention.query = nn.Linear(c.hidden_size, c.hidden_size)
        source = c.encoder_hidden_size if cross else c.hidden_size
        self.attention.key, self.attention.value = (
            nn.Linear(source, c.hidden_size),
            nn.Linear(source, c.hidden_size),
        )
        self.output = BertResidual(c, c.hidden_size)

    def forward(self, x, mask, source=None):
        source = x if source is None else source
        c = self.config

        def split(tensor):
            return tensor.view(
                *tensor.shape[:2], c.num_attention_heads, c.hidden_size // c.num_attention_heads
            ).transpose(1, 2)

        q, k, v = (
            split(self.attention.query(x)),
            split(self.attention.key(source)),
            split(self.attention.value(source)),
        )
        value = scaled_attention(
            q,
            k,
            v,
            mask,
            dropout=c.attention_probs_dropout_prob,
            training=self.training,
            softmax_in_fp32=False,
        )
        return self.output(value.transpose(1, 2).reshape_as(x), x)


class QFormerIntermediate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.intermediate_size)
        self.act = c.hidden_act

    def forward(self, x):
        return F.gelu(
            self.dense(x), approximate="tanh" if self.act == "gelu_pytorch_tanh" else "none"
        )


class QFormerLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.attention = QFormerAttention(c)
        if index % c.cross_attention_frequency == 0:
            self.crossattention = QFormerAttention(c, cross=True)
        self.intermediate_query, self.output_query = (
            QFormerIntermediate(c),
            BertResidual(c, c.intermediate_size),
        )
        if c.use_qformer_text_input:
            self.intermediate, self.output = (
                QFormerIntermediate(c),
                BertResidual(c, c.intermediate_size),
            )

    def forward(self, x, mask, source, source_mask, queries):
        x = self.attention(x, mask)
        query, text = x[:, :queries], x[:, queries:]
        if queries:
            if hasattr(self, "crossattention"):
                if source is None:
                    raise ValueError("Query cross-attention requires visual encoder states")
                query = self.crossattention(query, source_mask, source)
            query = self.output_query(self.intermediate_query(query), query)
        if text.shape[1]:
            text = self.output(self.intermediate(text), text)
        return torch.cat((query, text), 1)


def _visible(mask, batch, queries, keys, device):
    if mask is None:
        return torch.ones(batch, 1, queries, keys, device=device, dtype=torch.bool)
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Q-Former masks must be binary valid-entry masks")
    if mask.shape == (batch, keys):
        return mask[:, None, None].bool().expand(batch, 1, queries, keys)
    if mask.shape == (batch, 1, queries, keys):
        return mask.bool()
    raise ValueError("Q-Former mask must be [B,K] padding or explicit [B,1,Q,K] visibility")


class Blip2QFormerModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layernorm, self.dropout = (
            LayerNorm(config.hidden_size, config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            QFormerLayer(config, i) for i in range(config.num_hidden_layers)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query_embeds,
        *,
        query_length=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_hidden_states=False,
        state=None,
        use_cache=False,
    ):
        if state is not None or use_cache:
            raise ValueError("Q-Former bidirectional queries do not have append-only KV state")
        if (
            query_embeds.ndim != 3
            or query_embeds.shape[-1] != self.config.hidden_size
            or not query_embeds.shape[1]
        ):
            raise ValueError("Q-Former query/text embeddings must be nonempty [B,S,H]")
        b, s = query_embeds.shape[:2]
        length = s if query_length is None else query_length
        if (
            type(length) is not int
            or not 0 <= length <= s
            or (length < s and not self.config.use_qformer_text_input)
        ):
            raise ValueError(
                "Text suffix requires its independent configured FFN; invalid query length"
            )
        x = self.dropout(self.layernorm(query_embeds.to(self.layernorm.weight.dtype)))
        source, cross_mask = encoder_hidden_states, None
        if source is not None:
            if (
                source.ndim != 3
                or source.shape[0] != b
                or source.shape[-1] != self.config.encoder_hidden_size
                or not source.shape[1]
            ):
                raise ValueError("Q-Former visual states differ from configured batch/source width")
            source = source.to(x.dtype)
            cross_mask = _visible(encoder_attention_mask, b, length, source.shape[1], x.device)
        elif encoder_attention_mask is not None:
            raise ValueError("Encoder mask without encoder states")
        mask = _visible(attention_mask, b, s, s, x.device)
        history = []
        for layer in self.encoder.layer:
            if output_hidden_states:
                history.append(x)
            x = layer(x, mask, source, cross_mask, length)
        if output_hidden_states:
            history.append(x)
        return VisionOutput(x, x[:, 0], tuple(history) if output_hidden_states else None)


@dataclass(frozen=True)
class Blip2VisionConfig:
    architecture: ClassVar[str] = "blip2_vision"
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    image_size: int = 8
    patch_size: int = 2
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    qkv_bias: bool = True

    def __post_init__(self):
        _validate_transformer(self, (self.attention_dropout,), self.layer_norm_eps)
        if (
            any(type(v) is not int or v < 1 for v in (self.image_size, self.patch_size))
            or self.image_size % self.patch_size
        ):
            raise ValueError("BLIP-2 image grid must be divisible by patch size")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.class_embedding = nn.Parameter(torch.empty(1, 1, c.hidden_size))
        self.position_embedding = nn.Parameter(
            torch.empty(1, (c.image_size // c.patch_size) ** 2 + 1, c.hidden_size)
        )
        self.patch_embedding = nn.Conv2d(3, c.hidden_size, c.patch_size, c.patch_size)
        nn.init.trunc_normal_(self.class_embedding, std=c.initializer_range)
        nn.init.trunc_normal_(self.position_embedding, std=c.initializer_range)

    def forward(self, pixels, interpolate):
        c = self.config
        if (
            pixels.ndim != 4
            or pixels.shape[1] != 3
            or not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
        ):
            raise ValueError("BLIP-2 requires finite normalized RGB BCHW")
        h, w = pixels.shape[-2:]
        if not h or not w or h % c.patch_size or w % c.patch_size:
            raise ValueError("Invalid BLIP-2 patch grid")
        if (h, w) != (c.image_size, c.image_size) and not interpolate:
            raise ValueError("Non-native image grid needs explicit position interpolation")
        patches = (
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        x = torch.cat(
            (self.class_embedding.expand(pixels.shape[0], -1, -1).to(patches), patches), 1
        )
        positions = self.position_embedding
        if (h, w) != (c.image_size, c.image_size):
            side = c.image_size // c.patch_size
            patch_positions = (
                positions[:, 1:].reshape(1, side, side, c.hidden_size).permute(0, 3, 1, 2)
            )
            patch_positions = F.interpolate(
                patch_positions,
                size=(h // c.patch_size, w // c.patch_size),
                mode="bicubic",
                align_corners=False,
            )
            positions = torch.cat((positions[:, :1], patch_positions.flatten(2).transpose(1, 2)), 1)
        return x + positions.to(x)


class Blip2VisionAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c

        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=c.qkv_bias)
        self.projection = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x):
        c, b, s = self.config, x.shape[0], x.shape[1]
        q, k, v = (
            self.qkv(x)
            .reshape(b, s, 3, c.num_attention_heads, c.hidden_size // c.num_attention_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        visible = torch.ones(b, 1, s, s, dtype=torch.bool, device=x.device)
        out = scaled_attention(
            q,
            k,
            v,
            visible,
            dropout=c.attention_dropout,
            training=self.training,
            softmax_in_fp32=False,
        )
        return self.projection(out.transpose(1, 2).reshape_as(x))


class Blip2VisionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Blip2VisionAttention(c), CLIPMLP(c)
        self.layer_norm1, self.layer_norm2 = (
            LayerNorm(c.hidden_size, c.layer_norm_eps),
            LayerNorm(c.hidden_size, c.layer_norm_eps),
        )

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Blip2VisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            Blip2VisionLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values, *, interpolate_pos_encoding=False, output_hidden_states=False):
        x = self.embeddings(pixel_values, interpolate_pos_encoding)
        history = []
        for layer in self.encoder.layers:
            if output_hidden_states:
                history.append(x)
            x = layer(x)
        if output_hidden_states:
            history.append(x)
        x = self.post_layernorm(x)

        return VisionOutput(
            x, self.post_layernorm(x[:, 0]), tuple(history) if output_hidden_states else None
        )


@dataclass(frozen=True)
class Blip2Config:
    architecture: ClassVar[str] = "blip2"
    vision_config: Blip2VisionConfig = field(default_factory=Blip2VisionConfig)
    qformer_config: Blip2QFormerConfig = field(default_factory=Blip2QFormerConfig)
    text_config: T5Config | LlamaConfig = field(
        default_factory=lambda: T5Config(feed_forward_proj="gated-gelu")
    )
    num_query_tokens: int = 4
    image_token_id: int = 31

    def __post_init__(self):
        if not isinstance(self.vision_config, Blip2VisionConfig) or not isinstance(
            self.qformer_config, Blip2QFormerConfig
        ):
            raise ValueError("BLIP-2 requires its actual vision/Q-Former configurations")
        if type(self.text_config) not in {
            T5Config,
            LlamaConfig,
            Qwen2Config,
            Qwen3Config,
            MistralConfig,
        }:
            raise ValueError(
                "This BLIP-2 composition supports native T5 or the admitted dense causal decoders"
            )
        if self.qformer_config.encoder_hidden_size != self.vision_config.hidden_size:
            raise ValueError("Q-Former cross-attention source width differs from vision")
        if self.qformer_config.use_qformer_text_input:
            raise ValueError(
                "Conditional-generation Q-Former takes learned queries only; text suffix is a separate training use"
            )
        if (
            type(self.num_query_tokens) is not int
            or self.num_query_tokens < 1
            or type(self.image_token_id) is not int
            or not 0 <= self.image_token_id < self.text_config.vocab_size
        ):
            raise ValueError(
                "Query count must be positive and image placeholder must exist in language vocabulary"
            )

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "vision_config": self.vision_config.to_dict(),
            "qformer_config": self.qformer_config.to_dict(),
            "text_config": self.text_config.to_dict(),
            "num_query_tokens": self.num_query_tokens,
            "image_token_id": self.image_token_id,
        }


@dataclass(frozen=True)
class Blip2State:
    language_state: object
    model_key: str
    kind: str = "blip2_language_state"

    @property
    def seen_tokens(self):
        return self.language_state.seen_tokens

    @property
    def capabilities(self):
        source = self.language_state.capabilities
        return StateCapabilities(
            self.kind,
            forkable=source.forkable,
            truncatable=source.truncatable,
            reorderable=source.reorderable,
            replayable=source.replayable,
        )

    def fork(self):
        return type(self)(self.language_state.fork(), self.model_key)

    def reorder(self, indices):
        return type(self)(self.language_state.reorder(indices), self.model_key)

    def truncate(self, length):
        return type(self)(self.language_state.truncate(length), self.model_key)


@dataclass
class Blip2ImageFeatures:
    vision_features: torch.Tensor
    query_features: torch.Tensor
    projected_features: torch.Tensor


class Blip2ForConditionalGeneration(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        from . import build_model

        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)
        width = (
            config.text_config.d_model
            if isinstance(config.text_config, T5Config)
            else config.text_config.hidden_size
        )
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, width)
        nn.init.normal_(
            self.language_projection.weight, std=config.qformer_config.initializer_range
        )
        nn.init.zeros_(self.language_projection.bias)
        self.language_model = build_model(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values, *, interpolate_pos_encoding=False):
        vision = self.vision_model(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        ).last_hidden_state
        queries = self.query_tokens.expand(vision.shape[0], -1, -1)

        values = self.qformer(queries, encoder_hidden_states=vision).last_hidden_state
        projected = self.language_projection(values.to(self.language_projection.weight.dtype))
        return Blip2ImageFeatures(vision, values, projected)

    def forward(
        self,
        input_ids=None,
        *,
        pixel_values=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
        interpolate_pos_encoding=False,
    ):
        seq2seq = isinstance(self.config.text_config, T5Config)
        if state is not None and (
            not isinstance(state, Blip2State) or state.model_key != self.model_key
        ):
            raise ValueError(
                "BLIP-2 state must belong to the complete visual/language configuration"
            )
        previous = None if state is None else state.language_state
        visual = None
        if state is None:
            if (
                input_ids is None
                or input_ids.ndim != 2
                or input_ids.dtype != torch.long
                or pixel_values is None
            ):
                raise ValueError("BLIP-2 prefill requires token IDs and image pixels")
            if pixel_values.shape[0] != input_ids.shape[0]:
                raise ValueError("BLIP-2 requires one image per text sample")
            mask = input_ids == self.config.image_token_id
            if not (mask.sum(-1) == self.config.num_query_tokens).all():
                raise ValueError("Each sample needs exactly num_query_tokens image placeholders")
            visual = self.get_image_features(
                pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
            )
            inputs = self.get_input_embeddings()(input_ids)
            inputs = inputs.masked_scatter(
                mask[..., None].expand_as(inputs), visual.projected_features.to(inputs)
            )
        else:
            if pixel_values is not None:
                raise ValueError(
                    "Cached visual condition is fixed; new images require a new prefill"
                )
            inputs = None
        if seq2seq:
            if position_ids is not None:
                raise ValueError(
                    "T5 uses bucketed relative positions, not external absolute position IDs"
                )
            if decoder_input_ids is None:
                raise ValueError(
                    "BLIP-2/T5 requires decoder_input_ids; labels and shift-right belong to the objective"
                )
            if state is not None and input_ids is not None:
                raise ValueError("Cached T5 encoder input is fixed")
            output = self.language_model(
                inputs_embeds=inputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                state=previous,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
            )
        else:
            if decoder_input_ids is not None or decoder_attention_mask is not None:
                raise ValueError("Causal BLIP-2 has no separate decoder input stream")
            if (
                state is not None
                and input_ids is not None
                and (input_ids == self.config.image_token_id).any()
            ):
                raise ValueError("New visual placeholders require a new image prefill")
            output = self.language_model(
                input_ids if state is not None else None,
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                state=previous,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
            )
        wrapped = Blip2State(output.state, self.model_key) if output.state is not None else None
        return TokenOutput(
            output.logits,
            wrapped,
            output.hidden_states,
            {"vision": visual, "language_auxiliary": output.auxiliary},
        )
