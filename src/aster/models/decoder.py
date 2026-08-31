"""Shared dense-decoder execution with explicit Llama, Qwen, and Mistral layer differences."""

import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn import RMSNorm, GroupedQueryAttention, KVState
from .config import LlamaConfig, Qwen2Config, Qwen3Config
from .serialization import LocalModelMixin, configuration_key


class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_head_dim,
            config.rope,
            qkv_bias=isinstance(config, Qwen2Config),
            qk_norm=isinstance(config, Qwen3Config),
            eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            window=config.window_for_layer(index),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

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
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), present, None


class DecoderBackbone(nn.Module):
    def __init__(self, config, layer_type=DecoderLayer):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            layer_type(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class CausalLM(LocalModelMixin, nn.Module):
    layer_type = DecoderLayer
    state_kind = "dense_kv"
    state_type = KVState

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.model = DecoderBackbone(config, self.layer_type)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.get_decoder().embed_tokens

    def get_decoder(self):
        return self.model

    @property
    def decoder_config(self):
        return self.config

    def create_state(self, layers, seen, kind):
        return KVState(layers, seen, self.model_key, kind)

    def validate_positions(self, positions, hidden):
        if positions.shape != hidden.shape[:2] or (positions < 0).any():
            raise ValueError("position_ids must align with the current token batch")

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
        layer_additions=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        backbone, config = self.get_decoder(), self.decoder_config
        hidden = backbone.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != config.hidden_size or hidden.shape[1] < 1:
            raise ValueError("Expected nonempty [batch,tokens,hidden] embeddings")
        kind = (
            "window_kv"
            if any(config.window_for_layer(i) for i in range(config.num_hidden_layers))
            else self.state_kind
        )
        seen = 0
        if state is not None:
            if (
                not isinstance(state, self.state_type)
                or state.model_key != self.model_key
                or state.kind != kind
                or len(state.layers) != len(backbone.layers)
            ):
                raise ValueError("State codec/model configuration mismatch")
            seen = state.seen_tokens
            if seen < 0:
                raise ValueError("Negative state position")
        if seen + hidden.shape[1] > config.max_position_embeddings:
            raise ValueError("Sequence exceeds the model's declared position support")
        if position_ids is None:
            position_ids = torch.arange(seen, seen + hidden.shape[1], device=hidden.device)[
                None
            ].expand(hidden.shape[0], -1)
        self.validate_positions(position_ids, hidden)
        if layer_additions is not None and (
            len(layer_additions) > len(backbone.layers)
            or any(x.shape != hidden.shape for x in layer_additions)
        ):
            raise ValueError(
                "Layer residual additions must explicitly match [batch,sequence,hidden]"
            )
        states, layers, auxiliary = [], [], []
        for index, layer in enumerate(backbone.layers):
            if output_hidden_states:
                states.append(hidden)
            hidden, present, extra = layer(
                hidden,
                position_ids,
                attention_mask,
                state.layers[index] if state is not None else None,
                seen,
                use_cache,
            )
            if layer_additions is not None and index < len(layer_additions):
                hidden = hidden + layer_additions[index].to(hidden)
            if use_cache:
                layers.append(present)
            if extra is not None:
                auxiliary.append(extra)
        hidden = backbone.norm(hidden)
        if output_hidden_states:
            states.append(hidden)
        updated = (
            self.create_state(tuple(layers), seen + hidden.shape[1], kind) if use_cache else None
        )
        return TokenOutput(
            self.lm_head(hidden),
            updated,
            tuple(states) if output_hidden_states else None,
            {"router": tuple(auxiliary)} if auxiliary else None,
        )
