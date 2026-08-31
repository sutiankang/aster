"""Gemma4 text with per-layer embeddings, heterogeneous head widths, and shared KV ownership."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn.normalization import FloatRMSNorm
from aster.nn.attention import attention_mask, scaled_attention
from aster.nn.position import RopeConfig, RotaryEmbedding
from aster.nn.experts import PackedExperts
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Gemma4TextConfig:
    architecture: ClassVar[str] = "gemma4_text"
    vocab_size: int = 48
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    global_head_dim: int = 16
    num_global_key_value_heads: int = 1
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 128
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 4
    layer_types: tuple[str, ...] = (
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    )
    final_logit_softcapping: float | None = None
    vocab_size_per_layer_input: int = 48
    hidden_size_per_layer_input: int = 8
    attention_k_eq_v: bool = True
    num_kv_shared_layers: int = 2
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = True
    num_experts: int = 4
    top_k_experts: int = 2
    moe_intermediate_size: int = 24
    local_rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    global_rotary_fraction: float = 0.25
    global_rope_factor: float = 1.0
    use_bidirectional_attention: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.global_head_dim,
            self.num_global_key_value_heads,
            self.max_position_embeddings,
            self.sliding_window,
            self.vocab_size_per_layer_input,
            self.num_experts,
            self.top_k_experts,
            self.moe_intermediate_size,
        )
        if any(type(v) is not int or v < 1 for v in dimensions):
            raise ValueError("Invalid Gemma4 dimensions")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.num_attention_heads % self.num_global_key_value_heads
            or self.head_dim % 2
            or self.global_head_dim % 2
        ):
            raise ValueError("Gemma4 GQA/head dimensions are incompatible")
        if (
            len(self.layer_types) != self.num_hidden_layers
            or self.layer_types[-1] != "full_attention"
            or any(v not in {"sliding_attention", "full_attention"} for v in self.layer_types)
        ):
            raise ValueError(
                "Gemma4 needs an explicit local/global schedule ending in full attention"
            )
        independent = self.num_hidden_layers - self.num_kv_shared_layers
        if (
            type(self.num_kv_shared_layers) is not int
            or not 1 <= independent <= self.num_hidden_layers
        ):
            raise ValueError("KV sharing must retain at least one independent owner layer")
        if not set(self.layer_types[independent:]) <= set(self.layer_types[:independent]):
            raise ValueError("Every shared KV type needs an earlier independent source")
        if (
            type(self.hidden_size_per_layer_input) is not int
            or self.hidden_size_per_layer_input < 0
            or self.top_k_experts > self.num_experts
        ):
            raise ValueError("Invalid PLE/expert dimensions")
        if self.use_bidirectional_attention not in {None, "vision"}:
            raise ValueError(
                "Gemma4 supports causal text or vision-local bidirectionality; not all-bidirectional text"
            )
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < min(
            self.vocab_size, self.vocab_size_per_layer_input
        ):
            raise ValueError("Padding token outside vocabulary")
        if (
            self.hidden_activation not in {"silu", "gelu", "gelu_pytorch_tanh"}
            or not 0 <= self.attention_dropout < 1
        ):
            raise ValueError("Unsupported Gemma4 activation/dropout")
        if not 0 <= self.global_rotary_fraction <= 1 or any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.initializer_range,
                self.rms_norm_eps,
                self.global_rope_theta,
                self.local_rope_theta,
                self.global_rope_factor,
            )
        ):
            raise ValueError("Invalid Gemma4 numeric/rotary configuration")
        if self.final_logit_softcapping is not None and (
            not math.isfinite(self.final_logit_softcapping) or self.final_logit_softcapping <= 0
        ):
            raise ValueError("Final logit softcap must be finite positive")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}

    @property
    def independent_layers(self):
        return self.num_hidden_layers - self.num_kv_shared_layers


@dataclass(frozen=True)
class Gemma4State:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    kind: str = "gemma4_shared_kv"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple((k.clone(), v.clone()) for k, v in self.layers), self.seen_tokens, self.model_key
        )

    def reorder(self, indices):
        return type(self)(
            tuple((k.index_select(0, indices), v.index_select(0, indices)) for k, v in self.layers),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError(
            "Gemma4 local shared KV loses older windows; checkpoint+replay is required"
        )


def activation(x, name):
    return (
        F.silu(x)
        if name == "silu"
        else F.gelu(x, approximate="tanh" if name == "gelu_pytorch_tanh" else "none")
    )


class Gemma4Embedding(nn.Embedding):
    _aster_semantic_buffers = ("embed_scale",)

    def __init__(self, vocab, width, padding, scale):
        super().__init__(vocab, width, padding_idx=padding)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, tokens):
        return super().forward(tokens) * self.embed_scale.to(self.weight.dtype)


class Gemma4Attention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config, self.index = c, index
        self.layer_type = c.layer_types[index]
        self.local = self.layer_type == "sliding_attention"
        self.head_dim = c.head_dim if self.local else c.global_head_dim
        self.kv_heads = (
            c.num_global_key_value_heads
            if c.attention_k_eq_v and not self.local
            else c.num_key_value_heads
        )
        self.shared = index >= c.independent_layers
        self.q_proj = nn.Linear(
            c.hidden_size, c.num_attention_heads * self.head_dim, bias=c.attention_bias
        )
        self.q_norm = FloatRMSNorm(self.head_dim, c.rms_norm_eps)
        if not self.shared:
            self.k_proj = nn.Linear(
                c.hidden_size, self.kv_heads * self.head_dim, bias=c.attention_bias
            )
            self.v_proj = (
                None
                if c.attention_k_eq_v and not self.local
                else nn.Linear(c.hidden_size, self.kv_heads * self.head_dim, bias=c.attention_bias)
            )
            self.k_norm = FloatRMSNorm(self.head_dim, c.rms_norm_eps)
            self.v_norm = FloatRMSNorm(self.head_dim, c.rms_norm_eps, with_scale=False)
        self.o_proj = nn.Linear(
            c.num_attention_heads * self.head_dim, c.hidden_size, bias=c.attention_bias
        )
        self.rope = RotaryEmbedding(
            self.head_dim,
            RopeConfig(theta=c.local_rope_theta if self.local else c.global_rope_theta),
        )
        if not self.local:
            rotated = int(c.global_rotary_fraction * self.head_dim // 2)
            frequency = self.rope.inv_freq.clone() / c.global_rope_factor
            frequency[rotated:] = 0
            self.rope.inv_freq = frequency

    def forward(self, x, positions, padding, previous, shared, seen, cache, vision_blocks=None):
        c, b, s = self.config, x.shape[0], x.shape[1]

        def split(tensor, heads):
            return tensor.view(b, s, heads, self.head_dim).transpose(1, 2)

        q = self.rope(self.q_norm(split(self.q_proj(x), c.num_attention_heads)), positions)
        if self.shared:
            k, v = shared[self.layer_type]
        else:
            raw = split(self.k_proj(x), self.kv_heads)
            v = self.v_norm(raw if self.v_proj is None else split(self.v_proj(x), self.kv_heads))
            k = self.rope(self.k_norm(raw), positions)
            if previous is not None:
                pk, pv = previous
                length = min(seen, c.sliding_window - 1) if self.local else seen
                if pk.shape != (b, self.kv_heads, length, self.head_dim) or pv.shape != pk.shape:
                    raise ValueError("Gemma4 KV owner shape differs from its local/global layout")
                k, v = torch.cat((pk, k), -2), torch.cat((pv, v), -2)
            shared[self.layer_type] = (k, v)
        visible = attention_mask(
            b,
            s,
            k.shape[-2],
            seen_tokens=seen,
            window=c.sliding_window if self.local else None,
            padding=padding,
            device=x.device,
        )
        if self.local and vision_blocks is not None:
            same = (vision_blocks[:, :, None] == vision_blocks[:, None, :]) & (
                vision_blocks[:, :, None] >= 0
            )
            extra = torch.zeros((b, 1, s, k.shape[-2]), dtype=torch.bool, device=x.device)
            extra[..., -s:] = same[:, None]
            local_window = attention_mask(
                b,
                s,
                k.shape[-2],
                seen_tokens=seen,
                window=c.sliding_window,
                padding=padding,
                device=x.device,
                causal=False,
            )
            visible = (visible | extra) & local_window
        result = scaled_attention(
            q, k, v, visible, scale=1.0, dropout=c.attention_dropout, training=self.training
        )
        present = None
        if cache and not self.shared:
            keep = c.sliding_window - 1
            present = (
                (k[..., -keep:, :], v[..., -keep:, :])
                if self.local and keep
                else ((k[..., :0, :], v[..., :0, :]) if self.local else (k, v))
            )
        return self.o_proj(result.transpose(1, 2).reshape(b, s, -1)), present


class Gemma4MLP(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.activation = c.hidden_activation
        width = c.intermediate_size * (
            2 if c.use_double_wide_mlp and index >= c.independent_layers else 1
        )
        self.gate_proj, self.up_proj, self.down_proj = (
            nn.Linear(c.hidden_size, width, bias=False),
            nn.Linear(c.hidden_size, width, bias=False),
            nn.Linear(width, c.hidden_size, bias=False),
        )

    def forward(self, x):
        return self.down_proj(activation(self.gate_proj(x), self.activation) * self.up_proj(x))


class Gemma4Router(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.norm = FloatRMSNorm(c.hidden_size, c.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.scale, self.per_expert_scale = (
            nn.Parameter(torch.ones(c.hidden_size)),
            nn.Parameter(torch.ones(c.num_experts)),
        )

    def forward(self, x):
        probabilities = (
            self.proj(self.norm(x) * self.scale * self.config.hidden_size**-0.5).float().softmax(-1)
        )
        weights, indices = probabilities.topk(self.config.top_k_experts, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        return probabilities, weights * self.per_expert_scale[indices], indices


class Gemma4Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config = c
        self.self_attn, self.mlp = Gemma4Attention(c, index), Gemma4MLP(c, index)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            self.add_module(name, FloatRMSNorm(c.hidden_size, c.rms_norm_eps))
        self.register_buffer("layer_scalar", torch.ones(1))
        if c.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                c.hidden_size, c.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                c.hidden_size_per_layer_input, c.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = FloatRMSNorm(c.hidden_size, c.rms_norm_eps)
        if c.enable_moe_block:
            self.router = Gemma4Router(c)
            self.experts = PackedExperts(
                c.num_experts,
                c.hidden_size,
                c.moe_intermediate_size,
                std=c.initializer_range,
                activation=c.hidden_activation,
            )
            for name in (
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ):
                self.add_module(name, FloatRMSNorm(c.hidden_size, c.rms_norm_eps))

    def forward(
        self, x, ple, positions, padding, previous, shared, seen, cache, vision_blocks=None
    ):
        value, state = self.self_attn(
            self.input_layernorm(x),
            positions,
            padding,
            previous,
            shared,
            seen,
            cache,
            vision_blocks,
        )
        x = x + self.post_attention_layernorm(value)
        value = self.mlp(self.pre_feedforward_layernorm(x))
        router = None
        if self.config.enable_moe_block:
            flat = x.flatten(0, 1)
            router, weights, indices = self.router(flat)
            sparse = self.experts(
                self.pre_feedforward_layernorm_2(flat), indices, weights
            ).reshape_as(x)
            value = self.post_feedforward_layernorm_1(value) + self.post_feedforward_layernorm_2(
                sparse
            )
        x = x + self.post_feedforward_layernorm(value)
        if ple is not None:
            branch = activation(self.per_layer_input_gate(x), self.config.hidden_activation) * ple
            x = x + self.post_per_layer_input_norm(self.per_layer_projection(branch))

        return (x * self.layer_scalar).to(x.dtype), state, router


class Gemma4ForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        c = config
        self.model = nn.Module()
        self.model.embed_tokens = Gemma4Embedding(
            c.vocab_size, c.hidden_size, c.pad_token_id, c.hidden_size**0.5
        )
        self.model.layers = nn.ModuleList(Gemma4Layer(c, i) for i in range(c.num_hidden_layers))
        self.model.norm = FloatRMSNorm(c.hidden_size, c.rms_norm_eps)
        if c.hidden_size_per_layer_input:
            self.model.embed_tokens_per_layer = Gemma4Embedding(
                c.vocab_size_per_layer_input,
                c.num_hidden_layers * c.hidden_size_per_layer_input,
                c.pad_token_id,
                c.hidden_size_per_layer_input**0.5,
            )
            self.model.per_layer_model_projection = nn.Linear(
                c.hidden_size, c.num_hidden_layers * c.hidden_size_per_layer_input, bias=False
            )
            self.model.per_layer_projection_norm = FloatRMSNorm(
                c.hidden_size_per_layer_input, c.rms_norm_eps
            )
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=c.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def text_config(self):
        return self.config

    def get_decoder(self):
        return self.model

    def get_input_embeddings(self):
        return self.get_decoder().embed_tokens

    def get_per_layer_inputs(self, input_ids):
        c = self.text_config
        if not c.hidden_size_per_layer_input:
            raise ValueError("PLE is disabled")
        return (
            self.get_decoder()
            .embed_tokens_per_layer(input_ids)
            .view(*input_ids.shape, c.num_hidden_layers, c.hidden_size_per_layer_input)
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        c = self.text_config
        if not c.hidden_size_per_layer_input:
            raise ValueError("PLE is disabled")
        values = self.get_decoder().per_layer_model_projection(inputs_embeds) * c.hidden_size**-0.5
        values = self.get_decoder().per_layer_projection_norm(
            values.view(
                *inputs_embeds.shape[:-1], c.num_hidden_layers, c.hidden_size_per_layer_input
            )
        )
        if per_layer_inputs is None:
            return values
        if per_layer_inputs.shape != values.shape:
            raise ValueError("Token-identity PLE shape differs from configured layers/width")
        return (values + per_layer_inputs) * 2**-0.5

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
        per_layer_inputs=None,
        vision_block_ids=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one token or embedding input")
        c = self.text_config
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("Input IDs already determine token-identity PLE")
        x = self.get_input_embeddings()(input_ids) if input_ids is not None else inputs_embeds
        if x.ndim != 3 or x.shape[-1] != c.hidden_size:
            raise ValueError("Invalid Gemma4 input embedding shape")
        b, s = x.shape[:2]
        if not s:
            raise ValueError("Empty Gemma4 input")
        if vision_block_ids is not None and (
            c.use_bidirectional_attention != "vision"
            or vision_block_ids.shape != (b, s)
            or vision_block_ids.dtype != torch.long
            or (vision_block_ids < -1).any()
        ):
            raise ValueError(
                "Vision block IDs require configured local bidirectionality and current integer [B,S] IDs"
            )
        ple = None
        if c.hidden_size_per_layer_input:
            if input_ids is not None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            elif per_layer_inputs is None:
                raise ValueError(
                    "Embedding-only Gemma4 requires explicit token-identity PLE; reverse-vocabulary guessing is disabled"
                )
            ple = self.project_per_layer_inputs(x, per_layer_inputs)
        elif per_layer_inputs is not None:
            raise ValueError("PLE input supplied to a model without PLE")
        if state is not None and (
            not isinstance(state, Gemma4State)
            or state.model_key != self.model_key
            or len(state.layers) != c.independent_layers
        ):
            raise ValueError("Gemma4 state/configuration/KV owner count mismatch")
        seen = 0 if state is None else state.seen_tokens
        if position_ids is None:
            position_ids = torch.arange(seen, seen + s, device=x.device)[None].expand(b, -1)
        if position_ids.shape == (1, s):
            position_ids = position_ids.expand(b, -1)
        if (
            position_ids.shape != (b, s)
            or position_ids.dtype not in {torch.int32, torch.int64}
            or (position_ids < 0).any()
        ):
            raise ValueError("Gemma4 requires nonnegative integer current token positions[B,S]")
        if attention_mask is not None and (
            attention_mask.shape != (b, seen + s)
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("Padding mask must cover physical past+current tokens")
        shared, states, routers, history = {}, [], [], []
        for index, layer in enumerate(self.get_decoder().layers):
            if output_hidden_states:
                history.append(x)
            previous = (
                state.layers[index] if state is not None and index < c.independent_layers else None
            )
            x, present, router = layer(
                x,
                None if ple is None else ple[:, :, index],
                position_ids,
                attention_mask,
                previous,
                shared,
                seen,
                use_cache,
                vision_block_ids,
            )
            if use_cache and present is not None:
                states.append(present)
            if router is not None:
                routers.append(router)
        x = self.get_decoder().norm(x)
        if output_hidden_states:
            history.append(x)
        logits = self.lm_head(x)
        if c.final_logit_softcapping is not None:
            logits = (logits / c.final_logit_softcapping).tanh() * c.final_logit_softcapping
        present = Gemma4State(tuple(states), seen + s, self.model_key) if use_cache else None
        return TokenOutput(
            logits,
            present,
            tuple(history) if output_hidden_states else None,
            {"router_probabilities": tuple(routers)},
        )
