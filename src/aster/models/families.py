"""Architecture-specific Gemma3 and Llama4 text components."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import RMSNorm, RopeConfig, RotaryEmbedding, GroupedQueryAttention
from aster.nn.attention import attention_mask, scaled_attention
from .config import LlamaConfig
from .decoder import CausalLM, GatedMLP


@dataclass(frozen=True)
class Gemma3TextConfig(LlamaConfig):
    architecture: ClassVar[str] = "gemma3_text"
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(theta=1_000_000.0))
    rope_local: RopeConfig = field(default_factory=RopeConfig)
    head_dim: int = 8
    tie_word_embeddings: bool = True
    sliding_window: int = 32
    layer_types: tuple[str, ...] = ("sliding_attention", "full_attention")
    query_pre_attn_scalar: int = 8
    attention_bias: bool = False
    final_logit_softcapping: float | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if len(self.layer_types) != self.num_hidden_layers or any(
            x not in {"sliding_attention", "full_attention"} for x in self.layer_types
        ):
            raise ValueError("Gemma3 requires each local/global layer type")
        if min(self.sliding_window, self.query_pre_attn_scalar) < 1 or not isinstance(
            self.rope_local, RopeConfig
        ):
            raise ValueError("Invalid Gemma3 position/attention scale")
        if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0:
            raise ValueError("Softcap must be positive")

    def window_for_layer(self, index):
        return self.sliding_window if self.layer_types[index] == "sliding_attention" else None


class ScaledWordEmbedding(nn.Embedding):
    def forward(self, token_ids):

        return super().forward(token_ids) * (self.embedding_dim**0.5)


class GemmaMLP(GatedMLP):
    def forward(self, hidden):
        return self.down_proj(
            F.gelu(self.gate_proj(hidden), approximate="tanh") * self.up_proj(hidden)
        )


class GemmaLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        rope = c.rope_local if c.window_for_layer(index) else c.rope
        self.self_attn = GroupedQueryAttention(
            c.hidden_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.attention_head_dim,
            rope,
            qkv_bias=c.attention_bias,
            output_bias=c.attention_bias,
            dropout=c.attention_dropout,
            window=c.window_for_layer(index),
        )
        self.self_attn.scale = c.query_pre_attn_scalar**-0.5
        self.self_attn.q_norm = RMSNorm(c.attention_head_dim, c.rms_norm_eps, zero_centered=True)
        self.self_attn.k_norm = RMSNorm(c.attention_head_dim, c.rms_norm_eps, zero_centered=True)
        self.mlp = GemmaMLP(c.hidden_size, c.intermediate_size)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            self.add_module(name, RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True))

    def forward(self, hidden, positions, padding, previous, seen, use_cache):
        value, present = self.self_attn(
            self.input_layernorm(hidden),
            positions,
            padding,
            previous,
            seen_tokens=seen,
            use_cache=use_cache,
        )
        hidden = hidden + self.post_attention_layernorm(value)
        hidden = hidden + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden))
        )
        return hidden, present, None


class Gemma3ForCausalLM(CausalLM):
    layer_type = GemmaLayer

    def __init__(self, config):
        super().__init__(config)
        self.model.embed_tokens = ScaledWordEmbedding(config.vocab_size, config.hidden_size)
        self._initialize(self.model.embed_tokens)
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            output.logits = (output.logits / cap).tanh() * cap
        return output


@dataclass(frozen=True)
class Llama4TextConfig(LlamaConfig):
    architecture: ClassVar[str] = "llama4_text"
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(theta=500000.0))
    head_dim: int = 8
    rms_norm_eps: float = 1e-5
    intermediate_size_mlp: int = 64
    num_local_experts: int = 4
    num_experts_per_tok: int = 1
    moe_layers: tuple[int, ...] = (0, 1)
    no_rope_layers: tuple[int, ...] = (1, 0)
    attention_chunk_size: int | None = 32
    use_qk_norm: bool = True
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    attention_bias: bool = False

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "moe_layers", tuple(self.moe_layers))
        object.__setattr__(self, "no_rope_layers", tuple(self.no_rope_layers))
        if len(self.no_rope_layers) != self.num_hidden_layers or any(
            x not in {0, 1} for x in self.no_rope_layers
        ):
            raise ValueError("Each Llama4 layer must declare RoPE(1)/NoPE(0)")
        if any(i not in range(self.num_hidden_layers) for i in self.moe_layers) or len(
            set(self.moe_layers)
        ) != len(self.moe_layers):
            raise ValueError("Invalid Llama4 MoE layer schedule")
        if (
            min(self.intermediate_size_mlp, self.num_local_experts, self.floor_scale) < 1
            or not 1 <= self.num_experts_per_tok <= self.num_local_experts
        ):
            raise ValueError("Invalid Llama4 experts/temperature")
        if (
            self.attention_chunk_size is not None
            and self.attention_chunk_size < 1
            or self.attn_scale < 0
        ):
            raise ValueError("Invalid chunk size or temperature scale")
        if self.rope.interleaved:
            raise ValueError(
                "Llama4 applies its own complex pair layout; provide canonical frequency config"
            )


class Llama4Attention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config, self.use_rope = c, bool(c.no_rope_layers[index])
        d = c.attention_head_dim
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * d, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * d, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * d, bias=c.attention_bias)
        self.o_proj = nn.Linear(c.num_attention_heads * d, c.hidden_size, bias=c.attention_bias)
        self.rope = RotaryEmbedding(d, c.rope)

    def forward(self, hidden, positions, padding, previous, seen, use_cache):
        c = self.config
        b, s, _ = hidden.shape
        d = c.attention_head_dim

        def split(projection, heads):
            return projection(hidden).reshape(b, s, heads, d).transpose(1, 2)

        q, k, v = (
            split(self.q_proj, c.num_attention_heads),
            split(self.k_proj, c.num_key_value_heads),
            split(self.v_proj, c.num_key_value_heads),
        )
        if self.use_rope:

            def complex_rope(value):

                arranged = torch.cat((value[..., ::2], value[..., 1::2]), -1)
                real, imag = self.rope(arranged, positions).chunk(2, -1)
                return torch.stack((real, imag), -1).flatten(-2)

            q, k = complex_rope(q), complex_rope(k)
            if c.use_qk_norm:

                def unit_rms(value):
                    work = value.float()
                    return (
                        work * torch.rsqrt(work.square().mean(-1, keepdim=True) + c.rms_norm_eps)
                    ).to(value.dtype)

                q, k = unit_rms(q), unit_rms(k)
        elif c.attn_temperature_tuning:
            absolute = torch.arange(seen, seen + s, device=hidden.device).float()
            factor = (torch.floor((absolute + 1) / c.floor_scale)).log1p() * c.attn_scale + 1
            q = (q * factor[None, None, :, None]).to(q.dtype)
        if previous is not None:
            if (
                previous[0].shape != (b, c.num_key_value_heads, seen, d)
                or previous[1].shape != previous[0].shape
            ):
                raise ValueError("Invalid Llama4 KV layout/length")
            k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)
        visible = attention_mask(
            b, s, seen + s, seen_tokens=seen, padding=padding, device=hidden.device
        )
        if self.use_rope and c.attention_chunk_size is not None:
            qpos = torch.arange(seen, seen + s, device=hidden.device) // c.attention_chunk_size
            kpos = torch.arange(seen + s, device=hidden.device) // c.attention_chunk_size
            visible = visible & (qpos[:, None] == kpos[None, :])[None, None]
        output = scaled_attention(
            q, k, v, visible, dropout=c.attention_dropout, training=self.training
        )
        return self.o_proj(output.transpose(1, 2).reshape(b, s, -1)), (k, v) if use_cache else None


class Llama4Experts(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.gate_up_proj = nn.Parameter(
            torch.empty(c.num_local_experts, c.hidden_size, 2 * c.intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(c.num_local_experts, c.intermediate_size, c.hidden_size)
        )
        nn.init.normal_(self.gate_up_proj, std=c.initializer_range)
        nn.init.normal_(self.down_proj, std=c.initializer_range)

    def forward(self, hidden, indices, scores):
        result = torch.zeros_like(hidden)
        for expert in range(self.gate_up_proj.shape[0]):
            tokens, slots = torch.where(indices == expert)
            if tokens.numel():
                value = hidden[tokens] * scores[tokens, slots, None]
                gate, up = (value @ self.gate_up_proj[expert]).chunk(2, -1)
                result.index_add_(0, tokens, (F.silu(gate) * up) @ self.down_proj[expert])
        return result


class Llama4MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k = c.num_experts_per_tok
        self.router = nn.Linear(c.hidden_size, c.num_local_experts, bias=False)
        self.experts = Llama4Experts(c)
        self.shared_expert = GatedMLP(c.hidden_size, c.intermediate_size)

    def forward(self, hidden):
        shape = hidden.shape
        hidden = hidden.reshape(-1, shape[-1])
        logits = self.router(hidden)
        values, indices = logits.topk(self.top_k, -1)
        weights = values.float().sigmoid().to(hidden.dtype)
        output = self.experts(hidden, indices, weights) + self.shared_expert(hidden)
        return output.reshape(shape), {"logits": logits, "indices": indices, "weights": weights}


class Llama4Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.self_attn = Llama4Attention(c, index)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.sparse = index in c.moe_layers
        self.feed_forward = (
            Llama4MoE(c) if self.sparse else GatedMLP(c.hidden_size, c.intermediate_size_mlp)
        )

    def forward(self, hidden, positions, padding, previous, seen, use_cache):
        value, present = self.self_attn(
            self.input_layernorm(hidden), positions, padding, previous, seen, use_cache
        )
        hidden = hidden + value
        result = self.feed_forward(self.post_attention_layernorm(hidden))
        value, auxiliary = result if self.sparse else (result, None)
        return hidden + value, present, auxiliary


class Llama4ForCausalLM(CausalLM):
    layer_type = Llama4Layer
