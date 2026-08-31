"""Qwen3Next hybrid Gated DeltaNet, gated full attention, and shared experts."""

from dataclasses import dataclass, field
from typing import ClassVar
import torch
from torch import nn
from aster.nn import RMSNorm, RotaryEmbedding, RopeConfig
from aster.nn.attention import attention_mask, scaled_attention
from aster.nn.delta import GatedDeltaNet, HybridState
from aster.nn.experts import TopKRouter, PackedExperts
from .config import LlamaConfig
from .decoder import CausalLM, GatedMLP


@dataclass(frozen=True)
class Qwen3NextConfig(LlamaConfig):
    architecture: ClassVar[str] = "qwen3_next"
    num_hidden_layers: int = 4
    head_dim: int = 8
    partial_rotary_factor: float = 0.25
    layer_types: tuple[str, ...] | None = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 4
    linear_value_head_dim: int = 4
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 4
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 16
    shared_expert_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    mlp_only_layers: tuple[int, ...] = ()
    attention_bias: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.layer_types is None:
            object.__setattr__(
                self,
                "layer_types",
                tuple(
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ),
            )
        else:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "mlp_only_layers", tuple(self.mlp_only_layers))
        if len(self.layer_types) != self.num_hidden_layers or any(
            t not in {"linear_attention", "full_attention"} for t in self.layer_types
        ):
            raise ValueError("Every hybrid layer must declare its actual mixer")
        if (
            min(
                self.linear_conv_kernel_dim,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
                self.linear_num_key_heads,
                self.linear_num_value_heads,
                self.decoder_sparse_step,
                self.moe_intermediate_size,
                self.shared_expert_intermediate_size,
            )
            < 1
        ):
            raise ValueError("Invalid DeltaNet/MoE dimensions")
        if self.linear_num_value_heads % self.linear_num_key_heads or self.num_experts < 0:
            raise ValueError("Invalid DeltaNet head ratio or expert count")
        if self.num_experts and not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("Invalid top-k expert count")
        rotary = int(self.attention_head_dim * self.partial_rotary_factor)
        if not 0 < self.partial_rotary_factor <= 1 or rotary < 2 or rotary % 2:
            raise ValueError("Partial rotary dimensions must be positive and even")


class GatedFullAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        h, d = c.num_attention_heads, c.attention_head_dim
        self.q_proj = nn.Linear(c.hidden_size, 2 * h * d, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * d, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * d, bias=c.attention_bias)
        self.o_proj = nn.Linear(h * d, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(d, c.rms_norm_eps, zero_centered=True)
        self.k_norm = RMSNorm(d, c.rms_norm_eps, zero_centered=True)
        self.rotary_dim = int(d * c.partial_rotary_factor)
        self.rope = RotaryEmbedding(self.rotary_dim, c.rope)

    def forward(self, hidden, positions, padding, previous, seen, use_cache):
        c = self.config
        b, s, _ = hidden.shape
        d = c.attention_head_dim
        q, gate = self.q_proj(hidden).reshape(b, s, c.num_attention_heads, 2 * d).chunk(2, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).reshape(b, s, c.num_key_value_heads, d)).transpose(1, 2)
        v = self.v_proj(hidden).reshape(b, s, c.num_key_value_heads, d).transpose(1, 2)

        def rotate(value):
            return torch.cat(
                (
                    self.rope(value[..., : self.rotary_dim], positions),
                    value[..., self.rotary_dim :],
                ),
                -1,
            )

        q, k = rotate(q), rotate(k)
        if previous is not None:
            if (
                previous[0].shape != (b, c.num_key_value_heads, seen, d)
                or previous[1].shape != previous[0].shape
            ):
                raise ValueError("Invalid full-attention layer cache inside hybrid state")
            k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)
        mask = attention_mask(
            b, s, seen + s, seen_tokens=seen, padding=padding, device=hidden.device
        )
        output = scaled_attention(
            q, k, v, mask, dropout=c.attention_dropout, training=self.training
        )
        output = output.transpose(1, 2).reshape(b, s, -1) * gate.reshape(b, s, -1).sigmoid()
        return self.o_proj(output), (k, v) if use_cache else None


class QwenNextMoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = TopKRouter(
            c.hidden_size,
            c.num_experts,
            c.num_experts_per_tok,
            normalize=c.norm_topk_prob,
            std=c.initializer_range,
        )
        self.experts = PackedExperts(
            c.num_experts, c.hidden_size, c.moe_intermediate_size, std=c.initializer_range
        )
        self.shared_expert = GatedMLP(c.hidden_size, c.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, weights, indices = self.gate(flat)
        value = (
            self.experts(flat, indices, weights)
            + self.shared_expert(flat) * self.shared_expert_gate(flat).sigmoid()
        )
        return value.reshape(shape), {"logits": logits, "weights": weights, "indices": indices}


class HybridLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.kind = c.layer_types[index]
        if self.kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = GatedFullAttention(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True)
        self.sparse = (
            c.num_experts > 0
            and index not in c.mlp_only_layers
            and (index + 1) % c.decoder_sparse_step == 0
        )
        self.mlp = QwenNextMoE(c) if self.sparse else GatedMLP(c.hidden_size, c.intermediate_size)

    def forward(self, hidden, positions, padding, previous, seen, use_cache):
        normal = self.input_layernorm(hidden)
        if self.kind == "linear_attention":
            value, present = self.linear_attn(
                normal, previous, padding, seen_tokens=seen, use_cache=use_cache
            )
        else:
            value, present = self.self_attn(normal, positions, padding, previous, seen, use_cache)
        hidden = hidden + value
        result = self.mlp(self.post_attention_layernorm(hidden))
        value, auxiliary = result if self.sparse else (result, None)
        return hidden + value, present, auxiliary


class Qwen3NextForCausalLM(CausalLM):
    layer_type = HybridLayer
    state_kind = "hybrid_delta"
    state_type = HybridState

    def __init__(self, config):
        super().__init__(config)
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)

    def create_state(self, layers, seen, kind):
        return HybridState(layers, seen, self.model_key, self.config.layer_types)
