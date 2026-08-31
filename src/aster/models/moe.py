"""Mixtral and DeepSeek V3 expert/MLA architectures using shared token output contracts."""

import torch
from torch import nn
from aster.nn import RMSNorm
from aster.nn.experts import PackedExperts, TopKRouter
from aster.nn.latent_attention import MultiheadLatentAttention
from .decoder import CausalLM, DecoderLayer, GatedMLP


class MixtralExpertsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.jitter = config.router_jitter_noise
        self.gate = TopKRouter(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            std=config.initializer_range,
        )
        self.experts = PackedExperts(
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
            std=config.initializer_range,
        )

    def forward(self, hidden):
        if self.training and self.jitter:
            hidden = hidden * torch.empty_like(hidden).uniform_(1 - self.jitter, 1 + self.jitter)
        shape = hidden.shape
        hidden = hidden.reshape(-1, shape[-1])
        logits, weights, indices = self.gate(hidden)
        return self.experts(hidden, indices, weights).reshape(shape), {
            "logits": logits,
            "indices": indices,
            "weights": weights,
        }


class MixtralLayer(DecoderLayer):
    def __init__(self, config, index):
        super().__init__(config, index)
        self.mlp = MixtralExpertsBlock(config)

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
        value, routing = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + value, present, routing


class MixtralForCausalLM(CausalLM):
    layer_type = MixtralLayer


class DeepSeekExpertsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.gate = TopKRouter(
            c.hidden_size,
            c.n_routed_experts,
            c.num_experts_per_tok,
            groups=c.n_group,
            topk_groups=c.topk_group,
            sigmoid=True,
            normalize=c.norm_topk_prob,
            scale=c.routed_scaling_factor,
            std=c.initializer_range,
        )
        self.experts = PackedExperts(
            c.n_routed_experts, c.hidden_size, c.moe_intermediate_size, std=c.initializer_range
        )
        self.shared_experts = GatedMLP(c.hidden_size, c.n_shared_experts * c.moe_intermediate_size)

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits, weights, indices = self.gate(flat)
        return self.experts(flat, indices, weights).reshape(shape) + self.shared_experts(hidden), {
            "logits": logits,
            "indices": indices,
            "weights": weights,
        }


class DeepSeekLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = MultiheadLatentAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.sparse = index >= config.first_k_dense_replace
        self.mlp = (
            DeepSeekExpertsBlock(config)
            if self.sparse
            else GatedMLP(config.hidden_size, config.intermediate_size)
        )

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
        result = self.mlp(self.post_attention_layernorm(hidden))
        value, auxiliary = result if self.sparse else (result, None)
        return hidden + value, present, auxiliary


class DeepSeekV3ForCausalLM(CausalLM):
    layer_type = DeepSeekLayer
    state_kind = "mla_latent"
