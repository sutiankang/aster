"""Kimi K3 KDA/NoPE-MLA, attention residuals, latent experts, and SiTU."""

from dataclasses import dataclass
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn import RMSNorm
from aster.nn.kda import KimiDeltaAttention, situ_glu
from aster.nn.latent_attention import MultiheadLatentAttention
from .config import DeepSeekV3Config
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class KimiK3TextConfig(DeepSeekV3Config):
    architecture: ClassVar[str] = "kimi_k3_text"
    num_hidden_layers: int = 4
    rms_norm_eps: float = 1e-5
    routed_scaling_factor: float = 1.0
    layer_types: tuple[str, ...] | None = None
    attn_res_block_size: int = 2
    routed_expert_hidden_size: int = 16
    latent_moe_use_norm: bool = True
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    linear_num_heads: int = 4
    linear_head_dim: int = 8
    linear_conv_kernel_dim: int = 4
    gate_lower_bound: float = -5.0
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.layer_types is None:
            object.__setattr__(
                self,
                "layer_types",
                tuple(
                    "mla" if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1 else "kda"
                    for i in range(self.num_hidden_layers)
                ),
            )
        else:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if len(self.layer_types) != self.num_hidden_layers or any(
            x not in {"kda", "mla"} for x in self.layer_types
        ):
            raise ValueError("K3 requires an explicit KDA/MLA layer schedule")
        dimensions = (
            self.attn_res_block_size,
            self.routed_expert_hidden_size,
            self.linear_num_heads,
            self.linear_head_dim,
            self.linear_conv_kernel_dim,
        )
        if any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("Invalid K3 residual/latent/KDA dimensions")
        if self.linear_head_dim != self.v_head_dim:
            raise ValueError("Published K3 KDA uses equal key/value head dimensions")
        if not self.mla_use_nope or not self.mla_use_output_gate or self.attention_bias:
            raise ValueError(
                "This branch implements published K3 NoPE + output-gated bias-free MLA"
            )
        if not self.latent_moe_use_norm:
            raise ValueError(
                "K3 stable latent experts require normalization after the expert mixture"
            )
        if (
            any(
                not math.isfinite(x) or x <= 0
                for x in (self.activation_situ_beta, self.activation_situ_linear_beta)
            )
            or not math.isfinite(self.gate_lower_bound)
            or self.gate_lower_bound >= 0
        ):
            raise ValueError("Invalid SiTU/safe-forget-gate constants")


@dataclass(frozen=True)
class KimiK3State:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    layer_types: tuple[str, ...]
    kind: str = "kimi_k3_hybrid"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple(tuple(x.clone() for x in row) for row in self.layers),
            self.seen_tokens,
            self.model_key,
            self.layer_types,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(tuple(x.index_select(0, indices) for x in row) for row in self.layers),
            self.seen_tokens,
            self.model_key,
            self.layer_types,
        )

    def truncate(self, length):
        raise ValueError("K3 KDA memory cannot be sliced; restore a snapshot and replay")


def attention_residual(prefix, bank, score_projection, score_norm):
    """Aggregate depth-bank and current-prefix states per token, not across tokens."""
    if not bank:
        return prefix
    values = torch.stack((*bank, prefix), -2)
    scores = score_projection(score_norm(values)).squeeze(-1).float()
    return (scores.softmax(-1)[..., None] * values.float()).sum(-2).to(prefix.dtype)


class KimiK3MLP(nn.Module):
    def __init__(self, hidden, intermediate, beta, linear_beta):
        super().__init__()
        self.beta, self.linear_beta = beta, linear_beta
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value):
        gate, up = self.gate_up_proj(value).chunk(2, -1)
        return self.down_proj(situ_glu(gate, up, self.beta, self.linear_beta))


class FloatRouter(nn.Linear):
    def forward(self, value):
        return F.linear(value.float(), self.weight.float())


class KimiK3MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.gate = FloatRouter(c.hidden_size, c.n_routed_experts, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(c.n_routed_experts))
        self.routed_expert_down_proj = nn.Linear(
            c.hidden_size, c.routed_expert_hidden_size, bias=False
        )
        self.routed_expert_norm = RMSNorm(c.routed_expert_hidden_size, c.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(
            c.routed_expert_hidden_size, c.hidden_size, bias=False
        )
        self.experts = nn.ModuleList(
            KimiK3MLP(
                c.routed_expert_hidden_size,
                c.moe_intermediate_size,
                c.activation_situ_beta,
                c.activation_situ_linear_beta,
            )
            for _ in range(c.n_routed_experts)
        )
        self.shared_experts = KimiK3MLP(
            c.hidden_size,
            c.n_shared_experts * c.moe_intermediate_size,
            c.activation_situ_beta,
            c.activation_situ_linear_beta,
        )

    def forward(self, value):
        c = self.config
        shape = value.shape
        flat = value.reshape(-1, shape[-1])

        logits = self.gate(flat)
        probabilities = logits.sigmoid()
        choice = probabilities + self.e_score_correction_bias
        groups = choice.reshape(len(flat), c.n_group, -1).topk(2, -1).values.sum(-1)
        chosen_groups = groups.topk(c.topk_group, sorted=False).indices
        active_groups = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, chosen_groups, True)
        choice = choice.masked_fill(
            ~active_groups[..., None]
            .expand(-1, -1, c.n_routed_experts // c.n_group)
            .reshape_as(choice),
            -torch.inf,
        )
        indices = choice.topk(c.num_experts_per_tok, sorted=False).indices
        weights = probabilities.gather(1, indices)
        if c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * c.routed_scaling_factor
        latent = self.routed_expert_down_proj(flat)
        mixture = torch.zeros_like(latent)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)
            if not rows.numel():
                continue
            contribution = expert(latent[rows]) * weights[rows, slots, None]
            mixture = mixture.index_add(0, rows, contribution.to(mixture.dtype))

        result = self.routed_expert_up_proj(self.routed_expert_norm(mixture)) + self.shared_experts(
            flat
        )
        return result.reshape(shape), {"logits": logits, "indices": indices, "weights": weights}


class KimiK3Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.config, self.index = config, index
        self.is_kda = config.layer_types[index] == "kda"
        self.self_attn = (
            KimiDeltaAttention(config)
            if self.is_kda
            else MultiheadLatentAttention(
                config, skip_rope=True, latent_norm_eps=config.rms_norm_eps, output_gate=True
            )
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attention_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.sparse = index >= config.first_k_dense_replace
        self.mlp = (
            KimiK3MoE(config)
            if self.sparse
            else KimiK3MLP(
                config.hidden_size,
                config.intermediate_size,
                config.activation_situ_beta,
                config.activation_situ_linear_beta,
            )
        )

    def forward(self, prefix, bank, positions, padding, previous, seen, use_cache):
        mixed = attention_residual(
            prefix, bank, self.self_attention_res_proj, self.self_attention_res_norm
        )

        boundary = self.index % self.config.attn_res_block_size == 0
        if boundary:
            bank = (*bank, prefix)
        normalized = self.input_layernorm(mixed)
        if self.is_kda:
            value, updated = self.self_attn(
                normalized, previous, padding, seen_tokens=seen, use_cache=use_cache
            )
        else:
            value, updated = self.self_attn(
                normalized,
                positions,
                padding,
                previous,
                seen_tokens=seen,
                use_cache=use_cache,
                implementation="expanded",
            )
        prefix = value if boundary else prefix + value
        mixed = self.post_attention_layernorm(
            attention_residual(prefix, bank, self.mlp_res_proj, self.mlp_res_norm)
        )
        result = self.mlp(mixed)
        value, extra = result if self.sparse else (result, None)
        return prefix + value, bank, updated, extra


class KimiK3ForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList(
            KimiK3Layer(config, i) for i in range(config.num_hidden_layers)
        )
        self.model.output_attn_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.model.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                nn.init.normal_(module.weight, std=config.initializer_range)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_decoder(self):
        return self.model

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
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("K3 needs exactly one token/embedding input")
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or hidden.shape[1] < 1:
            raise ValueError("K3 expects a nonempty [B,S,H] sequence")
        seen = 0
        if state is not None:
            if (
                not isinstance(state, KimiK3State)
                or state.model_key != self.model_key
                or state.layer_types != self.config.layer_types
                or len(state.layers) != len(self.model.layers)
            ):
                raise ValueError("K3 hybrid state/config mismatch")
            seen = state.seen_tokens
        if seen < 0 or seen + hidden.shape[1] > self.config.max_position_embeddings:
            raise ValueError("K3 context capacity exceeded")
        if position_ids is not None and (
            position_ids.shape != hidden.shape[:2]
            or position_ids.dtype not in {torch.int32, torch.int64}
            or (position_ids < 0).any()
        ):
            raise ValueError(
                "Position metadata must align with current tokens; published K3 attention itself is NoPE"
            )
        bank, states, history, routers = (), [], [], []
        for index, layer in enumerate(self.model.layers):
            if output_hidden_states:
                history.append(hidden)
            hidden, bank, present, extra = layer(
                hidden,
                bank,
                position_ids,
                attention_mask,
                None if state is None else state.layers[index],
                seen,
                use_cache,
            )
            if use_cache:
                states.append(present)
            if extra is not None:
                routers.append(extra)
        hidden = self.model.norm(
            attention_residual(
                hidden, bank, self.model.output_attn_res_proj, self.model.output_attn_res_norm
            )
        )
        if output_hidden_states:
            history.append(hidden)
        updated = (
            KimiK3State(
                tuple(states), seen + hidden.shape[1], self.model_key, self.config.layer_types
            )
            if use_cache
            else None
        )
        return TokenOutput(
            self.lm_head(hidden),
            updated,
            tuple(history) if output_hidden_states else None,
            {"router": tuple(routers)},
        )
