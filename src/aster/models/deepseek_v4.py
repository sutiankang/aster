"""DeepSeek V4 floating-point reference: mHC, shared-KV MQA, compression, and routed experts."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn.normalization import RMSNorm
from aster.nn.position import RopeConfig, RotaryEmbedding
from aster.nn.hyperconnection import HyperConnection, HyperHead, UnweightedRMSNorm
from aster.nn.experts import PackedExperts
from aster.nn.compression import CompressedAttentionState, CompressedLayerState, compress_windows
from .serialization import LocalModelMixin
from .decoder import configuration_key


@dataclass(frozen=True)
class DeepSeekV4Config:
    architecture: ClassVar[str] = "deepseek_v4"
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 24
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    head_dim: int = 16
    q_lora_rank: int = 12
    qk_rope_head_dim: int = 4
    max_position_embeddings: int = 256
    layer_types: tuple[str, ...] = (
        "sliding_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    )
    mlp_layer_types: tuple[str, ...] = ("hash_moe", "moe", "moe")
    compress_rates: dict[str, int] = field(
        default_factory=lambda: {
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        }
    )
    rope: RopeConfig = field(default_factory=RopeConfig)
    compress_rope: RopeConfig = field(
        default_factory=lambda: RopeConfig(theta=160000, attention_factor=1.0)
    )
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    sliding_window: int = 4
    o_groups: int = 2
    o_lora_rank: int = 8
    index_n_heads: int = 2
    index_head_dim: int = 8
    index_topk: int = 2
    hc_mult: int = 3
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None

    def __post_init__(self):
        for name in ("rope", "compress_rope"):
            if isinstance(getattr(self, name), dict):
                object.__setattr__(self, name, RopeConfig(**getattr(self, name)))
        for name in ("layer_types", "mlp_layer_types"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            min(
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.head_dim,
                self.q_lora_rank,
                self.qk_rope_head_dim,
                self.max_position_embeddings,
                self.o_groups,
                self.o_lora_rank,
                self.sliding_window,
                self.index_n_heads,
                self.index_head_dim,
                self.index_topk,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.num_local_experts,
            )
            < 1
        ):
            raise ValueError("Invalid V4 dimensions")
        if (
            len(self.layer_types) != self.num_hidden_layers
            or len(self.mlp_layer_types) != self.num_hidden_layers
        ):
            raise ValueError("V4 attention/MoE schedules must explicitly cover every layer")
        if set(self.layer_types) - {
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        } or set(self.mlp_layer_types) - {"hash_moe", "moe"}:
            raise ValueError("Unknown V4 attention/MoE formula")
        if (
            set(self.compress_rates)
            != {"compressed_sparse_attention", "heavily_compressed_attention"}
            or min(self.compress_rates.values()) < 1
        ):
            raise ValueError("Both compressor rates must be positive")
        if self.qk_rope_head_dim % 2 or self.qk_rope_head_dim > min(
            self.head_dim, self.index_head_dim
        ):
            raise ValueError("Invalid partial RoPE width")
        if (
            self.num_attention_heads * self.head_dim % self.o_groups
            or not 1 <= self.num_experts_per_tok <= self.num_local_experts
        ):
            raise ValueError("Invalid grouped output/expert routing shape")
        if (
            min(
                self.rms_norm_eps,
                self.hc_eps,
                self.swiglu_limit,
                self.routed_scaling_factor,
                self.initializer_range,
            )
            <= 0
            or not 0 <= self.attention_dropout < 1
        ):
            raise ValueError("Invalid V4 numerical configuration")
        if self.scoring_func not in {"sqrtsoftplus", "sigmoid", "softmax"}:
            raise ValueError("Unsupported V4 router scoring")
        if self.rope.kind != "default" or self.compress_rope.kind not in {"default", "yarn"}:
            raise ValueError("V4 main RoPE is default; compressor supports explicit default/YaRN")
        if self.compress_rope.attention_factor not in (
            None,
            1.0,
        ) or self.rope.attention_factor not in (None, 1.0):
            raise ValueError("V4 rotations are norm-preserving: no YaRN amplitude scaling")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("Invalid padding token")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class V4Rotary(nn.Module):
    _aster_semantic_buffers = ("main_inv_freq", "compress_inv_freq")

    def __init__(self, c):
        super().__init__()
        self.dimension = c.qk_rope_head_dim
        for name, config in (("main", c.rope), ("compress", c.compress_rope)):
            self.register_buffer(
                name + "_inv_freq",
                RotaryEmbedding(self.dimension, config).inv_freq,
                persistent=False,
            )

    def forward(self, x, positions, kind):
        with torch.autocast(device_type=x.device.type, enabled=False):
            angle = (
                positions.float()[..., None] * getattr(self, kind + "_inv_freq").float()[None, None]
            )
            return angle.cos().to(x), angle.sin().to(x)


def rotate_v4(x, cos, sin):
    """Rotate the final rotary channels in interleaved layout rather than V3 split-half output layout."""
    width = cos.shape[-1] * 2
    ordinary, rotary = x[..., :-width], x[..., -width:]
    pair_rotate = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), -1).flatten(-2)
    cos, sin = cos[:, None].repeat_interleave(2, -1), sin[:, None].repeat_interleave(2, -1)
    rotary = (rotary.float() * cos.float() + pair_rotate.float() * sin.float()).to(x.dtype)
    return torch.cat((ordinary, rotary), -1)


class GroupedLinear(nn.Linear):
    def __init__(self, per_group_input, per_group_output, groups):
        super().__init__(per_group_input, per_group_output * groups, bias=False)
        self.groups = groups

    def forward(self, x):
        shape = x.shape
        grouped = x.reshape(-1, self.groups, shape[-1]).transpose(0, 1)
        result = grouped @ self.weight.view(self.groups, -1, shape[-1]).transpose(1, 2)
        return result.transpose(0, 1).reshape(*shape[:-1], -1)


class V4Compressor(nn.Module):
    def __init__(self, c, *, sparse=False, indexer=False):
        super().__init__()
        self.config = c
        self.overlap = sparse
        self.dimension = c.index_head_dim if indexer else c.head_dim
        self.ratio = c.compress_rates[
            "compressed_sparse_attention" if sparse else "heavily_compressed_attention"
        ]
        width = self.dimension * (2 if sparse else 1)
        self.kv_proj = nn.Linear(c.hidden_size, width, bias=False)
        self.gate_proj = nn.Linear(c.hidden_size, width, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.ratio, width))
        self.kv_norm = RMSNorm(self.dimension, c.rms_norm_eps)
        self.rotary_emb = V4Rotary(c)

    def compress(self, hidden, previous):
        def rotate(values, positions):
            return rotate_v4(values, *self.rotary_emb(values, positions, "compress"))

        return compress_windows(
            self.kv_proj(hidden),
            self.gate_proj(hidden),
            self.position_bias,
            self.ratio,
            self.dimension,
            self.kv_norm,
            rotate,
            overlap=self.overlap,
            previous=previous,
        )


class V4Indexer(V4Compressor):
    def __init__(self, c):
        super().__init__(c, sparse=True, indexer=True)
        self.q_b_proj = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
        self.scorer = nn.Module()
        self.scorer.weights_proj = nn.Linear(c.hidden_size, c.index_n_heads, bias=False)

    def forward(self, hidden, query_latent, positions, previous):
        c = self.config
        batch, length, _ = hidden.shape
        keys, updated = self.compress(hidden, previous)
        q = (
            self.q_b_proj(query_latent)
            .reshape(batch, length, c.index_n_heads, c.index_head_dim)
            .transpose(1, 2)
        )
        q = rotate_v4(q, *self.rotary_emb(hidden, positions, "compress")).transpose(1, 2)
        scores = F.relu(
            (q.float() @ keys.float().transpose(-1, -2)[:, None]) * c.index_head_dim**-0.5
        )
        weights = self.scorer.weights_proj(hidden).float() * c.index_n_heads**-0.5
        scores = (weights.unsqueeze(-2) @ scores).squeeze(-2)
        visible = (
            torch.arange(keys.shape[1], device=hidden.device)[None, None]
            < ((positions + 1) // self.ratio)[..., None]
        )
        scores = scores.masked_fill(~visible, -torch.inf)
        indices = scores.detach().topk(min(c.index_topk, keys.shape[1]), -1).indices
        selected = torch.zeros_like(visible).scatter(-1, indices, True) & visible
        return selected, updated, {"scores": scores, "visible": visible, "indices": indices}


class V4CSA(V4Compressor):
    def __init__(self, c):
        super().__init__(c, sparse=True)
        self.indexer = V4Indexer(c)


class V4Attention(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.config = c
        self.layer_type = c.layer_types[layer]
        self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_attention_heads * c.head_dim, bias=False)
        self.q_b_norm = UnweightedRMSNorm(c.rms_norm_eps)
        self.kv_proj = nn.Linear(c.hidden_size, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.o_a_proj = GroupedLinear(
            c.num_attention_heads * c.head_dim // c.o_groups, c.o_lora_rank, c.o_groups
        )
        self.o_b_proj = nn.Linear(c.o_groups * c.o_lora_rank, c.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(c.num_attention_heads))
        self.compressor = (
            V4CSA(c)
            if self.layer_type == "compressed_sparse_attention"
            else V4Compressor(c)
            if self.layer_type == "heavily_compressed_attention"
            else None
        )

    def forward(self, hidden, positions, embedding, previous, seen):
        c = self.config
        batch, length, _ = hidden.shape
        cos, sin = embedding["main" if self.layer_type == "sliding_attention" else "compress"]
        latent = self.q_a_norm(self.q_a_proj(hidden))
        q = self.q_b_norm(
            self.q_b_proj(latent)
            .reshape(batch, length, c.num_attention_heads, c.head_dim)
            .transpose(1, 2)
        )
        q = rotate_v4(q, cos, sin)
        kv = rotate_v4(self.kv_norm(self.kv_proj(hidden))[:, None], cos, sin)
        prior_length = 0 if previous is None else previous.window_kv.shape[-2]
        if previous is not None:
            kv = torch.cat((previous.window_kv, kv), -2)
        retained = kv[..., -(c.sliding_window - 1) :, :] if c.sliding_window > 1 else kv[..., :0, :]
        key_positions = torch.arange(seen - prior_length, seen + length, device=hidden.device)
        visible = (key_positions[None, None] <= positions[..., None]) & (
            key_positions[None, None] > positions[..., None] - c.sliding_window
        )
        compressor_state = indexer_state = index_info = None
        if self.compressor is not None:
            entries, compressor_state = self.compressor.compress(
                hidden, previous.compressor if previous else None
            )
            if self.layer_type == "compressed_sparse_attention":
                extra_visible, indexer_state, index_info = self.compressor.indexer(
                    hidden, latent, positions, previous.indexer if previous else None
                )
            else:
                extra_visible = (
                    torch.arange(entries.shape[1], device=hidden.device)[None, None]
                    < ((positions + 1) // self.compressor.ratio)[..., None]
                )
            visible = torch.cat((visible, extra_visible), -1)
            kv = torch.cat((kv, entries[:, None]), -2)
        logits = (q @ kv.transpose(-1, -2)) * c.head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None], -torch.inf)

        logits = torch.cat(
            (logits, self.sinks[None, :, None, None].expand(batch, -1, length, -1)), -1
        )
        logits = logits - logits.max(-1, keepdim=True).values
        probabilities = logits.softmax(-1)[..., :-1]
        probabilities = F.dropout(probabilities, c.attention_dropout, self.training)
        output = probabilities.to(kv.dtype) @ kv
        output = rotate_v4(output, cos, -sin).transpose(1, 2)
        output = self.o_a_proj(output.reshape(batch, length, c.o_groups, -1)).flatten(2)
        return (
            self.o_b_proj(output),
            CompressedLayerState(retained, compressor_state, indexer_state),
            index_info,
        )


class V4Router(nn.Module):
    def __init__(self, c, hash_route):
        super().__init__()
        self.config = c
        self.hash_route = hash_route
        self.weight = nn.Parameter(torch.empty(c.num_local_experts, c.hidden_size))
        if hash_route:
            self.register_buffer(
                "tid2eid", torch.zeros(c.vocab_size, c.num_experts_per_tok, dtype=torch.long)
            )
        else:
            self.register_buffer("e_score_correction_bias", torch.zeros(c.num_local_experts))
        nn.init.normal_(self.weight, std=c.initializer_range)

    def forward(self, hidden, input_ids):
        c = self.config
        logits = F.linear(hidden.reshape(-1, c.hidden_size), self.weight)
        scores = (
            F.softplus(logits).sqrt()
            if c.scoring_func == "sqrtsoftplus"
            else logits.sigmoid()
            if c.scoring_func == "sigmoid"
            else logits.softmax(-1)
        )
        if self.hash_route:
            if input_ids is None:
                raise ValueError(
                    "Hash-MoE needs explicit token IDs even with precomputed embeddings"
                )
            indices = self.tid2eid[input_ids.reshape(-1)]
            if (indices < 0).any() or (indices >= c.num_local_experts).any():
                raise ValueError("Hash route table has invalid expert IDs")
        else:
            indices = (
                (scores + self.e_score_correction_bias)
                .topk(c.num_experts_per_tok, -1, sorted=False)
                .indices
            )
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return logits, weights * c.routed_scaling_factor, indices


class V4MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.limit = c.swiglu_limit
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x).clamp(max=self.limit))
            * self.up_proj(x).clamp(-self.limit, self.limit)
        )


class V4MoE(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.gate = V4Router(c, c.mlp_layer_types[layer] == "hash_moe")
        self.experts = PackedExperts(
            c.num_local_experts,
            c.hidden_size,
            c.intermediate_size,
            std=c.initializer_range,
            swiglu_limit=c.swiglu_limit,
        )
        self.shared_experts = V4MLP(c)

    def forward(self, hidden, ids):
        logits, weights, indices = self.gate(hidden, ids)
        routed = self.experts(hidden.reshape(-1, hidden.shape[-1]), indices, weights).reshape_as(
            hidden
        )
        return routed + self.shared_experts(hidden), {
            "logits": logits,
            "weights": weights,
            "indices": indices,
        }


class V4Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.self_attn, self.mlp = V4Attention(c, index), V4MoE(c, index)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.attn_hc = HyperConnection(
            c.hidden_size, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps, c.rms_norm_eps
        )
        self.ffn_hc = HyperConnection(
            c.hidden_size, c.hc_mult, c.hc_sinkhorn_iters, c.hc_eps, c.rms_norm_eps
        )

    def forward(self, streams, ids, positions, embedding, previous, seen):
        post, mix, collapsed = self.attn_hc(streams)
        update, present, indexer = self.self_attn(
            self.input_layernorm(collapsed), positions, embedding, previous, seen
        )
        streams = self.attn_hc.expand(update, streams, post, mix)
        post, mix, collapsed = self.ffn_hc(streams)
        update, router = self.mlp(self.post_attention_layernorm(collapsed), ids)
        return self.ffn_hc.expand(update, streams, post, mix), present, router, indexer


class DeepSeekV4ForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.model.layers = nn.ModuleList(
            V4Layer(config, i) for i in range(config.num_hidden_layers)
        )
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.model.rotary_emb = V4Rotary(config)
        self.model.hc_head = HyperHead(
            config.hidden_size, config.hc_mult, config.hc_eps, config.rms_norm_eps
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=config.initializer_range)
            if isinstance(module, HyperConnection):
                nn.init.normal_(module.fn, std=config.initializer_range)
            if isinstance(module, HyperHead):
                nn.init.normal_(module.hc_fn, std=config.initializer_range)
        if config.pad_token_id is not None:
            with torch.no_grad():
                self.model.embed_tokens.weight[config.pad_token_id].zero_()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_hash_routes(self, layer, table):
        """Load the explicit token-to-expert table; token modulo expert count is not equivalent."""
        gate = self.model.layers[layer].mlp.gate
        if not gate.hash_route or table.shape != gate.tid2eid.shape or table.dtype != torch.long:
            raise ValueError("Hash route table shape/type mismatch")
        if (table < 0).any() or (table >= self.config.num_local_experts).any():
            raise ValueError("Invalid hash expert IDs")
        with torch.no_grad():
            gate.tid2eid.copy_(table)

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
        routing_input_ids=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one input representation")
        hidden = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        c = self.config
        if hidden.ndim != 3 or hidden.shape[-1] != c.hidden_size or hidden.shape[1] == 0:
            raise ValueError("Invalid V4 input shape")
        batch, length, _ = hidden.shape
        seen = 0
        if state is not None:
            if (
                not isinstance(state, CompressedAttentionState)
                or state.kind != "compressed_window_mqa"
                or state.model_key != self.model_key
                or len(state.layers) != c.num_hidden_layers
                or state.seen_tokens < 0
            ):
                raise ValueError("V4 compressed state type/model mismatch")
            seen = state.seen_tokens
            use_cache = True
            for layer in state.layers:
                if layer.window_kv.shape != (batch, 1, min(seen, c.sliding_window - 1), c.head_dim):
                    raise ValueError("Invalid rolling KV shape")
        if seen + length > c.max_position_embeddings:
            raise ValueError("Sequence exceeds explicit maximum context")
        if attention_mask is not None and (
            attention_mask.shape != (batch, seen + length) or not (attention_mask == 1).all()
        ):
            raise ValueError(
                "Compressed V4 reference requires unpadded contiguous sequences; group by valid length"
            )
        expected = torch.arange(seen, seen + length, device=hidden.device)[None].expand(batch, -1)
        if position_ids is not None and (
            position_ids.shape not in ((1, length), (batch, length))
            or not (position_ids == expected).all()
        ):
            raise ValueError("Compressed windows require zero-based consecutive position IDs")
        positions = expected
        ids = input_ids if routing_input_ids is None else routing_input_ids
        if ids is not None and (ids.shape != (batch, length) or ids.dtype != torch.long):
            raise ValueError("Invalid routing token IDs")
        if (
            input_ids is not None
            and routing_input_ids is not None
            and not torch.equal(input_ids, routing_input_ids)
        ):
            raise ValueError("Embedding and hash-routing token IDs disagree")
        embedding = {
            kind: self.model.rotary_emb(hidden, positions, kind) for kind in ("main", "compress")
        }
        streams = hidden[:, :, None].expand(-1, -1, c.hc_mult, -1).contiguous()
        states, history, routers, indexers = [], [], [], []
        for i, layer in enumerate(self.model.layers):
            streams, updated, router, indexer = layer(
                streams, ids, positions, embedding, state.layers[i] if state else None, seen
            )
            if use_cache:
                states.append(updated)
            if output_hidden_states:
                history.append(streams)
            routers.append(router)
            if indexer is not None:
                indexers.append(indexer)
        hidden = self.model.norm(self.model.hc_head(streams))
        if output_hidden_states:
            history.append(hidden)
        updated = (
            CompressedAttentionState(tuple(states), seen + length, self.model_key)
            if use_cache
            else None
        )
        return TokenOutput(
            self.lm_head(hidden),
            updated,
            tuple(history) if output_hidden_states else None,
            {"router": tuple(routers), "indexer": tuple(indexers)},
        )
