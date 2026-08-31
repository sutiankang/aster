"""Qwen3.8-Flash-Next text architecture with explicit Qwen4Exp components."""

from dataclasses import dataclass, field
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn import RMSNorm, RopeConfig
from aster.nn.attention import attention_mask, scaled_attention
from aster.nn.delta import GatedDeltaNet
from aster.nn.experts import PackedExperts
from .config import LlamaConfig
from .decoder import GatedMLP
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Qwen4ExpTextConfig(LlamaConfig):
    architecture: ClassVar[str] = "qwen4_exp_text"
    num_hidden_layers: int = 4
    head_dim: int = 12
    partial_rotary_factor: float = 0.5
    mrope_section: tuple[int, int, int] = (1, 1, 1)
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(theta=10000000))
    layer_types: tuple[str, ...] = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    )
    attention_bias: bool = False
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 4
    linear_value_head_dim: int = 4
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 4
    output_gate_type: str = "sigmoid"
    moe_intermediate_size: int = 16
    shared_expert_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 8
    indexer_n_heads: int = 2
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 8
    indexer_budget: int = 4
    indexer_compress_ratio: int = 2
    ngram_size: int = 3
    heads_per_ngram: int = 2
    ngram_vocab_size_base: int = 17
    make_ngram_vocab_size_divisible_by: int = 8
    ple_embed_dim: int = 32
    ple_layer_ids: tuple[int, ...] = (2,)
    ple_conv_kernel_size: int = 3
    eos_token_id: int = 2
    seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "ple_layer_ids", tuple(self.ple_layer_ids))
        object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if len(self.layer_types) != self.num_hidden_layers or any(
            x not in {"linear_attention", "qwen_sparse_attention"} for x in self.layer_types
        ):
            raise ValueError(
                "Qwen4Exp declares GDN/QSA layers explicitly, not ordinary full attention"
            )
        sizes = (
            self.hc_lowrank,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
            self.num_experts,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.indexer_n_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
            self.heads_per_ngram,
            self.ngram_vocab_size_base,
            self.make_ngram_vocab_size_divisible_by,
            self.ple_embed_dim,
            self.ple_conv_kernel_size,
        )
        if min(sizes) < 1 or self.hc_count < 2 or self.ngram_size < 2:
            raise ValueError("Invalid Flash-Next residual/indexer/PLE dimensions")
        if (
            self.linear_num_value_heads % self.linear_num_key_heads
            or not 1 <= self.num_experts_per_tok <= self.num_experts
        ):
            raise ValueError("Invalid Delta head ratio or expert top-k")
        if self.indexer_kv_heads != 1 or self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("QSA uses one raw key head and a whole-microblock budget")
        rotary = int(self.attention_head_dim * self.partial_rotary_factor)
        if (
            not 0 < self.partial_rotary_factor <= 1
            or rotary < 2
            or rotary % 2
            or rotary > self.indexer_head_dim
        ):
            raise ValueError(
                "Partial rotary dimensions must fit both attention and QSA index heads"
            )
        if (
            len(self.mrope_section) != 3
            or min(self.mrope_section) < 0
            or sum(self.mrope_section) != rotary // 2
        ):
            raise ValueError("MRoPE sections must cover the partial rotary half-width")
        if self.rope.kind != "default" or self.rope.interleaved:
            raise ValueError("This Flash-Next branch verifies default-frequency split-half MRoPE")
        if (
            self.output_gate_type not in {"sigmoid", "silu"}
            or not 0 <= self.eos_token_id < self.vocab_size
        ):
            raise ValueError("Invalid output activation or EOS token")
        if self.ple_embed_dim % ((self.ngram_size - 1) * self.heads_per_ngram):
            raise ValueError("PLE width must be divisible by all n-gram heads")
        if tuple(sorted(set(self.ple_layer_ids))) != self.ple_layer_ids or any(
            i < 1 or i > self.num_hidden_layers or self.layer_types[i - 1] != "linear_attention"
            for i in self.ple_layer_ids
        ):
            raise ValueError("PLE layer IDs are distinct one-indexed GDN layers")


class GroupRMSNorm(RMSNorm):
    def __init__(self, width, group_size, eps):
        super().__init__(width, eps, zero_centered=True)
        self.group_size = group_size

    def forward(self, hidden):
        value = hidden.float().unflatten(-1, (-1, self.group_size))
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (value.flatten(-2) * (1 + self.weight.float())).to(hidden.dtype)


class GatedResidual(nn.Module):
    def __init__(self, c, combine=True):
        super().__init__()
        self.count, self.width = c.hc_count, c.hidden_size
        size = c.hc_count * c.hidden_size
        self.hc_norm = GroupRMSNorm(size, c.hidden_size, c.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(size, c.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(c.hc_lowrank, size, bias=False)
        self.block_inject_weight = nn.Linear(size, c.hc_count, bias=False) if combine else None

    def forward(self, hidden):
        normal = self.hc_norm(hidden)
        mix = self.input_mix_weight_up(
            F.silu(self.input_mix_weight_down(normal) / self.count)
        ).sigmoid()
        value = (
            mix.unflatten(-1, (self.count, self.width))
            * normal.unflatten(-1, (self.count, self.width))
        ).mean(-2)
        if self.block_inject_weight is None:
            return value
        return value, 2 * (self.block_inject_weight(normal) / self.count).sigmoid()


def _prime_after(start, ordinal):
    def prime(n):
        return n >= 2 and (
            n == 2 or n % 2 != 0 and all(n % d for d in range(3, math.isqrt(n) + 1, 2))
        )

    for _ in range(ordinal):
        start += 1
        while not prime(start):
            start += 1
    return start


def _multipliers(vocab, ngrams, index, seed):

    mask = (1 << 64) - 1
    half = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    result = []
    for position in range(ngrams):
        x = (seed + 10007 * index + 0x9E3779B97F4A7C15 * (position + 1)) & mask
        x = (x + 0x9E3779B97F4A7C15) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        result.append(2 * ((x ^ (x >> 31)) % half) + 1)
    return torch.tensor(result, dtype=torch.long)


class NGramTable(nn.Embedding):
    def forward(self, ids):

        return super().forward(ids.to(self.weight.device)).to(ids.device)


class NGramEmbedding(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config = c
        heads = (c.ngram_size - 1) * c.heads_per_ngram
        sizes = [
            _prime_after(c.ngram_vocab_size_base - 1, index * heads + i + 1) for i in range(heads)
        ]
        offsets = [sum(sizes[:i]) for i in range(heads)]
        self.register_buffer(
            "layer_multipliers", _multipliers(c.vocab_size, c.ngram_size, index, c.seed)
        )
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(sizes))
        self.register_buffer("ngram_heads_offsets", torch.tensor(offsets))
        rows = (
            math.ceil(sum(sizes) / c.make_ngram_vocab_size_divisible_by)
            * c.make_ngram_vocab_size_divisible_by
        )
        self.ngram_embedding = NGramTable(rows, c.ple_embed_dim // heads)

    def lookup_ids(self, tokens, previous=None):
        c = self.config
        context = (
            tokens.new_full((len(tokens), c.ngram_size - 1), c.eos_token_id)
            if previous is None
            else previous
        )
        if context.shape != (len(tokens), c.ngram_size - 1):
            raise ValueError("Invalid PLE token context")
        history = torch.cat((context, tokens), -1)
        length = history.shape[1]
        positions = torch.arange(length, device=tokens.device)
        eos = torch.where(history == c.eos_token_id, positions, -1)
        boundary = (
            torch.cat((eos.new_full((len(tokens), 1), -1), eos.cummax(1).values[:, :-1]), 1) + 1
        )
        shifted = [history]
        for offset in range(1, c.ngram_size):
            source = positions - offset
            value = history[:, source.clamp_min(0)]
            shifted.append(
                torch.where((positions - boundary >= offset) & (source >= 0), value, c.eos_token_id)
            )
        hashes = []
        for degree in range(2, c.ngram_size + 1):
            value = shifted[0] * self.layer_multipliers[0]
            for offset in range(1, degree):
                value = value ^ (shifted[offset] * self.layer_multipliers[offset])
            start, end = (degree - 2) * c.heads_per_ngram, (degree - 1) * c.heads_per_ngram
            hashes.append(
                value[..., None] % self.ngram_heads_vocab_sizes[start:end]
                + self.ngram_heads_offsets[start:end]
            )
        return torch.cat(hashes, -1)[:, -tokens.shape[1] :], history[:, -(c.ngram_size - 1) :]

    def forward(self, tokens, previous=None):
        indices, context = self.lookup_ids(tokens, previous)
        return self.ngram_embedding(indices).flatten(-2), context


class PLELayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config = c
        width = c.hidden_size * c.hc_count
        self.history_size = (c.ple_conv_kernel_size - 1) * c.ngram_size
        self.ple_embedding = NGramEmbedding(c, index)
        self.key_proj = nn.Linear(c.ple_embed_dim, width, bias=False)
        self.value_proj = nn.Linear(c.ple_embed_dim, c.hidden_size, bias=False)
        for name in ("norm_key", "norm_query", "norm_conv"):
            setattr(self, name, GroupRMSNorm(width, c.hidden_size, c.rms_norm_eps))
        self.conv1d = nn.Conv1d(
            width, width, c.ple_conv_kernel_size, groups=width, dilation=c.ngram_size, bias=False
        )
        nn.init.zeros_(self.conv1d.weight)

    def forward(self, hidden, tokens, conv_history=None, token_context=None, padding=None):
        c = self.config
        embeddings, context = self.ple_embedding(tokens, token_context)
        keys = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (c.hc_count, c.hidden_size))
        queries = self.norm_query(hidden).unflatten(-1, (c.hc_count, c.hidden_size))
        similarity = (keys * queries).sum(-1, keepdim=True) / math.sqrt(c.hidden_size)

        gate = similarity.abs().clamp_min(1e-6).sqrt() * similarity.sign()
        values = (gate.sigmoid() * self.value_proj(embeddings).unsqueeze(-2)).flatten(-2)
        normal = self.norm_conv(values)
        if padding is not None:
            values = values * padding[..., None].to(values.dtype)
            normal = normal * padding[..., None].to(normal.dtype)
        inputs = normal.transpose(1, 2)
        if conv_history is None:
            conv_history = inputs.new_zeros(len(inputs), inputs.shape[1], self.history_size)
        if conv_history.shape != (*inputs.shape[:2], self.history_size):
            raise ValueError("Invalid PLE convolution history")
        extended = torch.cat((conv_history, inputs), -1)
        result = values + F.silu(self.conv1d(extended)).transpose(1, 2)
        return (
            result,
            extended[..., -self.history_size :] if self.history_size else extended[..., :0],
            context,
        )


class Qwen4Rotary(nn.Module):
    _aster_semantic_buffers = ("inv_freq",)

    def __init__(self, c):
        super().__init__()
        dimension = int(c.attention_head_dim * c.partial_rotary_factor)
        self.sections = c.mrope_section
        self.register_buffer(
            "inv_freq",
            c.rope.theta ** (-torch.arange(0, dimension, 2).float() / dimension),
            persistent=False,
        )

    def forward(self, positions, dtype):
        angles = positions.float()[..., None] * self.inv_freq
        values = angles[0].clone()
        for axis in (1, 2):
            values[..., axis : self.sections[axis] * 3 : 3] = angles[
                axis, ..., axis : self.sections[axis] * 3 : 3
            ]
        angles = torch.cat((values, values), -1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate(value, cos, sin):
    dimension = cos.shape[-1]
    rotary, rest = value[..., :dimension], value[..., dimension:]
    first, second = rotary.chunk(2, -1)
    return torch.cat((rotary * cos + torch.cat((-second, first), -1) * sin, rest), -1)


class QSAIndexer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.index_qk_proj = nn.Linear(
            c.hidden_size, (c.indexer_n_heads + 1) * c.indexer_head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(c.indexer_head_dim, c.rms_norm_eps, zero_centered=True)
        self.k_layernorm = RMSNorm(c.indexer_head_dim, c.rms_norm_eps, zero_centered=True)

    def forward(self, hidden, cos, sin, visible, previous=None):
        c = self.config
        batch, length, _ = hidden.shape
        q, raw = self.index_qk_proj(hidden).split(
            (c.indexer_n_heads * c.indexer_head_dim, c.indexer_head_dim), -1
        )
        q = self.q_layernorm(q.reshape(batch, length, c.indexer_n_heads, c.indexer_head_dim))
        q = _rotate(q, cos[:, -length:, None], sin[:, -length:, None])
        if previous is not None:
            raw = torch.cat((previous, raw), 1)
        if raw.shape != (batch, visible.shape[-1], c.indexer_head_dim):
            raise ValueError("QSA raw index-key history mismatch")
        selected = torch.zeros_like(visible)
        records = []
        for row in range(batch):
            for query in range(length):
                indices = visible[row, 0, query].nonzero().flatten()
                count = len(indices) // c.indexer_compress_ratio
                blocks = indices[: count * c.indexer_compress_ratio].reshape(
                    count, c.indexer_compress_ratio
                )
                if count:
                    pooled = raw[row, blocks].float().mean(1).to(raw.dtype)
                    pooled = self.k_layernorm(pooled)
                    keys = _rotate(pooled, cos[row, blocks[:, 0]], sin[row, blocks[:, 0]])
                    scores = (q[row, query].float() @ keys.float().T).T.relu().sum(-1) / math.sqrt(
                        c.indexer_head_dim
                    )
                    choice = scores.topk(
                        min(c.indexer_budget // c.indexer_compress_ratio, count)
                    ).indices
                    selected[row, 0, query, blocks[choice].flatten()] = True

                    records.append((row, query, blocks, scores))
                selected[row, 0, query, indices[count * c.indexer_compress_ratio :]] = True
        return selected, raw, tuple(records)


class QSAAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        dimension = c.attention_head_dim
        for name, size in (
            ("q_proj", 2 * c.num_attention_heads * dimension),
            ("k_proj", c.num_key_value_heads * dimension),
            ("v_proj", c.num_key_value_heads * dimension),
        ):
            setattr(self, name, nn.Linear(c.hidden_size, size, bias=c.attention_bias))
        self.o_proj = nn.Linear(
            c.num_attention_heads * dimension, c.hidden_size, bias=c.attention_bias
        )
        self.q_norm = RMSNorm(dimension, c.rms_norm_eps, zero_centered=True)
        self.k_norm = RMSNorm(dimension, c.rms_norm_eps, zero_centered=True)
        self.indexer = QSAIndexer(c)

    def forward(self, hidden, cos, sin, visible, previous=None):
        c = self.config
        b, s, _ = hidden.shape
        d = c.attention_head_dim
        selected, raw, records = self.indexer(
            hidden, cos, sin, visible, None if previous is None else previous[2]
        )
        q, gate = self.q_proj(hidden).reshape(b, s, c.num_attention_heads, 2 * d).chunk(2, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).reshape(b, s, c.num_key_value_heads, d)).transpose(1, 2)
        v = self.v_proj(hidden).reshape(b, s, c.num_key_value_heads, d).transpose(1, 2)
        q, k = (_rotate(x, cos[:, None, -s:], sin[:, None, -s:]) for x in (q, k))
        if previous is not None:
            expected = (b, c.num_key_value_heads, visible.shape[-1] - s, d)
            if previous[0].shape != expected or previous[1].shape != expected:
                raise ValueError("Invalid QSA dense KV leaves")
            k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)
        output = scaled_attention(
            q, k, v, selected & visible, dropout=c.attention_dropout, training=self.training
        )
        output = output.transpose(1, 2).reshape(b, s, -1) * gate.reshape(b, s, -1).sigmoid()
        return self.o_proj(output), (k, v, raw), records


class Qwen4MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c

        self.gate = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.experts = PackedExperts(
            c.num_experts, c.hidden_size, c.moe_intermediate_size, std=c.initializer_range
        )
        self.shared_expert = GatedMLP(c.hidden_size, c.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, hidden):
        c = self.config
        flat = hidden.flatten(0, 1)
        logits = self.gate(flat)
        weights, indices = logits.float().softmax(-1).topk(c.num_experts_per_tok, -1)
        if c.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(logits.dtype)
        output = (
            self.experts(flat, indices, weights)
            + self.shared_expert(flat) * self.shared_expert_gate(flat).sigmoid()
        )
        return output.reshape_as(hidden), logits


@dataclass(frozen=True)
class Qwen4LayerState:
    attention: tuple[torch.Tensor, ...]
    ple_convolution: torch.Tensor | None = None
    ple_tokens: torch.Tensor | None = None

    def map(self, function):
        return type(self)(
            tuple(function(x) for x in self.attention),
            None if self.ple_convolution is None else function(self.ple_convolution),
            None if self.ple_tokens is None else function(self.ple_tokens),
        )


@dataclass(frozen=True)
class Qwen4ExpState:
    layers: tuple[Qwen4LayerState, ...]
    position_ids: torch.Tensor
    seen_tokens: int
    model_key: str
    kind: str = "qwen4_exp_hybrid"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple(x.map(lambda y: y.clone()) for x in self.layers),
            self.position_ids.clone(),
            self.seen_tokens,
            self.model_key,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(x.map(lambda y: y.index_select(0, indices)) for x in self.layers),
            self.position_ids.index_select(1, indices),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError("Qwen4Exp recurrent/PLE states need snapshot+replay, not KV truncation")


class Qwen4Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.kind = c.layer_types[index]
        if self.kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                c, projection_layout="separate", output_gate=c.output_gate_type
            )
        else:
            self.self_attn = QSAAttention(c)
        self.ple = (
            PLELayer(c, c.ple_layer_ids.index(index + 1)) if index + 1 in c.ple_layer_ids else None
        )
        self.attn_hyper_connection, self.mlp_hyper_connection = GatedResidual(c), GatedResidual(c)
        self.mlp = Qwen4MoE(c)

    def forward(self, hidden, tokens, cos, sin, visible, padding, seen, previous):
        conv, context = None, None
        if self.ple is not None:
            value, conv, context = self.ple(
                hidden,
                tokens,
                None if previous is None else previous.ple_convolution,
                None if previous is None else previous.ple_tokens,
                None if padding is None else padding[:, -hidden.shape[1] :],
            )
            hidden = hidden + value
        mixed, weights = self.attn_hyper_connection(hidden)
        old = None if previous is None else previous.attention
        if self.kind == "linear_attention":
            value, state = self.linear_attn(mixed, old, padding, seen_tokens=seen, use_cache=True)
            records = ()
        else:
            value, state, records = self.self_attn(mixed, cos, sin, visible, old)
        hidden = hidden + (value.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        mixed, weights = self.mlp_hyper_connection(hidden)
        value, router = self.mlp(mixed)
        hidden = hidden + (value.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        return hidden, Qwen4LayerState(state, conv, context), router, records


class Qwen4ExpForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList(
            Qwen4Layer(config, i) for i in range(config.num_hidden_layers)
        )
        self.model.rotary_emb = Qwen4Rotary(config)
        self.model.hyper_connection_mixer = GatedResidual(config, combine=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        ple_input_ids=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        c = self.config
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one token or embedding input")
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != c.hidden_size or min(hidden.shape[:2]) < 1:
            raise ValueError("Expected nonempty BSH embeddings")
        b, s, _ = hidden.shape
        seen = 0
        if state is not None:
            if (
                not isinstance(state, Qwen4ExpState)
                or state.model_key != self.model_key
                or state.kind != "qwen4_exp_hybrid"
                or len(state.layers) != len(self.model.layers)
            ):
                raise ValueError("Qwen4Exp state/config mismatch")
            seen = state.seen_tokens
            if seen < 0 or state.position_ids.shape != (3, b, seen):
                raise ValueError("Invalid QSA historical positions")
        if seen + s > c.max_position_embeddings:
            raise ValueError("Sequence exceeds declared position support")
        visible = attention_mask_fn(
            b, s, seen + s, seen_tokens=seen, padding=attention_mask, device=hidden.device
        )
        if position_ids is None:
            position_ids = torch.arange(seen, seen + s, device=hidden.device)[None, None].expand(
                3, b, -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if (
            position_ids.shape != (3, b, s)
            or position_ids.dtype not in (torch.int32, torch.int64)
            or (position_ids < 0).any()
        ):
            raise ValueError("Current MRoPE positions must be integer B,S or 3,B,S")
        positions = (
            position_ids if state is None else torch.cat((state.position_ids, position_ids), -1)
        )
        cos, sin = self.model.rotary_emb(positions, hidden.dtype)
        tokens = input_ids if ple_input_ids is None else ple_input_ids
        if c.ple_layer_ids:
            if (
                tokens is None
                or tokens.shape != (b, s)
                or tokens.dtype not in (torch.int32, torch.int64)
                or (tokens < 0).any()
                or (tokens >= c.vocab_size).any()
            ):
                raise ValueError("PLE needs explicit valid original token IDs with inputs_embeds")
            if attention_mask is not None:
                tokens = torch.where(attention_mask[:, -s:].bool(), tokens, c.eos_token_id)
        hidden = hidden.repeat(1, 1, c.hc_count)
        present, states, routers, indexes = [], [], [], []
        for index, layer in enumerate(self.model.layers):
            if output_hidden_states:
                states.append(hidden)
            hidden, cached, router, records = layer(
                hidden,
                tokens,
                cos,
                sin,
                visible,
                attention_mask,
                seen,
                None if state is None else state.layers[index],
            )
            if use_cache:
                present.append(cached)
            routers.append(router)
            indexes.append(records)
        hidden = self.model.hyper_connection_mixer(hidden)
        if output_hidden_states:
            states.append(hidden)
        updated = (
            Qwen4ExpState(tuple(present), positions, seen + s, self.model_key)
            if use_cache
            else None
        )
        return TokenOutput(
            self.lm_head(hidden),
            updated,
            tuple(states) if output_hidden_states else None,
            {
                "router_logits": tuple(routers),
                "qsa_indexer": tuple(indexes),
                "qsa_layer_indices": tuple(
                    i for i, kind in enumerate(c.layer_types) if kind == "qwen_sparse_attention"
                ),
            },
        )


attention_mask_fn = attention_mask
