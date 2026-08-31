"""LLaDA bidirectional mask prediction over Llama-style blocks."""

from dataclasses import asdict, dataclass
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn import RMSNorm, RopeConfig, RotaryEmbedding
from aster.nn.attention import attention_mask, scaled_attention
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class LLaDAConfig:
    architecture: ClassVar[str] = "llada"
    d_model: int = 32
    n_heads: int = 4
    n_kv_heads: int = 4
    n_layers: int = 2
    mlp_hidden_size: int = 64
    vocab_size: int = 32
    embedding_size: int | None = None
    max_sequence_length: int = 128
    mask_token_id: int = 31
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_full_precision: bool = True
    include_bias: bool = False
    include_qkv_bias: bool = False
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    weight_tying: bool = False
    input_emb_norm: bool = False
    scale_logits: bool = False

    def __post_init__(self):
        if self.embedding_size is None:
            object.__setattr__(self, "embedding_size", self.vocab_size)
        if (
            min(
                self.d_model,
                self.n_heads,
                self.n_kv_heads,
                self.n_layers,
                self.mlp_hidden_size,
                self.vocab_size,
                self.embedding_size,
                self.max_sequence_length,
            )
            < 1
        ):
            raise ValueError("Invalid LLaDA dimensions")
        if (
            self.d_model % self.n_heads
            or self.n_heads % self.n_kv_heads
            or (self.d_model // self.n_heads) % 2
        ):
            raise ValueError("Invalid LLaDA GQA/RoPE dimensions")
        if (
            self.embedding_size < self.vocab_size
            or not 0 <= self.mask_token_id < self.embedding_size
        ):
            raise ValueError("Invalid LLaDA embedding or MASK vocabulary")
        if min(self.rms_norm_eps, self.rope_theta) <= 0 or any(
            not 0 <= p < 1
            for p in (self.attention_dropout, self.residual_dropout, self.embedding_dropout)
        ):
            raise ValueError("Invalid LLaDA numerics")

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_hidden_layers(self):
        return self.n_layers

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class LLaDABlock(nn.Module):
    def __init__(self, c, layer_index):
        super().__init__()
        self.config = c
        dim = c.d_model // c.n_heads
        self.attn_norm = RMSNorm(c.d_model, c.rms_norm_eps)
        self.ff_norm = RMSNorm(c.d_model, c.rms_norm_eps)
        bias = c.include_bias or c.include_qkv_bias
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=bias)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * dim, bias=bias)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * dim, bias=bias)
        self.attn_out = nn.Linear(c.d_model, c.d_model, bias=c.include_bias)
        self.ff_proj = nn.Linear(c.d_model, c.mlp_hidden_size, bias=c.include_bias)
        self.up_proj = nn.Linear(c.d_model, c.mlp_hidden_size, bias=c.include_bias)
        self.ff_out = nn.Linear(c.mlp_hidden_size, c.d_model, bias=c.include_bias)
        self.rotary_emb = RotaryEmbedding(dim, RopeConfig(theta=c.rope_theta))
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.ff_proj, self.up_proj):
            _mitchell(projection, c.d_model)
        _mitchell(self.attn_out, c.d_model, layer_index)
        _mitchell(self.ff_out, c.mlp_hidden_size, layer_index)

    def forward(self, hidden, positions, mask):
        c = self.config
        b, length, _ = hidden.shape
        normalized = self.attn_norm(hidden)

        def split(proj, heads):
            return proj(normalized).reshape(b, length, heads, -1).transpose(1, 2)

        query, key, value = (
            split(self.q_proj, c.n_heads),
            split(self.k_proj, c.n_kv_heads),
            split(self.v_proj, c.n_kv_heads),
        )
        dtype = query.dtype
        if c.rope_full_precision:
            query, key = query.float(), key.float()
        query, key = (
            self.rotary_emb(query, positions).to(dtype),
            self.rotary_emb(key, positions).to(dtype),
        )
        attended = scaled_attention(
            query, key, value, mask, dropout=c.attention_dropout, training=self.training
        )
        hidden = hidden + F.dropout(
            self.attn_out(attended.transpose(1, 2).reshape(b, length, c.d_model)),
            c.residual_dropout,
            self.training,
        )
        normalized = self.ff_norm(hidden)

        update = self.ff_out(F.silu(self.ff_proj(normalized)) * self.up_proj(normalized))
        return hidden + F.dropout(update, c.residual_dropout, self.training)


def _mitchell(module, input_size, layer_index=None, scale=1.0):
    std = scale / math.sqrt(input_size)
    if layer_index is not None:
        std /= math.sqrt(2 * (layer_index + 1))
    nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


class LLaDAForMaskedLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        backbone = self.model.transformer
        backbone.wte = nn.Embedding(c.embedding_size, c.d_model)
        backbone.blocks = nn.ModuleList(LLaDABlock(c, i) for i in range(c.n_layers))
        backbone.ln_f = RMSNorm(c.d_model, c.rms_norm_eps)
        if not c.weight_tying:
            backbone.ff_out = nn.Linear(c.d_model, c.embedding_size, bias=c.include_bias)
            _mitchell(backbone.ff_out, c.d_model)
        _mitchell(
            backbone.wte, c.d_model, scale=0.5 * math.sqrt(c.d_model) if c.scale_logits else 1.0
        )

    def get_input_embeddings(self):
        return self.model.transformer.wte

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
        if state is not None or use_cache:
            raise ValueError(
                "LLaDA bidirectional denoising invalidates token states; KV cache is not supported"
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one input representation")
        c = self.config
        backbone = self.model.transformer
        hidden = backbone.wte(input_ids) if inputs_embeds is None else inputs_embeds
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != c.d_model
            or not 0 < hidden.shape[1] <= c.max_sequence_length
        ):
            raise ValueError("Invalid LLaDA input dimensions/context")
        b, length, _ = hidden.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device)[None].expand(b, -1)
        if position_ids.shape != (b, length) or (position_ids < 0).any():
            raise ValueError("Invalid LLaDA positions")
        if c.input_emb_norm:
            hidden = hidden * math.sqrt(c.d_model)
        hidden = F.dropout(hidden, c.embedding_dropout, self.training)
        from aster.nn.attention import attention_mask as make_mask

        visible = make_mask(
            b, length, length, padding=attention_mask, device=hidden.device, causal=False
        )
        states = []
        for block in backbone.blocks:
            if output_hidden_states:
                states.append(hidden)
            hidden = block(hidden, position_ids, visible)
        hidden = backbone.ln_f(hidden)
        if output_hidden_states:
            states.append(hidden)
        logits = (
            F.linear(hidden, backbone.wte.weight) if c.weight_tying else backbone.ff_out(hidden)
        )
        if c.scale_logits:
            logits = logits / math.sqrt(c.d_model)
        return TokenOutput(logits, None, tuple(states) if output_hidden_states else None)
