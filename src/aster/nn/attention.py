"""Shared attention and immutable state on one absolute prefill/decode timeline."""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from aster.core.contracts import StateCapabilities
from .normalization import RMSNorm
from .position import RotaryEmbedding


@dataclass(frozen=True)
class KVState:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    kind: str = "dense_kv"

    @property
    def capabilities(self):
        return StateCapabilities(
            self.kind,
            forkable=True,
            truncatable=self.kind in {"dense_kv", "mla_latent"},
            reorderable=True,
        )

    def fork(self):

        return type(self)(
            tuple((k.clone(), v.clone()) for k, v in self.layers),
            self.seen_tokens,
            self.model_key,
            self.kind,
        )

    def reorder(self, indices):
        return type(self)(
            tuple((k.index_select(0, indices), v.index_select(0, indices)) for k, v in self.layers),
            self.seen_tokens,
            self.model_key,
            self.kind,
        )

    def truncate(self, length):
        if not self.capabilities.truncatable or not 0 <= length <= self.seen_tokens:
            raise ValueError("State cannot be truncated to this position; replay from a checkpoint")
        return type(self)(
            tuple((k[..., :length, :].clone(), v[..., :length, :].clone()) for k, v in self.layers),
            length,
            self.model_key,
            self.kind,
        )


def attention_mask(
    batch,
    query_length,
    key_length,
    *,
    seen_tokens=0,
    window=None,
    padding=None,
    device=None,
    causal=True,
):
    """Use True for visible positions and an absolute cache timeline.
    Padding and causality are independent constraints."""
    key_start = seen_tokens + query_length - key_length
    query_positions = torch.arange(seen_tokens, seen_tokens + query_length, device=device)
    key_positions = torch.arange(key_start, seen_tokens + query_length, device=device)
    visible = (
        (key_positions[None] <= query_positions[:, None])
        if causal
        else torch.ones(query_length, key_length, device=device, dtype=torch.bool)
    )
    if window is not None:
        visible &= key_positions[None] > query_positions[:, None] - window
    visible = visible[None, None].expand(batch, 1, -1, -1)
    if padding is not None:
        if padding.shape != (batch, seen_tokens + query_length):
            raise ValueError("Padding mask must cover all past/current token positions")
        if not ((padding == 0) | (padding == 1)).all():
            raise ValueError("Padding mask contains values other than zero/one")
        visible = visible & padding[:, None, None, key_start:].bool()
    return visible


def scaled_attention(
    q, k, v, mask, *, scale=None, dropout=0.0, training=False, softmax_in_fp32=True
):
    """Explicit softmax reference. Fully masked query rows return zero instead of NaN."""
    if q.shape[1] % k.shape[1] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Incompatible query/KV groups")
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    scores = (q @ k.transpose(-1, -2)) * (scale if scale is not None else q.shape[-1] ** -0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))

    probabilities = (
        F.softmax(scores.float() if softmax_in_fp32 else scores, -1)
        .to(q.dtype)
        .masked_fill(~mask, 0)
    )
    return F.dropout(probabilities, p=dropout, training=training) @ v


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        rope,
        *,
        qkv_bias=False,
        output_bias=False,
        qk_norm=False,
        eps=1e-6,
        dropout=0.0,
        window=None,
    ):
        super().__init__()
        if num_heads % num_kv_heads or min(hidden_size, num_heads, num_kv_heads, head_dim) < 1:
            raise ValueError("Attention dimensions/group ratio invalid")
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.dropout, self.window = dropout, window
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=output_bias)
        self.q_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(head_dim, rope)

    def forward(
        self, hidden, positions, padding=None, previous=None, *, seen_tokens=0, use_cache=False
    ):
        batch, length, _ = hidden.shape

        def split(value, heads):
            return value.reshape(batch, length, heads, self.head_dim).transpose(1, 2)

        q = self.rope(self.q_norm(split(self.q_proj(hidden), self.num_heads)), positions)
        k = self.rope(self.k_norm(split(self.k_proj(hidden), self.num_kv_heads)), positions)
        v = split(self.v_proj(hidden), self.num_kv_heads)
        if previous is not None:
            pk, pv = previous
            expected = (batch, self.num_kv_heads)
            expected_length = (
                seen_tokens if self.window is None else min(seen_tokens, max(self.window - 1, 0))
            )
            if (
                pk.ndim != 4
                or pk.shape != pv.shape
                or pk.shape[:2] != expected
                or pk.shape[-1] != self.head_dim
                or pk.shape[-2] != expected_length
            ):
                raise ValueError("KV tensor layout differs from this attention layer")
            k, v = torch.cat((pk, k), -2), torch.cat((pv, v), -2)
        provider = getattr(self, "attention_backend", None)
        if provider is None:
            mask = attention_mask(
                batch,
                length,
                k.shape[-2],
                seen_tokens=seen_tokens,
                window=self.window,
                padding=padding,
                device=hidden.device,
            )
            result = scaled_attention(
                q, k, v, mask, scale=self.scale, dropout=self.dropout, training=self.training
            )
        else:
            provider.validate_attention(self)

            key_start = seen_tokens + length - k.shape[-2]
            if padding is not None:
                if padding.shape != (batch, seen_tokens + length):
                    raise ValueError("Padding mask must cover all past/current token positions")
                if padding.requires_grad or not ((padding == 0) | (padding == 1)).all():
                    raise ValueError(
                        "Padding mask must be fixed binary data on the full physical axis"
                    )
            result = provider(
                q,
                k,
                v,
                query_positions=torch.arange(
                    seen_tokens, seen_tokens + length, device=hidden.device
                ).expand(batch, -1),
                key_offset=key_start,
                key_padding=None if padding is None else padding[:, key_start:],
                window=self.window,
                scale=self.scale,
                dropout=self.dropout,
            )

        present = (k, v) if use_cache else None
        if use_cache and self.window is not None:
            keep = max(self.window - 1, 0)
            present = (
                (k[..., -keep:, :], v[..., -keep:, :]) if keep else (k[..., :0, :], v[..., :0, :])
            )
        return self.o_proj(result.transpose(1, 2).reshape(batch, length, -1)), present
