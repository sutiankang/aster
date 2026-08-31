"""DeepSeek MLA with low-rank latent/RoPE caches and decompressed or absorbed execution."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .normalization import RMSNorm
from .position import RotaryEmbedding
from .attention import attention_mask, scaled_attention


class LatentKVProjection(nn.Linear):
    """Keep one low-rank K/V parameter owner across decompressed and absorbed paths."""

    def __init__(self, config):
        self.heads = config.num_attention_heads
        self.key_dim, self.value_dim = config.qk_nope_head_dim, config.v_head_dim
        super().__init__(
            config.kv_lora_rank, self.heads * (self.key_dim + self.value_dim), bias=False
        )

    def forward(self, value, *, projection="expanded"):
        if projection == "expanded":
            return F.linear(value, self.weight)
        weights = self.weight.view(self.heads, self.key_dim + self.value_dim, self.in_features)
        key_weight, value_weight = weights.split((self.key_dim, self.value_dim), 1)
        if projection == "query":
            return torch.einsum("bhqd,hdr->bhqr", value, key_weight)
        if projection == "value":
            return torch.einsum("bhqr,hvr->bhqv", value, value_weight)
        raise ValueError("Unknown latent K/V projection")


class MultiheadLatentAttention(nn.Module):
    def __init__(self, config, *, skip_rope=False, latent_norm_eps=1e-6, output_gate=False):
        super().__init__()
        self.config = config
        self.skip_rope, self.output_gate = skip_rope, output_gate
        c = config
        width = c.qk_nope_head_dim + c.qk_rope_head_dim
        if c.q_lora_rank is None:
            self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * width, bias=False)
        else:
            self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=c.attention_bias)

            self.q_a_layernorm = RMSNorm(c.q_lora_rank, latent_norm_eps)
            self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_attention_heads * width, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=c.attention_bias
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, latent_norm_eps)
        self.kv_b_proj = LatentKVProjection(c)
        self.o_proj = nn.Linear(
            c.num_attention_heads * c.v_head_dim, c.hidden_size, bias=c.attention_bias
        )
        self.rope = None if skip_rope else RotaryEmbedding(c.qk_rope_head_dim, c.rope)
        if output_gate:
            self.g_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.v_head_dim, bias=False)
        self.scale = width**-0.5
        if not skip_rope and c.rope.kind != "default" and c.rope.mscale_all_dim:
            mscale = (
                0.1 * c.rope.mscale_all_dim * math.log(c.rope.factor) + 1
                if c.rope.factor > 1
                else 1
            )
            self.scale *= mscale**2

    def forward(
        self,
        hidden,
        positions,
        padding=None,
        previous=None,
        *,
        seen_tokens=0,
        use_cache=False,
        implementation="absorbed",
        visibility=None,
        return_attention=False,
    ):
        c = self.config
        batch, length, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            if c.q_lora_rank is None
            else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        )
        q = q.reshape(batch, length, c.num_attention_heads, -1).transpose(1, 2)
        q_content, q_rotary = q.split((c.qk_nope_head_dim, c.qk_rope_head_dim), -1)
        latent, key_rope = self.kv_a_proj_with_mqa(hidden).split(
            (c.kv_lora_rank, c.qk_rope_head_dim), -1
        )
        latent = self.kv_a_layernorm(latent)[:, None]
        key_rope = key_rope[:, None]
        if not self.skip_rope:
            q_rotary, key_rope = self.rope(q_rotary, positions), self.rope(key_rope, positions)
        if previous is not None:
            old_latent, old_rope = previous
            if (
                old_latent.ndim != 4
                or old_rope.ndim != 4
                or old_latent.shape[:2] != (batch, 1)
                or old_rope.shape[:3] != old_latent.shape[:3]
                or old_latent.shape[-1] != c.kv_lora_rank
                or old_rope.shape[-1] != c.qk_rope_head_dim
                or old_latent.shape[-2] != seen_tokens
            ):
                raise ValueError(
                    "MLA cache must contain compressed latent plus separate rotary key"
                )
            latent, key_rope = (
                torch.cat((old_latent, latent), -2),
                torch.cat((old_rope, key_rope), -2),
            )
        present = (latent, key_rope) if use_cache else None
        mask = attention_mask(
            batch,
            length,
            latent.shape[-2],
            seen_tokens=seen_tokens,
            padding=padding,
            device=hidden.device,
        )
        if visibility is not None:
            if visibility.dtype != torch.bool or visibility.shape != mask.shape:
                raise ValueError("Sparse visibility must be explicit boolean [B,1,Q,K]")
            mask = mask & visibility
        if implementation == "expanded":
            decoded = (
                self.kv_b_proj(latent)
                .reshape(batch, -1, c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim)
                .transpose(1, 2)
            )
            keys, values = decoded.split((c.qk_nope_head_dim, c.v_head_dim), -1)
            keys = torch.cat((keys, key_rope.expand(-1, c.num_attention_heads, -1, -1)), -1)
            attention_query, attention_key = torch.cat((q_content, q_rotary), -1), keys
            attended = scaled_attention(
                attention_query,
                attention_key,
                values,
                mask,
                scale=self.scale,
                dropout=c.attention_dropout,
                training=self.training,
            )
        elif implementation == "absorbed":
            q_latent = self.kv_b_proj(q_content, projection="query")
            attention_query, attention_key = (
                torch.cat((q_latent, q_rotary), -1),
                torch.cat((latent, key_rope), -1),
            )
            attended_latent = scaled_attention(
                attention_query,
                attention_key,
                latent,
                mask,
                scale=self.scale,
                dropout=c.attention_dropout,
                training=self.training,
            )
            attended = self.kv_b_proj(attended_latent, projection="value")
        else:
            raise ValueError("Unknown MLA implementation")
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        if self.output_gate:
            attended = attended * self.g_proj(hidden).sigmoid()
        result = self.o_proj(attended), present
        if not return_attention:
            return result

        with torch.no_grad(), torch.autocast(hidden.device.type, enabled=False):
            scores = attention_query.float() @ attention_key.float().transpose(-1, -2) * self.scale
            scores = scores.masked_fill(~mask, -torch.inf)
            valid = mask.any(-1, keepdim=True)
            probabilities = torch.where(valid, scores, torch.zeros_like(scores)).softmax(-1)
            probabilities = probabilities.masked_fill(~mask, 0)
        return *result, probabilities
