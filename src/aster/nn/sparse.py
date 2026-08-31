"""DeepSeek sparse indexers and compressed state with explicit visibility."""

from dataclasses import dataclass, replace
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import StateCapabilities
from .normalization import LayerNorm
from .position import RotaryEmbedding


@dataclass(frozen=True)
class IndexedMLAState:
    layers: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    kind: str = "indexed_mla"

    @property
    def capabilities(self):
        return StateCapabilities(
            self.kind, forkable=True, truncatable=True, reorderable=True, replayable=True
        )

    def fork(self):
        return type(self)(
            tuple(tuple(x.clone() for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(tuple(x.index_select(0, indices) for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        if not 0 <= length <= self.seen_tokens:
            raise ValueError("Invalid indexed state truncation")
        return type(self)(
            tuple(tuple(x[..., :length, :].clone() for x in layer) for layer in self.layers),
            length,
            self.model_key,
        )


class LightningIndexer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.wq_b = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
        self.wk = nn.Linear(c.hidden_size, c.index_head_dim, bias=False)
        self.k_norm = LayerNorm(c.index_head_dim, 1e-6)
        self.weights_proj = nn.Linear(c.hidden_size, c.index_n_heads, bias=False)
        self.rope = RotaryEmbedding(c.qk_rope_head_dim, replace(c.rope, interleaved=False))

    def forward(self, hidden, query_latent, positions, visible, previous=None):
        c = self.config
        b, length, _ = hidden.shape
        query = (
            self.wq_b(query_latent)
            .reshape(b, length, c.index_n_heads, c.index_head_dim)
            .transpose(1, 2)
        )
        key = self.k_norm(self.wk(hidden))[:, None]
        rot = c.qk_rope_head_dim
        query = torch.cat((self.rope(query[..., :rot], positions), query[..., rot:]), -1).transpose(
            1, 2
        )
        key = torch.cat((self.rope(key[..., :rot], positions), key[..., rot:]), -1)
        if previous is not None:
            key = torch.cat((previous, key), -2)
        scores = F.relu(
            (query.float() @ key[:, 0].float().transpose(-1, -2).unsqueeze(1))
            * c.index_head_dim**-0.5
        )
        weights = (
            self.weights_proj(hidden.to(self.weights_proj.weight.dtype)).float()
            * c.index_n_heads**-0.5
        )
        scores = (weights.unsqueeze(-2) @ scores).squeeze(-2)
        scores = scores.masked_fill(~visible[:, 0], -torch.inf)

        indices = scores.detach().topk(min(c.index_topk, key.shape[-2]), -1).indices
        selected = torch.zeros_like(visible).scatter(-1, indices[:, None], True) & visible
        return selected, key, {"scores": scores, "visible": visible[:, 0], "indices": indices}
