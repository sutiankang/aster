"""DeepSeek V3.2 MLA with learned top-k lightning-indexer visibility."""

from dataclasses import dataclass
from typing import ClassVar
import torch
from aster.nn.attention import attention_mask
from aster.nn.latent_attention import MultiheadLatentAttention
from aster.nn.sparse import IndexedMLAState, LightningIndexer
from .config import DeepSeekV3Config
from .decoder import CausalLM
from .moe import DeepSeekLayer


@dataclass(frozen=True)
class DeepSeekV32Config(DeepSeekV3Config):
    architecture: ClassVar[str] = "deepseek_v32"
    index_topk: int = 3
    index_head_dim: int = 8
    index_n_heads: int = 4

    def __post_init__(self):
        super().__post_init__()
        if (
            self.q_lora_rank is None
            or min(self.index_topk, self.index_head_dim, self.index_n_heads) < 1
            or self.index_head_dim < self.qk_rope_head_dim
        ):
            raise ValueError(
                "DSA indexer requires query latent and a valid rotary/content head split"
            )
        if not self.rope.interleaved:
            raise ValueError(
                "V3.2 MLA checkpoint uses interleaved RoPE; its indexer independently uses half-split"
            )


class SparseMLA(MultiheadLatentAttention):
    def __init__(self, c):
        super().__init__(c)
        self.indexer = LightningIndexer(c)

        self.indexer_stage = None

    def forward(
        self, hidden, positions, padding=None, previous=None, *, seen_tokens=0, use_cache=False
    ):
        c = self.config
        b, length, _ = hidden.shape
        if self.indexer_stage is not None and (use_cache or previous is not None):
            raise ValueError(
                "DSA training stage forbids inference cache; export a fresh deployment model"
            )
        if previous is not None:
            if len(previous) != 3 or previous[2].shape != (b, 1, seen_tokens, c.index_head_dim):
                raise ValueError("Indexed MLA state must include aligned indexer key history")
        visible = attention_mask(
            b,
            length,
            seen_tokens + length,
            seen_tokens=seen_tokens,
            padding=padding,
            device=hidden.device,
        )

        with torch.no_grad():
            query_latent = self.q_a_layernorm(self.q_a_proj(hidden))
        selected, keys, info = self.indexer(
            hidden.detach(),
            query_latent,
            positions,
            visible,
            previous[2] if previous is not None else None,
        )
        training_visible = visible if self.indexer_stage == "dense_warmup" else selected
        result = super().forward(
            hidden,
            positions,
            padding,
            previous[:2] if previous is not None else None,
            seen_tokens=seen_tokens,
            use_cache=use_cache,
            visibility=training_visible,
            return_attention=self.indexer_stage is not None,
        )
        value, present = result[:2]
        if self.indexer_stage is not None:
            info = {
                **info,
                "teacher_probabilities": result[2],
                "training_visible": training_visible[:, 0],
            }
        return value, (*present, keys) if use_cache else None, info


class DSALayer(DeepSeekLayer):
    def __init__(self, c, index):
        super().__init__(c, index)
        self.self_attn = SparseMLA(c)

    def forward(self, hidden, positions, padding, previous, seen_tokens, use_cache):
        update, present, info = self.self_attn(
            self.input_layernorm(hidden),
            positions,
            padding,
            previous,
            seen_tokens=seen_tokens,
            use_cache=use_cache,
        )
        hidden = hidden + update
        result = self.mlp(self.post_attention_layernorm(hidden))
        update, routing = result if self.sparse else (result, None)
        return hidden + update, present, {"routing": routing, "indexer": info}


class DeepSeekV32ForCausalLM(CausalLM):
    layer_type = DSALayer
    state_kind = "indexed_mla"
    state_type = IndexedMLAState

    def create_state(self, layers, seen, kind):
        return IndexedMLAState(layers, seen, self.model_key, kind)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        entries = output.auxiliary["router"]
        output.auxiliary = {
            "router": tuple(x["routing"] for x in entries if x["routing"] is not None),
            "indexer": tuple(x["indexer"] for x in entries),
        }
        return output
