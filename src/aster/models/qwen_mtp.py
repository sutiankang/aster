"""Shared-backbone Qwen multi-token heads with sequential depth-specific attention."""

from dataclasses import dataclass, field, replace
from typing import ClassVar
import torch
from torch import nn
from aster.core import TokenOutput, StateCapabilities
from aster.nn import RMSNorm
from .qwen35 import Qwen35TextConfig, Qwen35MoETextConfig, Qwen35ForCausalLM, Qwen35Layer
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class QwenMTPConfig:
    architecture: ClassVar[str] = "qwen_mtp"
    text_config: Qwen35TextConfig = field(default_factory=Qwen35TextConfig)
    num_mtp_layers: int = 1
    share_embeddings: bool = True

    def __post_init__(self):
        if type(self.text_config) not in {Qwen35TextConfig, Qwen35MoETextConfig}:
            raise ValueError(
                "Qwen MTP requires actual native Qwen3.5-compatible dense/MoE text architecture"
            )
        if (
            type(self.num_mtp_layers) is not int
            or self.num_mtp_layers < 1
            or type(self.share_embeddings) is not bool
        ):
            raise ValueError("Invalid Qwen MTP layer/sharing configuration")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "num_mtp_layers": self.num_mtp_layers,
            "share_embeddings": self.share_embeddings,
        }


@dataclass(frozen=True)
class QwenMTPState:
    kv: tuple[torch.Tensor, torch.Tensor]
    seen_tokens: int
    layer_index: int
    model_key: str
    kind: str = "qwen_mtp_draft_kv"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, truncatable=True)

    def fork(self):
        return type(self)(
            tuple(x.clone() for x in self.kv), self.seen_tokens, self.layer_index, self.model_key
        )

    def reorder(self, indices):
        return type(self)(
            tuple(x.index_select(0, indices) for x in self.kv),
            self.seen_tokens,
            self.layer_index,
            self.model_key,
        )

    def truncate(self, length):
        if type(length) is not int or not 0 <= length <= self.seen_tokens:
            raise ValueError("Invalid draft cache rollback")
        return type(self)(
            tuple(x[..., :length, :].clone() for x in self.kv),
            length,
            self.layer_index,
            self.model_key,
        )


class QwenMTPHead(nn.Module):
    """Combine h_t and the known/draft token_(t+1) to predict token_(t+2)."""

    def __init__(self, config, embeddings, lm_head):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config) + ":mtp"
        c = config.text_config
        self.embed_tokens, self.lm_head = embeddings, lm_head
        self.fc = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True)
        self.pre_fc_norm_hidden = RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True)
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps, zero_centered=True)
        self.layer_config = replace(
            c,
            num_hidden_layers=config.num_mtp_layers,
            layer_types=("full_attention",) * config.num_mtp_layers,
        )
        self.layers = nn.ModuleList(
            Qwen35Layer(self.layer_config, index) for index in range(config.num_mtp_layers)
        )
        nn.init.normal_(self.fc.weight, std=c.initializer_range)

        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=c.initializer_range)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids=None,
        *,
        hidden_states,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        spec_step_idx=0,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("MTP needs exactly one next-token ID or embedding input")
        if type(spec_step_idx) is not int or spec_step_idx < 0:
            raise ValueError("MTP speculative step must be a nonnegative integer")
        index = spec_step_idx % len(self.layers)
        c = self.config.text_config
        embeddings = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if (
            hidden_states.shape != embeddings.shape
            or embeddings.ndim != 3
            or embeddings.shape[-1] != c.hidden_size
            or not embeddings.shape[1]
        ):
            raise ValueError(
                "Base/draft hidden states and next-token embeddings must align [B,S,H]"
            )
        if state is not None and (
            not isinstance(state, QwenMTPState)
            or state.model_key != self.model_key
            or state.layer_index != index
        ):
            raise ValueError(
                "Draft state must match this MTP head and selected layer; never reuse base/hybrid KV"
            )
        seen = 0 if state is None else state.seen_tokens
        b, s = embeddings.shape[:2]
        if seen < 0 or seen + s > c.max_position_embeddings:
            raise ValueError("Draft context exceeds declared capacity")
        if position_ids is None:
            position_ids = torch.arange(seen, seen + s, device=embeddings.device)[None].expand(
                b, -1
            )
        if (
            position_ids.shape not in {(b, s), (3, b, s)}
            or position_ids.dtype not in {torch.int32, torch.int64}
            or (position_ids < 0).any()
        ):
            raise ValueError("MTP requires integer current token coordinates[B,S] or MRoPE[3,B,S]")

        merged = torch.cat(
            (self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden_states)), -1
        )
        merged = self.fc(merged)
        value, present, extra = self.layers[index](
            merged,
            position_ids,
            attention_mask,
            None if state is None else state.kv,
            seen,
            use_cache,
        )
        value = self.norm(value)
        updated = QwenMTPState(present, seen + s, index, self.model_key) if use_cache else None
        return TokenOutput(
            self.lm_head(value),
            updated,
            (merged, value) if output_hidden_states else None,
            {"draft_hidden": value, "layer_index": index, "router": extra},
        )


class QwenMTPForCausalLM(LocalModelMixin, nn.Module):
    """Keep the backbone and MTP heads under one model and optimizer ownership graph."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = Qwen35ForCausalLM(config.text_config)
        c = config.text_config
        if config.share_embeddings:
            embeddings = nn.Embedding(c.vocab_size, c.hidden_size, device="meta")
            head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, device="meta")
            embeddings.weight = self.backbone.get_input_embeddings().weight
            head.weight = self.backbone.lm_head.weight
        else:
            embeddings, head = (
                nn.Embedding(c.vocab_size, c.hidden_size),
                nn.Linear(c.hidden_size, c.vocab_size, bias=False),
            )
            nn.init.normal_(embeddings.weight, std=c.initializer_range)
            nn.init.normal_(head.weight, std=c.initializer_range)
            if c.tie_word_embeddings:
                head.weight = embeddings.weight
        self.mtp = QwenMTPHead(config, embeddings, head)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def get_decoder(self):
        return self.backbone.get_decoder()

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
        mtp_depth=0,
        detach_mtp_base=False,
    ):
        if type(mtp_depth) is not int or mtp_depth < 0:
            raise ValueError("MTP training depth must be a nonnegative integer")
        if mtp_depth and (
            input_ids is None
            or inputs_embeds is not None
            or state is not None
            or use_cache
            or input_ids.shape[1] < mtp_depth + 2
        ):
            raise ValueError(
                "MTP teacher forcing requires uncached complete token IDs with at least depth+2 tokens"
            )
        output = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states or mtp_depth > 0,
        )
        if not mtp_depth:
            return output
        hidden = output.hidden_states[-1]
        if detach_mtp_base:
            hidden = hidden.detach()
        predictions, offsets = [], []
        for step in range(mtp_depth):
            offset = step + 1
            ids = input_ids[:, offset:]
            padding = None if attention_mask is None else attention_mask[:, offset:]
            positions = (
                torch.arange(offset, input_ids.shape[1], device=input_ids.device)[None].expand_as(
                    ids
                )
                if position_ids is None
                else position_ids[..., offset:]
            )
            prediction = self.mtp(
                ids,
                hidden_states=hidden[:, :-1],
                attention_mask=padding,
                position_ids=positions,
                spec_step_idx=step,
            )
            hidden = prediction.auxiliary["draft_hidden"]
            predictions.append(prediction.logits)
            offsets.append(offset)
        return TokenOutput(
            output.logits,
            None,
            output.hidden_states if output_hidden_states else None,
            {
                **(output.auxiliary or {}),
                "mtp_logits": tuple(predictions),
                "mtp_offsets": tuple(offsets),
            },
        )
