"""Qwen3 DSpark drafts with parallel backbone prediction and compact autoregressive heads."""

from dataclasses import dataclass
import hashlib
from typing import ClassVar

import torch
from torch import nn
import torch.nn.functional as F

from ..nn.attention import scaled_attention
from ..nn.markov import MarkovHead
from ..nn.normalization import RMSNorm
from ..nn.position import RotaryEmbedding
from .config import Qwen3Config, config_from_dict
from .decoder import GatedMLP
from .serialization import LocalModelMixin, semantic_buffers
from ..core import digest_json


def target_state_identity(model):

    digest = hashlib.sha256(digest_json(model.config.to_dict()).encode())
    state = {
        **{"weight/" + k: v for k, v in model.state_dict().items()},
        **{"runtime/" + k: v for k, v in semantic_buffers(model).items()},
    }
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(
            digest_json(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DSparkConfig:
    architecture: ClassVar[str] = "dspark_qwen3"
    target: Qwen3Config
    num_draft_layers: int = 1
    target_layer_ids: tuple[int, ...] = (-1,)
    block_size: int = 7
    num_anchors: int = 32
    mask_token_id: int = 0
    markov_rank: int = 128
    markov_head_type: str = "gated"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    freeze_embedding_head: bool = True

    def __post_init__(self):
        if isinstance(self.target, dict):
            object.__setattr__(self, "target", config_from_dict(self.target))
        object.__setattr__(self, "target_layer_ids", tuple(self.target_layer_ids))
        if type(self.target) is not Qwen3Config:
            raise ValueError("DSpark Qwen3 profile requires exact Qwen3 target config")
        if any(
            type(x) is not int or x < 1
            for x in (self.num_draft_layers, self.block_size, self.num_anchors)
        ):
            raise ValueError("DSpark layer/block/anchor counts must be positive integers")
        if (
            not self.target_layer_ids
            or any(
                type(x) is not int or x < -1 or x >= self.target.num_hidden_layers
                for x in self.target_layer_ids
            )
            or tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids
        ):
            raise ValueError(
                "DSpark target layer IDs must be strictly increasing and in {-1} union target layers"
            )
        if (
            type(self.mask_token_id) is not int
            or not 0 <= self.mask_token_id < self.target.vocab_size
        ):
            raise ValueError("DSpark mask token must have explicit target vocabulary semantics")
        if (
            type(self.markov_rank) is not int
            or self.markov_rank < 0
            or self.markov_head_type not in {"vanilla", "gated", "rnn"}
        ):
            raise ValueError("Invalid DSpark Markov head")
        if any(
            type(x) is not bool
            for x in (
                self.enable_confidence_head,
                self.confidence_head_with_markov,
                self.freeze_embedding_head,
            )
        ):
            raise ValueError("DSpark flags must be booleans")
        if (
            self.enable_confidence_head
            and self.confidence_head_with_markov
            and not self.markov_rank
        ):
            raise ValueError("Confidence with Markov embedding requires a Markov head")

    def to_dict(self):
        return dict(
            architecture=self.architecture,
            target=self.target.to_dict(),
            num_draft_layers=self.num_draft_layers,
            target_layer_ids=list(self.target_layer_ids),
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            mask_token_id=self.mask_token_id,
            markov_rank=self.markov_rank,
            markov_head_type=self.markov_head_type,
            enable_confidence_head=self.enable_confidence_head,
            confidence_head_with_markov=self.confidence_head_with_markov,
            freeze_embedding_head=self.freeze_embedding_head,
        )


def sample_anchors(loss_mask, num_anchors, *, generator=None):

    valid = loss_mask[:, :-1].bool() & loss_mask[:, 1:].bool()
    b, candidates = valid.shape
    if not candidates:
        return (
            torch.zeros(b, num_anchors, dtype=torch.long, device=loss_mask.device),
            torch.zeros(b, num_anchors, dtype=torch.bool, device=loss_mask.device),
        )
    scores = torch.rand(b, candidates, device=loss_mask.device, generator=generator).masked_fill(
        ~valid, 2.0
    )
    positions = (
        torch.arange(candidates, device=loss_mask.device)[None]
        .expand(b, -1)
        .masked_fill(~valid, loss_mask.shape[1] + 1)
    )
    chosen = positions.gather(1, scores.argsort(1))
    if candidates < num_anchors:
        chosen = torch.cat(
            (chosen, chosen.new_full((b, num_anchors - candidates), loss_mask.shape[1] + 1)), 1
        )
    chosen = chosen[:, :num_anchors].sort(1).values
    keep = torch.arange(num_anchors, device=loss_mask.device)[None] < valid.sum(1, keepdim=True)
    return chosen.masked_fill(~keep, 0), keep


def block_attention_mask(anchors, keep, seq_len, block_size):
    blocks = torch.arange(anchors.shape[1], device=anchors.device).repeat_interleave(block_size)
    context = torch.arange(seq_len, device=anchors.device)[None, None] < anchors[:, blocks, None]
    draft = (blocks[:, None] == blocks[None, :])[None].expand(len(anchors), -1, -1)
    return (torch.cat((context, draft), -1) & keep[:, blocks, None])[:, None]


class DSparkAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.hk, self.d = c.num_attention_heads, c.num_key_value_heads, c.attention_head_dim
        self.dropout = c.attention_dropout
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.hk * self.d, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.hk * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d, c.rms_norm_eps), RMSNorm(self.d, c.rms_norm_eps)

    def forward(self, hidden, context, positions, mask, rotary):
        b, length, _ = hidden.shape

        def split(x, heads):
            return x.reshape(b, -1, heads, self.d).transpose(1, 2)

        q = rotary(self.q_norm(split(self.q_proj(hidden), self.h)), positions[:, -length:])

        k = torch.cat((self.k_proj(context), self.k_proj(hidden)), 1)
        v = torch.cat((self.v_proj(context), self.v_proj(hidden)), 1)
        k, v = rotary(self.k_norm(split(k, self.hk)), positions), split(v, self.hk)
        output = scaled_attention(q, k, v, mask, dropout=self.dropout, training=self.training)
        return self.o_proj(output.transpose(1, 2).reshape(b, length, -1))

    def forward_cached(self, hidden, context, positions, rotary, past=None):
        b, length, _ = hidden.shape

        def split(x, heads):
            return x.reshape(b, -1, heads, self.d).transpose(1, 2)

        q = rotary(self.q_norm(split(self.q_proj(hidden), self.h)), positions[:, -length:])
        k = torch.cat((self.k_proj(context), self.k_proj(hidden)), 1)
        v = torch.cat((self.v_proj(context), self.v_proj(hidden)), 1)
        k, v = rotary(self.k_norm(split(k, self.hk)), positions), split(v, self.hk)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)

        visible = torch.ones(b, 1, length, k.shape[2], device=q.device, dtype=torch.bool)
        output = scaled_attention(q, k, v, visible)
        return self.o_proj(output.transpose(1, 2).reshape(b, length, -1)), (
            k[:, :, :-length],
            v[:, :, :-length],
        )


class DSparkLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = (
            RMSNorm(c.hidden_size, c.rms_norm_eps),
            RMSNorm(c.hidden_size, c.rms_norm_eps),
        )
        self.self_attn, self.mlp = DSparkAttention(c), GatedMLP(c.hidden_size, c.intermediate_size)

    def forward(self, hidden, context, positions, mask, rotary):
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), context, positions, mask, rotary
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


@dataclass
class DSparkOutput:
    draft_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_pred: torch.Tensor | None
    aligned_target_logits: torch.Tensor | None
    anchor_positions: torch.Tensor


class _FrozenEmbedding(nn.Module):
    def __init__(self, size, width):
        super().__init__()
        self.register_buffer("weight", torch.empty(size, width))

    def forward(self, ids):
        return F.embedding(ids, self.weight)


class _FrozenHead(nn.Module):
    def __init__(self, width, size):
        super().__init__()
        self.register_buffer("weight", torch.empty(size, width))

    def forward(self, hidden):

        return F.linear(hidden, self.weight)


class DSparkDraft(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, c = config, config.target
        self.embed_tokens = (
            _FrozenEmbedding(c.vocab_size, c.hidden_size)
            if config.freeze_embedding_head
            else nn.Embedding(c.vocab_size, c.hidden_size)
        )
        self.layers = nn.ModuleList(DSparkLayer(c) for _ in range(config.num_draft_layers))
        self.norm, self.hidden_norm = (
            RMSNorm(c.hidden_size, c.rms_norm_eps),
            RMSNorm(c.hidden_size, c.rms_norm_eps),
        )
        self.fc = nn.Linear(len(config.target_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.lm_head = (
            _FrozenHead(c.hidden_size, c.vocab_size)
            if config.freeze_embedding_head
            else nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        )
        self.rotary_emb = RotaryEmbedding(c.attention_head_dim, c.rope)
        self.markov_head = (
            MarkovHead(c.vocab_size, config.markov_rank, c.hidden_size, config.markov_head_type)
            if config.markov_rank
            else None
        )
        width = c.hidden_size + (config.markov_rank if config.confidence_head_with_markov else 0)
        self.confidence_head = nn.Linear(width, 1) if config.enable_confidence_head else None
        self.register_buffer("teacher_weights_loaded", torch.tensor(False))
        self.register_buffer("teacher_fingerprint", torch.zeros(32, dtype=torch.uint8))
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Embedding, _FrozenEmbedding, _FrozenHead)):
                nn.init.normal_(layer.weight, 0.0, c.initializer_range)
                if getattr(layer, "bias", None) is not None:
                    nn.init.zeros_(layer.bias)
        self.embed_tokens.requires_grad_(not config.freeze_embedding_head)
        self.lm_head.requires_grad_(not config.freeze_embedding_head)

    @torch.no_grad()
    def initialize_from_target(self, target):
        if getattr(self, "_aster_training_owned", False):
            raise ValueError("Initialize target weights before Trainer ownership")
        if target.config.to_dict() != self.config.target.to_dict():
            raise ValueError("Target configuration differs from DSpark declaration")
        self.embed_tokens.weight.copy_(target.get_input_embeddings().weight)
        self.lm_head.weight.copy_(target.lm_head.weight)
        self.teacher_fingerprint.copy_(
            torch.tensor(
                list(bytes.fromhex(target_state_identity(target))),
                dtype=torch.uint8,
                device=self.teacher_fingerprint.device,
            )
        )
        self.teacher_weights_loaded.fill_(True)
        return self

    @property
    def teacher_identity(self):
        return bytes(self.teacher_fingerprint.cpu().tolist()).hex()

    def backbone(self, noise_ids, context_features, positions, mask):
        hidden = self.embed_tokens(noise_ids)
        context = self.hidden_norm(self.fc(context_features))
        for layer in self.layers:
            hidden = layer(hidden, context, positions, mask, self.rotary_emb)
        return self.norm(hidden)

    @torch.no_grad()
    def backbone_cached(self, noise_ids, new_context_features, *, state=None):

        c, target = self.config, self.config.target
        if self.training or noise_ids.ndim != 2 or noise_ids.shape[1] != c.block_size:
            raise ValueError(
                "Cached DSpark inference requires eval mode and one complete draft block"
            )
        if (
            noise_ids.dtype != torch.int64
            or (noise_ids < 0).any()
            or (noise_ids >= target.vocab_size).any()
        ):
            raise ValueError("Invalid DSpark inference token IDs")
        if (
            new_context_features.ndim != 3
            or new_context_features.shape[0] != noise_ids.shape[0]
            or new_context_features.shape[2] != len(c.target_layer_ids) * target.hidden_size
        ):
            raise ValueError("Cached DSpark context feature shape differs")
        old = 0
        if state is not None:
            if not isinstance(state, tuple) or len(state) != len(self.layers):
                raise ValueError("DSpark cached layer count differs")
            old = state[0][0].shape[2]
            shape = (noise_ids.shape[0], target.num_key_value_heads, old, target.attention_head_dim)
            if any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(t.shape != shape or t.device != noise_ids.device for t in pair)
                for pair in state
            ):
                raise ValueError("DSpark cached K/V layout differs")
        start = old + new_context_features.shape[1]
        if start + c.block_size > target.max_position_embeddings:
            raise ValueError("DSpark draft positions exceed target context capacity")
        positions = torch.arange(old, start + c.block_size, device=noise_ids.device)[None].expand(
            len(noise_ids), -1
        )
        hidden, context = (
            self.embed_tokens(noise_ids),
            self.hidden_norm(self.fc(new_context_features)),
        )
        caches = []
        for index, layer in enumerate(self.layers):
            attended, kv = layer.self_attn.forward_cached(
                layer.input_layernorm(hidden),
                context,
                positions,
                self.rotary_emb,
                None if state is None else state[index],
            )
            hidden = hidden + attended
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            caches.append(kv)
        return self.norm(hidden), tuple(caches)

    def confidence(self, hidden, previous_ids):
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            hidden = torch.cat(
                (hidden, self.markov_head.get_prev_embeddings(previous_ids).to(hidden)), -1
            )
        return self.confidence_head(hidden).squeeze(-1).float()

    def compute_logits(self, hidden):
        return self.lm_head(hidden)

    def validate_batch(
        self, input_ids, target_hidden_states, loss_mask, target_last_hidden_states=None
    ):
        c = self.config.target
        if (
            input_ids.ndim != 2
            or min(input_ids.shape) < 1
            or input_ids.dtype != torch.int64
            or (input_ids < 0).any()
            or (input_ids >= c.vocab_size).any()
        ):
            raise ValueError("DSpark input IDs must be nonempty int64 target tokens")
        if input_ids.shape[1] + self.config.block_size > c.max_position_embeddings:
            raise ValueError("DSpark context plus draft block exceeds declared positions")
        expected = (*input_ids.shape, len(self.config.target_layer_ids) * c.hidden_size)
        if (
            target_hidden_states.shape != expected
            or not target_hidden_states.is_floating_point()
            or not torch.isfinite(target_hidden_states).all()
            or target_hidden_states.requires_grad
        ):
            raise ValueError("DSpark requires aligned detached target context features")
        if loss_mask.shape != input_ids.shape or not ((loss_mask == 0) | (loss_mask == 1)).all():
            raise ValueError("DSpark loss mask must be aligned binary values")
        if target_last_hidden_states is not None and (
            target_last_hidden_states.shape != (*input_ids.shape, c.hidden_size)
            or not target_last_hidden_states.is_floating_point()
            or not torch.isfinite(target_last_hidden_states).all()
            or target_last_hidden_states.requires_grad
        ):
            raise ValueError("DSpark aligned final target hidden state is invalid")
        if any(x.device != input_ids.device for x in (target_hidden_states, loss_mask)) or (
            target_last_hidden_states is not None
            and target_last_hidden_states.device != input_ids.device
        ):
            raise ValueError("DSpark batch tensors must share device")
        if not bool(self.teacher_weights_loaded):
            raise ValueError("Initialize DSpark embedding/head from the bound target first")

    def validate_anchors(self, input_ids, loss_mask, anchors, keep):
        b, length = input_ids.shape
        if (
            anchors.shape != (b, self.config.num_anchors)
            or anchors.dtype != torch.int64
            or keep is None
            or keep.shape != anchors.shape
            or keep.dtype != torch.bool
        ):
            raise ValueError("Explicit anchors/keep mask must match configured block layout")
        if anchors.device != input_ids.device or keep.device != input_ids.device:
            raise ValueError("Explicit DSpark anchors must share input device")
        if (
            (anchors < 0).any()
            or (anchors >= length).any()
            or (keep & (anchors + 1 >= length)).any()
        ):
            raise ValueError("DSpark anchors outside eligible target range")
        eligible = (
            loss_mask.gather(1, anchors).bool()
            & loss_mask.gather(1, (anchors + 1).clamp_max(length - 1)).bool()
        )
        if (keep & ~eligible).any():
            raise ValueError("DSpark kept anchor must have enabled anchor and first target")
        for row, selected in zip(anchors, keep):
            values = row[selected]
            if len(values) > 1 and not (values[1:] > values[:-1]).all():
                raise ValueError(
                    "Explicit kept anchors must be distinct and increasing, as in the source sampler"
                )

    def forward(
        self,
        input_ids,
        target_hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        *,
        anchor_positions=None,
        block_keep_mask=None,
    ):
        self.validate_batch(input_ids, target_hidden_states, loss_mask, target_last_hidden_states)
        b, length = input_ids.shape
        c = self.config
        k = c.block_size
        if anchor_positions is None:
            if block_keep_mask is not None:
                raise ValueError("Keep mask requires explicit anchors")
            anchor_positions, block_keep_mask = sample_anchors(loss_mask, c.num_anchors)
        anchors, keep = anchor_positions, block_keep_mask
        self.validate_anchors(input_ids, loss_mask, anchors, keep)
        noise = input_ids.new_full((b, c.num_anchors, k), c.mask_token_id)
        anchor_tokens = input_ids.gather(1, anchors)
        noise[:, :, 0] = torch.where(keep, anchor_tokens, noise[:, :, 0])
        positions = torch.cat(
            (
                torch.arange(length, device=input_ids.device)[None].expand(b, -1),
                (anchors[:, :, None] + torch.arange(k, device=input_ids.device)).reshape(b, -1),
            ),
            1,
        )
        hidden = self.backbone(
            noise.reshape(b, -1),
            target_hidden_states,
            positions,
            block_attention_mask(anchors, keep, length, k),
        ).reshape(b, c.num_anchors, k, -1)
        indices = anchors[:, :, None] + torch.arange(1, k + 1, device=input_ids.device)
        safe = indices.clamp_max(length - 1).masked_fill(~keep[:, :, None], 0)
        targets = input_ids[:, None].expand(-1, c.num_anchors, -1).gather(2, safe)
        valid = (
            (indices < length)
            & loss_mask[:, None].expand(-1, c.num_anchors, -1).gather(2, safe).bool()
            & keep[:, :, None]
        )
        valid = valid.int().cumprod(-1).bool()
        previous = torch.cat((anchor_tokens[:, :, None], targets[:, :, :-1]), -1)
        logits = self.lm_head(hidden)
        if self.markov_head is not None:
            logits = logits + self.markov_head(hidden, previous)
        teacher_logits = None
        if target_last_hidden_states is not None:
            if not c.freeze_embedding_head:
                raise ValueError(
                    "Teacher-logit supervision requires the target head to remain frozen"
                )
            teacher_hidden = (
                target_last_hidden_states[:, None]
                .expand(-1, c.num_anchors, -1, -1)
                .gather(
                    2,
                    (safe - 1)
                    .clamp_min(0)[:, :, :, None]
                    .expand(-1, -1, -1, target_last_hidden_states.shape[-1]),
                )
            )
            teacher_logits = self.lm_head(teacher_hidden)
        return DSparkOutput(
            logits, targets, valid, keep, self.confidence(hidden, previous), teacher_logits, anchors
        )
