"""Gemma4 DSpark drafts with shared K/V, RMS normalization, and softcapped Markov logits."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar
import torch
from torch import nn
from ..nn.attention import scaled_attention
from ..nn.normalization import FloatRMSNorm
from ..nn.position import RopeConfig, RotaryEmbedding
from ..nn.markov import MarkovHead
from .gemma4 import Gemma4TextConfig, Gemma4MLP
from .serialization import LocalModelMixin
from .dspark import (
    DSparkOutput,
    sample_anchors,
    block_attention_mask,
    target_state_identity,
    _FrozenEmbedding,
    _FrozenHead,
)


@dataclass(frozen=True)
class Gemma4DSparkConfig:
    architecture: ClassVar[str] = "dspark_gemma4"
    target: Gemma4TextConfig
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
            values = dict(self.target)
            if values.pop("architecture", None) != "gemma4_text":
                raise ValueError("DSpark target must be genuine Gemma4 text")
            object.__setattr__(self, "target", Gemma4TextConfig(**values))
        object.__setattr__(self, "target_layer_ids", tuple(self.target_layer_ids))
        if type(self.target) is not Gemma4TextConfig:
            raise ValueError("DSpark target must be genuine Gemma4TextConfig")
        if self.target.enable_moe_block or self.target.hidden_size_per_layer_input:
            raise ValueError(
                "Pinned Gemma4 DSpark explicitly rejects MoE and per-layer input gates"
            )
        if any(
            type(v) is not int or v < 1
            for v in (self.num_draft_layers, self.block_size, self.num_anchors)
        ):
            raise ValueError("Invalid Gemma4 DSpark layer/block/anchor count")
        if (
            not self.target_layer_ids
            or any(
                type(v) is not int or v < -1 or v >= self.target.num_hidden_layers
                for v in self.target_layer_ids
            )
            or tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids
        ):
            raise ValueError("Gemma4 DSpark target layers must be unique sorted valid indices")
        if (
            type(self.mask_token_id) is not int
            or not 0 <= self.mask_token_id < self.target.vocab_size
        ):
            raise ValueError("Declare an explicit valid mask token")
        if (
            type(self.markov_rank) is not int
            or self.markov_rank < 0
            or self.markov_head_type not in {"vanilla", "gated", "rnn"}
        ):
            raise ValueError("Invalid Gemma4 DSpark Markov head")
        if any(
            type(v) is not bool
            for v in (
                self.enable_confidence_head,
                self.confidence_head_with_markov,
                self.freeze_embedding_head,
            )
        ):
            raise ValueError("Gemma4 DSpark flags must be bool")
        if (
            self.enable_confidence_head
            and self.confidence_head_with_markov
            and not self.markov_rank
        ):
            raise ValueError("Markov confidence requires a Markov head")

    def to_dict(self):

        target = self.target.to_dict()
        target["layer_types"] = list(target["layer_types"])
        return dict(
            architecture=self.architecture,
            target=target,
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


class Gemma4DSparkAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c.num_attention_heads, c.global_head_dim
        self.hk = c.num_global_key_value_heads if c.attention_k_eq_v else c.num_key_value_heads
        self.dropout = c.attention_dropout
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, self.hk * self.d, bias=c.attention_bias)
        self.v_proj = (
            None
            if c.attention_k_eq_v
            else nn.Linear(c.hidden_size, self.hk * self.d, bias=c.attention_bias)
        )
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=c.attention_bias)
        self.q_norm, self.k_norm = (
            FloatRMSNorm(self.d, c.rms_norm_eps),
            FloatRMSNorm(self.d, c.rms_norm_eps),
        )
        self.v_norm = FloatRMSNorm(self.d, c.rms_norm_eps, with_scale=False)

    def project(self, hidden, context, positions, rotary):
        b, length, _ = hidden.shape

        def split(value, heads):
            return value.reshape(b, -1, heads, self.d)

        q = rotary(
            self.q_norm(split(self.q_proj(hidden), self.h)).transpose(1, 2), positions[:, -length:]
        )
        raw_k = torch.cat((self.k_proj(context), self.k_proj(hidden)), 1)
        raw_v = (
            raw_k
            if self.v_proj is None
            else torch.cat((self.v_proj(context), self.v_proj(hidden)), 1)
        )
        k = rotary(self.k_norm(split(raw_k, self.hk)).transpose(1, 2), positions)
        v = self.v_norm(split(raw_v, self.hk)).transpose(1, 2)
        return q, k, v

    def forward(self, hidden, context, positions, mask, rotary):
        b, length, _ = hidden.shape
        q, k, v = self.project(hidden, context, positions, rotary)

        value = scaled_attention(q, k, v, mask, scale=1.0, dropout=0.0, training=self.training)
        return self.o_proj(value.transpose(1, 2).reshape(b, length, -1))

    def forward_cached(self, hidden, context, positions, rotary, previous=None):
        b, length, _ = hidden.shape
        q, k, v = self.project(hidden, context, positions, rotary)
        if previous is not None:
            k, v = torch.cat((previous[0], k), 2), torch.cat((previous[1], v), 2)

        visible = torch.ones(b, 1, length, k.shape[2], device=q.device, dtype=torch.bool)
        value = scaled_attention(q, k, v, visible, scale=1.0)

        cache = (k[:, :, :-length].clone(), v[:, :, :-length].clone())
        return self.o_proj(value.transpose(1, 2).reshape(b, length, -1)), cache


class Gemma4DSparkLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        c = config.target
        self.self_attn = Gemma4DSparkAttention(c)

        threshold = config.num_draft_layers - c.num_kv_shared_layers

        view = SimpleNamespace(
            hidden_activation=c.hidden_activation,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            use_double_wide_mlp=c.use_double_wide_mlp and threshold > 0,
            independent_layers=threshold,
        )
        self.mlp = Gemma4MLP(view, index)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            self.add_module(name, FloatRMSNorm(c.hidden_size, c.rms_norm_eps))
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden, context, positions, mask, rotary):
        attended = self.self_attn(self.input_layernorm(hidden), context, positions, mask, rotary)
        hidden = hidden + self.post_attention_layernorm(attended)
        hidden = hidden + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden))
        )
        return hidden * self.layer_scalar


class Gemma4DSparkDraft(LocalModelMixin, nn.Module):
    _aster_semantic_buffers = ("embed_scale",)

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.target
        self.embed_tokens = (
            _FrozenEmbedding(c.vocab_size, c.hidden_size)
            if config.freeze_embedding_head
            else nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=c.pad_token_id)
        )
        self.lm_head = (
            _FrozenHead(c.hidden_size, c.vocab_size)
            if config.freeze_embedding_head
            else nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        )
        self.register_buffer("embed_scale", torch.tensor(c.hidden_size**0.5), persistent=False)
        self.layers = nn.ModuleList(
            Gemma4DSparkLayer(config, i) for i in range(config.num_draft_layers)
        )
        self.norm, self.hidden_norm = (
            FloatRMSNorm(c.hidden_size, c.rms_norm_eps),
            FloatRMSNorm(c.hidden_size, c.rms_norm_eps),
        )
        self.fc = nn.Linear(len(config.target_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(c.global_head_dim, RopeConfig(theta=c.global_rope_theta))
        frequency = self.rotary_emb.inv_freq.clone() / c.global_rope_factor
        frequency[int(c.global_rotary_fraction * c.global_head_dim // 2) :] = 0
        self.rotary_emb.inv_freq = frequency
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
        if c.pad_token_id is not None:
            with torch.no_grad():
                self.embed_tokens.weight[c.pad_token_id].zero_()

    @torch.no_grad()
    def initialize_from_target(self, target):
        if getattr(self, "_aster_training_owned", False):
            raise ValueError("Bind target before Trainer takes parameter ownership")
        if target.config.to_dict() != self.config.target.to_dict():
            raise ValueError("Gemma4 target configuration differs")
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

    def compute_logits(self, hidden):
        logits = self.lm_head(hidden)
        cap = self.config.target.final_logit_softcapping
        return logits if cap is None else torch.tanh(logits / cap) * cap

    def backbone(self, noise_ids, context_features, positions, mask):
        hidden = self.embed_tokens(noise_ids) * self.embed_scale.to(self.embed_tokens.weight.dtype)
        context = self.hidden_norm(self.fc(context_features))
        for layer in self.layers:
            hidden = layer(hidden, context, positions, mask, self.rotary_emb)
        return self.norm(hidden)

    @torch.no_grad()
    def backbone_cached(self, noise_ids, new_context_features, *, state=None):

        c, target = self.config, self.config.target
        if self.training or getattr(self, "_aster_training_owned", False):
            raise ValueError("Gemma4 cached draft requires an idle dense eval snapshot")
        if (
            noise_ids.ndim != 2
            or noise_ids.shape[0] < 1
            or noise_ids.shape[1] != c.block_size
            or noise_ids.dtype != torch.int64
            or (noise_ids < 0).any()
            or (noise_ids >= target.vocab_size).any()
        ):
            raise ValueError("Gemma4 cached draft needs one complete valid int64 block per row")
        features = new_context_features
        if (
            features.ndim != 3
            or features.shape[0] != noise_ids.shape[0]
            or features.shape[2] != len(c.target_layer_ids) * target.hidden_size
            or not features.is_floating_point()
            or not torch.isfinite(features).all()
            or features.requires_grad
        ):
            raise ValueError("Gemma4 cached context must contain finite detached aligned features")
        if (
            features.device != noise_ids.device
            or features.device != self.fc.weight.device
            or features.dtype != self.fc.weight.dtype
        ):
            raise ValueError(
                "Gemma4 cached context must match declared model device/stored precision"
            )
        if not bool(self.teacher_weights_loaded):
            raise ValueError("Bind Gemma4 target before cached inference")
        old = 0
        if state is not None:
            if (
                not isinstance(state, tuple)
                or len(state) != len(self.layers)
                or any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in pair)
                    for pair in state
                )
            ):
                raise ValueError("Gemma4 draft cache needs explicit per-layer K/V tuples")
            old = state[0][0].shape[2]
            heads = (
                target.num_global_key_value_heads
                if target.attention_k_eq_v
                else target.num_key_value_heads
            )
            expected = (noise_ids.shape[0], heads, old, target.global_head_dim)
            if any(
                t.shape != expected
                or t.device != features.device
                or t.dtype != features.dtype
                or t.requires_grad
                or not torch.isfinite(t).all()
                for pair in state
                for t in pair
            ):
                raise ValueError("Gemma4 draft cache K/V schema/device/precision differs")
        start = old + features.shape[1]
        if start + c.block_size > target.max_position_embeddings:
            raise ValueError("Gemma4 draft positions exceed declared context capacity")
        positions = torch.arange(old, start + c.block_size, device=noise_ids.device)[None].expand(
            len(noise_ids), -1
        )

        with torch.autocast(noise_ids.device.type, enabled=False):
            hidden = self.embed_tokens(noise_ids) * self.embed_scale.to(
                self.embed_tokens.weight.dtype
            )
            context = self.hidden_norm(self.fc(features))
            caches = []
            for index, layer in enumerate(self.layers):
                attended, cache = layer.self_attn.forward_cached(
                    layer.input_layernorm(hidden),
                    context,
                    positions,
                    self.rotary_emb,
                    None if state is None else state[index],
                )
                hidden = hidden + layer.post_attention_layernorm(attended)
                hidden = hidden + layer.post_feedforward_layernorm(
                    layer.mlp(layer.pre_feedforward_layernorm(hidden))
                )
                hidden = hidden * layer.layer_scalar
                caches.append(cache)
            return self.norm(hidden), tuple(caches)

    def confidence(self, hidden, previous_ids):
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            hidden = torch.cat(
                (hidden, self.markov_head.get_prev_embeddings(previous_ids).to(hidden)), -1
            )
        return self.confidence_head(hidden).squeeze(-1).float()

    def validate_batch(
        self, input_ids, target_hidden_states, loss_mask, target_last_hidden_states=None
    ):
        c, t = self.config, self.config.target
        if (
            input_ids.ndim != 2
            or min(input_ids.shape) < 1
            or input_ids.dtype != torch.int64
            or (input_ids < 0).any()
            or (input_ids >= t.vocab_size).any()
        ):
            raise ValueError("Invalid Gemma4 DSpark int64 tokens")
        if input_ids.shape[1] + c.block_size > t.max_position_embeddings:
            raise ValueError("Gemma4 draft positions exceed context capacity")
        if (
            target_hidden_states.shape
            != (*input_ids.shape, len(c.target_layer_ids) * t.hidden_size)
            or target_hidden_states.requires_grad
            or not target_hidden_states.is_floating_point()
            or not torch.isfinite(target_hidden_states).all()
        ):
            raise ValueError("Gemma4 DSpark needs finite detached aligned target features")
        if loss_mask.shape != input_ids.shape or not ((loss_mask == 0) | (loss_mask == 1)).all():
            raise ValueError("Invalid Gemma4 DSpark binary loss mask")
        if target_last_hidden_states is not None and (
            target_last_hidden_states.shape != (*input_ids.shape, t.hidden_size)
            or target_last_hidden_states.requires_grad
            or not target_last_hidden_states.is_floating_point()
            or not torch.isfinite(target_last_hidden_states).all()
        ):
            raise ValueError("Invalid final Gemma4 teacher hidden state")
        if any(value.device != input_ids.device for value in (target_hidden_states, loss_mask)) or (
            target_last_hidden_states is not None
            and target_last_hidden_states.device != input_ids.device
        ):
            raise ValueError("Gemma4 DSpark tensors need one device")
        if not bool(self.teacher_weights_loaded):
            raise ValueError("Bind Gemma4 target embedding/head first")

    def validate_anchors(self, input_ids, loss_mask, anchors, keep):

        from .dspark import DSparkDraft

        return DSparkDraft.validate_anchors(self, input_ids, loss_mask, anchors, keep)

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
        c = self.config
        b, length = input_ids.shape
        k = c.block_size
        if anchor_positions is None:
            if block_keep_mask is not None:
                raise ValueError("Keep mask needs explicit anchors")
            anchors, keep = sample_anchors(loss_mask, c.num_anchors)
        else:
            self.validate_anchors(input_ids, loss_mask, anchor_positions, block_keep_mask)
            anchors, keep = anchor_positions, block_keep_mask
        anchor_tokens = input_ids.gather(1, anchors)
        noise = input_ids.new_full((b, c.num_anchors, k), c.mask_token_id)
        noise[:, :, 0] = anchor_tokens
        noise = noise.masked_fill(~keep[:, :, None], c.mask_token_id)
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

        logits = self.compute_logits(hidden)
        if self.markov_head is not None:
            logits = logits + self.markov_head(hidden, previous)
        teacher_logits = None
        if target_last_hidden_states is not None:
            if not c.freeze_embedding_head:
                raise ValueError("Gemma4 teacher-logit supervision requires a frozen target head")
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
            teacher_logits = self.compute_logits(teacher_hidden)
        return DSparkOutput(
            logits, targets, valid, keep, self.confidence(hidden, previous), teacher_logits, anchors
        )
