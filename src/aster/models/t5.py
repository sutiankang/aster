"""T5 encoder-decoder with unscaled attention, relative buckets, and cross-attention state."""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput, StateCapabilities
from aster.nn import RMSNorm
from aster.nn.attention import attention_mask as make_attention_mask
from .serialization import LocalModelMixin, configuration_key


def relative_position_bucket(relative, *, bidirectional, num_buckets, max_distance):
    """Use exact small-distance buckets and logarithmic large-distance buckets;
    retain signs only for bidirectional attention."""
    offset = torch.zeros_like(relative)
    if bidirectional:
        num_buckets //= 2
        offset = (relative > 0).long() * num_buckets
        distance = relative.abs()
    else:
        distance = (-relative).clamp_min(0)
    exact = num_buckets // 2
    large = (
        exact
        + (
            torch.log(distance.float().clamp_min(1) / exact)
            / math.log(max_distance / exact)
            * (num_buckets - exact)
        ).long()
    )
    return offset + torch.where(distance < exact, distance, large.clamp_max(num_buckets - 1))


@dataclass(frozen=True)
class Seq2SeqState:
    layers: tuple[tuple[torch.Tensor, ...], ...]
    seen_tokens: int
    model_key: str
    encoder_hidden: torch.Tensor
    encoder_mask: torch.Tensor
    kind: str = "encoder_decoder_kv"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, truncatable=True, reorderable=True)

    def fork(self):
        return type(self)(
            tuple(tuple(x.clone() for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
            self.encoder_hidden.clone(),
            self.encoder_mask.clone(),
        )

    def reorder(self, indices):
        return type(self)(
            tuple(tuple(x.index_select(0, indices) for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
            self.encoder_hidden.index_select(0, indices),
            self.encoder_mask.index_select(0, indices),
        )

    def truncate(self, length):
        if not 0 <= length <= self.seen_tokens:
            raise ValueError("Invalid decoder truncation length")
        return type(self)(
            tuple(
                tuple(
                    x[..., :length, :].clone() if i < 2 else x.clone() for i, x in enumerate(layer)
                )
                for layer in self.layers
            ),
            length,
            self.model_key,
            self.encoder_hidden.clone(),
            self.encoder_mask.clone(),
        )


class T5Attention(nn.Module):
    def __init__(self, c, decoder=False, relative_bias=False):
        super().__init__()
        self.config, self.decoder = c, decoder
        inner = c.num_heads * c.d_kv
        self.q = nn.Linear(c.d_model, inner, bias=False)
        self.k = nn.Linear(c.d_model, inner, bias=False)
        self.v = nn.Linear(c.d_model, inner, bias=False)
        self.o = nn.Linear(inner, c.d_model, bias=False)
        if relative_bias:
            self.relative_attention_bias = nn.Embedding(
                c.relative_attention_num_buckets, c.num_heads
            )

    def forward(self, hidden, mask, bias=None, previous=None, source=None, seen=0):
        c = self.config

        def split(value):
            return value.reshape(value.shape[0], value.shape[1], c.num_heads, c.d_kv).transpose(
                1, 2
            )

        q = split(self.q(hidden))
        if source is not None and previous is not None:
            k, v = previous
        else:
            value = hidden if source is None else source
            k, v = split(self.k(value)), split(self.v(value))
            if previous is not None:
                k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)
        if bias is None:
            if hasattr(self, "relative_attention_bias"):
                queries = torch.arange(hidden.shape[1], device=hidden.device) + seen
                keys = torch.arange(k.shape[-2], device=hidden.device)
                buckets = relative_position_bucket(
                    keys[None] - queries[:, None],
                    bidirectional=not self.decoder,
                    num_buckets=c.relative_attention_num_buckets,
                    max_distance=c.relative_attention_max_distance,
                )
                bias = self.relative_attention_bias(buckets).permute(2, 0, 1)[None]
            else:
                bias = hidden.new_zeros(1, c.num_heads, hidden.shape[1], k.shape[-2])

        scores = (q @ k.transpose(-1, -2)) + bias
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~mask, 0)
        output = F.dropout(weights, c.dropout_rate, self.training) @ v
        output = output.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        return self.o(output), bias, (k, v)


class T5AttentionResidual(nn.Module):
    def __init__(self, c, *, decoder=False, relative_bias=False, cross=False):
        super().__init__()
        self.cross, self.dropout = cross, nn.Dropout(c.dropout_rate)
        self.layer_norm = RMSNorm(c.d_model, c.layer_norm_epsilon)
        name = "EncDecAttention" if cross else "SelfAttention"
        self.add_module(name, T5Attention(c, decoder=decoder, relative_bias=relative_bias))

    def forward(self, hidden, mask, **kwargs):
        attention = self.EncDecAttention if self.cross else self.SelfAttention
        output, bias, present = attention(self.layer_norm(hidden), mask, **kwargs)
        return hidden + self.dropout(output), bias, present


class T5Dense(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.kind, self.dropout = c.feed_forward_proj, nn.Dropout(c.dropout_rate)
        if self.kind.startswith("gated-"):
            self.wi_0 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.wi_1 = nn.Linear(c.d_model, c.d_ff, bias=False)
        else:
            self.wi = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.wo = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, hidden):
        if self.kind == "relu":
            hidden = F.relu(self.wi(hidden))
        else:
            gate = (
                F.gelu(self.wi_0(hidden), approximate="tanh")
                if self.kind == "gated-gelu"
                else F.silu(self.wi_0(hidden))
            )
            hidden = gate * self.wi_1(hidden)
        return self.wo(self.dropout(hidden).to(self.wo.weight.dtype))


class T5FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.DenseReluDense = T5Dense(c)
        self.layer_norm = RMSNorm(c.d_model, c.layer_norm_epsilon)
        self.dropout = nn.Dropout(c.dropout_rate)

    def forward(self, hidden):
        return hidden + self.dropout(self.DenseReluDense(self.layer_norm(hidden)))


class T5Block(nn.Module):
    def __init__(self, c, decoder, index):
        super().__init__()
        blocks = [T5AttentionResidual(c, decoder=decoder, relative_bias=index == 0)]
        if decoder:
            blocks.append(T5AttentionResidual(c, cross=True))
        self.layer = nn.ModuleList(blocks + [T5FeedForward(c)])


class T5Stack(nn.Module):
    def __init__(self, c, shared, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.embed_tokens = shared
        self.block = nn.ModuleList(
            T5Block(c, decoder, i) for i in range(c.num_decoder_layers if decoder else c.num_layers)
        )
        self.final_layer_norm = RMSNorm(c.d_model, c.layer_norm_epsilon)
        self.dropout = nn.Dropout(c.dropout_rate)

    def forward(
        self,
        ids,
        padding=None,
        previous=None,
        source=None,
        source_mask=None,
        seen=0,
        return_hidden=False,
        *,
        inputs_embeds=None,
    ):
        if (ids is None) == (inputs_embeds is None):
            raise ValueError("T5 stack needs exactly one token or embedding input")
        hidden = self.dropout(self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds)
        b, s, _ = hidden.shape
        mask = make_attention_mask(
            b,
            s,
            seen + s,
            seen_tokens=seen,
            padding=padding,
            causal=self.decoder,
            device=hidden.device,
        )
        cross_mask = (
            None if source is None else source_mask[:, None, None, :].bool().expand(b, 1, s, -1)
        )
        bias = cross_bias = None
        states, present = [], []
        for i, block in enumerate(self.block):
            if return_hidden:
                states.append(hidden)
            layer_previous = None if previous is None else previous[i]
            hidden, bias, self_kv = block.layer[0](
                hidden,
                mask,
                bias=bias,
                previous=None if layer_previous is None else layer_previous[:2],
                seen=seen,
            )
            if self.decoder:
                hidden, cross_bias, cross_kv = block.layer[1](
                    hidden,
                    cross_mask,
                    bias=cross_bias,
                    source=source,
                    previous=None if layer_previous is None else layer_previous[2:],
                )
                present.append(self_kv + cross_kv)
            hidden = block.layer[-1](hidden)
        hidden = self.dropout(self.final_layer_norm(hidden))
        if return_hidden:
            states.append(hidden)
        return hidden, tuple(present), tuple(states) if return_hidden else None


class T5ForConditionalGeneration(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)
        self.decoder = T5Stack(config, self.shared, decoder=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        c, factor = config, config.initializer_factor
        nn.init.normal_(self.shared.weight, std=factor)
        nn.init.normal_(self.lm_head.weight, std=factor)
        for module in self.modules():
            if isinstance(module, RMSNorm):
                nn.init.constant_(module.weight, factor)
            elif isinstance(module, T5Dense):
                for name in ("wi", "wi_0", "wi_1"):
                    if hasattr(module, name):
                        nn.init.normal_(getattr(module, name).weight, std=factor * c.d_model**-0.5)
                nn.init.normal_(module.wo.weight, std=factor * c.d_ff**-0.5)
            elif isinstance(module, T5Attention):
                for name, std in (
                    ("q", (c.d_model * c.d_kv) ** -0.5),
                    ("k", c.d_model**-0.5),
                    ("v", c.d_model**-0.5),
                    ("o", (c.num_heads * c.d_kv) ** -0.5),
                ):
                    nn.init.normal_(getattr(module, name).weight, std=factor * std)
                if hasattr(module, "relative_attention_bias"):
                    nn.init.normal_(
                        module.relative_attention_bias.weight, std=factor * c.d_model**-0.5
                    )
        if c.tie_word_embeddings:
            self.lm_head.weight = self.shared.weight

    def shift_right(self, labels):
        """Shift decoder inputs only; replace -100 with padding on the input side, not in labels."""
        result = labels.new_full(labels.shape, self.config.pad_token_id)
        result[:, 1:] = labels[:, :-1]
        result[:, 0] = self.config.decoder_start_token_id
        return result.masked_fill(result == -100, self.config.pad_token_id)

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] < 1:
            raise ValueError("Decoder requires nonempty [batch,sequence] token IDs")
        if state is None:
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError(
                    "First T5 call requires exactly one encoder token or embedding input"
                )
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if (
                len(shape) != 2
                or shape[1] < 1
                or (inputs_embeds is not None and inputs_embeds.shape[-1] != self.config.d_model)
            ):
                raise ValueError("Invalid T5 encoder token/embedding shape")
            memory_mask = (
                torch.ones(shape, dtype=torch.long, device=device)
                if attention_mask is None
                else attention_mask
            )
            memory, _, _ = self.encoder(input_ids, memory_mask, inputs_embeds=inputs_embeds)
            previous, seen = None, 0
        else:
            if not isinstance(state, Seq2SeqState) or state.model_key != self.model_key:
                raise ValueError("T5 state/config mismatch")
            if input_ids is not None or inputs_embeds is not None or attention_mask is not None:
                raise ValueError("Cached encoder condition is fixed; use a new state to change it")
            memory, memory_mask, previous, seen = (
                state.encoder_hidden,
                state.encoder_mask,
                state.layers,
                state.seen_tokens,
            )
            if len(previous) != self.config.num_decoder_layers or any(
                len(layer) != 4 or layer[0].shape[-2] != seen for layer in previous
            ):
                raise ValueError("Malformed T5 state layout/length")
        if decoder_input_ids.shape[0] != memory.shape[0]:
            raise ValueError("Encoder/decoder batch sizes differ")
        hidden, present, states = self.decoder(
            decoder_input_ids,
            decoder_attention_mask,
            previous,
            memory,
            memory_mask,
            seen,
            output_hidden_states,
        )
        if self.config.scale_decoder_outputs:
            hidden = hidden * self.config.d_model**-0.5
        output_state = (
            Seq2SeqState(
                present, seen + decoder_input_ids.shape[1], self.model_key, memory, memory_mask
            )
            if use_cache
            else None
        )
        return TokenOutput(self.lm_head(hidden), output_state, states)
