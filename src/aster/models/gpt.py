"""GPT-2 causal decoding with Conv1D checkpoint layout and absolute positions."""

from dataclasses import asdict, dataclass
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn import LayerNorm
from aster.nn.attention import KVState, attention_mask
from .serialization import LocalModelMixin
from .decoder import configuration_key


@dataclass(frozen=True)
class GPT2Config:
    architecture: ClassVar[str] = "gpt2"
    vocab_size: int = 32
    n_positions: int = 128
    n_embd: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_inner: int | None = 64
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    tie_word_embeddings: bool = True
    add_cross_attention: bool = False

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.n_positions,
                self.n_embd,
                self.n_layer,
                self.n_head,
                self.n_inner or 4 * self.n_embd,
            )
            < 1
            or self.n_embd % self.n_head
        ):
            raise ValueError("Invalid GPT2 dimensions")
        if self.activation_function not in {"gelu_new", "gelu", "relu", "silu"}:
            raise ValueError("Unsupported GPT2 activation")
        if (
            any(not 0 <= p < 1 for p in (self.resid_pdrop, self.embd_pdrop, self.attn_pdrop))
            or min(self.layer_norm_epsilon, self.initializer_range) <= 0
        ):
            raise ValueError("Invalid GPT2 numerics")
        if self.add_cross_attention:
            raise ValueError(
                "GPT2 cross-attention fine-tuning variant is not admitted by this causal branch"
            )

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_hidden_layers(self):
        return self.n_layer

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def max_position_embeddings(self):
        return self.n_positions

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class TransposedLinear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, hidden):
        return torch.addmm(self.bias, hidden.reshape(-1, hidden.shape[-1]), self.weight).reshape(
            *hidden.shape[:-1], self.weight.shape[1]
        )


class GPTAttention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.config = c
        self.c_attn = TransposedLinear(c.n_embd, 3 * c.n_embd)
        self.c_proj = TransposedLinear(c.n_embd, c.n_embd)
        self.scale = (c.n_embd // c.n_head) ** -0.5 if c.scale_attn_weights else 1.0
        if c.scale_attn_by_inverse_layer_idx:
            self.scale /= index + 1

    def forward(self, hidden, mask, previous):
        c = self.config
        b, s, h = hidden.shape
        q, k, v = (
            x.reshape(b, s, c.n_head, h // c.n_head).transpose(1, 2)
            for x in self.c_attn(hidden).chunk(3, -1)
        )
        if previous is not None:
            k, v = torch.cat((previous[0], k), -2), torch.cat((previous[1], v), -2)

        if c.reorder_and_upcast_attn:
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = (q.float() @ k.float().transpose(-1, -2)) * self.scale
        else:
            scores = (q @ k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = scores.softmax(-1).to(v.dtype).masked_fill(~mask, 0)
        result = F.dropout(probabilities, c.attn_pdrop, self.training) @ v
        output = self.c_proj(result.transpose(1, 2).reshape(b, s, h))
        return F.dropout(output, c.resid_pdrop, self.training), (k, v)


class GPTMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        width = c.n_inner or 4 * c.n_embd
        self.c_fc, self.c_proj = (
            TransposedLinear(c.n_embd, width),
            TransposedLinear(width, c.n_embd),
        )

    def forward(self, hidden):
        hidden = self.c_fc(hidden)
        activation = self.config.activation_function
        if activation == "gelu_new":
            hidden = (
                0.5
                * hidden
                * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden.pow(3))))
            )
        elif activation == "gelu":
            hidden = F.gelu(hidden)
        elif activation == "relu":
            hidden = F.relu(hidden)
        else:
            hidden = F.silu(hidden)
        return F.dropout(self.c_proj(hidden), self.config.resid_pdrop, self.training)


class GPTBlock(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.ln_1, self.ln_2 = (
            LayerNorm(c.n_embd, c.layer_norm_epsilon),
            LayerNorm(c.n_embd, c.layer_norm_epsilon),
        )
        self.attn, self.mlp = GPTAttention(c, index), GPTMLP(c)

    def forward(self, hidden, mask, previous):
        update, present = self.attn(self.ln_1(hidden), mask, previous)
        hidden = hidden + update
        return hidden + self.mlp(self.ln_2(hidden)), present


class GPT2ForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.transformer.h = nn.ModuleList(GPTBlock(config, i) for i in range(config.n_layer))
        self.transformer.ln_f = LayerNorm(config.n_embd, config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (TransposedLinear, nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for layer in self.transformer.h:
            for projection in (layer.attn.c_proj, layer.mlp.c_proj):
                nn.init.normal_(
                    projection.weight, std=config.initializer_range / math.sqrt(2 * config.n_layer)
                )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def get_input_embeddings(self):
        return self.transformer.wte

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one token/embedding input")
        c = self.config
        hidden = self.transformer.wte(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != c.n_embd or hidden.shape[1] == 0:
            raise ValueError("Invalid GPT2 input shape")
        b, s, _ = hidden.shape
        seen = 0
        if state is not None:
            if (
                not isinstance(state, KVState)
                or state.kind != "dense_kv"
                or state.model_key != self.model_key
                or len(state.layers) != c.n_layer
            ):
                raise ValueError("GPT2 state type/model mismatch")
            seen = state.seen_tokens
            use_cache = True
            shape = (b, c.n_head, seen, c.n_embd // c.n_head)
            if seen < 0 or any(k.shape != shape or v.shape != shape for k, v in state.layers):
                raise ValueError("Invalid GPT2 cache tensor layout")
        if seen + s > c.n_positions:
            raise ValueError("GPT2 context exceeds learned position table")
        if position_ids is None:
            position_ids = torch.arange(seen, seen + s, device=hidden.device)[None]
        if (
            position_ids.shape not in {(1, s), (b, s)}
            or (position_ids < 0).any()
            or (position_ids >= c.n_positions).any()
        ):
            raise ValueError("GPT2 positions exceed learned table")
        hidden = hidden + self.transformer.wpe(position_ids)
        if token_type_ids is not None:
            if token_type_ids.shape != (b, s):
                raise ValueError("GPT2 token types must match input shape")
            hidden = hidden + self.transformer.wte(token_type_ids)
        hidden = F.dropout(hidden, c.embd_pdrop, self.training)

        from aster.nn.attention import attention_mask as make_mask

        mask = make_mask(
            b, s, seen + s, seen_tokens=seen, padding=attention_mask, device=hidden.device
        )
        states, layers = [], []
        for i, layer in enumerate(self.transformer.h):
            if output_hidden_states:
                states.append(hidden)
            hidden, present = layer(hidden, mask, state.layers[i] if state is not None else None)
            if use_cache:
                layers.append(present)
        hidden = self.transformer.ln_f(hidden)
        if output_hidden_states:
            states.append(hidden)
        updated = KVState(tuple(layers), seen + s, self.model_key) if use_cache else None
        return TokenOutput(
            self.lm_head(hidden), updated, tuple(states) if output_hidden_states else None
        )
