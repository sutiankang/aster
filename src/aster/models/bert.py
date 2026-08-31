"""Bidirectional BERT encoding and masked-language-model prediction."""

import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn import LayerNorm
from aster.nn.attention import attention_mask as make_attention_mask, scaled_attention
from .serialization import LocalModelMixin


class BertEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=c.pad_token_id)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.token_type_embeddings = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.LayerNorm = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)

    def forward(self, hidden, positions, segments):
        return self.dropout(
            self.LayerNorm(
                hidden + self.position_embeddings(positions) + self.token_type_embeddings(segments)
            )
        )


class BertSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        self.query = nn.Linear(c.hidden_size, c.hidden_size)
        self.key = nn.Linear(c.hidden_size, c.hidden_size)
        self.value = nn.Linear(c.hidden_size, c.hidden_size)
        self.dropout = c.attention_probs_dropout_prob

    def forward(self, hidden, mask):
        b, s, d = hidden.shape

        def split(proj):
            return proj(hidden).reshape(b, s, self.heads, self.dim).transpose(1, 2)

        value = scaled_attention(
            split(self.query),
            split(self.key),
            split(self.value),
            mask,
            dropout=self.dropout,
            training=self.training,
        )
        return value.transpose(1, 2).reshape(b, s, d)


class BertResidual(nn.Module):
    def __init__(self, c, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, c.hidden_size)
        self.LayerNorm = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.dropout = nn.Dropout(c.hidden_dropout_prob)

    def forward(self, value, residual):
        return self.LayerNorm(residual + self.dropout(self.dense(value)))


class BertAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self = BertSelfAttention(c)
        self.output = BertResidual(c, c.hidden_size)

    def forward(self, hidden, mask):
        return self.output(self.self(hidden, mask), hidden)


class BertIntermediate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.intermediate_size)

    def forward(self, hidden):
        return F.gelu(self.dense(hidden))


class BertLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention = BertAttention(c)
        self.intermediate = BertIntermediate(c)
        self.output = BertResidual(c, c.intermediate_size)

    def forward(self, hidden, mask):
        hidden = self.attention(hidden, mask)
        return self.output(self.intermediate(hidden), hidden)


class BertPredictionTransform(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.LayerNorm = LayerNorm(c.hidden_size, c.layer_norm_eps)

    def forward(self, hidden):
        return self.LayerNorm(F.gelu(self.dense(hidden)))


class BertForMaskedLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = nn.Module()
        self.bert.embeddings = BertEmbeddings(config)
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList(
            BertLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        self.cls.predictions.transform = BertPredictionTransform(config)
        self.cls.predictions.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.cls.predictions.bias = self.cls.predictions.decoder.bias
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.bert.embeddings.word_embeddings.weight[config.pad_token_id].zero_()
        if config.tie_word_embeddings:
            self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

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
        if state is not None or use_cache:
            raise ValueError("Bidirectional BERT has no append-only causal KV state")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = (
            self.bert.embeddings.word_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or hidden.shape[1] == 0:
            raise ValueError("BERT expects nonempty [batch,sequence,hidden]")
        b, s, _ = hidden.shape
        positions = (
            torch.arange(s, device=hidden.device)[None].expand(b, -1)
            if position_ids is None
            else position_ids
        )
        segments = (
            torch.zeros((b, s), device=hidden.device, dtype=torch.long)
            if token_type_ids is None
            else token_type_ids
        )
        if positions.shape not in {(1, s), (b, s)} or segments.shape != (b, s):
            raise ValueError("Invalid BERT position/segment layout")
        mask = make_attention_mask(
            b, s, s, padding=attention_mask, causal=False, device=hidden.device
        )
        hidden = self.bert.embeddings(hidden, positions, segments)
        states = [hidden] if output_hidden_states else None
        for layer in self.bert.encoder.layer:
            hidden = layer(hidden, mask)
            if states is not None:
                states.append(hidden)
        logits = self.cls.predictions.decoder(self.cls.predictions.transform(hidden))
        return TokenOutput(logits, hidden_states=tuple(states) if states is not None else None)
