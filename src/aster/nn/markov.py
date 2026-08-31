"""DSpark low-rank vocabulary correction with per-token Markov conditioning."""

import torch
from torch import nn


class MarkovHead(nn.Module):
    def __init__(self, vocab_size, rank, hidden_size, kind="gated"):
        super().__init__()
        if any(
            type(x) is not int or x < 1 for x in (vocab_size, rank, hidden_size)
        ) or kind not in {"vanilla", "gated", "rnn"}:
            raise ValueError("Invalid DSpark Markov head configuration")
        self.vocab_size, self.rank, self.hidden_size, self.kind = (
            vocab_size,
            rank,
            hidden_size,
            kind,
        )
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)
        if kind == "gated":
            self.gate_proj = nn.Linear(hidden_size + rank, rank)
        if kind == "rnn":
            self.joint_proj = nn.Linear(hidden_size + 2 * rank, 3 * rank)

    def get_prev_embeddings(self, token_ids):
        if (
            token_ids.dtype != torch.int64
            or (token_ids < 0).any()
            or (token_ids >= self.vocab_size).any()
        ):
            raise ValueError("Previous token IDs must be int64 in the declared vocabulary")
        return self.markov_w1(token_ids)

    def step(self, hidden, previous_ids, state=None):
        if hidden.shape[:-1] != previous_ids.shape or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Markov hidden states and previous token layout differ")
        previous = self.get_prev_embeddings(previous_ids)
        if self.kind == "vanilla":
            if state is not None:
                raise ValueError("Vanilla Markov head has no recurrent state")
            value, updated = previous, None
        elif self.kind == "gated":
            if state is not None:
                raise ValueError("Gated Markov head has no recurrent state")
            gate = self.gate_proj(torch.cat((hidden, previous), -1)).sigmoid().to(previous)
            value, updated = gate * previous, None
        else:
            if state is None:
                state = torch.zeros_like(previous)
            if state.shape != previous.shape or state.device != previous.device:
                raise ValueError("DSpark block-local RNN state layout differs")
            gate, candidate, output = self.joint_proj(
                torch.cat((state, previous, hidden), -1)
            ).chunk(3, -1)
            gate = gate.sigmoid()
            updated = gate * state + (1 - gate) * candidate.tanh()

            value = output.tanh()
        return self.markov_w2(value), updated

    def forward(self, hidden, previous_ids):

        if hidden.ndim < 3 or hidden.shape[:-1] != previous_ids.shape or hidden.shape[-2] < 1:
            raise ValueError(
                "Markov block needs nonempty [...,positions,hidden] and aligned previous IDs"
            )
        if self.kind != "rnn":
            return self.step(hidden, previous_ids)[0]
        state, bias = None, []
        for index in range(hidden.shape[-2]):
            value, state = self.step(hidden[..., index, :], previous_ids[..., index], state)
            bias.append(value)
        return torch.stack(bias, -2)
