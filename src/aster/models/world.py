"""Stochastic state-space world models with distinct reset, terminal, and truncation semantics."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F
from ..core import StateCapabilities
from .serialization import LocalModelMixin, configuration_key


def symlog(value):
    return value.sign() * value.abs().log1p()


def symexp(value):
    return value.sign() * value.abs().expm1()


def two_hot(values, support):
    if support.ndim != 1 or len(support) < 2 or not (support[1:] > support[:-1]).all():
        raise ValueError("Two-hot support must strictly increase")
    values = values.clamp(support[0], support[-1])
    upper = torch.searchsorted(support, values.contiguous()).clamp(1, len(support) - 1)
    lower = upper - 1
    upper_weight = (values - support[lower]) / (support[upper] - support[lower])
    targets = values.new_zeros(*values.shape, len(support))
    targets.scatter_add_(-1, lower[..., None], (1 - upper_weight)[..., None])
    targets.scatter_add_(-1, upper[..., None], upper_weight[..., None])
    return targets


class BlockLinear(nn.Module):
    def __init__(self, incoming, outgoing, blocks):
        super().__init__()
        if incoming % blocks or outgoing % blocks:
            raise ValueError("Block dimensions must divide evenly")
        self.blocks, self.incoming, self.outgoing = blocks, incoming, outgoing
        self.weight = nn.Parameter(torch.empty(blocks, incoming // blocks, outgoing // blocks))
        self.bias = nn.Parameter(torch.zeros(blocks, outgoing // blocks))
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, value):
        blocks = value.reshape(*value.shape[:-1], self.blocks, self.incoming // self.blocks)
        return (torch.einsum("...gi,gio->...go", blocks, self.weight) + self.bias).flatten(-2)


def _mlp(incoming, outgoing, hidden, layers=2):
    modules = []
    for _ in range(layers):
        modules += [
            nn.Linear(incoming, hidden),
            nn.RMSNorm(hidden, eps=1e-5),
            nn.GELU(approximate="tanh"),
        ]
        incoming = hidden
    modules += [nn.Linear(incoming, outgoing)]
    return nn.Sequential(*modules)


@dataclass(frozen=True)
class RSSMConfig:
    observation_dim: int = 16
    action_dim: int = 4
    deter_dim: int = 64
    stochastic_variables: int = 8
    classes: int = 8
    hidden_size: int = 64
    blocks: int = 4
    unimix: float = 0.01
    reward_bins: int = 255
    reward_low: float = -20.0
    reward_high: float = 20.0

    def __post_init__(self):
        if (
            min(
                self.observation_dim,
                self.action_dim,
                self.deter_dim,
                self.stochastic_variables,
                self.hidden_size,
                self.blocks,
            )
            < 1
            or self.classes < 2
            or self.reward_bins < 2
            or self.deter_dim % self.blocks
        ):
            raise ValueError("Invalid RSSM dimensions")
        if not 0 <= self.unimix < 1 or self.reward_low >= self.reward_high:
            raise ValueError("Invalid distribution support")

    @property
    def feature_dim(self):
        return self.deter_dim + self.stochastic_variables * self.classes

    def to_dict(self):
        return {"architecture": "rssm", **asdict(self)}


@dataclass
class RSSMState:
    deter: torch.Tensor
    stochastic: torch.Tensor
    logits: torch.Tensor
    config_key: str
    capabilities = StateCapabilities("rssm", forkable=True, reorderable=True, replayable=True)

    @property
    def features(self):
        return torch.cat((self.deter, self.stochastic.flatten(-2)), -1)

    def detach(self):
        return RSSMState(
            self.deter.detach(), self.stochastic.detach(), self.logits.detach(), self.config_key
        )

    def fork(self):
        return RSSMState(
            self.deter.clone(), self.stochastic.clone(), self.logits.clone(), self.config_key
        )

    def reorder(self, indices):
        return RSSMState(
            self.deter[indices], self.stochastic[indices], self.logits[indices], self.config_key
        )


class RSSMWorldModel(LocalModelMixin, nn.Module):
    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        self.config_key = configuration_key(config)
        h, d, z, g = (
            config.hidden_size,
            config.deter_dim,
            config.stochastic_variables * config.classes,
            config.blocks,
        )
        self.encoder = _mlp(config.observation_dim, h, h)
        self.deter_in, self.stochastic_in, self.action_in = [
            _mlp(incoming, h, h, layers=0) for incoming in (d, z, config.action_dim)
        ]
        self.core_norms = nn.ModuleList([nn.RMSNorm(h, eps=1e-5) for _ in range(3)])
        self.block_hidden = BlockLinear(d + g * 3 * h, d, g)
        self.block_norm = nn.RMSNorm(d, eps=1e-5)
        self.gates = BlockLinear(d, 3 * d, g)
        self.prior = _mlp(d, z, h)
        self.posterior = _mlp(d + h, z, h, layers=1)
        self.decoder = _mlp(config.feature_dim, config.observation_dim, h)
        self.reward_head = _mlp(config.feature_dim, config.reward_bins, h)
        self.continue_head = _mlp(config.feature_dim, 1, h)
        self.register_buffer(
            "reward_support",
            torch.linspace(config.reward_low, config.reward_high, config.reward_bins),
        )

    def initial(self, batch_size, *, device=None, dtype=None):
        p = next(self.parameters())
        device = p.device if device is None else device
        dtype = p.dtype if dtype is None else dtype
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device, dtype=dtype)
        latent = torch.zeros(
            batch_size,
            self.config.stochastic_variables,
            self.config.classes,
            device=device,
            dtype=dtype,
        )
        return RSSMState(deter, latent, latent.clone(), self.config_key)

    def log_probs(self, logits):

        probabilities = (
            logits.float().softmax(-1) * (1 - self.config.unimix)
            + self.config.unimix / self.config.classes
        )
        return probabilities.log()

    def _sample(self, logits, *, sample=True, generator=None):
        probabilities = self.log_probs(logits).exp()
        index = (
            torch.multinomial(
                probabilities.reshape(-1, self.config.classes), 1, generator=generator
            ).reshape(logits.shape[:-1])
            if sample
            else probabilities.argmax(-1)
        )
        hard = F.one_hot(index, self.config.classes).to(probabilities)

        return (hard + probabilities - probabilities.detach()).to(logits.dtype)

    def _core(self, state, action):
        if state.config_key != self.config_key or action.shape != (
            *state.deter.shape[:-1],
            self.config.action_dim,
        ):
            raise ValueError("RSSM state/action mismatch")
        action = action / action.abs().clamp_min(1).detach()
        inputs = (state.deter, state.stochastic.flatten(-2), action)
        projections = (self.deter_in, self.stochastic_in, self.action_in)
        parts = [
            F.gelu(norm(projection(value)), approximate="tanh")
            for value, projection, norm in zip(inputs, projections, self.core_norms)
        ]
        shared = torch.cat(parts, -1)[:, None].expand(-1, self.config.blocks, -1)
        grouped = torch.cat(
            (state.deter.reshape(len(action), self.config.blocks, -1), shared), -1
        ).flatten(1)
        hidden = F.gelu(self.block_norm(self.block_hidden(grouped)), approximate="tanh")
        grouped_gates = self.gates(hidden).reshape(len(action), self.config.blocks, -1)
        reset, candidate, update = [x.flatten(1) for x in grouped_gates.chunk(3, -1)]
        candidate = torch.tanh(reset.sigmoid() * candidate)
        update = (update - 1).sigmoid()
        return update * candidate + (1 - update) * state.deter

    def step(self, state, action, observation=None, *, reset=None, sample=True, generator=None):
        if reset is not None:
            if reset.shape != (len(action),) or reset.dtype != torch.bool:
                raise ValueError("Reset mask must be bool B")
            keep = (~reset).to(action.dtype)
            state = RSSMState(
                state.deter * keep[:, None],
                state.stochastic * keep[:, None, None],
                state.logits * keep[:, None, None],
                state.config_key,
            )
            action = action * keep[:, None]
        deter = self._core(state, action)
        prior = self.prior(deter).reshape(
            len(action), self.config.stochastic_variables, self.config.classes
        )
        logits = prior
        if observation is not None:
            embedded = self.encoder(symlog(observation))
            logits = self.posterior(torch.cat((deter, embedded), -1)).reshape_as(prior)
        state = RSSMState(
            deter, self._sample(logits, sample=sample, generator=generator), logits, self.config_key
        )
        return state, prior

    def observe(self, observations, actions, is_first, state=None, *, sample=True, generator=None):
        if (
            observations.ndim != 3
            or actions.shape[:2] != observations.shape[:2]
            or is_first.shape != observations.shape[:2]
            or is_first.dtype != torch.bool
        ):
            raise ValueError("Observe requires aligned BTD/BTA/BT sequence inputs")
        if observations.shape[-1] != self.config.observation_dim or observations.shape[1] < 1:
            raise ValueError("Invalid observation sequence")
        state = (
            self.initial(len(observations), device=observations.device, dtype=observations.dtype)
            if state is None
            else state
        )
        states, priors = [], []
        for t in range(observations.shape[1]):
            state, prior = self.step(
                state,
                actions[:, t],
                observations[:, t],
                reset=is_first[:, t],
                sample=sample,
                generator=generator,
            )
            states.append(state)
            priors.append(prior)
        stacked = RSSMState(
            *(
                torch.stack([getattr(s, name) for s in states], 1)
                for name in ("deter", "stochastic", "logits")
            ),
            self.config_key,
        )
        return stacked, torch.stack(priors, 1), state

    def imagine(self, state, actions, *, sample=True, generator=None):
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError("Imagine requires BTA actions")
        states = []
        for action in actions.unbind(1):
            state, _ = self.step(state, action, sample=sample, generator=generator)
            states.append(state)
        return RSSMState(
            *(
                torch.stack([getattr(s, name) for s in states], 1)
                for name in ("deter", "stochastic", "logits")
            ),
            self.config_key,
        )

    def predictions(self, state):
        features = state.features
        reward_logits = self.reward_head(features)
        return {
            "reconstruction_symlog": self.decoder(features),
            "reward_logits": reward_logits,
            "reward": symexp((reward_logits.softmax(-1) * self.reward_support).sum(-1)),
            "continue_logits": self.continue_head(features).squeeze(-1),
        }

    def forward(self, observations, actions, is_first, state=None):
        sequence, prior, final = self.observe(observations, actions, is_first, state)
        return {
            "state": sequence,
            "prior_logits": prior,
            "final_state": final,
            **self.predictions(sequence),
        }
