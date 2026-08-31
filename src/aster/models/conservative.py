"""Native continuous CQL policies and independent twin-Q networks."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from .serialization import LocalModelMixin


def _initialize_hidden(layer):

    nn.init.uniform_(
        layer.weight, -1 / math.sqrt(layer.out_features), 1 / math.sqrt(layer.out_features)
    )
    nn.init.constant_(layer.bias, 0.1)


@dataclass(frozen=True)
class CQLPolicyConfig:
    observation_dim: int
    action_dim: int
    hidden: int = 256
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    epsilon: float = 1e-6
    output_init: float = 1e-3

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.observation_dim, self.action_dim, self.hidden)
        ):
            raise ValueError("CQL dimensions must be positive integers")
        if not all(
            math.isfinite(value)
            for value in (self.log_std_min, self.log_std_max, self.epsilon, self.output_init)
        ):
            raise ValueError("CQL distribution parameters must be finite")
        if self.log_std_min >= self.log_std_max or self.epsilon <= 0 or self.output_init <= 0:
            raise ValueError("Invalid CQL distribution support/initialization")

    def to_dict(self):
        return {"architecture": "cql_policy", **asdict(self)}


class CQLPolicy(LocalModelMixin, nn.Module):
    def __init__(self, config: CQLPolicyConfig):
        super().__init__()
        self.config = config
        self.hidden = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden, config.action_dim)
        self.log_std = nn.Linear(config.hidden, config.action_dim)
        for layer in (self.hidden[0], self.hidden[2]):
            _initialize_hidden(layer)
        for layer in (self.mean, self.log_std):
            nn.init.uniform_(layer.weight, -config.output_init, config.output_init)
            nn.init.uniform_(layer.bias, -config.output_init, config.output_init)

    def _parameters_at(self, observations):
        hidden = self.hidden(observations)

        mean = self.mean(hidden).float()
        log_std = (
            self.log_std(hidden).float().clamp(self.config.log_std_min, self.config.log_std_max)
        )
        return mean, log_std

    def _density(self, raw, action, mean, log_std):
        normal = (
            -0.5 * ((raw - mean) * (-log_std).exp()).square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
        return (normal - (1 - action.square() + self.config.epsilon).log()).sum(-1)

    def forward(self, observations, *, noise=None, deterministic=False):
        mean, log_std = self._parameters_at(observations)
        if deterministic:
            if noise is not None:
                raise ValueError("Deterministic policy does not consume noise")
            raw = mean
        else:
            noise = torch.randn_like(mean) if noise is None else noise
            if noise.shape != mean.shape or noise.device != mean.device:
                raise ValueError("Policy noise must match each action coordinate")
            raw = mean + log_std.exp() * noise
        action = raw.tanh()
        return action, self._density(raw, action, mean, log_std)

    def log_prob(self, observations, actions):
        mean, log_std = self._parameters_at(observations)

        raw = (
            0.5
            * (
                (1 + actions).clamp_min(self.config.epsilon)
                / (1 - actions).clamp_min(self.config.epsilon)
            ).log()
        )
        return self._density(raw, actions, mean.clamp(-9, 9), log_std)


class CQLTwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=256):
        super().__init__()
        if min(observation_dim, action_dim, hidden) < 1:
            raise ValueError("CQL critic dimensions must be positive")
        self.dimensions = (observation_dim, action_dim, hidden)

        def network():
            return nn.Sequential(
                nn.Linear(observation_dim + action_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1, self.q2 = network(), network()
        for network in (self.q1, self.q2):
            for layer in (network[0], network[2]):
                _initialize_hidden(layer)
            nn.init.uniform_(network[4].weight, -0.003, 0.003)
            nn.init.uniform_(network[4].bias, -0.003, 0.003)

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), -1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)
