"""Differentiable MuZero representation, dynamics, prediction, and categorical value support."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .serialization import LocalModelMixin


def scalar_transform(value, epsilon=0.001):
    """Apply sign(x) * (sqrt(abs(x) + 1) - 1) + epsilon * x, not symlog."""
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("Value transform epsilon must be finite and nonnegative")
    return value.sign() * (value.abs().add(1).sqrt() - 1) + epsilon * value


def inverse_scalar_transform(value, epsilon=0.001):
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("Value transform epsilon must be finite and nonnegative")

    constant = value.abs() + 1 + epsilon
    root = 2 * constant / (1 + (1 + 4 * epsilon * constant).sqrt())
    return value.sign() * (root.square() - 1).clamp_min(0)


def support_targets(value, support_size, epsilon=0.001):
    """Clip transformed values to categorical support and interpolate adjacent integer bins."""
    if type(support_size) is not int or support_size < 1:
        raise ValueError("MuZero categorical support must have positive radius")
    transformed = scalar_transform(value.float(), epsilon).clamp(-support_size, support_size)
    lower = transformed.floor()
    fraction = transformed - lower
    low_index = (lower + support_size).long()
    high_index = (low_index + 1).clamp_max(2 * support_size)
    target = transformed.new_zeros(*transformed.shape, 2 * support_size + 1)
    target.scatter_add_(-1, low_index[..., None], (1 - fraction)[..., None])
    target.scatter_add_(-1, high_index[..., None], fraction[..., None])
    return target


def scale_gradient(value, scale):
    """Keep forward values unchanged and scale only their backward gradient."""
    return value * scale + value.detach() * (1 - scale)


@dataclass(frozen=True)
class MuZeroConfig:
    observation_dim: int = 16
    num_actions: int = 4
    latent_dim: int = 32
    hidden_size: int = 64
    support_size: int = 20
    transform_epsilon: float = 0.001
    state_epsilon: float = 1e-5
    discount: float = 0.997
    dynamics_gradient_scale: float = 0.5

    def __post_init__(self):
        dimensions = (
            self.observation_dim,
            self.num_actions,
            self.latent_dim,
            self.hidden_size,
            self.support_size,
        )
        if any(type(value) is not int or value < 1 for value in dimensions) or self.latent_dim < 2:
            raise ValueError("MuZero dimensions must be positive and latent_dim >= 2")
        values = (
            self.transform_epsilon,
            self.state_epsilon,
            self.discount,
            self.dynamics_gradient_scale,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or self.transform_epsilon < 0
            or self.state_epsilon <= 0
            or not 0 <= self.discount <= 1
            or not 0 < self.dynamics_gradient_scale <= 1
        ):
            raise ValueError("Invalid MuZero normalization/discount/gradient scale")

    def to_dict(self):
        return {"architecture": "muzero_vector", **asdict(self)}


@dataclass
class MuZeroOutput:
    embedding: torch.Tensor
    prior_logits: torch.Tensor
    value_logits: torch.Tensor
    reward_logits: torch.Tensor | None


def _network(incoming, outgoing, hidden):
    return nn.Sequential(
        nn.Linear(incoming, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outgoing),
    )


class MuZeroModel(LocalModelMixin, nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.representation = _network(
            config.observation_dim, config.latent_dim, config.hidden_size
        )
        self.dynamics = _network(
            config.latent_dim + config.num_actions, config.latent_dim, config.hidden_size
        )
        self.policy = _network(config.latent_dim, config.num_actions, config.hidden_size)
        self.value = _network(config.latent_dim, 2 * config.support_size + 1, config.hidden_size)
        self.reward = _network(config.latent_dim, 2 * config.support_size + 1, config.hidden_size)

    def _normalize(self, latent):
        minimum, maximum = latent.amin(-1, keepdim=True), latent.amax(-1, keepdim=True)
        return (latent - minimum) / (maximum - minimum).clamp_min(self.config.state_epsilon)

    def initial(self, observations):
        if (
            observations.ndim != 2
            or observations.shape[-1] != self.config.observation_dim
            or not len(observations)
        ):
            raise ValueError("MuZero vector representation needs nonempty B,observation_dim input")
        embedding = self._normalize(self.representation(observations))
        return MuZeroOutput(embedding, self.policy(embedding), self.value(embedding), None)

    def recurrent(self, action, embedding, *, scale_dynamics_gradient=True):
        if (
            embedding.ndim != 2
            or embedding.shape[-1] != self.config.latent_dim
            or action.shape != (len(embedding),)
            or action.dtype not in (torch.int32, torch.int64)
            or (action < 0).any()
            or (action >= self.config.num_actions).any()
        ):
            raise ValueError(
                "MuZero recurrence needs aligned latent vectors and valid discrete actions"
            )
        encoded = F.one_hot(action.long(), self.config.num_actions).to(embedding)

        previous = (
            scale_gradient(embedding, self.config.dynamics_gradient_scale)
            if scale_dynamics_gradient
            else embedding
        )
        raw_next = self.dynamics(torch.cat((previous, encoded), -1))
        reward_logits = self.reward(raw_next)
        next_embedding = self._normalize(raw_next)
        return MuZeroOutput(
            next_embedding, self.policy(next_embedding), self.value(next_embedding), reward_logits
        )

    def decode_value(self, logits):
        support = torch.arange(
            -self.config.support_size,
            self.config.support_size + 1,
            device=logits.device,
            dtype=torch.float32,
        )
        return inverse_scalar_transform(
            (logits.float().softmax(-1) * support).sum(-1), self.config.transform_epsilon
        )

    def forward(self, observations, actions=None):
        output = self.initial(observations)
        if actions is None:
            return output
        if actions.ndim != 2 or actions.shape[0] != len(observations):
            raise ValueError("MuZero training action unroll must have shape B,K")
        predictions = [output]
        for step in range(actions.shape[1]):
            output = self.recurrent(actions[:, step], output.embedding)
            predictions.append(output)
        return tuple(predictions)

    def search_root(self, observations):
        from ..planning.mcts import RootOutput

        prediction = self.initial(observations)
        return RootOutput(
            prediction.prior_logits,
            self.decode_value(prediction.value_logits),
            prediction.embedding,
        )

    def search_step(self, action, embedding):
        from ..planning.mcts import RecurrentOutput

        prediction = self.recurrent(action, embedding, scale_dynamics_gradient=False)
        value = self.decode_value(prediction.value_logits)
        return RecurrentOutput(
            self.decode_value(prediction.reward_logits),
            torch.full_like(value, self.config.discount),
            prediction.prior_logits,
            value,
            prediction.embedding,
        )
