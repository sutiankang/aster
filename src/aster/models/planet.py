"""PlaNet continuous Gaussian RSSM with explicit action/observation alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import StateCapabilities
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class PlaNetConfig:
    action_dim: int = 6
    state_size: int = 30
    belief_size: int = 200
    hidden_size: int = 200
    model_layers: int = 1
    activation: str = "relu"
    min_stddev: float = 0.1
    mean_only: bool = False
    future_rnn: bool = True
    image_channels: int = 3
    conv_channels: int = 32
    reward_hidden_size: int = 300
    reward_layers: int = 3

    observation_dim: int = 0

    def __post_init__(self):
        integers = (
            self.action_dim,
            self.state_size,
            self.belief_size,
            self.hidden_size,
            self.model_layers,
            self.image_channels,
            self.conv_channels,
            self.reward_hidden_size,
            self.reward_layers,
        )
        if any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("PlaNet dimensions must be positive integers")
        if type(self.observation_dim) is not int or self.observation_dim < 0:
            raise ValueError("PlaNet observation_dim must be nonnegative")
        if self.activation not in {"relu", "elu", "tanh", "swish", "softplus", "none"}:
            raise ValueError("Unsupported PlaNet activation")
        if not math.isfinite(self.min_stddev) or self.min_stddev <= 0:
            raise ValueError("PlaNet min_stddev must be finite and positive")
        if type(self.mean_only) is not bool or type(self.future_rnn) is not bool:
            raise ValueError("PlaNet mode flags must be boolean")

    @property
    def feature_dim(self):
        return self.state_size + self.belief_size

    @property
    def observation_shape(self):
        return (self.observation_dim,) if self.observation_dim else (self.image_channels, 64, 64)

    def to_dict(self):
        return dict(architecture="planet", **asdict(self))


def _activation(name):
    return {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "swish": nn.SiLU,
        "softplus": nn.Softplus,
        "none": nn.Identity,
    }[name]()


def _hidden(incoming, width, layers, activation):
    blocks = []
    for _ in range(layers):
        blocks.extend((nn.Linear(incoming, width), _activation(activation)))
        incoming = width
    return nn.Sequential(*blocks)


class PlaNetGRU(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.width = width
        self.gate_kernel = nn.Parameter(torch.empty(incoming + width, 2 * width))
        self.gate_bias = nn.Parameter(torch.ones(2 * width))
        self.candidate_kernel = nn.Parameter(torch.empty(incoming + width, width))
        self.candidate_bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.gate_kernel)
        nn.init.xavier_uniform_(self.candidate_kernel)

    def forward(self, value, previous):
        reset, update = (
            (torch.cat((value, previous), -1) @ self.gate_kernel + self.gate_bias)
            .sigmoid()
            .chunk(2, -1)
        )
        candidate = (
            torch.cat((value, reset * previous), -1) @ self.candidate_kernel + self.candidate_bias
        ).tanh()
        return update * previous + (1 - update) * candidate


class PlaNetImageEncoder(nn.Module):
    def __init__(self, channels, base):
        super().__init__()
        sequence = []
        for outgoing in (base, 2 * base, 4 * base, 8 * base):
            sequence.extend((nn.Conv2d(channels, outgoing, 4, stride=2), nn.ReLU()))
            channels = outgoing
        self.convolutions = nn.Sequential(*sequence)

    def forward(self, value):

        return self.convolutions(value).permute(0, 2, 3, 1).flatten(1)


class PlaNetImageDecoder(nn.Module):
    def __init__(self, features, channels, base):
        super().__init__()
        self.projection = nn.Linear(features, 32 * base)
        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(32 * base, 4 * base, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * base, 2 * base, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * base, base, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(base, channels, 6, stride=2),
        )

    def forward(self, features):
        return self.deconvolutions(self.projection(features)[..., None, None])


@dataclass
class PlaNetState:
    mean: torch.Tensor
    stddev: torch.Tensor
    sample: torch.Tensor
    belief: torch.Tensor
    config_key: str
    capabilities = StateCapabilities(
        "planet_gaussian_rssm", forkable=True, reorderable=True, replayable=True
    )

    @property
    def features(self):
        return torch.cat((self.sample, self.belief), -1)

    def map(self, fn):
        return PlaNetState(
            *(fn(v) for v in (self.mean, self.stddev, self.sample, self.belief)), self.config_key
        )

    def detach(self):
        return self.map(torch.Tensor.detach)

    def fork(self):
        return self.map(torch.Tensor.clone)

    def reorder(self, indices):
        return self.map(lambda v: v[indices])


def stack_planet_states(states, dim=1):
    if not states or any(s.config_key != states[0].config_key for s in states):
        raise ValueError("Cannot combine incompatible PlaNet states")
    return PlaNetState(
        *(
            torch.stack([getattr(s, k) for s in states], dim)
            for k in ("mean", "stddev", "sample", "belief")
        ),
        states[0].config_key,
    )


class PlaNetWorldModel(LocalModelMixin, nn.Module):
    def __init__(self, config: PlaNetConfig):
        super().__init__()
        self.config, self.config_key = config, configuration_key(config)
        c = config
        if c.observation_dim:
            self.encoder = _hidden(c.observation_dim, c.hidden_size, 2, c.activation)
            self.decoder = nn.Sequential(
                _hidden(c.feature_dim, c.hidden_size, 2, c.activation),
                nn.Linear(c.hidden_size, c.observation_dim),
            )
            encoded = c.hidden_size
        else:
            self.encoder = PlaNetImageEncoder(c.image_channels, c.conv_channels)
            self.decoder = PlaNetImageDecoder(c.feature_dim, c.image_channels, c.conv_channels)
            encoded = 32 * c.conv_channels
        self.transition_input = _hidden(
            c.state_size + c.action_dim, c.hidden_size, c.model_layers, c.activation
        )
        self.gru = PlaNetGRU(c.hidden_size, c.belief_size)
        self.prior_hidden = _hidden(
            c.belief_size if c.future_rnn else c.hidden_size,
            c.hidden_size,
            c.model_layers,
            c.activation,
        )
        self.prior_mean, self.prior_stddev = (
            nn.Linear(c.hidden_size, c.state_size),
            nn.Linear(c.hidden_size, c.state_size),
        )
        self.posterior_hidden = _hidden(
            c.belief_size + encoded, c.hidden_size, c.model_layers, c.activation
        )
        self.posterior_mean, self.posterior_stddev = (
            nn.Linear(c.hidden_size, c.state_size),
            nn.Linear(c.hidden_size, c.state_size),
        )
        self.reward_head = nn.Sequential(
            _hidden(c.feature_dim, c.reward_hidden_size, c.reward_layers, c.activation),
            nn.Linear(c.reward_hidden_size, 1),
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def initial(self, batch_size, *, device=None, dtype=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("PlaNet state needs a positive batch size")
        reference = next(self.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        z = torch.zeros(batch_size, self.config.state_size, device=device, dtype=dtype)
        h = torch.zeros(batch_size, self.config.belief_size, device=device, dtype=dtype)

        return PlaNetState(z, z.clone(), z.clone(), h, self.config_key)

    def _check_state(self, state, batch_size):
        if not isinstance(state, PlaNetState) or state.config_key != self.config_key:
            raise ValueError("PlaNet state configuration differs")
        if state.sample.shape != (batch_size, self.config.state_size) or state.belief.shape != (
            batch_size,
            self.config.belief_size,
        ):
            raise ValueError("PlaNet step requires a matching single-time state")

    def _distribution(self, hidden, belief, mean_head, stddev_head, noise, generator):

        mean = mean_head(hidden).float()
        stddev = F.softplus(stddev_head(hidden).float()) + self.config.min_stddev
        if self.config.mean_only:
            sample = mean
        else:
            if noise is None:
                noise = torch.randn(
                    mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
                )
            if (
                noise.shape != mean.shape
                or noise.device != mean.device
                or not torch.isfinite(noise).all()
            ):
                raise ValueError("PlaNet latent noise shape/device/values differ")
            sample = mean + stddev * noise
        return PlaNetState(mean, stddev, sample, belief, self.config_key)

    def transition(self, state, action, *, noise=None, generator=None):
        self._check_state(state, len(action))
        if action.shape != (len(action), self.config.action_dim):
            raise ValueError("PlaNet action shape differs")
        hidden = self.transition_input(torch.cat((state.sample, action), -1))
        belief = self.gru(hidden, state.belief)
        hidden = self.prior_hidden(belief if self.config.future_rnn else hidden)
        return self._distribution(
            hidden, belief, self.prior_mean, self.prior_stddev, noise, generator
        )

    def posterior(self, prior, embedded, *, noise=None, generator=None):
        hidden = self.posterior_hidden(torch.cat((prior.belief, embedded), -1))
        return self._distribution(
            hidden, prior.belief, self.posterior_mean, self.posterior_stddev, noise, generator
        )

    def observe(
        self,
        observations,
        previous_actions,
        is_first,
        *,
        initial=None,
        prior_noise=None,
        posterior_noise=None,
        generator=None,
    ):

        b, t = observations.shape[:2]
        if (
            tuple(observations.shape[2:]) != self.config.observation_shape
            or previous_actions.shape != (b, t, self.config.action_dim)
            or is_first.shape != (b, t)
            or is_first.dtype != torch.bool
            or min(b, t) < 1
        ):
            raise ValueError("PlaNet observation/action/reset shapes differ")
        encoded = self.encoder(observations.flatten(0, 1)).reshape(b, t, -1)
        state = (
            self.initial(b, device=observations.device, dtype=observations.dtype)
            if initial is None
            else initial
        )
        self._check_state(state, b)
        priors, posteriors = [], []
        for index in range(t):
            keep = (~is_first[:, index]).unsqueeze(-1)
            state = state.map(lambda v: v * keep)
            action = previous_actions[:, index] * keep
            prior = self.transition(
                state,
                action,
                noise=None if prior_noise is None else prior_noise[:, index],
                generator=generator,
            )
            state = self.posterior(
                prior,
                encoded[:, index],
                noise=None if posterior_noise is None else posterior_noise[:, index],
                generator=generator,
            )
            priors.append(prior)
            posteriors.append(state)
        return stack_planet_states(posteriors), stack_planet_states(priors)

    def predictions(self, state):
        features = state.features
        leading = features.shape[:-1]
        flattened = features.reshape(-1, self.config.feature_dim)
        reconstruction = self.decoder(flattened).reshape(*leading, *self.config.observation_shape)
        reward = self.reward_head(flattened).reshape(leading)
        return dict(reconstruction=reconstruction, reward=reward)

    def forward(self, observations, previous_actions, is_first, **kwargs):
        posterior, prior = self.observe(observations, previous_actions, is_first, **kwargs)
        return dict(state=posterior, prior=prior, **self.predictions(posterior))

    def imagine(self, initial, actions, *, noise=None, generator=None):
        if actions.ndim != 3 or actions.shape[-1] != self.config.action_dim or actions.shape[1] < 1:
            raise ValueError("PlaNet imagination requires [B,T,A] actions")
        state, states = initial, []
        for index in range(actions.shape[1]):
            state = self.transition(
                state,
                actions[:, index],
                noise=None if noise is None else noise[:, index],
                generator=generator,
            )
            states.append(state)
        return stack_planet_states(states)
