"""TD-MPC2 implicit world model and action prior."""

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .serialization import LocalModelMixin
from .world import symlog, symexp, two_hot


class SimNorm(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        if group_size < 1:
            raise ValueError("SimNorm group size must be positive")
        self.group_size = group_size

    def forward(self, value):
        if value.shape[-1] % self.group_size:
            raise ValueError("Latent width must divide into complete simplex groups")
        return value.reshape(*value.shape[:-1], -1, self.group_size).softmax(-1).reshape_as(value)


def normed_mlp(incoming, hidden, outgoing, *, final_activation=None, dropout=0.0):
    modules = []
    for index, width in enumerate(hidden):
        modules.extend(
            (
                nn.Linear(incoming, width),
                nn.Dropout(dropout if index == 0 else 0.0),
                nn.LayerNorm(width),
                nn.Mish(),
            )
        )
        incoming = width
    modules.append(nn.Linear(incoming, outgoing))
    if final_activation is not None:
        modules.extend((nn.LayerNorm(outgoing), final_activation))
    return nn.Sequential(*modules)


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.02, 0.02)


@dataclass(frozen=True)
class TDMPC2Config:
    observation_dim: int = 16
    action_dim: int = 4
    latent_dim: int = 64
    simnorm_dim: int = 8
    hidden_size: int = 64
    encoder_size: int = 64
    encoder_layers: int = 2
    num_q: int = 5
    num_bins: int = 101
    value_low: float = -10.0
    value_high: float = 10.0
    q_dropout: float = 0.01
    episodic: bool = False
    observation_kind: str = "state"
    image_channels: int = 3
    conv_channels: int = 4
    task_dim: int = 0
    action_dimensions: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "action_dimensions", tuple(self.action_dimensions))
        if (
            min(
                self.observation_dim,
                self.action_dim,
                self.latent_dim,
                self.simnorm_dim,
                self.hidden_size,
                self.encoder_size,
                self.encoder_layers,
            )
            < 1
            or self.num_q < 2
        ):
            raise ValueError("Invalid TD-MPC2 dimensions")
        if (
            self.latent_dim % self.simnorm_dim
            or self.num_bins < 2
            or self.value_low >= self.value_high
            or not 0 <= self.q_dropout < 1
        ):
            raise ValueError("Invalid simplex/value distribution")
        if self.observation_kind not in {"state", "rgb"}:
            raise ValueError("TD-MPC2 observation must be state or rgb")
        if self.observation_kind == "rgb" and (
            self.latent_dim != 16 * self.conv_channels or self.task_dim
        ):
            raise ValueError(
                "Official RGB encoder requires 64x64 images, latent=16*conv_channels and no task embedding"
            )
        if bool(self.task_dim) != bool(self.action_dimensions) or any(
            not 0 < dim <= self.action_dim for dim in self.action_dimensions
        ):
            raise ValueError("Multitask embedding and action dimensions must be declared together")
        if self.episodic and self.task_dim:
            raise ValueError("Public episodic termination path does not support multitask")

    def to_dict(self):
        return {"architecture": "tdmpc2_world", **asdict(self)}


@dataclass(frozen=True)
class TDMPC2PolicyConfig:
    feature_dim: int = 64
    action_dim: int = 4
    hidden_size: int = 64
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    def __post_init__(self):
        if (
            min(self.feature_dim, self.action_dim, self.hidden_size) < 1
            or self.log_std_min >= self.log_std_max
        ):
            raise ValueError("Invalid TD-MPC2 policy dimensions/variance support")

    def to_dict(self):
        return {"architecture": "tdmpc2_policy", **asdict(self)}


class QEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.networks = nn.ModuleList(
            normed_mlp(
                config.latent_dim + config.task_dim + config.action_dim,
                [config.hidden_size] * 2,
                config.num_bins,
                dropout=config.q_dropout,
            )
            for _ in range(config.num_q)
        )

    def forward(self, features):

        return torch.stack([network(features) for network in self.networks])


def random_shift(images, *, pad=3, generator=None):
    if images.ndim != 4 or images.shape[-2:] != (64, 64):
        raise ValueError("Public TD-MPC2 pixel encoder expects B,C,64,64")
    n, _, h, _ = images.shape
    padded = F.pad(images.float(), (pad, pad, pad, pad), mode="replicate")
    axis = torch.arange(h, device=images.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    shifts = torch.randint(2 * pad + 1, (n, 1, 1, 2), device=images.device, generator=generator)
    grid = (torch.stack((xx, yy), -1)[None] + shifts + 0.5) * (2 / (h + 2 * pad)) - 1
    return F.grid_sample(padded, grid, align_corners=False)


class TDMPC2WorldModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.task_embedding = (
            nn.Embedding(len(c.action_dimensions), c.task_dim, max_norm=None)
            if c.task_dim
            else None
        )
        mask = (
            torch.arange(c.action_dim)[None] < torch.tensor(c.action_dimensions)[:, None]
            if c.task_dim
            else torch.ones(1, c.action_dim, dtype=torch.bool)
        )
        self.register_buffer("action_masks", mask)
        if c.observation_kind == "state":
            self.encoder = normed_mlp(
                c.observation_dim + c.task_dim,
                [c.encoder_size] * max(c.encoder_layers - 1, 1),
                c.latent_dim,
                final_activation=SimNorm(c.simnorm_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(c.image_channels, c.conv_channels, 7, stride=2),
                nn.ReLU(),
                nn.Conv2d(c.conv_channels, c.conv_channels, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(c.conv_channels, c.conv_channels, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(c.conv_channels, c.conv_channels, 3),
                nn.Flatten(),
                SimNorm(c.simnorm_dim),
            )
        incoming = c.latent_dim + c.task_dim + c.action_dim
        self.dynamics = normed_mlp(
            incoming, [c.hidden_size] * 2, c.latent_dim, final_activation=SimNorm(c.simnorm_dim)
        )
        self.reward_head = normed_mlp(incoming, [c.hidden_size] * 2, c.num_bins)
        self.termination_head = (
            normed_mlp(c.latent_dim + c.task_dim, [c.hidden_size] * 2, 1) if c.episodic else None
        )
        self.q_heads = QEnsemble(c)
        self.register_buffer("value_support", torch.linspace(c.value_low, c.value_high, c.num_bins))
        self.apply(initialize)
        nn.init.zeros_(self.reward_head[-1].weight)
        for network in self.q_heads.networks:
            nn.init.zeros_(network[-1].weight)

    def condition(self, features, task=None):
        if self.task_embedding is None:
            if task is not None:
                raise ValueError("Single-task model must not silently accept task IDs")
            return features
        if task is None or task.dtype != torch.long or task.shape != features.shape[:-1]:
            raise ValueError("Task IDs must align with every feature position")
        return torch.cat((features, self.task_embedding(task)), -1)

    def action_mask(self, features, task=None):
        return (
            self.action_masks[task]
            if self.task_embedding is not None
            else self.action_masks[0].expand(*features.shape[:-1], -1)
        )

    def encode(self, observations, task=None, *, generator=None):
        if self.config.observation_kind == "state":
            return self.encoder(self.condition(observations, task))
        leading = observations.shape[:-3]
        pixels = observations.reshape(-1, *observations.shape[-3:])
        if (
            pixels.shape[1] != self.config.image_channels
            or not torch.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 255
        ):
            raise ValueError(
                "Pixel values must be explicitly in [0,255] before TD-MPC2 preprocessing"
            )
        value = self.encoder(random_shift(pixels, generator=generator) / 255.0 - 0.5)
        return value.reshape(*leading, self.config.latent_dim)

    def next(self, latent, action, task=None):
        return self.dynamics(torch.cat((self.condition(latent, task), action), -1))

    def reward(self, latent, action, task=None):
        return self.reward_head(torch.cat((self.condition(latent, task), action), -1))

    def termination(self, latent, task=None):
        if self.termination_head is None:
            raise ValueError("No episodic termination head configured")
        return self.termination_head(self.condition(latent, task)).squeeze(-1)

    def decode_value(self, logits):
        return symexp((logits.float().softmax(-1) * self.value_support).sum(-1))

    def value_loss(self, logits, values):
        targets = two_hot(symlog(values.float()), self.value_support)
        return -(targets * logits.float().log_softmax(-1)).sum(-1)

    def q(self, latent, action, task=None, *, ensemble=None, reduction="all", generator=None):
        logits = (self.q_heads if ensemble is None else ensemble)(
            torch.cat((self.condition(latent, task), action), -1)
        )
        if reduction == "all":
            return logits
        if reduction not in {"min", "avg"}:
            raise ValueError("Q reduction must be all/min/avg")
        selected = torch.randperm(self.config.num_q, device=logits.device, generator=generator)[:2]
        values = self.decode_value(logits[selected])
        return values.min(0).values if reduction == "min" else values.mean(0)

    def forward(self, observations, actions, task=None):
        latent = self.encode(observations, task)
        return {
            "latent": latent,
            "next_latent": self.next(latent, actions, task),
            "reward_logits": self.reward(latent, actions, task),
            "q_logits": self.q(latent, actions, task),
        }


class TDMPC2Policy(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = normed_mlp(
            config.feature_dim, [config.hidden_size] * 2, config.action_dim * 2
        )
        self.apply(initialize)

    def forward(self, features, *, action_mask=None, noise=None, generator=None):
        mean, raw_std = self.network(features).chunk(2, -1)
        c = self.config
        log_std = c.log_std_min + (c.log_std_max - c.log_std_min) * (raw_std.tanh() + 1) / 2
        noise = (
            torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            if noise is None
            else noise
        )
        if noise.shape != mean.shape:
            raise ValueError("Policy reparameterization noise shape mismatch")
        mask = torch.ones_like(mean) if action_mask is None else action_mask.to(mean)
        if (
            mask.shape != mean.shape
            or not ((mask == 0) | (mask == 1)).all()
            or not mask.sum(-1).gt(0).all()
        ):
            raise ValueError("Action mask must select a nonempty set of binary active dimensions")
        mean, log_std, noise = mean * mask, log_std * mask, noise * mask
        gaussian_logp = (-0.5 * noise.square() - log_std - 0.9189385175704956).sum(-1)
        action = (mean + noise * log_std.exp()).tanh()
        logp = gaussian_logp - (F.relu(1 - action.square()) + 1e-6).log().sum(-1)

        scaled_logp = gaussian_logp * mask.sum(-1)
        entropy_scale = scaled_logp / (logp + 1e-8)
        return action, {
            "mean": mean.tanh(),
            "log_std": log_std,
            "entropy": -logp,
            "scaled_entropy": -logp * entropy_scale,
        }
