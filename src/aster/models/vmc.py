"""Native World Models vision, memory, and controller components for non-driving tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import StateCapabilities
from .planet import PlaNetImageEncoder, PlaNetImageDecoder
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class VMCVAEConfig:
    latent_size: int = 64
    image_channels: int = 3
    conv_channels: int = 32

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.latent_size, self.image_channels, self.conv_channels)
        ):
            raise ValueError("VMC VAE dimensions must be positive integers")

    def to_dict(self):
        return dict(architecture="vmc_vae", **asdict(self))


@dataclass
class VMCVAEOutput:
    reconstruction: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor


class VMCVAE(LocalModelMixin, nn.Module):
    def __init__(self, config: VMCVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = PlaNetImageEncoder(config.image_channels, config.conv_channels)
        self.mean = nn.Linear(32 * config.conv_channels, config.latent_size)
        self.logvar = nn.Linear(32 * config.conv_channels, config.latent_size)
        self.decoder = PlaNetImageDecoder(
            config.latent_size, config.image_channels, config.conv_channels
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (self.config.image_channels, 64, 64):
            raise ValueError("World Models VAE requires 64x64 BCHW images")
        hidden = self.encoder(images)
        return self.mean(hidden).float(), self.logvar(hidden).float()

    def decode(self, latent):
        return self.decoder(latent).sigmoid()

    def forward(self, images, *, noise=None, generator=None):
        mean, logvar = self.encode(images)
        if noise is None:
            noise = torch.randn(mean.shape, device=mean.device, generator=generator)
        if (
            noise.shape != mean.shape
            or noise.device != mean.device
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("VMC VAE noise must match its latent distribution")
        latent = mean + (0.5 * logvar).exp() * noise
        return VMCVAEOutput(self.decode(latent), mean, logvar, latent)


@dataclass(frozen=True)
class MDNRNNConfig:
    latent_size: int = 64
    action_dim: int = 1
    hidden_size: int = 512
    mixtures: int = 5
    layer_norm: bool = False
    input_dropout: float = 0.0
    output_dropout: float = 0.0
    recurrent_dropout: float = 0.0

    def __post_init__(self):
        if any(
            type(v) is not int or v < 1
            for v in (self.latent_size, self.action_dim, self.hidden_size, self.mixtures)
        ):
            raise ValueError("MDN-RNN dimensions must be positive integers")
        if type(self.layer_norm) is not bool or any(
            not math.isfinite(v) or not 0 <= v < 1
            for v in (self.input_dropout, self.output_dropout, self.recurrent_dropout)
        ):
            raise ValueError("Invalid MDN-RNN normalization/dropout")

    def to_dict(self):
        return dict(architecture="vmc_mdn_rnn", **asdict(self))


@dataclass
class MDNState:
    cell: torch.Tensor
    hidden: torch.Tensor
    config_key: str
    capabilities = StateCapabilities(
        "vmc_mdn_lstm", forkable=True, reorderable=True, replayable=True
    )

    def detach(self):
        return MDNState(self.cell.detach(), self.hidden.detach(), self.config_key)

    def fork(self):
        return MDNState(self.cell.clone(), self.hidden.clone(), self.config_key)

    def reorder(self, indices):
        return MDNState(self.cell[indices], self.hidden[indices], self.config_key)


class VMCLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.projection = nn.Linear(
            config.latent_size + config.action_dim + 1 + h, 4 * h, bias=not config.layer_norm
        )
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        self.norms = (
            nn.ModuleList([nn.LayerNorm(h, eps=1e-12) for _ in range(5)])
            if config.layer_norm
            else nn.ModuleList()
        )
        self.input_dropout, self.output_dropout, self.recurrent_dropout = (
            config.input_dropout,
            config.output_dropout,
            config.recurrent_dropout,
        )

    def forward(self, value, cell, hidden):
        value = F.dropout(value, self.input_dropout, self.training)
        gates = self.projection(torch.cat((value, hidden), -1)).chunk(4, -1)
        if self.norms:
            gates = tuple(self.norms[k](v) for k, v in enumerate(gates))
        incoming, candidate, forget, outgoing = gates
        candidate = F.dropout(candidate.tanh(), self.recurrent_dropout, self.training)
        cell = cell * (forget + 1).sigmoid() + incoming.sigmoid() * candidate
        if self.norms:
            cell = self.norms[4](cell)
        hidden = cell.tanh() * outgoing.sigmoid()
        output = F.dropout(hidden, self.output_dropout, self.training)

        return output, cell, hidden


@dataclass
class MDNOutput:
    logmix: torch.Tensor
    mean: torch.Tensor
    logstd: torch.Tensor
    restart_logits: torch.Tensor
    state: MDNState


class MDNRNN(LocalModelMixin, nn.Module):
    def __init__(self, config: MDNRNNConfig):
        super().__init__()
        self.config, self.config_key = config, configuration_key(config)
        self.lstm = VMCLSTM(config)
        self.output = nn.Linear(config.hidden_size, 1 + 3 * config.latent_size * config.mixtures)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initial(self, batch_size, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("MDN-RNN needs positive batch size")
        zeros = torch.zeros(batch_size, self.config.hidden_size, device=device, dtype=dtype)
        return MDNState(zeros, zeros.clone(), self.config_key)

    def forward(self, latents, actions, restart, *, state=None):
        c = self.config
        if latents.ndim != 3 or latents.shape[-1] != c.latent_size or min(latents.shape[:2]) < 1:
            raise ValueError("MDN-RNN latents must be [B,T,Z]")
        b, t = latents.shape[:2]
        if (
            actions.shape != (b, t, c.action_dim)
            or restart.shape != (b, t)
            or restart.dtype != torch.bool
        ):
            raise ValueError("MDN-RNN current-action/restart dimensions differ")
        state = (
            self.initial(b, device=latents.device, dtype=latents.dtype) if state is None else state
        )
        if (
            not isinstance(state, MDNState)
            or state.config_key != self.config_key
            or state.cell.shape != (b, c.hidden_size)
            or state.hidden.shape != state.cell.shape
        ):
            raise ValueError("MDN-RNN state configuration/shape differs")
        cell, hidden = state.cell, state.hidden
        outputs = []
        for index in range(t):
            keep = (~restart[:, index, None]).to(cell)
            cell, hidden = cell * keep, hidden * keep
            value = torch.cat(
                (latents[:, index], actions[:, index], restart[:, index, None].to(latents)), -1
            )
            output, cell, hidden = self.lstm(value, cell, hidden)
            outputs.append(output)
        projected = self.output(torch.stack(outputs, 1)).float()

        mixture = projected[..., 1:].reshape(b, t, c.latent_size, 3 * c.mixtures)
        logits, mean, logstd = mixture.chunk(3, -1)
        return MDNOutput(
            logits.log_softmax(-1),
            mean,
            logstd,
            projected[..., 0],
            MDNState(cell, hidden, self.config_key),
        )


def sample_mdn(output, *, temperature=1.25, generator=None):
    """Scale mixture logits by 1/T and Gaussian standard deviation by sqrt(T)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MDN sampling temperature must be finite and positive")
    if any(
        not torch.isfinite(v).all()
        for v in (output.logmix, output.mean, output.logstd, output.restart_logits)
    ):
        raise ValueError("Cannot sample a nonfinite MDN distribution")
    probabilities = (output.logmix / temperature).softmax(-1)
    selected = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator
    ).reshape(*probabilities.shape[:-1], 1)
    mean = output.mean.gather(-1, selected).squeeze(-1)
    stddev = output.logstd.gather(-1, selected).squeeze(-1).exp() * math.sqrt(temperature)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    sample = mean + stddev * noise
    if not torch.isfinite(sample).all():
        raise ValueError("MDN sampling overflow; distribution must be repaired before deployment")
    return sample, output.restart_logits > 0


@dataclass(frozen=True)
class VMCControllerConfig:
    latent_size: int = 64
    hidden_size: int = 512
    action_dim: int = 1
    include_cell: bool = True

    def __post_init__(self):
        if (
            any(
                type(v) is not int or v < 1
                for v in (self.latent_size, self.hidden_size, self.action_dim)
            )
            or type(self.include_cell) is not bool
        ):
            raise ValueError("Invalid VMC controller configuration")

    @property
    def feature_size(self):
        return self.latent_size + self.hidden_size * (2 if self.include_cell else 1)

    def to_dict(self):
        return dict(architecture="vmc_controller", **asdict(self))


class VMCController(LocalModelMixin, nn.Module):
    def __init__(self, config: VMCControllerConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.zeros(config.feature_size, config.action_dim))

    def forward(self, latent, cell, hidden):
        features = torch.cat(
            (latent, cell, hidden) if self.config.include_cell else (latent, hidden), -1
        )
        return (features @ self.weight).tanh()
