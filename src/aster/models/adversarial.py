"""PatchGAN and explicitly initialized ActNorm for multi-role autoencoder training."""

from dataclasses import dataclass, asdict
from typing import ClassVar
import hashlib

import torch
from torch import nn

from ..core import digest_json
from .serialization import LocalModelMixin


class _ActAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, value, *, reverse=False, logdet=False):
        if logdet:
            return (
                value.shape[-2]
                * value.shape[-1]
                * self.scale.abs().log().sum()
                * value.new_ones(value.shape[0])
            )
        return value / self.scale - self.loc if reverse else self.scale * (value + self.loc)


class ActNorm2d(nn.Module):
    def __init__(self, channels, *, logdet=False):
        super().__init__()
        if type(channels) is not int or channels < 1 or type(logdet) is not bool:
            raise ValueError("Invalid ActNorm dimensions")
        self.channels, self.logdet = channels, logdet
        self.affine = _ActAffine(channels)
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def initialize(self, value, *, group=None):

        error = None
        try:
            if bool(self.initialized):
                raise ValueError(
                    "ActNorm is already initialized; do not reset trained optimizer parameters"
                )
            if (
                value.ndim not in (2, 4)
                or value.shape[0] < 1
                or value.shape[1] != self.channels
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("ActNorm calibration requires finite NC or NCHW activations")
            if (
                not isinstance(self.affine.loc, nn.Parameter)
                or value.device != self.affine.loc.device
            ):
                raise ValueError(
                    "Calibrate unsharded ActNorm on its parameter device before creating Trainer"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        declarations = (
            [(error, self.channels)]
            if group is None
            else group.gather_objects((error, self.channels))
        )
        if any(item[0] or item[1] != self.channels for item in declarations):
            raise ValueError("ActNorm collective calibration failed: " + str(declarations))
        x = value.reshape(value.shape[0], self.channels, -1).double()
        stats = torch.cat((x.sum((0, 2)), x.new_tensor([x.shape[0] * x.shape[2]])))
        if group is not None:
            group.all_reduce(stats)
        count = stats[-1]
        if count <= 1:
            raise ValueError(
                "ActNorm unbiased standard deviation requires at least two global coordinates"
            )
        mean = stats[:-1] / count
        centered = (x - mean[None, :, None]).square().sum((0, 2))
        if group is not None:
            group.all_reduce(centered)
        std = (centered / (count - 1)).sqrt()
        self.affine.loc.copy_(-mean.reshape(1, -1, 1, 1))
        self.affine.scale.copy_((std + 1e-6).reciprocal().reshape(1, -1, 1, 1))
        self.initialized.fill_(1)

    def forward(self, value, *, reverse=False):
        if not bool(self.initialized):
            raise ValueError("ActNorm requires explicit calibration before train/inference")
        if value.ndim not in (2, 4) or value.shape[1] != self.channels:
            raise ValueError("ActNorm expects NC/NCHW with its configured channels")
        squeeze = value.ndim == 2
        if squeeze:
            value = value[:, :, None, None]
        output = self.affine(value, reverse=reverse)
        if squeeze:
            output = output[:, :, 0, 0]
        if self.logdet and not reverse:
            return output, self.affine(value, logdet=True)
        return output


@dataclass(frozen=True)
class PatchDiscriminatorConfig:
    architecture: ClassVar[str] = "patch_discriminator"
    in_channels: int = 3
    base_channels: int = 64
    num_layers: int = 3
    normalization: str = "actnorm"

    def __post_init__(self):
        if (
            any(
                type(v) is not int or v < 1
                for v in (self.in_channels, self.base_channels, self.num_layers)
            )
            or self.num_layers > 8
        ):
            raise ValueError("Invalid PatchGAN dimensions")
        if self.normalization not in {"batchnorm", "actnorm"}:
            raise ValueError("PatchGAN requires explicit BatchNorm or ActNorm")

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))


class PatchDiscriminator(LocalModelMixin, nn.Module):
    def __init__(self, config=PatchDiscriminatorConfig()):
        super().__init__()
        self.config = config
        c = config.base_channels
        normalizer = nn.BatchNorm2d if config.normalization == "batchnorm" else ActNorm2d
        use_bias = config.normalization == "actnorm"
        layers = [nn.Conv2d(config.in_channels, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=False)]
        previous = 1
        for level in range(1, config.num_layers):
            current = min(2**level, 8)
            layers.extend(
                (
                    nn.Conv2d(c * previous, c * current, 4, 2, 1, bias=use_bias),
                    normalizer(c * current),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            )
            previous = current
        current = min(2**config.num_layers, 8)
        layers.extend(
            (
                nn.Conv2d(c * previous, c * current, 4, 1, 1, bias=use_bias),
                normalizer(c * current),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(c * current, 1, 4, 1, 1),
            )
        )
        self.main = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0.0, 0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1.0, 0.02)
                nn.init.zeros_(layer.bias)

    def _validate(self, images):
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1] != self.config.in_channels
            or min(images.shape[-2:]) < 3 * 2**self.config.num_layers
        ):
            raise ValueError(
                "PatchGAN input dimensions are too small for the declared receptive field"
            )
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("PatchGAN expects finite floating images")

    @torch.no_grad()
    def initialize(self, calibration, *, group=None):
        error = None
        try:
            self._validate(calibration)
            if self.config.normalization != "actnorm":
                raise ValueError("Explicit data calibration is for ActNorm only")
            if any(bool(m.initialized) for m in self.modules() if isinstance(m, ActNorm2d)):
                raise ValueError("Calibrate a fresh discriminator, not trained ActNorm state")
            digest = hashlib.sha256(digest_json(self.config.to_dict()).encode())
            for name, parameter in self.named_parameters():
                digest.update(name.encode())
                digest.update(
                    parameter.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
            identity = digest.hexdigest()
        except Exception as exc:
            error, identity = f"{type(exc).__name__}: {exc}", None
        declarations = (
            [(error, identity)] if group is None else group.gather_objects((error, identity))
        )
        if any(item[0] for item in declarations) or any(
            item[1] != identity for item in declarations
        ):
            raise ValueError(
                "PatchGAN calibration configuration/weights differ: " + str(declarations)
            )
        value = calibration
        for layer in self.main:
            if isinstance(layer, ActNorm2d):
                layer.initialize(value, group=group)
            value = layer(value)
        return self

    def forward(self, images):
        self._validate(images)
        return self.main(images)

    def load_reference_state(self, state):

        converted = {}
        for name, value in state.items():
            if self.config.normalization == "actnorm" and name.rsplit(".", 1)[-1] in {
                "loc",
                "scale",
            }:
                owner, _, suffix = name.rpartition(".")
                name = owner + ".affine." + suffix
            if name in converted:
                raise ValueError("Duplicate converted PatchGAN key")
            converted[name] = value
        expected = self.state_dict()
        if set(converted) != set(expected):
            raise ValueError("Reference discriminator state is incomplete or has unexpected keys")
        for name, value in converted.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or (value.is_floating_point() and not torch.isfinite(value).all())
            ):
                raise ValueError(f"Invalid reference discriminator tensor: {name}")
        self.load_state_dict(converted, strict=True)
        return self
