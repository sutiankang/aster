"""Native LPIPS feature distances with local weights and differentiable image inputs."""

from dataclasses import dataclass, asdict
from typing import ClassVar
import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from ..core import digest_json
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class LPIPSConfig:
    architecture: ClassVar[str] = "lpips"
    backbone: str = "vgg"
    version: str = "0.1"
    channels: tuple[int, ...] | None = None
    learned: bool = True
    spatial: bool = False
    allow_untrained: bool = False

    def __post_init__(self):
        if self.backbone not in {"vgg", "alex"} or self.version not in {"0.0", "0.1"}:
            raise ValueError("LPIPS supports explicit VGG16/AlexNet and version 0.0/0.1")
        standard = (64, 128, 256, 512, 512) if self.backbone == "vgg" else (64, 192, 384, 256, 256)
        object.__setattr__(
            self, "channels", standard if self.channels is None else tuple(self.channels)
        )
        if len(self.channels) != 5 or any(type(x) is not int or x < 1 for x in self.channels):
            raise ValueError("LPIPS requires five positive channel widths")
        if any(type(x) is not bool for x in (self.learned, self.spatial, self.allow_untrained)):
            raise ValueError("LPIPS switches must be boolean")

    @property
    def standard_architecture(self):
        return self.channels == (
            (64, 128, 256, 512, 512) if self.backbone == "vgg" else (64, 192, 384, 256, 256)
        )

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))


class LPIPS(LocalModelMixin, nn.Module):
    """Freeze feature-network parameters while preserving gradients to reconstructed images."""

    def __init__(self, config=LPIPSConfig()):
        super().__init__()
        self.config = config
        layers, endpoints, in_channels = [], [], 3
        if config.backbone == "vgg":
            for block, (width, depth) in enumerate(zip(config.channels, (2, 2, 3, 3, 3))):
                if block:
                    layers.append(nn.MaxPool2d(2, 2))
                for _ in range(depth):
                    layers.extend(
                        (nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(inplace=False))
                    )
                    in_channels = width
                endpoints.append(len(layers) - 1)
        else:
            for block, (width, kernel, stride, padding) in enumerate(
                zip(config.channels, (11, 5, 3, 3, 3), (4, 1, 1, 1, 1), (2, 2, 1, 1, 1))
            ):
                if block in (1, 2):
                    layers.append(nn.MaxPool2d(3, 2))
                layers.extend(
                    (nn.Conv2d(in_channels, width, kernel, stride, padding), nn.ReLU(inplace=False))
                )
                endpoints.append(len(layers) - 1)
                in_channels = width
        self.features, self.endpoints = nn.ModuleList(layers), tuple(endpoints)
        self.linear = (
            nn.ModuleList(nn.Conv2d(width, 1, 1, bias=False) for width in config.channels)
            if config.learned
            else nn.ModuleList()
        )

        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188]).reshape(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450]).reshape(1, 3, 1, 1))
        self.register_buffer("weights_loaded", torch.tensor(False))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):

        return super().train(False)

    def load_reference_weights(self, backbone_state, calibration_state=None):

        if not isinstance(backbone_state, dict) or not isinstance(calibration_state or {}, dict):
            raise TypeError(
                "Reference weights must be tensor mappings loaded with weights_only=True"
            )
        mapped = {}
        expected_backbone = {f"features.{name}" for name in self.features.state_dict()}
        if set(backbone_state) - expected_backbone and any(
            not name.startswith("classifier.") for name in set(backbone_state) - expected_backbone
        ):
            raise ValueError("Unknown LPIPS backbone weight keys")
        if not expected_backbone <= set(backbone_state):
            raise ValueError("Incomplete LPIPS backbone weights")
        for name in expected_backbone:
            mapped[name] = backbone_state[name]
        calibration_state = calibration_state or {}
        consumed = set()
        for index in range(len(self.linear)):
            options = [f"lin{index}.model.{offset}.weight" for offset in (0, 1)]
            found = [name for name in options if name in calibration_state]
            if len(found) != 1:
                raise ValueError("Each LPIPS stage needs exactly one calibration tensor")
            consumed.add(found[0])
            mapped[f"linear.{index}.weight"] = calibration_state[found[0]]
        if set(calibration_state) != consumed:
            raise ValueError("Unknown/duplicate LPIPS calibration keys")
        state = self.state_dict()
        for name, value in mapped.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != state[name].shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid LPIPS tensor: {name}")
        with torch.no_grad():
            for name, value in mapped.items():
                state[name].copy_(value)
            self.weights_loaded.fill_(True)
        return self

    def weight_identity(self):

        digest = hashlib.sha256(digest_json(self.config.to_dict()).encode())
        for name, value in sorted(self.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(
                digest_json(
                    dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                ).encode()
            )
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _features(self, value):
        result = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in self.endpoints:
                result.append(value)
        return result

    def forward(self, left, right, *, normalize=False, return_layers=False):
        if not bool(self.weights_loaded) and not self.config.allow_untrained:
            raise RuntimeError(
                "LPIPS weights not loaded; random features are not a perceptual metric"
            )
        if left.shape != right.shape or left.ndim != 4 or left.shape[0] < 1 or left.shape[1] != 3:
            raise ValueError("LPIPS requires matching nonempty BCHW RGB pairs")
        minimum = 16 if self.config.backbone == "vgg" else 31
        if (
            min(left.shape[-2:]) < minimum
            or not left.is_floating_point()
            or not right.is_floating_point()
        ):
            raise ValueError("LPIPS input is too small or not floating-point RGB")
        if (
            left.device != right.device
            or left.device != self.shift.device
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()
        ):
            raise ValueError("LPIPS inputs and frozen weights must share device and be finite")
        dtype = next(self.parameters()).dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Frozen LPIPS weights stay FP32/FP64; outer AMP does not quantize the metric"
            )

        with torch.autocast(left.device.type, enabled=False):
            left, right = left.to(dtype), right.to(dtype)
            if normalize:
                left, right = 2 * left - 1, 2 * right - 1
            if self.config.version == "0.1":
                left, right = (left - self.shift) / self.scale, (right - self.shift) / self.scale
            left_features, right_features = self._features(left), self._features(right)
            distances = []
            for index, (a, b) in enumerate(zip(left_features, right_features)):
                a = a / (torch.linalg.vector_norm(a, dim=1, keepdim=True) + 1e-10)
                b = b / (torch.linalg.vector_norm(b, dim=1, keepdim=True) + 1e-10)
                difference = (a - b).square()
                distance = (
                    self.linear[index](difference)
                    if self.config.learned
                    else difference.sum(1, keepdim=True)
                )
                distances.append(
                    F.interpolate(distance, left.shape[-2:], mode="bilinear", align_corners=False)
                    if self.config.spatial
                    else distance.mean((2, 3), keepdim=True)
                )
            result = torch.stack(distances).sum(0)
        return (result, tuple(distances)) if return_layers else result
