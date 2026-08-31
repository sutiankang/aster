"""Trainable MAE-ResNet feature encoders for Drifting."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .serialization import LocalModelMixin


def patch_input(samples, size):
    b, c, h, w = samples.shape
    if h % size or w % size:
        raise ValueError("MAE image geometry must divide its input patch size")
    return (
        samples.reshape(b, c, h // size, size, w // size, size)
        .permute(0, 3, 5, 1, 2, 4)
        .reshape(b, size * size * c, h // size, w // size)
    )


def _groups(channels, maximum=32):
    value = min(channels, maximum)
    while channels % value:
        value -= 1
    return value


def _norm(channels):
    return nn.GroupNorm(_groups(channels), channels, eps=1e-6)


def _conv(incoming, outgoing, kernel=3, stride=1, bias=False):
    return nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2, bias=bias)


class MAEBasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride, dropout):
        super().__init__()
        self.conv1, self.conv2 = _conv(incoming, outgoing, stride=stride), _conv(outgoing, outgoing)
        self.gn1, self.gn2 = _norm(outgoing), _norm(outgoing)
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Sequential(_conv(incoming, outgoing, 1, stride), _norm(outgoing))
            if incoming != outgoing or stride != 1
            else nn.Identity()
        )

    def forward(self, value):
        hidden = self.dropout(F.relu(self.gn1(self.conv1(value))))
        return F.relu(self.projection(value) + self.gn2(self.conv2(hidden)))


class MAEResNetEncoder(nn.Module):
    def __init__(self, config, *, every_k_block=2):
        super().__init__()
        if type(every_k_block) is not int or every_k_block < 0:
            raise ValueError("every_k_block must be an integer; 0 disables block features")
        self.config, self.every_k_block = config, every_k_block
        base = config.base_channels
        self.conv1, self.gn1 = (
            _conv(config.in_channels * config.input_patch_size**2, base),
            _norm(base),
        )
        incoming, stages, norms = base, [], []
        for level, count in enumerate(config.layers):
            outgoing = base * 2**level
            blocks = []
            for block in range(count):
                stride = 2 if level > 0 and block == 0 else 1
                blocks.append(MAEBasicBlock(incoming, outgoing, stride, config.dropout))
                incoming = outgoing
            stages.append(nn.ModuleList(blocks))
            norms.append(_norm(outgoing))
        self.stages, self.stage_norms = nn.ModuleList(stages), nn.ModuleList(norms)

    def config_dict(self):
        return dict(
            type="drifting_mae_encoder",
            model=self.config.to_dict(),
            every_k_block=self.every_k_block,
        )

    def forward(self, value):
        value = F.relu(self.gn1(self.conv1(value)))
        result = dict(conv1=value)
        for level, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms), start=1):
            for index, block in enumerate(blocks, start=1):
                value = block(value)
                if self.every_k_block and index % self.every_k_block == 0:
                    result[f"layer{level}_blk{index}"] = value

            value = norm(value)
            result[f"layer{level}"] = value
        return result


class MAEConvGNReLU(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(_conv(incoming, outgoing), _norm(outgoing), nn.ReLU())


class MAEUpBlock(nn.Module):
    def __init__(self, incoming, skip_channels, outgoing):
        super().__init__()

        self.concat_norm = nn.GroupNorm(32, incoming + skip_channels, eps=1e-6)
        self.proj = MAEConvGNReLU(incoming + skip_channels, outgoing)
        self.refine = MAEConvGNReLU(outgoing, outgoing)

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.proj(self.concat_norm(torch.cat((value, skip), 1))))


class MAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.base_channels
        self.bridge = MAEConvGNReLU(8 * c, 8 * c)
        self.up43, self.up32 = MAEUpBlock(8 * c, 4 * c, 4 * c), MAEUpBlock(4 * c, 2 * c, 2 * c)
        self.up21, self.up10 = MAEUpBlock(2 * c, c, c), MAEUpBlock(c, c, c)
        self.head = _conv(c, config.in_channels * config.input_patch_size**2, 1, bias=True)

    def forward(self, features):
        x = self.bridge(features["layer4"])
        x = self.up43(x, features["layer3"])
        x = self.up32(x, features["layer2"])
        x = self.up21(x, features["layer1"])
        return self.head(self.up10(x, features["conv1"]))


@dataclass(frozen=True)
class MAEResNetConfig:
    in_channels: int = 3
    num_classes: int = 1000
    base_channels: int = 64
    layers: tuple[int, int, int, int] = (2, 2, 2, 2)
    patch_size: int = 4
    input_patch_size: int = 1
    dropout: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))
        if (
            len(self.layers) != 4
            or any(
                type(x) is not int or x < 1
                for x in (
                    self.in_channels,
                    self.num_classes,
                    self.base_channels,
                    self.patch_size,
                    self.input_patch_size,
                    *self.layers,
                )
            )
            or self.base_channels % 32
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("MAE needs four nonempty stages and base_channels divisible by 32")

    def to_dict(self):
        return {"architecture": "drifting_mae", **asdict(self)}


@dataclass
class MAEOutput:
    reconstruction: torch.Tensor
    logits: torch.Tensor
    patched_target: torch.Tensor
    mask: torch.Tensor


class MAEResNet(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.decoder = MAEResNetEncoder(config), MAEDecoder(config)
        self.classifier = nn.Linear(8 * config.base_channels, config.num_classes)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                fan_in = layer.weight.shape[1] * math.prod(layer.weight.shape[2:])
                std = math.sqrt(1 / fan_in) / 0.87962566103423978
                nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, samples, mask):
        c = self.config
        if (
            samples.ndim != 4
            or samples.shape[1] != c.in_channels
            or not samples.is_floating_point()
            or not torch.isfinite(samples).all()
        ):
            raise ValueError("MAE expects finite BCHW inputs with its declared channels")
        target = patch_input(samples, c.input_patch_size)
        if min(target.shape[-2:]) < 8:
            raise ValueError("Four MAE stages require patched spatial dimensions at least 8")
        if (
            mask.shape != (len(samples), 1, *target.shape[-2:])
            or mask.dtype != torch.bool
            or mask.device != samples.device
        ):
            raise ValueError("MAE mask must be explicit bool B1HW in patched coordinates")
        features = self.encoder(target * (~mask))
        return MAEOutput(
            self.decoder(features), self.classifier(features["layer4"].mean((2, 3))), target, mask
        )
