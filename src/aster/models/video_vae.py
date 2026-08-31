"""Wan2.1 causal video VAE with incremental temporal decoding."""

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .generative import DiagonalGaussian
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class WanVAEConfig:
    base_channels: int = 32
    latent_channels: int = 16
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    attention_scales: tuple[float, ...] = ()
    dropout: float = 0.0
    latent_mean: tuple[float, ...] = ()
    latent_std: tuple[float, ...] = ()

    def __post_init__(self):
        for key in (
            "channel_mult",
            "temporal_downsample",
            "attention_scales",
            "latent_mean",
            "latent_std",
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if len(self.channel_mult) < 2 or any(
            type(v) is not int or v < 1
            for v in (
                self.base_channels,
                self.latent_channels,
                self.num_res_blocks,
                *self.channel_mult,
            )
        ):
            raise ValueError("Invalid causal video VAE dimensions")
        if any(self.base_channels * mult % 2 for mult in self.channel_mult):
            raise ValueError("Upsampling halves channels, so each stage width must be even")
        if len(self.temporal_downsample) != len(self.channel_mult) - 1 or any(
            type(v) is not bool for v in self.temporal_downsample
        ):
            raise ValueError("Declare one temporal downsample flag per transition")
        if not 0 <= self.dropout < 1 or any(
            not math.isfinite(v) or v <= 0 for v in self.attention_scales
        ):
            raise ValueError("Invalid video VAE dropout or attention scales")
        if bool(self.latent_mean) != bool(self.latent_std):
            raise ValueError("Latent mean and std must be declared together")
        if self.latent_mean and (
            len(self.latent_mean) != self.latent_channels
            or len(self.latent_std) != self.latent_channels
            or any(not math.isfinite(v) for v in self.latent_mean)
            or any(not math.isfinite(v) or v <= 0 for v in self.latent_std)
        ):
            raise ValueError("Latent normalization must match every latent channel")

    @property
    def temporal_stride(self):
        return 2 ** sum(self.temporal_downsample)

    @property
    def spatial_stride(self):
        return 2 ** (len(self.channel_mult) - 1)

    def to_dict(self):
        return {"architecture": "wan21_vae", **asdict(self)}

    @classmethod
    def public_wan21(cls):

        return cls(
            base_channels=96,
            latent_mean=(
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ),
            latent_std=(
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ),
        )


class CausalConv3D(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t, h, w = self.padding
        self.causal_padding = (w, w, h, h, 2 * t, 0)
        self.padding = (0, 0, 0)

    def forward(self, value, previous=None):
        padding = list(self.causal_padding)
        if previous is not None:
            if previous.shape[2] > padding[4]:
                raise ValueError("Causal cache exceeds convolution receptive history")
            value = torch.cat((previous, value), 2)
            padding[4] -= previous.shape[2]
        return super().forward(F.pad(value, padding))


class ChannelRMS(nn.Module):
    def __init__(self, channels, *, images=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels, *((1, 1) if images else (1, 1, 1))))
        self.scale = math.sqrt(channels)

    def forward(self, value):

        return F.normalize(value, dim=1) * self.scale * self.gamma


def causal_layer(layer, value, cache):
    previous = cache.get(layer)
    history = value if previous is None else torch.cat((previous, value), 2)
    result = layer(value, previous)
    cache[layer] = history[:, :, -layer.causal_padding[4] :].clone()
    return result


class VideoResidual(nn.Module):
    def __init__(self, incoming, outgoing, dropout):
        super().__init__()
        self.residual = nn.ModuleList(
            (
                ChannelRMS(incoming),
                nn.SiLU(),
                CausalConv3D(incoming, outgoing, 3, padding=1),
                ChannelRMS(outgoing),
                nn.SiLU(),
                nn.Dropout(dropout),
                CausalConv3D(outgoing, outgoing, 3, padding=1),
            )
        )
        self.shortcut = (
            CausalConv3D(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()
        )

    def forward(self, value, cache):
        shortcut = self.shortcut(value)
        for layer in self.residual:
            value = (
                causal_layer(layer, value, cache)
                if isinstance(layer, CausalConv3D)
                else layer(value)
            )
        return value + shortcut


class VideoSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelRMS(channels, images=True)
        self.to_qkv, self.proj = (
            nn.Conv2d(channels, 3 * channels, 1),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.proj.weight)

    def forward(self, value):
        b, c, t, h, w = value.shape
        frames = value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        q, k, v = self.to_qkv(self.norm(frames)).flatten(2).transpose(1, 2).chunk(3, -1)
        out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        out = self.proj(out.transpose(1, 2).reshape(b * t, c, h, w))
        return value + out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class VideoUpsample(nn.Module):
    def forward(self, value):
        return F.interpolate(value.float(), scale_factor=2, mode="nearest-exact").to(value.dtype)


class VideoResample(nn.Module):
    def __init__(self, channels, *, up, temporal):
        super().__init__()
        self.up, self.temporal = up, temporal
        if up:
            self.resample = nn.Sequential(
                VideoUpsample(), nn.Conv2d(channels, channels // 2, 3, padding=1)
            )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(channels, channels, 3, stride=2)
            )
        if temporal:
            self.time_conv = CausalConv3D(
                channels,
                2 * channels if up else channels,
                (3, 1, 1),
                stride=(1 if up else 2, 1, 1),
                padding=(1 if up else 0, 0, 0),
            )

    def forward(self, value, cache):
        b, c, t, h, w = value.shape
        if self.up and self.temporal:
            if self not in cache:
                cache[self] = "first"
            else:
                previous = cache[self]
                history = (
                    torch.zeros_like(value[:, :, :1]).expand(-1, -1, 2, -1, -1)
                    if isinstance(previous, str)
                    else previous
                )
                out = self.time_conv(value, None if isinstance(previous, str) else previous)
                cache[self] = torch.cat((history, value), 2)[:, :, -2:].clone()
                value = (
                    out.reshape(b, 2, c, t, h, w)
                    .permute(0, 2, 3, 1, 4, 5)
                    .reshape(b, c, 2 * t, h, w)
                )
        t = value.shape[2]
        frames = self.resample(value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        value = frames.reshape(b, t, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        if not self.up and self.temporal:
            if self in cache:
                previous = cache[self]
                cache[self] = value[:, :, -1:].clone()
                value = self.time_conv(torch.cat((previous, value), 2))
            else:
                cache[self] = value[:, :, -1:].clone()
        return value


def run_video_layers(layers, value, cache):
    for layer in layers:
        if isinstance(layer, (VideoResidual, VideoResample)):
            value = layer(value, cache)
        elif isinstance(layer, CausalConv3D):
            value = causal_layer(layer, value, cache)
        else:
            value = layer(value)
    return value


class WanVideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        current = config.base_channels
        self.conv1 = CausalConv3D(3, current, 3, padding=1)
        layers, scale = [], 1.0
        for level, multiplier in enumerate(config.channel_mult):
            outgoing = config.base_channels * multiplier
            for _ in range(config.num_res_blocks):
                layers.append(VideoResidual(current, outgoing, config.dropout))
                if scale in config.attention_scales:
                    layers.append(VideoSpatialAttention(outgoing))
                current = outgoing
            if level + 1 < len(config.channel_mult):
                layers.append(
                    VideoResample(current, up=False, temporal=config.temporal_downsample[level])
                )
                scale /= 2
        self.downsamples = nn.ModuleList(layers)
        self.middle = nn.ModuleList(
            (
                VideoResidual(current, current, config.dropout),
                VideoSpatialAttention(current),
                VideoResidual(current, current, config.dropout),
            )
        )
        self.head = nn.ModuleList(
            (
                ChannelRMS(current),
                nn.SiLU(),
                CausalConv3D(current, config.latent_channels * 2, 3, padding=1),
            )
        )

    def forward(self, value, cache):
        value = causal_layer(self.conv1, value, cache)
        for layers in (self.downsamples, self.middle, self.head):
            value = run_video_layers(layers, value, cache)
        return value


class WanVideoDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        current = config.base_channels * config.channel_mult[-1]
        self.conv1 = CausalConv3D(config.latent_channels, current, 3, padding=1)
        self.middle = nn.ModuleList(
            (
                VideoResidual(current, current, config.dropout),
                VideoSpatialAttention(current),
                VideoResidual(current, current, config.dropout),
            )
        )
        layers, scale = [], 1 / 2 ** (len(config.channel_mult) - 2)
        for level, multiplier in enumerate(reversed(config.channel_mult)):
            outgoing = config.base_channels * multiplier
            for _ in range(config.num_res_blocks + 1):
                layers.append(VideoResidual(current, outgoing, config.dropout))
                if scale in config.attention_scales:
                    layers.append(VideoSpatialAttention(outgoing))
                current = outgoing
            if level + 1 < len(config.channel_mult):
                layers.append(
                    VideoResample(
                        current, up=True, temporal=config.temporal_downsample[::-1][level]
                    )
                )
                current //= 2
                scale *= 2
        self.upsamples = nn.ModuleList(layers)
        self.head = nn.ModuleList(
            (ChannelRMS(current), nn.SiLU(), CausalConv3D(current, 3, 3, padding=1))
        )

    def forward(self, value, cache):
        value = causal_layer(self.conv1, value, cache)
        for layers in (self.middle, self.upsamples, self.head):
            value = run_video_layers(layers, value, cache)
        return value


class WanVideoVAE(LocalModelMixin, nn.Module):
    def __init__(self, config: WanVAEConfig):
        super().__init__()
        self.config = config
        z = config.latent_channels
        self.encoder, self.decoder = WanVideoEncoder(config), WanVideoDecoder(config)
        self.conv1, self.conv2 = CausalConv3D(2 * z, 2 * z, 1), CausalConv3D(z, z, 1)

    def encode(self, video):
        c = self.config
        if (
            video.ndim != 5
            or video.shape[1] != 3
            or min(video.shape) < 1
            or not video.is_floating_point()
        ):
            raise ValueError("Video VAE requires floating B,3,T,H,W")
        if (video.shape[2] - 1) % c.temporal_stride or any(
            n % c.spatial_stride for n in video.shape[-2:]
        ):
            raise ValueError(
                "Video must have 1+k*temporal_stride frames and stride-divisible spatial dimensions"
            )
        cache = {}
        chunks = [self.encoder(video[:, :, :1], cache)]
        for start in range(1, video.shape[2], c.temporal_stride):
            chunks.append(self.encoder(video[:, :, start : start + c.temporal_stride], cache))
        mean, logvar = self.conv1(torch.cat(chunks, 2)).chunk(2, 1)
        return DiagonalGaussian(mean, logvar.clamp(-30, 20))

    def transform(self, latent, *, inverse=False):
        mean = latent.new_tensor(self.config.latent_mean or (0.0,) * self.config.latent_channels)[
            None, :, None, None, None
        ]
        std = latent.new_tensor(self.config.latent_std or (1.0,) * self.config.latent_channels)[
            None, :, None, None, None
        ]
        return latent * std + mean if inverse else (latent - mean) / std

    def latent(self, video, *, sample=False, generator=None):
        posterior = self.encode(video)
        return self.transform(posterior.sample(generator) if sample else posterior.mode())

    def decode_chunks(self, latent, *, scaled=False):

        if (
            latent.ndim != 5
            or min(latent.shape) < 1
            or latent.shape[1] != self.config.latent_channels
        ):
            raise ValueError("Invalid video latent shape")
        latent = self.transform(latent, inverse=True) if scaled else latent
        latent, cache = self.conv2(latent), {}
        for index in range(latent.shape[2]):
            yield self.decoder(latent[:, :, index : index + 1], cache)

    def decode(self, latent, *, scaled=False):

        return torch.cat(tuple(self.decode_chunks(latent, scaled=scaled)), 2)

    def forward(self, video, *, sample_posterior=True, generator=None):
        posterior = self.encode(video)
        latent = posterior.sample(generator) if sample_posterior else posterior.mode()
        return self.decode(latent), posterior
