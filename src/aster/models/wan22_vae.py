"""Wan2.2 residual video codec with explicit Cosmos3-compatible configuration."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F

from .generative import DiagonalGaussian
from .serialization import LocalModelMixin
from .video_vae import CausalConv3D, ChannelRMS, VideoResample, VideoSpatialAttention, causal_layer


@dataclass(frozen=True)
class Wan22VAEConfig:
    architecture: ClassVar[str] = "wan22_vae"
    base_dim: int = 4
    decoder_base_dim: int = 8
    z_dim: int = 2
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 1
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    patch_size: int = 2
    latents_mean: tuple[float, ...] = ()
    latents_std: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("dim_mult", "temperal_downsample", "latents_mean", "latents_std"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            any(
                type(x) is not int or x < 1
                for x in (
                    self.base_dim,
                    self.decoder_base_dim,
                    self.z_dim,
                    self.num_res_blocks,
                    self.patch_size,
                    *self.dim_mult,
                )
            )
            or len(self.dim_mult) < 2
        ):
            raise ValueError("Wan2.2 codec widths/depths must be positive")
        if (
            len(self.temperal_downsample) != len(self.dim_mult) - 1
            or any(type(x) is not bool for x in self.temperal_downsample)
            or sum(self.temperal_downsample) != 2
        ):
            raise ValueError(
                "This source-compatible Wan2.2 codec requires explicit total temporal stride 4"
            )
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("Invalid Wan2.2 dropout")
        if bool(self.latents_mean) != bool(self.latents_std) or (
            self.latents_mean
            and (len(self.latents_mean) != self.z_dim or len(self.latents_std) != self.z_dim)
        ):
            raise ValueError("Wan2.2 latent statistics must cover every channel")
        if any(not math.isfinite(x) for x in self.latents_mean) or any(
            not math.isfinite(x) or x <= 0 for x in self.latents_std
        ):
            raise ValueError("Invalid Wan2.2 latent normalization statistics")
        dims = [self.base_dim * x for x in (1, *self.dim_mult)]
        for i, (incoming, outgoing) in enumerate(zip(dims, dims[1:])):
            factor = (8 if self.temperal_downsample[i] else 4) if i < len(self.dim_mult) - 1 else 1
            if incoming * factor % outgoing:
                raise ValueError("Wan2.2 AvgDown channels/factor must divide exactly")
        dims = [self.decoder_base_dim * x for x in (self.dim_mult[-1], *self.dim_mult[::-1])]
        for i, (incoming, outgoing) in enumerate(zip(dims[:-2], dims[1:-1])):
            factor = 8 if self.temperal_downsample[::-1][i] else 4
            if outgoing * factor % incoming:
                raise ValueError("Wan2.2 DupUp channels/factor must divide exactly")

    @property
    def spatial_stride(self):
        return self.patch_size * 2 ** (len(self.dim_mult) - 1)

    @property
    def temporal_stride(self):
        return 4

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))

    @classmethod
    def from_diffusers_config(cls, values):

        values = dict(values)
        allowed = {
            "base_dim",
            "decoder_base_dim",
            "z_dim",
            "dim_mult",
            "num_res_blocks",
            "temperal_downsample",
            "dropout",
            "patch_size",
            "latents_mean",
            "latents_std",
            "is_residual",
            "in_channels",
            "out_channels",
            "attn_scales",
            "scale_factor_spatial",
            "scale_factor_temporal",
            "clip_output",
        }
        unknown = {key for key in values if not key.startswith("_")} - allowed
        if unknown:
            raise ValueError(f"Unsupported Wan2.2 configuration fields: {sorted(unknown)}")
        if values.get("is_residual") is not True or values.get("attn_scales", []) != []:
            raise ValueError(
                "Only verified Wan2.2 residual blocks with default mid attention are supported"
            )
        p = values.get("patch_size")
        if (
            type(p) is not int
            or p < 1
            or values.get("in_channels") != 3 * p * p
            or values.get("out_channels") != 3 * p * p
        ):
            raise ValueError("Wan2.2 codec requires explicit patchified RGB channel layout")
        if "clip_output" in values and type(values["clip_output"]) is not bool:
            raise ValueError("Invalid upstream clip_output metadata")
        fields = {
            "base_dim",
            "decoder_base_dim",
            "z_dim",
            "dim_mult",
            "num_res_blocks",
            "temperal_downsample",
            "dropout",
            "patch_size",
            "latents_mean",
            "latents_std",
        }
        if not fields <= values.keys():
            raise ValueError("Explicit upstream Wan2.2 architectural fields are required")
        c = cls(**{key: values[key] for key in fields})
        if (
            values.get("scale_factor_spatial") != c.spatial_stride
            or values.get("scale_factor_temporal") != c.temporal_stride
        ):
            raise ValueError("Wan2.2 declared strides differ from the actual architecture")

        return c


class Wan22RMS(ChannelRMS):
    def forward(self, value):

        normalized = F.normalize(
            value.float() if value.dtype in (torch.float16, torch.bfloat16) else value, dim=1
        ).to(value.dtype)
        return normalized * self.scale * self.gamma


class Wan22Attention(VideoSpatialAttention):
    def __init__(self, channels):
        super().__init__(channels)
        self.norm = Wan22RMS(channels, images=True)

        self.proj.reset_parameters()

    def forward(self, x):
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        qkv = self.to_qkv(self.norm(frames)).reshape(b * t, 1, c * 3, h * w)
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, -1)
        out = (
            F.scaled_dot_product_attention(q, k, v)
            .squeeze(1)
            .permute(0, 2, 1)
            .reshape(b * t, c, h, w)
        )
        return x + self.proj(out).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class Wan22Residual(nn.Module):
    def __init__(self, incoming, outgoing, dropout):
        super().__init__()
        self.norm1, self.norm2 = Wan22RMS(incoming), Wan22RMS(outgoing)
        self.conv1 = CausalConv3D(incoming, outgoing, 3, padding=1)
        self.conv2 = CausalConv3D(outgoing, outgoing, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.conv_shortcut = (
            CausalConv3D(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()
        )

    def forward(self, x, cache):
        residual = self.conv_shortcut(x)
        x = causal_layer(self.conv1, F.silu(self.norm1(x)), cache)
        x = causal_layer(self.conv2, self.dropout(F.silu(self.norm2(x))), cache)
        return x + residual


class Wan22Resample(VideoResample):
    def __init__(self, width, *, up, temporal):
        super().__init__(width, up=up, temporal=temporal)
        if up:
            self.resample[1] = nn.Conv2d(width, width, 3, padding=1)


class Wan22AvgDown(nn.Module):
    def __init__(self, incoming, outgoing, temporal, spatial):
        super().__init__()
        self.outgoing, self.temporal, self.spatial = outgoing, temporal, spatial
        self.groups = incoming * temporal * spatial**2 // outgoing

    def forward(self, x):
        r, p = self.temporal, self.spatial
        x = F.pad(x, (0, 0, 0, 0, (-x.shape[2]) % r, 0))
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t // r, r, h // p, p, w // p, p).permute(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(b, self.outgoing, self.groups, t // r, h // p, w // p).mean(2)


class Wan22DupUp(nn.Module):
    def __init__(self, incoming, outgoing, temporal, spatial):
        super().__init__()
        self.outgoing, self.temporal, self.spatial = outgoing, temporal, spatial
        self.repeats = outgoing * temporal * spatial**2 // incoming

    def forward(self, x, first_chunk):
        b, _, t, h, w = x.shape
        r, p = self.temporal, self.spatial
        x = x.repeat_interleave(self.repeats, 1).reshape(b, self.outgoing, r, p, p, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, self.outgoing, t * r, h * p, w * p)
        return x[:, :, r - 1 :] if first_chunk else x


class Wan22DownBlock(nn.Module):
    def __init__(self, incoming, outgoing, c, down, temporal):
        super().__init__()
        self.avg_shortcut = Wan22AvgDown(incoming, outgoing, 2 if temporal else 1, 2 if down else 1)
        self.resnets = nn.ModuleList(
            Wan22Residual(incoming if i == 0 else outgoing, outgoing, c.dropout)
            for i in range(c.num_res_blocks)
        )
        self.downsampler = Wan22Resample(outgoing, up=False, temporal=temporal) if down else None

    def forward(self, x, cache):
        residual = self.avg_shortcut(x)
        for block in self.resnets:
            x = block(x, cache)
        if self.downsampler is not None:
            x = self.downsampler(x, cache)
        return x + residual


class Wan22UpBlock(nn.Module):
    def __init__(self, incoming, outgoing, c, up, temporal):
        super().__init__()
        self.avg_shortcut = Wan22DupUp(incoming, outgoing, 2 if temporal else 1, 2) if up else None
        self.resnets = nn.ModuleList(
            Wan22Residual(incoming if i == 0 else outgoing, outgoing, c.dropout)
            for i in range(c.num_res_blocks + 1)
        )
        self.upsampler = Wan22Resample(outgoing, up=True, temporal=temporal) if up else None

    def forward(self, x, cache, first_chunk):
        residual = self.avg_shortcut(x, first_chunk) if self.avg_shortcut is not None else None
        for block in self.resnets:
            x = block(x, cache)
        if self.upsampler is not None:
            x = self.upsampler(x, cache)
        return x if residual is None else x + residual


class Wan22MidBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.resnets = nn.ModuleList(
            (Wan22Residual(width, width, dropout), Wan22Residual(width, width, dropout))
        )
        self.attentions = nn.ModuleList((Wan22Attention(width),))

    def forward(self, x, cache):
        return self.resnets[1](self.attentions[0](self.resnets[0](x, cache)), cache)


class Wan22Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        dims = [c.base_dim * x for x in (1, *c.dim_mult)]
        self.conv_in = CausalConv3D(3 * c.patch_size**2, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            Wan22DownBlock(
                incoming,
                outgoing,
                c,
                i < len(c.dim_mult) - 1,
                c.temperal_downsample[i] if i < len(c.dim_mult) - 1 else False,
            )
            for i, (incoming, outgoing) in enumerate(zip(dims, dims[1:]))
        )
        self.mid_block = Wan22MidBlock(dims[-1], c.dropout)
        self.norm_out = Wan22RMS(dims[-1])
        self.conv_out = CausalConv3D(dims[-1], c.z_dim * 2, 3, padding=1)

    def forward(self, x, cache):
        x = causal_layer(self.conv_in, x, cache)
        for block in self.down_blocks:
            x = block(x, cache)
        return causal_layer(self.conv_out, F.silu(self.norm_out(self.mid_block(x, cache))), cache)


class Wan22Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        dims = [c.decoder_base_dim * x for x in (c.dim_mult[-1], *c.dim_mult[::-1])]
        self.conv_in = CausalConv3D(c.z_dim, dims[0], 3, padding=1)
        self.mid_block = Wan22MidBlock(dims[0], c.dropout)
        self.up_blocks = nn.ModuleList(
            Wan22UpBlock(
                incoming,
                outgoing,
                c,
                i < len(c.dim_mult) - 1,
                c.temperal_downsample[::-1][i] if i < len(c.dim_mult) - 1 else False,
            )
            for i, (incoming, outgoing) in enumerate(zip(dims, dims[1:]))
        )
        self.norm_out = Wan22RMS(dims[-1])
        self.conv_out = CausalConv3D(dims[-1], 3 * c.patch_size**2, 3, padding=1)

    def forward(self, x, cache, first_chunk):
        x = self.mid_block(causal_layer(self.conv_in, x, cache), cache)
        for block in self.up_blocks:
            x = block(x, cache, first_chunk)
        return causal_layer(self.conv_out, F.silu(self.norm_out(x)), cache)


def wan22_patchify(video, patch_size):
    b, c, t, h, w = video.shape
    p = patch_size

    return (
        video.reshape(b, c, t, h // p, p, w // p, p)
        .permute(0, 1, 6, 4, 2, 3, 5)
        .reshape(b, c * p * p, t, h // p, w // p)
    )


def wan22_unpatchify(value, patch_size):
    b, c, t, h, w = value.shape
    p = patch_size
    return (
        value.reshape(b, c // (p * p), p, p, t, h, w)
        .permute(0, 1, 4, 5, 3, 6, 2)
        .reshape(b, c // (p * p), t, h * p, w * p)
    )


class Wan22VideoVAE(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.decoder = Wan22Encoder(config), Wan22Decoder(config)
        self.quant_conv = CausalConv3D(2 * config.z_dim, 2 * config.z_dim, 1)
        self.post_quant_conv = CausalConv3D(config.z_dim, config.z_dim, 1)

    def encode(self, video):
        c = self.config
        if (
            video.ndim != 5
            or video.shape[1] != 3
            or min(video.shape) < 1
            or not video.is_floating_point()
        ):
            raise ValueError("Wan2.2 expects RGB BCTHW video")
        if (video.shape[2] - 1) % 4 or any(x % c.spatial_stride for x in video.shape[-2:]):
            raise ValueError("Wan2.2 video needs 1+4k frames and stride-divisible spatial sizes")
        x = wan22_patchify(video, c.patch_size)
        cache = {}
        pieces = [self.encoder(x[:, :, :1], cache)]
        for start in range(1, x.shape[2], 4):
            pieces.append(self.encoder(x[:, :, start : start + 4], cache))
        mean, logvar = self.quant_conv(torch.cat(pieces, 2)).chunk(2, 1)
        return DiagonalGaussian(mean, logvar.clamp(-30, 20))

    def transform(self, latent, *, inverse=False):
        if (
            type(inverse) is not bool
            or latent.ndim != 5
            or latent.shape[1] != self.config.z_dim
            or not latent.is_floating_point()
        ):
            raise ValueError(
                "Wan2.2 normalization requires configured floating BCTHW latent and boolean inverse"
            )
        mean = latent.new_tensor(self.config.latents_mean or (0.0,) * self.config.z_dim)[
            None, :, None, None, None
        ]
        std = latent.new_tensor(self.config.latents_std or (1.0,) * self.config.z_dim)[
            None, :, None, None, None
        ]

        inv_std = 1.0 / std
        return latent / inv_std + mean if inverse else (latent - mean) * inv_std

    def latent(self, video, *, sample=False, generator=None):
        if type(sample) is not bool:
            raise ValueError("Wan2.2 posterior sampling flag must be boolean")
        posterior = self.encode(video)
        return self.transform(posterior.sample(generator) if sample else posterior.mode())

    def decode_chunks(self, latent, *, scaled=False, clip_output=True):
        if (
            latent.ndim != 5
            or min(latent.shape) < 1
            or latent.shape[1] != self.config.z_dim
            or not latent.is_floating_point()
        ):
            raise ValueError("Wan2.2 decoder needs configured floating latent BCTHW")
        if type(scaled) is not bool or type(clip_output) is not bool:
            raise ValueError("Wan2.2 decode flags must be boolean")
        latent = self.transform(latent, inverse=True) if scaled else latent
        latent, cache = self.post_quant_conv(latent), {}
        for i in range(latent.shape[2]):
            output = wan22_unpatchify(
                self.decoder(latent[:, :, i : i + 1], cache, i == 0), self.config.patch_size
            )
            yield output.clamp(-1, 1) if clip_output else output

    def decode(self, latent, *, scaled=False, clip_output=True):

        return torch.cat(
            tuple(self.decode_chunks(latent, scaled=scaled, clip_output=clip_output)), 2
        )

    def forward(self, video, *, sample_posterior=True, generator=None, clip_output=True):
        if type(sample_posterior) is not bool:
            raise ValueError("Wan2.2 posterior sampling flag must be boolean")
        posterior = self.encode(video)
        latent = posterior.sample(generator) if sample_posterior else posterior.mode()
        return self.decode(latent, clip_output=clip_output), posterior
