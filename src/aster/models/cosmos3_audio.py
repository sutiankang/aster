"""Cosmos3 AVAE2 waveform/STFT encoding and Snake-Oobleck decoding."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class Cosmos3AudioConfig:
    architecture: ClassVar[str] = "cosmos3_avae2"
    sampling_rate: int = 48000
    vocoder_input_dim: int = 4
    dec_dim: int = 4
    dec_c_mults: tuple[int, ...] = (1, 2, 4)
    dec_strides: tuple[int, ...] = (2, 3, 2)
    dec_out_channels: int = 2
    stereo: bool = True
    normalize_volume: bool = True
    hop_size: int | None = None
    input_channels: int = 1
    enc_dim: int = 4
    enc_num_blocks: int = 1
    enc_n_fft: int = 8
    enc_hop_length: int = 2
    enc_latent_dim: int = 8
    enc_c_mults: tuple[int, ...] = (1, 2)
    enc_strides: tuple[int, ...] = (2, 3)
    enc_identity_init: bool = False
    enc_use_snake: bool = True
    padding_mode: str = "zeros"
    encoder_enabled: bool = True

    def __post_init__(self):
        for name in ("dec_c_mults", "dec_strides", "enc_c_mults", "enc_strides"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.hop_size is None:
            object.__setattr__(self, "hop_size", math.prod(self.dec_strides))
        integers = (
            self.sampling_rate,
            self.vocoder_input_dim,
            self.dec_dim,
            self.dec_out_channels,
            self.hop_size,
            self.input_channels,
            self.enc_dim,
            self.enc_num_blocks,
            self.enc_n_fft,
            self.enc_hop_length,
            self.enc_latent_dim,
            *self.dec_c_mults,
            *self.dec_strides,
            *self.enc_c_mults,
            *self.enc_strides,
        )
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("AVAE2 dimensions/strides must be positive integers")
        if (
            not self.dec_c_mults
            or not self.enc_c_mults
            or len(self.dec_c_mults) != len(self.dec_strides)
            or len(self.enc_c_mults) != len(self.enc_strides)
        ):
            raise ValueError("AVAE2 encoder/decoder channel and stride lengths must agree")
        if self.hop_size != math.prod(
            self.dec_strides
        ) or self.hop_size != self.enc_hop_length * math.prod(self.enc_strides):
            raise ValueError(
                "AVAE2 declared compression must equal both actual encoder and decoder strides"
            )
        if (
            self.enc_latent_dim != 2 * self.vocoder_input_dim
            or self.enc_n_fft % 2
            or self.enc_hop_length > self.enc_n_fft
        ):
            raise ValueError(
                "AVAE2 requires even FFT and mean/scale channels for every decoder latent"
            )
        if self.dec_out_channels != self.input_channels * (2 if self.stereo else 1):
            raise ValueError("AVAE2 audio channel count must agree at both ends")
        if self.padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("Unsupported AVAE2 convolution padding mode")
        for name in (
            "stereo",
            "normalize_volume",
            "enc_identity_init",
            "enc_use_snake",
            "encoder_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError("AVAE2 architecture switches must be boolean")

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))

    @classmethod
    def from_diffusers_config(cls, values):

        values = dict(values)
        fixed = dict(
            model_type="autoencoder_v2",
            use_wav_as_input=True,
            enc_type="spec_convnext",
            dec_type="oobleck",
            dec_use_snake=True,
            dec_final_tanh=False,
            dec_anti_aliasing=False,
            dec_use_nearest_upsample=False,
            dec_use_tanh_at_final=False,
            bottleneck_type="vae",
            activation="snakebeta",
            snake_logscale=True,
            anti_aliasing=False,
            use_cuda_kernel=False,
            causal=False,
            latent_mean=None,
            latent_std=None,
        )
        for name, expected in fixed.items():
            if name in values and values[name] != expected:
                raise ValueError("Unsupported AVAE2 source configuration: " + name)
        if values.get("bottleneck") not in (None, {"type": "vae"}):
            raise ValueError("Unsupported AVAE2 bottleneck")
        fields = set(cls.__dataclass_fields__) - {"architecture"}
        unknown = (
            {name for name in values if not name.startswith("_")}
            - fields
            - set(fixed)
            - {"enc_intermediate_dim", "enc_num_layers", "bottleneck"}
        )
        if unknown:
            raise ValueError("Unknown AVAE2 configuration fields: " + str(sorted(unknown)))
        required = {
            "vocoder_input_dim",
            "dec_dim",
            "dec_c_mults",
            "dec_strides",
            "dec_out_channels",
            "enc_dim",
            "enc_num_blocks",
            "enc_n_fft",
            "enc_hop_length",
            "enc_latent_dim",
            "enc_c_mults",
            "enc_strides",
        }
        if not required <= values.keys():
            raise ValueError("Explicit AVAE2 architectural dimensions are required")
        return cls(**{name: values[name] for name in fields if name in values})


class AudioWeightNormConv(nn.Module):
    def __init__(
        self,
        incoming,
        outgoing,
        kernel,
        *,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        transpose=False,
        output_padding=0,
        padding_mode="zeros",
    ):
        super().__init__()
        constructor = nn.ConvTranspose1d if transpose else nn.Conv1d
        options = dict(stride=stride, padding=padding, dilation=dilation, bias=bias)
        if transpose:
            options["output_padding"] = output_padding
        layer = constructor(incoming, outgoing, kernel, **options)
        self.weight_v = nn.Parameter(layer.weight.detach().clone())
        self.weight_g = nn.Parameter(
            torch.linalg.vector_norm(layer.weight.detach(), dim=(1, 2), keepdim=True)
        )
        self.bias = nn.Parameter(layer.bias.detach().clone()) if bias else None
        self.transpose, self.stride, self.padding = transpose, stride, padding
        self.dilation, self.output_padding, self.padding_mode = (
            dilation,
            output_padding,
            padding_mode,
        )

    def forward(self, sample):
        dtype = self.weight_v.dtype
        v = self.weight_v.float() if dtype in (torch.float16, torch.bfloat16) else self.weight_v
        g = self.weight_g.to(v.dtype)
        weight = (v * (g / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True))).to(dtype)
        if self.transpose:
            return F.conv_transpose1d(
                sample,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                dilation=self.dilation,
            )
        padding = self.padding
        if self.padding_mode != "zeros" and padding:
            sample = F.pad(sample, (padding, padding), mode=self.padding_mode)
            padding = 0
        return F.conv1d(sample, weight, self.bias, self.stride, padding, self.dilation)


class AudioSnake(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, width, 1))
        self.beta = nn.Parameter(torch.zeros(1, width, 1))

    def forward(self, sample):

        return (
            sample
            + (self.beta.exp() + 1e-9).reciprocal() * torch.sin(self.alpha.exp() * sample).square()
        )


class AudioFP32LayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, sample):
        return F.layer_norm(sample.float(), self.weight.shape, self.weight.float(), None, 1e-5).to(
            sample.dtype
        )


class AudioConvNeXt(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.ConstantPad1d((3, 3), 0), nn.Conv1d(width, width, 7, groups=width)
        )
        self.norm = AudioFP32LayerNorm(width)
        self.pwconv1, self.pwconv2 = nn.Conv1d(width, 4 * width, 1), nn.Conv1d(4 * width, width, 1)
        self.act = AudioSnake(4 * width) if config.enc_use_snake else nn.GELU()
        if config.enc_identity_init:
            nn.init.zeros_(self.pwconv2.weight)
            nn.init.zeros_(self.pwconv2.bias)

    def forward(self, sample):
        value = self.norm(self.dwconv(sample).transpose(1, 2)).transpose(1, 2)
        return sample + self.pwconv2(self.act(self.pwconv1(value)))


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        layers = [
            AudioWeightNormConv(
                (c.enc_n_fft + 2) * c.dec_out_channels, c.enc_dim * c.enc_c_mults[0], 1, bias=False
            )
        ]
        for index, stride in enumerate(c.enc_strides):
            width = c.enc_dim * c.enc_c_mults[index]
            output = c.enc_dim * c.enc_c_mults[min(index + 1, len(c.enc_c_mults) - 1)]
            layers.extend(AudioConvNeXt(width, c) for _ in range(c.enc_num_blocks))
            layers.append(
                AudioWeightNormConv(
                    width,
                    output,
                    2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    padding_mode=c.padding_mode,
                )
            )
        layers.append(
            AudioWeightNormConv(c.enc_dim * c.enc_c_mults[-1], c.enc_latent_dim, 1, bias=False)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        c = self.config
        b, channels, length = sample.shape
        left = (c.enc_n_fft - c.enc_hop_length) // 2
        right = c.enc_n_fft - c.enc_hop_length - left

        waveform = F.pad(sample.reshape(b * channels, length), (left, right)).float()
        spectrum = torch.stft(
            waveform,
            c.enc_n_fft,
            hop_length=c.enc_hop_length,
            window=torch.hann_window(c.enc_n_fft, device=sample.device),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        real, imaginary = torch.view_as_real(spectrum).chunk(2, -1)

        features = torch.cat((real, imaginary), 1).squeeze(-1).to(sample.dtype)
        features = features.reshape(b, channels * (c.enc_n_fft + 2), features.shape[-1])
        return self.layers(features)


class AudioResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.snake1, self.snake2 = AudioSnake(width), AudioSnake(width)
        self.conv1 = AudioWeightNormConv(width, width, 7, dilation=dilation, padding=3 * dilation)
        self.conv2 = AudioWeightNormConv(width, width, 1)

    def forward(self, sample):
        return sample + self.conv2(self.snake2(self.conv1(self.snake1(sample))))


class AudioDecoderBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.snake1 = AudioSnake(incoming)
        self.conv_t1 = AudioWeightNormConv(
            incoming,
            outgoing,
            2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,
            transpose=True,
        )
        self.res_unit1, self.res_unit2, self.res_unit3 = (
            AudioResidual(outgoing, d) for d in (1, 3, 9)
        )

    def forward(self, sample):
        return self.res_unit3(self.res_unit2(self.res_unit1(self.conv_t1(self.snake1(sample)))))


class AudioDecoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        mult = (1, *c.dec_c_mults)
        strides = c.dec_strides[::-1]
        self.conv1 = AudioWeightNormConv(c.vocoder_input_dim, c.dec_dim * mult[-1], 7, padding=3)
        self.block = nn.ModuleList(
            AudioDecoderBlock(
                c.dec_dim * mult[len(strides) - i], c.dec_dim * mult[len(strides) - i - 1], s
            )
            for i, s in enumerate(strides)
        )
        self.snake1 = AudioSnake(c.dec_dim)
        self.conv2 = AudioWeightNormConv(c.dec_dim, c.dec_out_channels, 7, padding=3, bias=False)

    def forward(self, sample):
        sample = self.conv1(sample)
        for block in self.block:
            sample = block(sample)
        return self.conv2(self.snake1(sample))


@dataclass
class AudioGaussian:
    parameters: torch.Tensor

    def __post_init__(self):
        self.mean, self.scale = self.parameters.chunk(2, 1)
        self.std = F.softplus(self.scale) + 1e-4
        self.var = self.std.square()
        self.logvar = self.var.log()

    def mode(self):
        return self.mean

    def sample(self, generator=None):
        return self.mean + self.std * torch.randn(
            self.mean.shape, dtype=self.mean.dtype, device=self.mean.device, generator=generator
        )

    def kl(self):

        return (self.mean.square() + self.var - self.logvar - 1).sum(1).mean()


class Cosmos3AudioCodec(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config) if config.encoder_enabled else None
        self.decoder = AudioDecoder(config)

    def encode(self, sample, *, force_pad=False):
        c = self.config
        if self.encoder is None:
            raise ValueError("Decoder-only AVAE2 cannot encode waveforms")
        if (
            sample.ndim != 3
            or sample.shape[1] != c.dec_out_channels
            or min(sample.shape) < 1
            or not sample.is_floating_point()
            or not torch.isfinite(sample).all()
        ):
            raise ValueError(
                "AVAE2 waveform must be finite floating [B,C,N] with configured channels"
            )
        if c.normalize_volume:
            sample = sample / (sample.abs().max() + 1e-5) * 0.95
        if force_pad or not self.training:
            sample = F.pad(sample, (0, (-sample.shape[-1]) % c.hop_size))
        sample = sample.to(next(self.encoder.parameters()).dtype)
        return AudioGaussian(self.encoder(sample))

    def decode(self, latents, *, clip_output=True):
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents[None]
        if (
            latents.ndim != 3
            or min(latents.shape) < 1
            or latents.shape[1] != self.config.vocoder_input_dim
        ):
            raise ValueError("AVAE2 latent must be BDT or DT with explicit acoustic width")
        waveform = self.decoder(latents)
        if clip_output:
            waveform = waveform.clamp(-1, 1)
        return waveform[0] if squeeze else waveform

    def forward(
        self, sample, *, sample_posterior=False, generator=None, force_pad=False, clip_output=True
    ):
        posterior = self.encode(sample, force_pad=force_pad)
        latent = posterior.sample(generator) if sample_posterior else posterior.mode()
        return self.decode(latent, clip_output=clip_output), posterior
