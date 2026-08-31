"""Native instantiation of published Genie mechanisms, not the unreleased Genie 3 system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F

from .serialization import LocalModelMixin


@dataclass(frozen=True)
class GenieTokenizerConfig:
    architecture: ClassVar[str] = "genie_tokenizer"
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    patch_size: int = 4
    hidden_size: int = 64
    num_heads: int = 4
    head_dim: int = 16
    encoder_layers: int = 2
    decoder_hidden_size: int = 64
    decoder_num_heads: int = 4
    decoder_head_dim: int = 16
    decoder_layers: int = 2
    intermediate_ratio: int = 4
    latent_dim: int = 32
    num_codes: int = 1024
    max_frames: int = 16
    qk_norm: bool = False
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in {"qk_norm", "norm_eps"} and (type(value) is not int or value < 1):
                raise ValueError(f"Genie {name} must be a positive integer")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("Genie images must be divisible by patch size")
        if type(self.qk_norm) is not bool or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("Invalid Genie normalization")

    @property
    def spatial_tokens(self):
        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))


@dataclass(frozen=True)
class GenieActionConfig(GenieTokenizerConfig):
    architecture: ClassVar[str] = "genie_action"
    patch_size: int = 16
    num_codes: int = 8


@dataclass(frozen=True)
class GenieDynamicsConfig:
    architecture: ClassVar[str] = "genie_dynamics"
    spatial_tokens: int = 256
    vocab_size: int = 1024
    action_dim: int = 32
    hidden_size: int = 64
    num_heads: int = 4
    head_dim: int = 16
    num_layers: int = 2
    intermediate_ratio: int = 4
    max_frames: int = 16
    qk_norm: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in {"qk_norm", "norm_eps"} and (type(value) is not int or value < 1):
                raise ValueError(f"Genie {name} must be a positive integer")
        if type(self.qk_norm) is not bool or not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("Invalid Genie normalization")

    @property
    def mask_token_id(self):
        return self.vocab_size

    def to_dict(self):
        return dict(architecture=self.architecture, **asdict(self))


class STAttention(nn.Module):
    def __init__(self, width, heads, head_dim, *, qk_norm=False, eps=1e-5):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.qkv = nn.Linear(width, 3 * heads * head_dim)
        self.query_norm = nn.LayerNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(head_dim, eps=eps) if qk_norm else nn.Identity()
        self.output = nn.Linear(heads * head_dim, width)

    def forward(self, value, *, causal=False):
        b, s, _ = value.shape
        q, k, v = self.qkv(value).reshape(b, s, 3, self.heads, self.head_dim).unbind(2)
        q, k = self.query_norm(q), self.key_norm(k)
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
        )
        return self.output(result.transpose(1, 2).reshape(b, s, self.heads * self.head_dim))


class STBlock(nn.Module):
    """Apply spatial attention, causal temporal attention, then one feed-forward block."""

    def __init__(self, width, heads, head_dim, ratio, qk_norm, eps):
        super().__init__()
        self.spatial_norm, self.temporal_norm, self.ffn_norm = (
            nn.LayerNorm(width, eps=eps) for _ in range(3)
        )
        self.spatial = STAttention(width, heads, head_dim, qk_norm=qk_norm, eps=eps)
        self.temporal = STAttention(width, heads, head_dim, qk_norm=qk_norm, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * ratio), nn.GELU(), nn.Linear(width * ratio, width)
        )

    def forward(self, value):
        b, t, n, d = value.shape
        spatial = self.spatial(self.spatial_norm(value).reshape(b * t, n, d)).reshape_as(value)
        value = value + spatial
        temporal = self.temporal_norm(value).transpose(1, 2).reshape(b * n, t, d)
        temporal = self.temporal(temporal, causal=True).reshape(b, n, t, d).transpose(1, 2)
        value = value + temporal
        return value + self.ffn(self.ffn_norm(value))


class STStack(nn.Module):
    def __init__(
        self, width, heads, head_dim, layers, spatial_tokens, max_frames, ratio, qk_norm, eps
    ):
        super().__init__()
        self.spatial_position = nn.Embedding(spatial_tokens, width)
        self.temporal_position = nn.Embedding(max_frames, width)
        nn.init.normal_(self.spatial_position.weight, std=0.02)
        nn.init.normal_(self.temporal_position.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [STBlock(width, heads, head_dim, ratio, qk_norm, eps) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(width, eps=eps)
        self.spatial_tokens, self.max_frames = spatial_tokens, max_frames

    def forward(self, value):
        if (
            value.ndim != 4
            or value.shape[2] != self.spatial_tokens
            or not 1 <= value.shape[1] <= self.max_frames
        ):
            raise ValueError("Genie stack spatial/temporal shape differs")
        spatial = self.spatial_position(torch.arange(value.shape[2], device=value.device))
        temporal = self.temporal_position(torch.arange(value.shape[1], device=value.device))
        value = value + spatial[None, None] + temporal[None, :, None]
        for block in self.blocks:
            value = block(value)
        return self.norm(value)


def _stack(c, *, decoder=False):
    return STStack(
        c.decoder_hidden_size if decoder else c.hidden_size,
        c.decoder_num_heads if decoder else c.num_heads,
        c.decoder_head_dim if decoder else c.head_dim,
        c.decoder_layers if decoder else c.encoder_layers,
        c.spatial_tokens,
        c.max_frames,
        c.intermediate_ratio,
        c.qk_norm,
        c.norm_eps,
    )


@dataclass
class VQEncoding:
    quantized: torch.Tensor
    indices: torch.Tensor
    commitment_errors: torch.Tensor
    codebook_errors: torch.Tensor


class VideoVectorQuantizer(nn.Module):
    """Use non-EMA VQ losses with explicit element counts for trainer normalization."""

    def __init__(self, num_codes, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, latent_dim)
        self.num_codes, self.latent_dim = num_codes, latent_dim
        nn.init.uniform_(
            self.embedding.weight, -math.sqrt(3 / latent_dim), math.sqrt(3 / latent_dim)
        )

    def lookup(self, indices):
        if indices.dtype != torch.int64 or (indices < 0).any() or (indices >= self.num_codes).any():
            raise ValueError("VQ code index outside codebook")
        return self.embedding(indices)

    def forward(self, value):
        if value.shape[-1] != self.latent_dim:
            raise ValueError("VQ latent width differs")
        with torch.autocast(device_type=value.device.type, enabled=False):
            codes = self.embedding(torch.arange(self.num_codes, device=value.device)).float()
            flat = value.float().reshape(-1, self.latent_dim)
            distances = (
                flat.square().sum(-1, keepdim=True) + codes.square().sum(-1) - 2 * flat @ codes.T
            )
            indices = distances.argmin(-1).reshape(value.shape[:-1])
            selected = F.embedding(indices, codes)
            source = value.float()
            commitment = (source - selected.detach()).square()
            codebook = (source.detach() - selected).square()
            quantized = source + (selected - source).detach()
        return VQEncoding(quantized.to(value.dtype), indices, commitment, codebook)


def _validate_video(video, c, *, minimum_frames=1):
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 5
        or video.shape[2:] != (c.image_channels, c.image_height, c.image_width)
        or not minimum_frames <= video.shape[1] <= c.max_frames
        or len(video) < 1
    ):
        raise ValueError("Genie video must have configured [B,T,C,H,W] shape")
    if (
        not video.is_floating_point()
        or not torch.isfinite(video).all()
        or video.min() < 0
        or video.max() > 1
    ):
        raise ValueError("Genie pixel video must be finite float in [0,1]")


def patch_video(video, c):
    b, t, channels, height, width = video.shape
    p = c.patch_size

    return (
        video.reshape(b, t, channels, height // p, p, width // p, p)
        .permute(0, 1, 3, 5, 4, 6, 2)
        .reshape(b, t, c.spatial_tokens, p * p * channels)
    )


def unpatch_video(patches, c):
    b, t = patches.shape[:2]
    p = c.patch_size
    return (
        patches.reshape(b, t, c.image_height // p, c.image_width // p, p, p, c.image_channels)
        .permute(0, 1, 6, 2, 4, 3, 5)
        .reshape(b, t, c.image_channels, c.image_height, c.image_width)
    )


@dataclass
class GenieVQOutput:
    reconstruction: torch.Tensor
    encoding: VQEncoding


class GenieTokenizer(LocalModelMixin, nn.Module):
    def __init__(self, config: GenieTokenizerConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.patch_size**2 * config.image_channels, config.hidden_size)
        self.encoder = _stack(config)
        self.to_latent = nn.Linear(config.hidden_size, config.latent_dim)
        self.quantizer = VideoVectorQuantizer(config.num_codes, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.decoder_hidden_size)
        self.decoder = _stack(config, decoder=True)
        self.output = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.image_channels
        )

    def encode(self, video):
        _validate_video(video, self.config)
        return self.quantizer(
            self.to_latent(self.encoder(self.input(patch_video(video, self.config))))
        )

    def decode_latents(self, latent):
        return unpatch_video(
            self.output(self.decoder(self.from_latent(latent))).sigmoid(), self.config
        )

    def decode(self, tokens):
        return self.decode_latents(self.quantizer.lookup(tokens))

    def forward(self, video):
        encoding = self.encode(video)
        return GenieVQOutput(self.decode_latents(encoding.quantized), encoding)


class GenieLatentAction(LocalModelMixin, nn.Module):
    def __init__(self, config: GenieActionConfig):
        super().__init__()
        self.config = config
        pixels = config.patch_size**2 * config.image_channels
        self.input = nn.Linear(pixels, config.hidden_size)
        self.encoder = _stack(config)
        self.to_latent = nn.Linear(config.hidden_size, config.latent_dim)
        self.quantizer = VideoVectorQuantizer(config.num_codes, config.latent_dim)
        self.context = nn.Linear(pixels, config.decoder_hidden_size)
        self.action = nn.Linear(config.latent_dim, config.decoder_hidden_size)
        self.decoder = _stack(config, decoder=True)
        self.output = nn.Linear(config.decoder_hidden_size, pixels)

    def encode(self, video):
        _validate_video(video, self.config, minimum_frames=2)
        hidden = self.encoder(self.input(patch_video(video, self.config)))
        return self.quantizer(self.to_latent(hidden[:, 1:].mean(2)))

    def decode(self, context, actions):
        _validate_video(context, self.config)
        if actions.shape != (*context.shape[:2], self.config.latent_dim):
            raise ValueError("Genie LAM action must align with previous frame")
        hidden = self.context(patch_video(context, self.config)) + self.action(actions)[:, :, None]
        return unpatch_video(self.output(self.decoder(hidden)).sigmoid(), self.config)

    def forward(self, video):
        encoding = self.encode(video)
        return GenieVQOutput(self.decode(video[:, :-1], encoding.quantized), encoding)


class GenieDynamics(LocalModelMixin, nn.Module):
    """Apply bidirectional token attention within each frame and causal attention between frames."""

    def __init__(self, config: GenieDynamicsConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.action = nn.Linear(config.action_dim, config.hidden_size, bias=False)
        self.backbone = STStack(
            config.hidden_size,
            config.num_heads,
            config.head_dim,
            config.num_layers,
            config.spatial_tokens,
            config.max_frames,
            config.intermediate_ratio,
            config.qk_norm,
            config.norm_eps,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, tokens, actions):
        c = self.config
        if (
            tokens.ndim != 3
            or tokens.shape[2] != c.spatial_tokens
            or len(tokens) < 1
            or not 1 <= tokens.shape[1] <= c.max_frames
            or tokens.dtype != torch.int64
            or (tokens < 0).any()
            or (tokens > c.mask_token_id).any()
        ):
            raise ValueError("Invalid Genie dynamics token grid")
        if (
            actions.shape != (len(tokens), tokens.shape[1] - 1, c.action_dim)
            or actions.device != tokens.device
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Genie actions must align previous-to-current frames")
        conditioning = self.action(actions)
        conditioning = F.pad(conditioning, (0, 0, 1, 0))
        return self.output(self.backbone(self.tokens(tokens) + conditioning[:, :, None])).float()


@dataclass(frozen=True)
class GenieWorldConfig:
    action: GenieActionConfig
    dynamics: GenieDynamicsConfig

    def __post_init__(self):
        if not isinstance(self.action, GenieActionConfig) or not isinstance(
            self.dynamics, GenieDynamicsConfig
        ):
            raise ValueError("Genie world needs explicit action/dynamics configurations")
        if (
            self.action.latent_dim != self.dynamics.action_dim
            or self.action.max_frames != self.dynamics.max_frames
        ):
            raise ValueError("Genie world latent-action/context configurations differ")

    def to_dict(self):
        return dict(
            architecture="genie_world",
            action=self.action.to_dict(),
            dynamics=self.dynamics.to_dict(),
        )


class GenieWorld(LocalModelMixin, nn.Module):
    """Train latent-action reconstruction/VQ jointly while dynamics reads detached action codes."""

    def __init__(self, config: GenieWorldConfig):
        super().__init__()
        self.config = config
        self.action_model = GenieLatentAction(config.action)
        self.dynamics = GenieDynamics(config.dynamics)

    def forward(self, video, masked_tokens):
        inferred = self.action_model(video)
        logits = self.dynamics(masked_tokens, inferred.encoding.quantized.detach())
        return inferred, logits
