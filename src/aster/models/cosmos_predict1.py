"""Native Cosmos-Predict1 GeneralDIT architecture and uncertainty head."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F

from aster.core import FieldOutput
from aster.nn.normalization import FloatRMSNorm
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class CosmosPredict1Config:
    architecture: ClassVar[str] = "cosmos_predict1"
    in_channels: int = 2
    out_channels: int = 2
    model_channels: int = 48
    num_blocks: int = 2
    num_heads: int = 4
    crossattn_emb_channels: int = 24
    mlp_ratio: float = 4.0
    patch_spatial: int = 2
    patch_temporal: int = 1
    max_img_h: int = 16
    max_img_w: int = 16
    max_frames: int = 8
    concat_padding_mask: bool = True
    affline_emb_norm: bool = True
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 16
    block_config: str = "FA-CA-MLP"
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    extra_per_block_abs_pos_emb: bool = False

    def __post_init__(self):
        dimensions = (
            self.in_channels,
            self.out_channels,
            self.model_channels,
            self.num_blocks,
            self.num_heads,
            self.crossattn_emb_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.max_img_h,
            self.max_img_w,
            self.max_frames,
            self.adaln_lora_dim,
        )
        if any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("Predict1 dimensions must be positive integers")
        if (
            self.model_channels % self.num_heads
            or self.model_channels % 2
            or self.model_channels // self.num_heads < 12
            or self.model_channels // self.num_heads % 2
        ):
            raise ValueError("Predict1 needs even head_dim>=12 to define all three NTK RoPE axes")
        if (
            self.max_img_h % self.patch_spatial
            or self.max_img_w % self.patch_spatial
            or self.max_frames % self.patch_temporal
        ):
            raise ValueError("Predict1 configured dimensions must divide patch sizes")
        if self.in_channels != self.out_channels:
            raise ValueError(
                "This Predict1 EDM residual contract requires equal input/output channels"
            )
        if (
            not math.isfinite(self.mlp_ratio)
            or self.mlp_ratio <= 0
            or int(self.model_channels * self.mlp_ratio) < 1
        ):
            raise ValueError("Invalid Predict1 MLP expansion")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (
                self.rope_h_extrapolation_ratio,
                self.rope_w_extrapolation_ratio,
                self.rope_t_extrapolation_ratio,
            )
        ):
            raise ValueError("Predict1 NTK ratios must be finite positive")
        if any(
            type(x) is not bool
            for x in (
                self.concat_padding_mask,
                self.affline_emb_norm,
                self.use_adaln_lora,
                self.extra_per_block_abs_pos_emb,
            )
        ):
            raise ValueError("Predict1 architecture switches must be boolean")
        if not self.block_config or any(
            x not in {"FA", "CA", "MLP"} for x in self.block_config.split("-")
        ):
            raise ValueError("Predict1 only implements explicit FA/CA/MLP building blocks")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class CosmosPredict1Condition:
    text_embeddings: torch.Tensor
    fps: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None


class Predict1PatchLayout(nn.Module):
    def __init__(self, spatial, temporal):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal

    def forward(self, x):
        b, c, t, h, w = x.shape
        p, r = self.spatial, self.temporal

        return (
            x.reshape(b, c, t // r, r, h // p, p, w // p, p)
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(b, -1, c * r * p * p)
        )


class Predict1PatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        width = (c.in_channels + int(c.concat_padding_mask)) * c.patch_spatial**2 * c.patch_temporal
        self.proj = nn.Sequential(
            Predict1PatchLayout(c.patch_spatial, c.patch_temporal),
            nn.Linear(width, c.model_channels, bias=False),
        )

    def forward(self, x):
        return self.proj(x)


class Predict1Timesteps(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, time):
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(self.width // 2, device=time.device).float()
            / (self.width // 2)
        )
        angles = time.float()[:, None] * frequencies
        return torch.cat((angles.cos(), angles.sin()), -1).to(time.dtype)


class Predict1TimeEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.use_lora = c.use_adaln_lora
        self.linear_1 = nn.Linear(c.model_channels, c.model_channels, bias=not self.use_lora)
        self.linear_2 = nn.Linear(
            c.model_channels, c.model_channels * (3 if self.use_lora else 1), bias=not self.use_lora
        )

    def forward(self, sample):
        value = self.linear_2(F.silu(self.linear_1(sample)))

        return (sample, value) if self.use_lora else (value, None)


class Predict1Rope(nn.Module):
    _aster_semantic_buffers = ("dim_spatial_range", "dim_temporal_range")

    def __init__(self, c):
        super().__init__()
        self.config = c
        head = c.model_channels // c.num_heads
        spatial = head // 6 * 2
        temporal = head - 2 * spatial
        length = max(
            c.max_img_h // c.patch_spatial,
            c.max_img_w // c.patch_spatial,
            c.max_frames // c.patch_temporal,
        )
        self.register_buffer("seq", torch.arange(length).float())
        self.register_buffer(
            "dim_spatial_range", torch.arange(0, spatial, 2).float() / spatial, persistent=False
        )
        self.register_buffer(
            "dim_temporal_range", torch.arange(0, temporal, 2).float() / temporal, persistent=False
        )
        self.factors = (
            c.rope_t_extrapolation_ratio ** (temporal / (temporal - 2)),
            c.rope_h_extrapolation_ratio ** (spatial / (spatial - 2)),
            c.rope_w_extrapolation_ratio ** (spatial / (spatial - 2)),
        )

    def forward(self, grid, fps):
        t, h, w = grid
        with torch.autocast(self.seq.device.type, enabled=False):
            frequencies = [
                (10000 * factor) ** (-axis.float())
                for factor, axis in zip(
                    self.factors,
                    (self.dim_temporal_range, self.dim_spatial_range, self.dim_spatial_range),
                )
            ]
            time_positions = (
                self.seq[:t].float() if fps is None else self.seq[:t].float() / fps[0].float() * 24
            )
            axes = [
                torch.outer(time_positions, frequencies[0]),
                torch.outer(self.seq[:h].float(), frequencies[1]),
                torch.outer(self.seq[:w].float(), frequencies[2]),
            ]
            phase = torch.cat(
                (
                    axes[0][:, None, None].expand(t, h, w, -1),
                    axes[1][None, :, None].expand(t, h, w, -1),
                    axes[2][None, None].expand(t, h, w, -1),
                ),
                -1,
            )
            return torch.cat((phase, phase), -1).reshape(t * h * w, -1)


class Predict1AxisPositions(nn.Module):
    def __init__(self, c):
        super().__init__()
        for axis, length in (
            ("h", c.max_img_h // c.patch_spatial),
            ("w", c.max_img_w // c.patch_spatial),
            ("t", c.max_frames // c.patch_temporal),
        ):
            parameter = nn.Parameter(torch.empty(length, c.model_channels))
            nn.init.trunc_normal_(parameter, std=0.02)
            setattr(self, "pos_emb_" + axis, parameter)

    def forward(self, grid):
        t, h, w = grid
        value = (
            self.pos_emb_t[:t, None, None]
            + self.pos_emb_h[None, :h, None]
            + self.pos_emb_w[None, None, :w]
        )

        denominator = (
            torch.linalg.vector_norm(value, dim=-1, keepdim=True, dtype=torch.float32)
            / math.sqrt(value.shape[-1])
            + 1e-6
        )
        return (value / denominator.to(value.dtype)).reshape(1, t * h * w, -1)


class Predict1Attention(nn.Module):
    def __init__(self, c, cross):
        super().__init__()
        self.heads = c.num_heads
        self.head_dim = c.model_channels // c.num_heads
        self.cross = cross
        self.to_q = nn.Sequential(
            nn.Linear(c.model_channels, c.model_channels, bias=False), FloatRMSNorm(self.head_dim)
        )
        self.to_k = nn.Sequential(
            nn.Linear(
                c.crossattn_emb_channels if cross else c.model_channels,
                c.model_channels,
                bias=False,
            ),
            FloatRMSNorm(self.head_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(
                c.crossattn_emb_channels if cross else c.model_channels,
                c.model_channels,
                bias=False,
            ),
            nn.Identity(),
        )
        self.to_out = nn.Sequential(
            nn.Linear(c.model_channels, c.model_channels, bias=False), nn.Dropout(0.0)
        )

    def forward(self, hidden, context, phase):
        b, length, width = hidden.shape
        source = context if self.cross else hidden

        def project(x, layers):
            return layers[1](
                layers[0](x).reshape(b, x.shape[1], self.heads, self.head_dim)
            ).transpose(1, 2)

        q, k, v = project(hidden, self.to_q), project(source, self.to_k), project(source, self.to_v)
        if not self.cross:

            def rotate(x):
                a, d = x.chunk(2, -1)
                return (
                    x * phase.cos().to(x.dtype)[None, None]
                    + torch.cat((-d, a), -1) * phase.sin().to(x.dtype)[None, None]
                )

            q, k = rotate(q), rotate(k)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, width)
        return self.to_out(out)


class Predict1VideoAttention(nn.Module):
    def __init__(self, c, cross):
        super().__init__()
        self.attn = Predict1Attention(c, cross)

    def forward(self, hidden, context, phase):
        return self.attn(hidden, context, phase)


class Predict1FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer1 = nn.Linear(c.model_channels, int(c.model_channels * c.mlp_ratio), bias=False)
        self.layer2 = nn.Linear(int(c.model_channels * c.mlp_ratio), c.model_channels, bias=False)

    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x), approximate="none"))


def _modulation(c, chunks):
    layers = [
        nn.SiLU(),
        nn.Linear(
            c.model_channels,
            c.adaln_lora_dim if c.use_adaln_lora else chunks * c.model_channels,
            bias=False,
        ),
    ]
    if c.use_adaln_lora:
        layers.append(nn.Linear(c.adaln_lora_dim, chunks * c.model_channels, bias=False))
    return nn.Sequential(*layers)


class Predict1BuildingBlock(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.block = (
            Predict1FeedForward(c) if kind == "MLP" else Predict1VideoAttention(c, kind == "CA")
        )
        self.norm_state = nn.LayerNorm(c.model_channels, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = _modulation(c, 3)

    def forward(self, x, emb, shared, context, phase):
        affine = self.adaLN_modulation(emb)
        if shared is not None:
            affine = affine + shared
        shift, scale, gate = affine.chunk(3, -1)
        transformed = self.norm_state(x) * (1 + scale[:, None]) + shift[:, None]
        value = (
            self.block(transformed)
            if self.kind == "MLP"
            else self.block(transformed, context, phase)
        )
        return x + gate[:, None] * value


class Predict1Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList(
            Predict1BuildingBlock(c, kind) for kind in c.block_config.split("-")
        )

    def forward(self, x, emb, shared, context, phase, positions):
        if positions is not None:
            x = x + positions
        for block in self.blocks:
            x = block(x, emb, shared, context, phase)
        return x


class Predict1FinalLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.width = c.model_channels
        self.norm_final = nn.LayerNorm(c.model_channels, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(
            c.model_channels, c.patch_spatial**2 * c.patch_temporal * c.out_channels, bias=False
        )
        self.adaLN_modulation = _modulation(c, 2)

    def forward(self, x, emb, shared):
        affine = self.adaLN_modulation(emb)
        if shared is not None:
            affine = affine + shared[:, : 2 * self.width]
        shift, scale = affine.chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale[:, None]) + shift[:, None])


class CosmosPredict1DiT(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.x_embedder = Predict1PatchEmbed(c)
        self.pos_embedder = Predict1Rope(c)
        self.t_embedder = nn.Sequential(
            Predict1Timesteps(c.model_channels), Predict1TimeEmbedding(c)
        )
        self.blocks = nn.ModuleDict(
            {f"block{index}": Predict1Block(c) for index in range(c.num_blocks)}
        )
        self.final_layer = Predict1FinalLayer(c)
        self.affline_norm = FloatRMSNorm(c.model_channels) if c.affline_emb_norm else nn.Identity()
        if c.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = Predict1AxisPositions(c)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for layer in (self.t_embedder[1].linear_1, self.t_embedder[1].linear_2):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks.values():
            for unit in block.blocks:
                nn.init.zeros_(unit.adaLN_modulation[-1].weight)

    def forward(self, sample, time, condition):
        c = self.config
        if (
            sample.ndim != 5
            or sample.shape[1] != c.in_channels
            or min(sample.shape[0], *sample.shape[2:]) < 1
            or not sample.is_floating_point()
        ):
            raise ValueError("Predict1 expects floating BCTHW latent")
        b, _, t, h, w = sample.shape
        p, r = c.patch_spatial, c.patch_temporal
        if t % r or h % p or w % p or t > c.max_frames or h > c.max_img_h or w > c.max_img_w:
            raise ValueError(
                "Predict1 latent dimensions must divide patches and stay inside configured grid"
            )
        if (
            time.shape != (b,)
            or time.device != sample.device
            or not time.is_floating_point()
            or not torch.isfinite(time).all()
        ):
            raise ValueError("Predict1 time is floating c_noise=log(sigma)/4 [B], not step indexes")
        if not isinstance(condition, CosmosPredict1Condition):
            raise ValueError("Predict1 requires explicit text/FPS/padding condition")
        text, fps, padding = condition.text_embeddings, condition.fps, condition.padding_mask
        if (
            text.ndim != 3
            or text.shape[0] != b
            or text.shape[1] < 1
            or text.shape[2] != c.crossattn_emb_channels
            or text.device != sample.device
            or not text.is_floating_point()
        ):
            raise ValueError("Predict1 text context must be floating [B,L,crossattn_emb_channels]")
        if fps is None:
            if t // r != 1:
                raise ValueError("Predict1 video requires explicit FPS")
        elif (
            fps.shape != (b,)
            or fps.device != sample.device
            or not torch.isfinite(fps).all()
            or (fps <= 0).any()
            or (t // r > 1 and not torch.equal(fps, fps[:1].expand_as(fps)))
        ):
            raise ValueError("Predict1 video batches require common finite positive FPS")
        if c.concat_padding_mask:
            if (
                padding is None
                or padding.ndim != 3
                or padding.shape[0] != b
                or padding.device != sample.device
                or not torch.isfinite(padding).all()
            ):
                raise ValueError("Predict1 needs explicit spatial padding_mask[B,H,W] channel")
            resized = F.interpolate(padding[:, None].to(sample.dtype), (h, w), mode="nearest")
            sample = torch.cat((sample, resized[:, :, None].expand(-1, -1, t, -1, -1)), 1)
        elif padding is not None:
            raise ValueError("Predict1 padding channel disabled by configuration")
        grid = t // r, h // p, w // p
        hidden = self.x_embedder(sample)
        emb, shared = self.t_embedder(time.to(hidden.dtype))
        emb = self.affline_norm(emb)
        phase = self.pos_embedder(grid, fps)
        positions = self.extra_pos_embedder(grid) if c.extra_per_block_abs_pos_emb else None
        for block in self.blocks.values():
            hidden = block(hidden, emb, shared, text.to(hidden.dtype), phase, positions)
        output = self.final_layer(hidden, emb, shared)
        output = (
            output.reshape(b, *grid, p, p, r, c.out_channels)
            .permute(0, 7, 1, 6, 2, 4, 3, 5)
            .reshape(b, c.out_channels, t, h, w)
        )
        return FieldOutput(output, "edm_residual")


@dataclass(frozen=True)
class CosmosPredict1ModelConfig:
    architecture: ClassVar[str] = "cosmos_predict1_model"
    net: CosmosPredict1Config = CosmosPredict1Config()
    logvar_channels: int = 128

    def __post_init__(self):
        if isinstance(self.net, dict):
            values = dict(self.net)
            if values.pop("architecture", "cosmos_predict1") != "cosmos_predict1":
                raise ValueError("Predict1 composite needs actual Predict1 net")
            object.__setattr__(self, "net", CosmosPredict1Config(**values))
        if (
            not isinstance(self.net, CosmosPredict1Config)
            or type(self.logvar_channels) is not int
            or self.logvar_channels < 1
        ):
            raise ValueError("Invalid Predict1 training composition")

    def to_dict(self):
        return dict(
            architecture=self.architecture,
            net=self.net.to_dict(),
            logvar_channels=self.logvar_channels,
        )


class Predict1FourierFeatures(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.register_buffer("freqs", 2 * math.pi * torch.randn(width))
        self.register_buffer("phases", 2 * math.pi * torch.rand(width))

    def forward(self, time):
        value = time.float()[:, None] * self.freqs.float() + self.phases.float()
        return (value.cos() * math.sqrt(2)).to(time.dtype)


class CosmosPredict1Model(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = CosmosPredict1DiT(config.net)
        self.logvar = nn.Sequential(
            Predict1FourierFeatures(config.logvar_channels),
            nn.Linear(config.logvar_channels, 1, bias=False),
        )

    def forward(self, sample, time, condition):
        return self.net(sample, time, condition)

    def predict_logvar(self, time):
        return self.logvar(time).squeeze(-1)
