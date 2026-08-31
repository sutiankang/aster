"""SAM/Vary window attention, decomposed relative positions, and channel-wise 2D normalization."""

import torch
from torch import nn
import torch.nn.functional as F
from .parameter_codec import register_parameter_codec


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial position."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(width)), nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, hidden):
        mean = hidden.mean(1, keepdim=True)
        variance = (hidden - mean).square().mean(1, keepdim=True)
        return (
            self.weight[:, None, None] * (hidden - mean) / torch.sqrt(variance + self.eps)
            + self.bias[:, None, None]
        )


class SAMPosition(nn.Module):
    def __init__(self, side, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, side, side, width))

    def forward(self, hidden):
        positions = self.pos_embed
        if positions.shape[1:3] != hidden.shape[1:3]:
            positions = (
                F.interpolate(
                    positions.permute(0, 3, 1, 2).float(),
                    size=hidden.shape[1:3],
                    mode="bicubic",
                    antialias=True,
                    align_corners=False,
                )
                .to(positions.dtype)
                .permute(0, 2, 3, 1)
            )
        return hidden + positions


class SAMRelativeBias(nn.Module):
    def __init__(self, side, width):
        super().__init__()
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * side - 1, width))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * side - 1, width))

    @staticmethod
    def table(parameter, side):
        if len(parameter) != 2 * side - 1:
            parameter = F.interpolate(
                parameter.T[None].float(), size=2 * side - 1, mode="linear", align_corners=False
            )[0].T.to(parameter.dtype)
        coordinates = torch.arange(side, device=parameter.device)
        return parameter[(coordinates[:, None] - coordinates[None] + side - 1).long()]

    def forward(self, query, height, width):
        b, heads, _, dimension = query.shape
        image_query = query.reshape(b, heads, height, width, dimension)
        height_bias = torch.einsum(
            "bnhwd,hkd->bnhwk", image_query, self.table(self.rel_pos_h, height)
        )
        width_bias = torch.einsum(
            "bnhwd,wkd->bnhwk", image_query, self.table(self.rel_pos_w, width)
        )

        return (height_bias[..., :, None] + width_bias[..., None, :]).reshape(
            b, heads, height * width, height * width
        )


class SAMAttention(nn.Module):
    def __init__(self, width, heads, side):
        super().__init__()
        self.heads = heads
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.relative = SAMRelativeBias(side, width // heads)
        register_parameter_codec(
            self, {"relative.rel_pos_h": "rel_pos_h", "relative.rel_pos_w": "rel_pos_w"}
        )

    def forward(self, hidden):
        b, h, w, width = hidden.shape

        q, k, v = (
            self.qkv(hidden)
            .reshape(b, h * w, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=self.relative(q, h, w))
        return self.proj(value.transpose(1, 2).reshape(b, h, w, width))


class SAMMLP(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(width, intermediate), nn.Linear(intermediate, width)

    def forward(self, hidden):
        return self.lin2(F.gelu(self.lin1(hidden)))


class SAMBlock(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.window = 0 if index in c.global_attn_indexes else c.window_size
        side = self.window or c.image_size // c.patch_size
        self.norm1, self.norm2 = (
            nn.LayerNorm(c.hidden_size, eps=c.norm_eps),
            nn.LayerNorm(c.hidden_size, eps=c.norm_eps),
        )
        self.attn = SAMAttention(c.hidden_size, c.num_heads, side)
        self.mlp = SAMMLP(c.hidden_size, c.intermediate_size)

    def forward(self, hidden):
        normal = self.norm1(hidden)
        if self.window:
            b, h, w, width = normal.shape
            size = self.window
            hp, wp = h + (-h) % size, w + (-w) % size

            padded = F.pad(normal, (0, 0, 0, wp - w, 0, hp - h))
            windows = (
                padded.reshape(b, hp // size, size, wp // size, size, width)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, size, size, width)
            )
            windows = self.attn(windows)
            value = (
                windows.reshape(b, hp // size, wp // size, size, size, width)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, hp, wp, width)[:, :h, :w]
            )
        else:
            value = self.attn(normal)
        hidden = hidden + value
        return hidden + self.mlp(self.norm2(hidden))


class SAMImageEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(
            c.in_channels, c.hidden_size, c.patch_size, stride=c.patch_size
        )
        self.position = SAMPosition(c.image_size // c.patch_size, c.hidden_size)
        register_parameter_codec(self, {"position.pos_embed": "pos_embed"})
        self.blocks = nn.ModuleList(SAMBlock(c, i) for i in range(c.depth))
        self.neck = nn.Sequential(
            nn.Conv2d(c.hidden_size, c.neck_channels, 1, bias=False),
            LayerNorm2d(c.neck_channels, c.norm_eps),
            nn.Conv2d(c.neck_channels, c.neck_channels, 3, padding=1, bias=False),
            LayerNorm2d(c.neck_channels, c.norm_eps),
        )
        self.net_2 = nn.Conv2d(
            c.neck_channels, c.downsample_channels, 3, stride=2, padding=1, bias=False
        )
        self.net_3 = nn.Conv2d(
            c.downsample_channels, c.output_channels, 3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values):
        c = self.config
        if (
            pixel_values.ndim != 4
            or pixel_values.shape[1] != c.in_channels
            or pixel_values.shape[2] != pixel_values.shape[3]
            or pixel_values.shape[2] % c.patch_size
        ):
            raise ValueError("SAM expects square BCHW views divisible by patch size")
        hidden = self.position(self.patch_embed.proj(pixel_values).permute(0, 2, 3, 1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.net_3(self.net_2(self.neck(hidden.permute(0, 3, 1, 2))))
