"""ACT conditional VAE with action chunks and DETR-style query decoding."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn
from .serialization import LocalModelMixin


def _position(indices, width):
    frequencies = 10000 ** (2 * (torch.arange(width, device=indices.device) // 2).float() / width)
    angles = indices.float()[:, None] / frequencies[None]
    return torch.where(
        (torch.arange(width, device=indices.device) % 2 == 0)[None], angles.sin(), angles.cos()
    )


class PositionEncoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(width, feedforward), nn.Linear(feedforward, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, position, padding=None):
        query = x + position
        attention = self.attention(query, query, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.norm1(x + self.dropout(attention))
        return self.norm2(x + self.dropout(self.linear2(self.dropout(self.linear1(x).relu()))))


class PositionDecoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward, dropout=0.0):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.linear1, self.linear2 = nn.Linear(width, feedforward), nn.Linear(feedforward, width)
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(width),
            nn.LayerNorm(width),
            nn.LayerNorm(width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, query_position, memory_position, padding=None):
        query = x + query_position
        x = self.norm1(
            x + self.dropout(self.self_attention(query, query, x, need_weights=False)[0])
        )
        x = self.norm2(
            x
            + self.dropout(
                self.cross_attention(
                    x + query_position,
                    memory + memory_position,
                    memory,
                    key_padding_mask=padding,
                    need_weights=False,
                )[0]
            )
        )
        return self.norm3(x + self.dropout(self.linear2(self.dropout(self.linear1(x).relu()))))


@dataclass(frozen=True)
class ACTConfig:
    proprio_dim: int = 7
    action_dim: int = 7
    vision_dim: int = 32
    hidden_size: int = 64
    latent_dim: int = 16
    horizon: int = 16
    num_heads: int = 4
    posterior_layers: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 3
    feedforward_size: int = 256
    dropout: float = 0.0

    def __post_init__(self):
        if (
            min(
                self.proprio_dim,
                self.action_dim,
                self.vision_dim,
                self.hidden_size,
                self.latent_dim,
                self.horizon,
                self.num_heads,
                self.posterior_layers,
                self.encoder_layers,
                self.decoder_layers,
                self.feedforward_size,
            )
            < 1
            or self.hidden_size % self.num_heads
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Invalid ACT dimensions")

    def to_dict(self):
        return {"architecture": "act", **asdict(self)}


@dataclass
class ActionOutput:
    actions: torch.Tensor
    pad_logits: torch.Tensor
    mean: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    state: object = None


class ACTPolicy(LocalModelMixin, nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.posterior_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.posterior_proprio = nn.Linear(config.proprio_dim, d)
        self.posterior_actions = nn.Linear(config.action_dim, d)
        self.posterior = nn.ModuleList(
            [
                PositionEncoderLayer(d, config.num_heads, config.feedforward_size, config.dropout)
                for _ in range(config.posterior_layers)
            ]
        )
        self.latent_statistics = nn.Linear(d, 2 * config.latent_dim)
        self.proprio_projection = nn.Linear(config.proprio_dim, d)
        self.vision_projection = nn.Linear(config.vision_dim, d)
        self.latent_projection = nn.Linear(config.latent_dim, d)
        self.extra_position = nn.Parameter(torch.randn(1, 2, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, config.horizon, d) * 0.02)
        self.encoder = nn.ModuleList(
            [
                PositionEncoderLayer(d, config.num_heads, config.feedforward_size, config.dropout)
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                PositionDecoderLayer(d, config.num_heads, config.feedforward_size, config.dropout)
                for _ in range(config.decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d)
        self.action_head, self.pad_head = nn.Linear(d, config.action_dim), nn.Linear(d, 1)

    def forward(
        self,
        proprio,
        vision_tokens,
        *,
        actions=None,
        action_padding=None,
        vision_padding=None,
        vision_positions=None,
        generator=None,
    ):
        c, b = self.config, len(proprio)
        if (
            proprio.shape != (b, c.proprio_dim)
            or vision_tokens.ndim != 3
            or vision_tokens.shape[0] != b
            or vision_tokens.shape[-1] != c.vision_dim
        ):
            raise ValueError("ACT observation dimensions mismatch")
        mean = logvar = None
        if actions is not None:
            if actions.shape != (b, c.horizon, c.action_dim):
                raise ValueError("ACT training requires full padded horizon")
            action_padding = (
                torch.zeros(b, c.horizon, dtype=torch.bool, device=actions.device)
                if action_padding is None
                else action_padding
            )
            if action_padding.shape != (b, c.horizon) or action_padding.dtype != torch.bool:
                raise ValueError("ACT padding is bool BH")
            inputs = torch.cat(
                (
                    self.posterior_cls.expand(b, -1, -1),
                    self.posterior_proprio(proprio)[:, None],
                    self.posterior_actions(actions),
                ),
                1,
            )
            positions = _position(
                torch.arange(c.horizon + 2, device=inputs.device), c.hidden_size
            ).to(inputs)[None]
            padding = torch.cat(
                (torch.zeros(b, 2, dtype=torch.bool, device=inputs.device), action_padding), 1
            )
            for layer in self.posterior:
                inputs = layer(inputs, positions, padding)
            mean, logvar = self.latent_statistics(inputs[:, 0]).chunk(2, -1)

            latent = mean + (0.5 * logvar).exp() * torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
            )
        else:
            if action_padding is not None:
                raise ValueError("Padding without training actions is ambiguous")
            latent = proprio.new_zeros(b, c.latent_dim)
        vision = self.vision_projection(vision_tokens)
        if vision_positions is None:
            vision_positions = _position(
                torch.arange(vision.shape[1], device=vision.device), c.hidden_size
            ).to(vision)[None]
        if vision_positions.shape not in {
            (1, vision.shape[1], c.hidden_size),
            (b, vision.shape[1], c.hidden_size),
        }:
            raise ValueError("Invalid explicit vision positions")
        memory = torch.cat(
            (
                self.latent_projection(latent)[:, None],
                self.proprio_projection(proprio)[:, None],
                vision,
            ),
            1,
        )
        positions = torch.cat(
            (self.extra_position.expand(b, -1, -1), vision_positions.expand(b, -1, -1)), 1
        )
        padding = None
        if vision_padding is not None:
            if vision_padding.shape != vision.shape[:2] or vision_padding.dtype != torch.bool:
                raise ValueError("Invalid vision padding mask")
            padding = torch.cat(
                (torch.zeros(b, 2, device=vision.device, dtype=torch.bool), vision_padding), 1
            )
        for layer in self.encoder:
            memory = layer(memory, positions, padding)
        query_position = self.queries.expand(b, -1, -1)
        decoded = torch.zeros_like(query_position)
        for layer in self.decoder:
            decoded = layer(decoded, memory, query_position, positions, padding)
        decoded = self.decoder_norm(decoded)
        return ActionOutput(
            self.action_head(decoded), self.pad_head(decoded).squeeze(-1), mean, logvar
        )

    @torch.no_grad()
    def predict_chunk(self, observation, state=None):
        if state is not None:
            raise ValueError(
                "ACT is stateless; execution/ensemble state belongs to the action controller"
            )
        return self(
            observation["proprio"],
            observation["vision_tokens"],
            vision_padding=observation.get("vision_padding"),
            vision_positions=observation.get("vision_positions"),
        )
