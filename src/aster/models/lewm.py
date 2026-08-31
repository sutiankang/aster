"""LeWorldModel ViT, batch-normalized projection, and action-conditioned causal prediction.

Reference: lucas-maes/le-wm at 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, MIT.
Copyright(c)2026 Lucas Maes.
ViT components follow the Google/Hugging Face Apache-2.0 implementations.
Training targets retain gradients; planning freezes the complete model."""

from dataclasses import asdict, dataclass, field
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from ..nn.normalization import LayerNorm, BatchNorm1d
from ..nn.parameter_codec import register_parameter_codec
from ..nn.attention import scaled_attention, attention_mask
from .serialization import LocalModelMixin
from .vit import ViTConfig, ViTModel, convert_vit_state_dict


@dataclass(frozen=True)
class LeWMConfig:
    architecture: ClassVar[str] = "lewm"
    encoder: ViTConfig = field(default_factory=ViTConfig)
    embed_dim: int = 32
    action_dim: int = 2
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4
    history_size: int = 3
    predictor_hidden_dim: int = 32
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_head_dim: int = 8
    predictor_mlp_dim: int = 64
    projector_hidden_dim: int = 64
    predictor_dropout: float = 0.0
    embedding_dropout: float = 0.0

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            values = dict(self.encoder)
            if values.pop("architecture", "vit") != "vit":
                raise ValueError("LeWM encoder must be a genuine standard ViT")
            object.__setattr__(self, "encoder", ViTConfig(**values))
        if not isinstance(self.encoder, ViTConfig):
            raise ValueError("LeWM encoder config must be ViTConfig")
        for name in (
            "embed_dim",
            "action_dim",
            "action_smoothed_dim",
            "action_mlp_scale",
            "history_size",
            "predictor_hidden_dim",
            "predictor_depth",
            "predictor_heads",
            "predictor_head_dim",
            "predictor_mlp_dim",
            "projector_hidden_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Invalid LeWM dimension")
        if any(
            not math.isfinite(x) or not 0 <= x < 1
            for x in (self.predictor_dropout, self.embedding_dropout)
        ):
            raise ValueError("Invalid LeWM dropout")

    def to_dict(self):
        values = asdict(self)
        values["encoder"] = self.encoder.to_dict()
        return dict(architecture=self.architecture, **values)

    @classmethod
    def official_tiny(cls, *, action_dim=10):
        """Construct the published 18,034,478-parameter LeWorldModel configuration without downloading weights."""
        return cls(
            encoder=ViTConfig(
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
                intermediate_size=768,
                image_size=224,
                patch_size=14,
            ),
            embed_dim=192,
            action_dim=action_dim,
            predictor_hidden_dim=192,
            predictor_depth=6,
            predictor_heads=16,
            predictor_head_dim=64,
            predictor_mlp_dim=2048,
            projector_hidden_dim=2048,
            predictor_dropout=0.1,
        )


class LeWMActionEmbedder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch_embed = nn.Conv1d(c.action_dim, c.action_smoothed_dim, 1)
        self.embed = nn.Sequential(
            nn.Linear(c.action_smoothed_dim, c.action_mlp_scale * c.embed_dim),
            nn.SiLU(),
            nn.Linear(c.action_mlp_scale * c.embed_dim, c.embed_dim),
        )

    def forward(self, action):

        return self.embed(
            self.patch_embed(action.to(self.patch_embed.weight.dtype).transpose(1, 2)).transpose(
                1, 2
            )
        )


class LeWMProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value):
        return self.net(value)


class _PredictorAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        d = c.predictor_hidden_dim
        inner = c.predictor_heads * c.predictor_head_dim
        self.norm = LayerNorm(d)
        self.to_qkv = nn.Linear(d, 3 * inner, bias=False)
        self.to_out = (
            nn.Identity()
            if c.predictor_heads == 1 and c.predictor_head_dim == d
            else nn.Sequential(nn.Linear(inner, d), nn.Dropout(c.predictor_dropout))
        )

    def forward(self, x):
        c = self.config
        b, t, _ = x.shape
        q, k, v = [
            part.reshape(b, t, c.predictor_heads, c.predictor_head_dim).transpose(1, 2)
            for part in self.to_qkv(self.norm(x)).chunk(3, -1)
        ]
        result = scaled_attention(
            q,
            k,
            v,
            attention_mask(b, t, t, device=x.device),
            dropout=c.predictor_dropout,
            training=self.training,
            softmax_in_fp32=False,
        )
        return self.to_out(result.transpose(1, 2).reshape(b, t, -1))


class _PredictorMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.predictor_hidden_dim
        self.net = nn.Sequential(
            LayerNorm(d),
            nn.Linear(d, c.predictor_mlp_dim),
            nn.GELU(),
            nn.Dropout(c.predictor_dropout),
            nn.Linear(c.predictor_mlp_dim, d),
            nn.Dropout(c.predictor_dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ConditionalBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.predictor_hidden_dim
        self.attn, self.mlp = _PredictorAttention(c), _PredictorMLP(c)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, condition):
        sa, ca, ga, sm, cm, gm = self.adaLN_modulation(condition).chunk(6, -1)

        x = x + ga * self.attn(self.norm1(x) * (1 + ca) + sa)
        return x + gm * self.mlp(self.norm2(x) * (1 + cm) + sm)


class _ActionTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.predictor_hidden_dim
        self.norm = LayerNorm(d)
        self.input_proj = nn.Linear(c.embed_dim, d) if c.embed_dim != d else nn.Identity()
        self.cond_proj = nn.Linear(c.embed_dim, d) if c.embed_dim != d else nn.Identity()
        self.output_proj = nn.Linear(d, c.embed_dim) if c.embed_dim != d else nn.Identity()
        self.layers = nn.ModuleList(_ConditionalBlock(c) for _ in range(c.predictor_depth))

    def forward(self, x, c):
        x, c = self.input_proj(x), self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class _PredictorPosition(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, c.history_size, c.embed_dim))

    def forward(self, x):
        return x + self.pos_embedding[:, : x.shape[1]]


class LeWMPredictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.positions = _PredictorPosition(c)
        self.dropout = nn.Dropout(c.embedding_dropout)
        self.transformer = _ActionTransformer(c)
        register_parameter_codec(self, {"positions.pos_embedding": "pos_embedding"})

    def forward(self, x, condition):
        return self.transformer(self.dropout(self.positions(x)), condition)


@dataclass(frozen=True)
class LeWMOutput:
    embeddings: torch.Tensor
    predictions: torch.Tensor


class LeWorldModel(LocalModelMixin, nn.Module):
    def __init__(self, config=LeWMConfig()):
        super().__init__()
        self.config = c = config
        self.encoder, self.predictor = ViTModel(c.encoder), LeWMPredictor(c)
        self.action_encoder = LeWMActionEmbedder(c)
        self.projector = LeWMProjector(c.encoder.hidden_size, c.projector_hidden_dim, c.embed_dim)
        self.pred_proj = LeWMProjector(c.embed_dim, c.projector_hidden_dim, c.embed_dim)

    def load_author_state_dict(self, state, *, vit_layout):

        if not isinstance(state, dict) or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError("LeWM author checkpoint must be a plain tensor state dictionary")
        vision = convert_vit_state_dict(
            {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")},
            layout=vit_layout,
        )
        mapped = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        mapped.update({"encoder." + k: v for k, v in vision.items()})
        expected = self.state_dict()
        if set(mapped) != set(expected) or any(
            mapped[k].shape != expected[k].shape or mapped[k].dtype != expected[k].dtype
            for k in mapped
        ):
            raise ValueError("LeWM complete parameter/buffer schema differs; model was not changed")
        if any(
            value.is_floating_point() and not torch.isfinite(value).all()
            for value in mapped.values()
        ):
            raise ValueError("Non-finite LeWM author weight")
        return self.load_state_dict(mapped, strict=True)

    def encode(self, pixels):
        if pixels.ndim != 5 or min(pixels.shape) < 1:
            raise ValueError("LeWM pixels require nonempty BTCHW")
        b, t = pixels.shape[:2]
        cls = self.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[
            :, 0
        ]
        return self.projector(cls).reshape(b, t, -1)

    def predict(self, embeddings, action_embeddings):
        c = self.config
        if (
            embeddings.ndim != 3
            or embeddings.shape != action_embeddings.shape
            or not 1 <= embeddings.shape[1] <= c.history_size
            or embeddings.shape[-1] != c.embed_dim
        ):
            raise ValueError(
                "LeWM predictor needs aligned latent/action embeddings and bounded history"
            )
        result = self.predictor(embeddings, action_embeddings)
        return self.pred_proj(result.flatten(0, 1)).reshape_as(embeddings)

    def forward(self, pixels, actions):
        c = self.config
        if (
            actions.ndim != 3
            or actions.shape != (len(pixels), pixels.shape[1] - 1, c.action_dim)
            or pixels.shape[1] != c.history_size + 1
        ):
            raise ValueError("LeWM teacher forcing needs H+1 frames and H transition actions")
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError("LeWM actions must be finite explicitly-normalized floats")
        embeddings = self.encode(pixels)
        return LeWMOutput(
            embeddings, self.predict(embeddings[:, :-1], self.action_encoder(actions))
        )

    def rollout_latents(self, history, actions):

        c = self.config
        if (
            history.ndim != 3
            or not 1 <= history.shape[1] <= c.history_size
            or history.shape[-1] != c.embed_dim
            or actions.ndim != 4
        ):
            raise ValueError("Invalid LeWM rollout history/candidate layout")
        b, s, t, a = actions.shape
        h = history.shape[1]
        if b != len(history) or min(b, s) < 1 or t < h or a != c.action_dim:
            raise ValueError("Candidate actions must include known history and a future step")
        latent = history[:, None].expand(b, s, h, -1).reshape(b * s, h, -1)
        actions = actions.reshape(b * s, t, a)
        for index in range(h - 1, t):
            length = min(c.history_size, latent.shape[1])
            condition = self.action_encoder(actions[:, index - length + 1 : index + 1])
            next_latent = self.predict(latent[:, -length:], condition)[:, -1:]
            latent = torch.cat((latent, next_latent), 1)
        return latent.reshape(b, s, t + 1, c.embed_dim)
