"""GR00T N1.7 embodiment-conditioned action DiT over native Qwen3-VL features.

Reference: NVIDIA Isaac-GR00T at 23ace64f17aa5015259b8609d371eb61a357c776,
gr00t/model/{gr00t_n1d7/gr00t_n1d7.py,modules/dit.py,modules/embodiment_conditioned_mlp.py}.
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
Apache-2.0; model weight terms are separate. Actions must already be normalized."""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import FieldOutput
from aster.nn.attention import scaled_attention
from .serialization import LocalModelMixin, configuration_key
from .qwen_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from .actions import ActionOutput


@dataclass(frozen=True)
class GrootActionConfig:
    architecture: ClassVar[str] = "groot_n17_action"
    max_state_dim: int = 6
    max_action_dim: int = 3
    action_horizon: int = 4
    state_history_length: int = 1
    max_num_embodiments: int = 3
    hidden_size: int = 24
    input_embedding_dim: int = 32
    backbone_embedding_dim: int = 32
    num_layers: int = 4
    num_attention_heads: int = 4
    attention_head_dim: int = 8
    dropout: float = 0.0
    state_dropout_prob: float = 0.0
    attention_bias: bool = True
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-5
    norm_type: str = "ada_norm"
    activation_fn: str = "gelu-approximate"
    final_dropout: bool = True
    positional_embeddings: str | None = None
    max_seq_len: int = 128
    interleave_self_attention: bool = True
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    use_vlln: bool = True
    add_pos_embed: bool = True
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4

    def __post_init__(self):
        integers = (
            self.max_state_dim,
            self.max_action_dim,
            self.action_horizon,
            self.state_history_length,
            self.max_num_embodiments,
            self.hidden_size,
            self.input_embedding_dim,
            self.backbone_embedding_dim,
            self.num_layers,
            self.num_attention_heads,
            self.attention_head_dim,
            self.max_seq_len,
            self.attend_text_every_n_blocks,
            self.num_timestep_buckets,
            self.num_inference_timesteps,
        )
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("GR00T dimensions/counts must be positive integers")
        if (
            self.input_embedding_dim != self.num_attention_heads * self.attention_head_dim
            or self.input_embedding_dim % 2
        ):
            raise ValueError("GR00T action width must equal heads*head_dim and be even")
        if (
            self.action_horizon + 1 > self.max_seq_len
            or self.norm_eps <= 0
            or not math.isfinite(self.norm_eps)
        ):
            raise ValueError("GR00T positional capacity or normalization is invalid")
        if any(not 0 <= x < 1 for x in (self.dropout, self.state_dropout_prob)):
            raise ValueError("Dropout probabilities must be in [0,1)")
        if self.use_alternate_vl_dit and not self.interleave_self_attention:
            raise ValueError("AlternateVLDiT requires interleaved self attention")
        if self.norm_type not in {"ada_norm", "layer_norm"} or self.activation_fn not in {
            "gelu",
            "gelu-approximate",
            "geglu",
        }:
            raise ValueError("Unsupported GR00T normalization/FFN formula")
        if self.positional_embeddings not in {None, "sinusoidal"}:
            raise ValueError("Unsupported GR00T DiT position embedding")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}

    @property
    def action_dim(self):
        return self.max_action_dim


@dataclass(frozen=True)
class GrootConfig:
    architecture: ClassVar[str] = "groot_n17"
    backbone_config: Qwen3VLConfig = field(default_factory=Qwen3VLConfig)
    action_config: GrootActionConfig = field(default_factory=GrootActionConfig)
    select_layer: int | None = None

    def __post_init__(self):
        if not isinstance(self.backbone_config, Qwen3VLConfig) or not isinstance(
            self.action_config, GrootActionConfig
        ):
            raise TypeError("GR00T N1.7 needs actual Qwen3-VL and action DiT configurations")
        layers = (
            self.backbone_config.text_config.num_hidden_layers
            if self.select_layer is None
            else self.select_layer
        )
        if (
            type(layers) is not int
            or not len(self.backbone_config.vision_config.deepstack_visual_indexes)
            <= layers
            <= self.backbone_config.text_config.num_hidden_layers
        ):
            raise ValueError("select_layer must retain all consumed DeepStack features")
        if (
            self.backbone_config.text_config.hidden_size
            != self.action_config.backbone_embedding_dim
        ):
            raise ValueError("Qwen3-VL hidden size must match GR00T cross-attention context")
        object.__setattr__(self, "select_layer", layers)

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "backbone_config": self.backbone_config.to_dict(),
            "action_config": self.action_config.to_dict(),
            "select_layer": self.select_layer,
        }

    @property
    def action_dim(self):
        return self.action_config.max_action_dim

    @property
    def action_horizon(self):
        return self.action_config.action_horizon


@dataclass(frozen=True)
class GrootCondition:
    features: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor
    proprio: torch.Tensor
    embodiment_id: torch.Tensor


@dataclass(frozen=True)
class PreparedGrootCondition:
    features: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    cross_kv: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None
    model_key: str


class CategoryLinear(nn.Module):
    def __init__(self, categories, inputs, outputs):
        super().__init__()

        self.W = nn.Parameter(torch.randn(categories, inputs, outputs) * 0.02)
        self.b = nn.Parameter(torch.zeros(categories, outputs))

    def forward(self, x, ids):
        return torch.bmm(x, self.W[ids]) + self.b[ids, None]


class CategoryMLP(nn.Module):
    def __init__(self, categories, inputs, hidden, outputs):
        super().__init__()
        self.layer1, self.layer2 = (
            CategoryLinear(categories, inputs, hidden),
            CategoryLinear(categories, hidden, outputs),
        )

    def forward(self, x, ids):
        return self.layer2(F.relu(self.layer1(x, ids)), ids)


def _time_features(time, width, *, cosine_first=False, frequency_shift=0):

    half = width // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / (half - frequency_shift)
    )
    angles = time.float()[..., None] * frequency
    left, right = (angles.cos(), angles.sin()) if cosine_first else (angles.sin(), angles.cos())
    return torch.cat((left, right), -1)


class ActionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.width = c.input_embedding_dim
        self.W1 = CategoryLinear(c.max_num_embodiments, c.max_action_dim, self.width)
        self.W2 = CategoryLinear(c.max_num_embodiments, 2 * self.width, self.width)
        self.W3 = CategoryLinear(c.max_num_embodiments, self.width, self.width)

    def forward(self, action, time, ids):
        values = self.W1(action, ids)
        t = _time_features(time, self.width).to(values)[:, None].expand(-1, action.shape[1], -1)
        return self.W3(F.silu(self.W2(torch.cat((values, t), -1), ids)), ids)


class TimestepEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(256, width)
        self.timestep_embedder.linear_2 = nn.Linear(width, width)

    def forward(self, time):
        linear = self.timestep_embedder
        t = _time_features(time, 256, cosine_first=True, frequency_shift=1).to(
            linear.linear_1.weight
        )
        return linear.linear_2(F.silu(linear.linear_1(t)))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 2 * width)

        self.norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)

    def forward(self, x, time):
        scale, shift = self.linear(F.silu(time)).chunk(2, -1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class GrootAttention(nn.Module):
    def __init__(self, c, cross):
        super().__init__()
        self.heads, self.dim = c.num_attention_heads, c.attention_head_dim
        width, context = (
            c.input_embedding_dim,
            c.backbone_embedding_dim if cross else c.input_embedding_dim,
        )
        self.to_q = nn.Linear(width, width, bias=c.attention_bias)
        self.to_k = nn.Linear(context, width, bias=c.attention_bias)
        self.to_v = nn.Linear(context, width, bias=c.attention_bias)
        self.to_out = nn.ModuleList((nn.Linear(width, width), nn.Dropout(c.dropout)))

    def _split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.dim).transpose(1, 2)

    def project_context(self, context):
        return self._split(self.to_k(context)), self._split(self.to_v(context))

    def forward(self, x, context, visible, kv=None):
        q = self._split(self.to_q(x))
        k, v = self.project_context(context) if kv is None else kv

        out = scaled_attention(q, k, v, visible[:, None, None, :])
        return self.to_out[1](self.to_out[0](out.transpose(1, 2).reshape_as(x)))


class _ActivationProjection(nn.Module):
    def __init__(self, width, activation):
        super().__init__()
        self.activation = activation
        self.proj = nn.Linear(width, width * (8 if activation == "geglu" else 4))

    def forward(self, x):
        values = self.proj(x)
        if self.activation == "geglu":
            value, gate = values.chunk(2, -1)
            return value * F.gelu(gate)
        return F.gelu(
            values, approximate="tanh" if self.activation == "gelu-approximate" else "none"
        )


class GrootBlock(nn.Module):
    def __init__(self, c, cross):
        super().__init__()
        self.cross, self.config = cross, c
        width = c.input_embedding_dim
        self.norm1 = (
            AdaptiveLayerNorm(width)
            if c.norm_type == "ada_norm"
            else nn.LayerNorm(width, eps=c.norm_eps, elementwise_affine=c.norm_elementwise_affine)
        )
        self.norm3 = nn.LayerNorm(
            width, eps=c.norm_eps, elementwise_affine=c.norm_elementwise_affine
        )
        self.attn1 = GrootAttention(c, cross)
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList(
            (
                _ActivationProjection(width, c.activation_fn),
                nn.Dropout(c.dropout),
                nn.Linear(4 * width, width),
            )
        )
        if c.final_dropout:
            self.ff.net.append(nn.Dropout(c.dropout))
        self.final_dropout = nn.Dropout(c.dropout) if c.final_dropout else nn.Identity()
        self.pos_embed = None
        if c.positional_embeddings:
            self.pos_embed = nn.Module()
            positions = torch.arange(c.max_seq_len)[:, None]
            frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            pe = torch.empty(1, c.max_seq_len, width)
            pe[0, :, 0::2], pe[0, :, 1::2] = (
                (positions * frequencies).sin(),
                (positions * frequencies).cos(),
            )
            self.pos_embed.register_buffer("pe", pe)

    def forward(self, x, time, context, mask, kv=None):
        value = self.norm1(x, time) if self.config.norm_type == "ada_norm" else self.norm1(x)
        if self.pos_embed is not None:
            value = value + self.pos_embed.pe[:, : value.shape[1]].to(value)
        if not self.cross:
            context, mask = (
                value,
                torch.ones(value.shape[:2], dtype=torch.bool, device=value.device),
            )
        x = x + self.final_dropout(self.attn1(value, context, mask, kv))
        value = self.norm3(x)
        for layer in self.ff.net:
            value = layer(value)
        return x + value


class GrootDiT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.timestep_encoder = TimestepEncoder(c.input_embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            GrootBlock(c, not (c.interleave_self_attention and index % 2))
            for index in range(c.num_layers)
        )
        self.norm_out = nn.LayerNorm(c.input_embedding_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(c.input_embedding_dim, 2 * c.input_embedding_dim)
        self.proj_out_2 = nn.Linear(c.input_embedding_dim, c.hidden_size)

    def forward(self, x, time, context):
        temb = self.timestep_encoder(time)
        for index, block in enumerate(self.transformer_blocks):
            mask = context.attention_mask
            if block.cross and self.config.use_alternate_vl_dit:
                is_text = index % (2 * self.config.attend_text_every_n_blocks) == 0
                mask = mask & (~context.image_mask if is_text else context.image_mask)
            kv = None if context.cross_kv is None else context.cross_kv[index]
            x = block(x, temb, context.features, mask, kv)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, -1)
        return self.proj_out_2(self.norm_out(x) * (1 + scale[:, None]) + shift[:, None])


class GrootActionHead(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        c = config
        self.model = GrootDiT(c)
        self.state_encoder = CategoryMLP(
            c.max_num_embodiments,
            c.max_state_dim * c.state_history_length,
            c.hidden_size,
            c.input_embedding_dim,
        )
        self.action_encoder = ActionEncoder(c)
        self.action_decoder = CategoryMLP(
            c.max_num_embodiments, c.hidden_size, c.hidden_size, c.max_action_dim
        )
        self.vlln = nn.LayerNorm(c.backbone_embedding_dim) if c.use_vlln else nn.Identity()
        if c.add_pos_embed:
            self.position_embedding = nn.Embedding(c.max_seq_len, c.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, std=0.02)

    def prepare_condition(self, condition, *, cache_cross_attention=False):
        if not isinstance(condition, GrootCondition):
            raise TypeError("GR00T needs an explicit GrootCondition")
        c, values = self.config, condition.features
        if (
            values.ndim != 3
            or values.shape[-1] != c.backbone_embedding_dim
            or not values.is_floating_point()
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Invalid GR00T backbone features")
        batch, length = values.shape[:2]
        if not length:
            raise ValueError("Empty GR00T context")
        for mask in (condition.attention_mask, condition.image_mask):
            if (
                mask.shape != (batch, length)
                or mask.dtype != torch.bool
                or mask.device != values.device
            ):
                raise ValueError("GR00T masks must be boolean [B,context] on the same device")
        if not condition.attention_mask.any(-1).all():
            raise ValueError("Every observation needs valid context")
        if not c.use_alternate_vl_dit and not condition.attention_mask.all():
            raise ValueError(
                "Upstream plain DiT ignores padding; only unpadded contexts have declared parity"
            )
        ids, proprio = condition.embodiment_id, condition.proprio
        if (
            ids.shape != (batch,)
            or ids.dtype != torch.long
            or ids.device != values.device
            or ((ids < 0) | (ids >= c.max_num_embodiments)).any()
        ):
            raise ValueError("Invalid GR00T embodiment IDs")
        if (
            proprio.shape != (batch, c.state_history_length, c.max_state_dim)
            or proprio.device != values.device
            or proprio.dtype != values.dtype
            or not torch.isfinite(proprio).all()
        ):
            raise ValueError(
                "State history must match the declared [B,T,D], device and floating dtype"
            )
        if cache_cross_attention and (self.training or torch.is_grad_enabled()):
            raise ValueError("Projected context caching is inference-only under eval/no_grad")
        features = self.vlln(values)
        state_features = self.state_encoder(proprio.reshape(batch, 1, -1), ids)
        if self.training and c.state_dropout_prob:
            keep = torch.rand(batch, device=values.device) >= c.state_dropout_prob
            state_features = state_features * keep[:, None, None]
        kv = (
            tuple(
                block.attn1.project_context(features) if block.cross else None
                for block in self.model.transformer_blocks
            )
            if cache_cross_attention
            else None
        )
        return PreparedGrootCondition(
            features,
            condition.attention_mask,
            condition.image_mask,
            state_features,
            ids,
            kv,
            self.model_key,
        )

    def forward(self, sample, time, condition=None):
        c = self.config
        prepared = (
            self.prepare_condition(condition)
            if isinstance(condition, GrootCondition)
            else condition
        )
        if not isinstance(prepared, PreparedGrootCondition) or prepared.model_key != self.model_key:
            raise ValueError("Prepared GR00T condition/configuration mismatch")
        if prepared.cross_kv is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("Inference cross KV must never bypass training gradients")
        batch = prepared.features.shape[0]
        if (
            sample.shape != (batch, c.action_horizon, c.max_action_dim)
            or sample.dtype != prepared.features.dtype
            or sample.device != prepared.features.device
            or not torch.isfinite(sample).all()
        ):
            raise ValueError("Noisy action must match fixed normalized [B,horizon,action_dim]")
        if (
            time.shape != (batch,)
            or time.device != sample.device
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("GR00T time must be finite [B] in [0,1], noise=0/data=1")
        discrete = (time * c.num_timestep_buckets).long()
        features = self.action_encoder(sample, discrete, prepared.embodiment_id)
        if c.add_pos_embed:
            features = (
                features
                + self.position_embedding(torch.arange(c.action_horizon, device=sample.device))[
                    None
                ]
            )
        states = torch.cat((prepared.state_features, features), 1)
        values = self.model(states, discrete, prepared)
        prediction = self.action_decoder(values, prepared.embodiment_id)[:, -c.action_horizon :]
        return FieldOutput(prediction, "velocity")

    @torch.no_grad()
    def sample_actions(
        self,
        condition,
        *,
        noise=None,
        steps=None,
        cache_cross_attention=True,
        previous_actions=None,
        overlap_steps=0,
        frozen_steps=0,
        ramp_rate=5.0,
    ):
        if self.training:
            raise ValueError("Call eval() before GR00T action sampling")
        c = self.config
        prepared = self.prepare_condition(condition, cache_cross_attention=cache_cross_attention)
        batch = len(prepared.features)
        steps = c.num_inference_timesteps if steps is None else steps
        if type(steps) is not int or steps < 1:
            raise ValueError("Euler step count must be a positive integer")
        shape = (batch, c.action_horizon, c.max_action_dim)
        noise = (
            torch.randn(shape, device=prepared.features.device, dtype=prepared.features.dtype)
            if noise is None
            else noise
        )
        if (
            noise.shape != shape
            or noise.dtype != prepared.features.dtype
            or noise.device != prepared.features.device
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("Invalid initial GR00T action noise")
        actions, strength = noise.clone(), torch.ones_like(noise)
        if (
            any(type(v) is not int for v in (overlap_steps, frozen_steps))
            or not 0 <= frozen_steps <= overlap_steps <= c.action_horizon
        ):
            raise ValueError("RTC requires 0 <= frozen <= overlap <= horizon")
        if overlap_steps:
            if (
                previous_actions is None
                or previous_actions.ndim != 3
                or previous_actions.shape[0] != batch
                or previous_actions.shape[-1] != c.max_action_dim
                or previous_actions.shape[1] < overlap_steps
                or not torch.isfinite(previous_actions).all()
            ):
                raise ValueError("RTC overlap needs a finite previous normalized action chunk")
            if previous_actions.device != actions.device or previous_actions.dtype != actions.dtype:
                raise ValueError("RTC previous action device/dtype mismatch")
            if not math.isfinite(ramp_rate) or ramp_rate <= 0:
                raise ValueError("RTC ramp rate must be finite positive")
            actions[:, :overlap_steps] = previous_actions[:, -overlap_steps:]
            strength[:, :frozen_steps] = 0
            intermediate = overlap_steps - frozen_steps
            ramp = 1 - torch.exp(
                -ramp_rate * torch.linspace(0, 1, intermediate + 2, device=actions.device)
            )
            strength[:, frozen_steps:overlap_steps] = (ramp / ramp[-1])[None, 1:-1, None].to(
                actions
            )
        elif previous_actions is not None:
            raise ValueError("Previous actions with zero overlap would be silently ignored")
        for index in range(steps):
            time = actions.new_full((batch,), index / steps)
            actions = actions + self(actions, time, prepared).prediction * strength / steps
        return actions


class GrootVLA(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        backbone_config = replace(
            config.backbone_config,
            text_config=replace(
                config.backbone_config.text_config, num_hidden_layers=config.select_layer
            ),
        )
        self.backbone = nn.Module()
        self.backbone.model = Qwen3VLForConditionalGeneration(backbone_config)
        self.action_head = GrootActionHead(config.action_config)

    def encode_observation(self, observation):
        required = {"input_ids", "attention_mask", "proprio", "embodiment_id"}
        allowed = required | {"pixel_values", "image_grid_thw"}
        if (
            not isinstance(observation, dict)
            or not required <= observation.keys()
            or observation.keys() - allowed
        ):
            raise ValueError(
                "GR00T observation requires explicit image/text/state/embodiment fields only"
            )
        ids, mask = observation["input_ids"], observation["attention_mask"]
        if mask.shape != ids.shape or mask.dtype != torch.bool:
            raise ValueError("GR00T backbone padding mask must be boolean [B,S]")
        image = ids == self.config.backbone_config.image_token_id
        model = self.backbone.model
        output = model(
            ids,
            attention_mask=mask,
            pixel_values=observation.get("pixel_values"),
            image_grid_thw=observation.get("image_grid_thw"),
            mm_token_type_ids=image.long(),
            output_hidden_states=True,
        )
        return GrootCondition(
            output.hidden_states[-1],
            mask,
            image,
            observation["proprio"],
            observation["embodiment_id"],
        )

    def forward(self, sample, time, condition=None):
        return self.action_head(sample, time, self.encode_observation(condition))

    @torch.no_grad()
    def sample_actions(self, observation, **options):
        if self.training:
            raise ValueError("Call eval() before GR00T VLA sampling")
        return self.action_head.sample_actions(self.encode_observation(observation), **options)

    @torch.no_grad()
    def predict_chunk(self, observation, state=None):
        if state is not None:
            raise ValueError(
                "GR00T condition/RTC belongs to the current observation, not a reusable token KV state"
            )
        actions = self.sample_actions(observation)
        return ActionOutput(actions, actions.new_full(actions.shape[:2], -torch.inf))
