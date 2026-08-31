"""Cosmos3 mixture-of-transformers with shared cross-modal attention and separate domain weights."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F

from aster.core import FieldOutput, TokenOutput, StateCapabilities
from aster.nn import RopeConfig
from aster.nn.parameter_codec import register_parameter_codec
from .qwen_vl import InterleavedMRope
from .serialization import LocalModelMixin, configuration_key


@dataclass(frozen=True)
class Cosmos3Config:
    architecture: ClassVar[str] = "cosmos3_mot"
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 12
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    qk_norm_for_text: bool = True
    use_und_k_norm_for_gen: bool = False
    attention_bias: bool = False
    rope_theta: float = 5000000.0
    rope_axes_dim: tuple[int, int, int] = (2, 2, 2)
    latent_channel: int = 2
    latent_patch_size: int = 2
    action_dim: int | None = 3
    sound_dim: int | None = 4
    num_embodiment_domains: int = 4
    timestep_scale: float = 0.001

    def __post_init__(self):
        object.__setattr__(self, "rope_axes_dim", tuple(self.rope_axes_dim))
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.latent_channel,
            self.latent_patch_size,
            self.num_embodiment_domains,
        )
        if (
            any(type(x) is not int or x < 1 for x in dimensions)
            or self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
        ):
            raise ValueError("Invalid Cosmos3 dimensions/head groups")
        if (
            len(self.rope_axes_dim) != 3
            or sum(self.rope_axes_dim) != self.head_dim // 2
            or any(type(x) is not int or x < 0 for x in self.rope_axes_dim)
        ):
            raise ValueError("Cosmos3 interleaved mRoPE sections must cover head_dim/2")
        if any(
            x is not None and (type(x) is not int or x < 1)
            for x in (self.action_dim, self.sound_dim)
        ):
            raise ValueError("Optional Cosmos3 modality widths must be positive")
        if (
            self.hidden_act not in {"silu", "relu2"}
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 1
        ):
            raise ValueError("Cosmos3 supports explicit Qwen SwiGLU or Nemotron ReLU-squared")
        if (
            not math.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
            or not math.isfinite(self.timestep_scale)
            or self.timestep_scale <= 0
        ):
            raise ValueError("Invalid Cosmos3 numerical scale")
        if any(
            type(x) is not bool
            for x in (self.qk_norm_for_text, self.use_und_k_norm_for_gen, self.attention_bias)
        ):
            raise ValueError("Cosmos3 architecture flags must be boolean")

    @property
    def rope(self):
        return RopeConfig(theta=self.rope_theta)

    @property
    def mrope_section(self):
        return self.rope_axes_dim

    @property
    def attention_head_dim(self):
        return self.head_dim

    @property
    def patch_latent_dim(self):
        return self.latent_channel * self.latent_patch_size**2

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class Cosmos3Vision:
    sample: torch.Tensor
    positions: torch.Tensor
    timesteps: torch.Tensor
    noisy_frames: torch.Tensor
    valid_frames: torch.Tensor | None = None


@dataclass(frozen=True)
class Cosmos3Sequence:
    sample: torch.Tensor
    positions: torch.Tensor
    timesteps: torch.Tensor  # [B,T]
    noisy_frames: torch.Tensor
    valid_frames: torch.Tensor | None = None
    domain_ids: torch.Tensor | None = None


@dataclass
class Cosmos3Output:
    text: TokenOutput
    vision: FieldOutput | None = None
    sound: FieldOutput | None = None
    action: FieldOutput | None = None


def cosmos3_positions(
    grid,
    *,
    batch_size=1,
    temporal_offset=15000.0,
    fps=None,
    base_fps=24.0,
    temporal_compression=4,
    base_temporal_compression=None,
    reset_spatial=True,
    start_frame_offset=0,
    device=None,
):

    if len(grid) != 3 or any(
        type(x) is not int or x < 1 for x in (*grid, batch_size, temporal_compression)
    ):
        raise ValueError("Invalid Cosmos3 position grid/compression")
    base_compression = (
        temporal_compression if base_temporal_compression is None else base_temporal_compression
    )
    if (
        type(base_compression) is not int
        or base_compression < 1
        or not math.isfinite(temporal_offset)
        or temporal_offset < 0
        or not math.isfinite(base_fps)
        or base_fps <= 0
    ):
        raise ValueError("Invalid Cosmos3 position offset/base rate")
    if (
        type(reset_spatial) is not bool
        or type(start_frame_offset) is not int
        or start_frame_offset < 0
    ):
        raise ValueError(
            "Cosmos3 frame offset must be nonnegative integer and spatial reset boolean"
        )
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("Cosmos3 FPS must be finite positive")
    t, h, w = grid
    times = torch.arange(t, device=device)
    if fps is not None and t > 1:
        times = (times.float() + start_frame_offset) / (fps / temporal_compression) * (
            base_fps / base_compression
        ) + temporal_offset
    else:
        times = times + int(temporal_offset) + start_frame_offset
    coordinates = torch.meshgrid(
        times,
        torch.arange(h, device=device, dtype=times.dtype),
        torch.arange(w, device=device, dtype=times.dtype),
        indexing="ij",
    )
    positions = torch.stack(tuple(value.flatten() for value in coordinates))
    if not reset_spatial:
        positions[1:] += int(temporal_offset)
    return positions[:, None].expand(-1, batch_size, -1).clone()


@dataclass(frozen=True)
class Cosmos3State:
    layers: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor
    seen_tokens: int
    model_key: str
    kind: str = "cosmos3_understanding"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple(tuple(x.clone() for x in layer) for layer in self.layers),
            self.attention_mask.clone(),
            self.seen_tokens,
            self.model_key,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(tuple(x.index_select(0, indices) for x in layer) for layer in self.layers),
            self.attention_mask.index_select(0, indices),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError("Cosmos3 context rollback requires explicit snapshot/replay")


class Cosmos3Norm(nn.Module):
    """Preserve the declared FP32 multiply/cast order for RMS normalization."""

    def __init__(self, width, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps, self.nemotron = c.rms_norm_eps, c.hidden_act == "relu2"

    def forward(self, hidden):
        values = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps
        )
        if self.nemotron:
            return (values * self.weight.float()).to(hidden.dtype)

        if self.weight.dtype in (torch.float16, torch.bfloat16):
            values = values.to(self.weight.dtype)
        return values * self.weight


class Cosmos3MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.activation = c.hidden_act
        if c.hidden_act == "silu":
            self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, hidden):
        up = self.up_proj(hidden)
        return self.down_proj(
            F.silu(self.gate_proj(hidden)) * up if self.activation == "silu" else up.relu().square()
        )


class Cosmos3Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        for prefix, width in (
            ("to_q", c.num_attention_heads * c.head_dim),
            ("to_k", c.num_key_value_heads * c.head_dim),
            ("to_v", c.num_key_value_heads * c.head_dim),
            ("add_q_proj", c.num_attention_heads * c.head_dim),
            ("add_k_proj", c.num_key_value_heads * c.head_dim),
            ("add_v_proj", c.num_key_value_heads * c.head_dim),
        ):
            setattr(self, prefix, nn.Linear(c.hidden_size, width, bias=c.attention_bias))
        self.to_out = nn.Linear(
            c.num_attention_heads * c.head_dim, c.hidden_size, bias=c.attention_bias
        )
        self.to_add_out = nn.Linear(
            c.num_attention_heads * c.head_dim, c.hidden_size, bias=c.attention_bias
        )
        self.norm_q = Cosmos3Norm(c.head_dim, c) if c.qk_norm_for_text else nn.Identity()
        self.norm_k = Cosmos3Norm(c.head_dim, c) if c.qk_norm_for_text else nn.Identity()
        self.norm_added_q, self.norm_added_k = (
            Cosmos3Norm(c.head_dim, c),
            Cosmos3Norm(c.head_dim, c),
        )
        self.k_norm_und_for_gen = (
            Cosmos3Norm(c.head_dim, c)
            if c.use_und_k_norm_for_gen and not c.qk_norm_for_text
            else None
        )

    def forward(self, und, gen, positions_u, positions_g, rope, mask_u, mask_g, previous, seen):
        c = self.config
        b, length, _ = und.shape

        def project(layer, x, heads):
            return layer(x).reshape(b, x.shape[1], heads, c.head_dim).transpose(1, 2)

        q = self.norm_q(project(self.to_q, und, c.num_attention_heads))
        k = self.norm_k(project(self.to_k, und, c.num_key_value_heads))
        v = project(self.to_v, und, c.num_key_value_heads)
        kg = self.k_norm_und_for_gen(k) if self.k_norm_und_for_gen is not None else k
        q, k, kg = rope(q, positions_u), rope(k, positions_u), rope(kg, positions_u)
        if previous is not None:
            k, kg, v = (torch.cat((old, new), -2) for old, new in zip(previous, (k, kg, v)))
        visible = (
            torch.arange(k.shape[-2], device=und.device)[None]
            <= (seen + torch.arange(length, device=und.device))[:, None]
        )
        visible = visible[None, None] & mask_u[:, None, None]

        def attend(query, key, value, mask):
            repeat = c.num_attention_heads // c.num_key_value_heads

            return F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(repeat, 1),
                value.repeat_interleave(repeat, 1),
                attn_mask=mask,
            )

        result_u = (
            attend(q, k, v, visible)
            .transpose(1, 2)
            .reshape(b, length, c.num_attention_heads * c.head_dim)
        )
        qg = self.norm_added_q(project(self.add_q_proj, gen, c.num_attention_heads))
        key_g = self.norm_added_k(project(self.add_k_proj, gen, c.num_key_value_heads))
        value_g = project(self.add_v_proj, gen, c.num_key_value_heads)
        qg, key_g = rope(qg, positions_g), rope(key_g, positions_g)
        all_keys, all_values = torch.cat((kg, key_g), -2), torch.cat((v, value_g), -2)
        visible_g = torch.cat((mask_u, mask_g), -1)[:, None, None]
        result_g = (
            attend(qg, all_keys, all_values, visible_g)
            .transpose(1, 2)
            .reshape(b, gen.shape[1], c.num_attention_heads * c.head_dim)
        )
        return self.to_out(result_u), self.to_add_out(result_g), (k, kg, v)


class Cosmos3Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Cosmos3Attention(c)
        self.mlp, self.mlp_moe_gen = Cosmos3MLP(c), Cosmos3MLP(c)
        for name in (
            "input_layernorm",
            "input_layernorm_moe_gen",
            "post_attention_layernorm",
            "post_attention_layernorm_moe_gen",
        ):
            setattr(self, name, Cosmos3Norm(c.hidden_size, c))

    def forward(self, und, gen, *args):
        update_u, update_g, state = self.self_attn(
            self.input_layernorm(und), self.input_layernorm_moe_gen(gen), *args
        )
        und, gen = und + update_u, gen + update_g
        return (
            und + self.mlp(self.post_attention_layernorm(und)),
            gen + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(gen)),
            state,
        )


class Cosmos3DomainLinear(nn.Module):
    """Interpret domain-embedding rows as [input, output], not transposed Linear weights."""

    def __init__(self, incoming, outgoing, domains):
        super().__init__()
        self.incoming, self.outgoing = incoming, outgoing
        self.fc, self.bias = (
            nn.Embedding(domains, incoming * outgoing),
            nn.Embedding(domains, outgoing),
        )

    def forward(self, sample, domain_ids):
        weight = self.fc(domain_ids).reshape(len(sample), self.incoming, self.outgoing)
        return torch.bmm(sample, weight) + self.bias(domain_ids)[:, None]


class Cosmos3ModalityBias(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, hidden):
        return hidden + self.weight


class Cosmos3TimeLinear(nn.Linear):
    def forward(self, features):

        with torch.autocast(features.device.type, enabled=False):
            return F.linear(features.to(self.weight.dtype), self.weight, self.bias)


class Cosmos3Time(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1, self.linear_2 = (
            Cosmos3TimeLinear(256, width),
            Cosmos3TimeLinear(width, width),
        )

    def forward(self, times):
        with torch.autocast(times.device.type, enabled=False):
            frequency = torch.exp(
                -math.log(10000) * torch.arange(128, device=times.device).float() / 128
            )
            angle = times.float()[..., None] * frequency
            features = torch.cat((angle.cos(), angle.sin()), -1)
        return self.linear_2(F.silu(self.linear_1(features)))


class Cosmos3MoT(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.model_key = config, configuration_key(config)
        c = config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Cosmos3Layer(c) for _ in range(c.num_hidden_layers))
        self.norm, self.norm_moe_gen = Cosmos3Norm(c.hidden_size, c), Cosmos3Norm(c.hidden_size, c)
        self.rotary_emb = InterleavedMRope(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.proj_in, self.proj_out = (
            nn.Linear(c.patch_latent_dim, c.hidden_size),
            nn.Linear(c.hidden_size, c.patch_latent_dim),
        )
        self.time_embedder = Cosmos3Time(c.hidden_size)
        parameter_map = {}
        if c.sound_dim is not None:
            self.audio_proj_in, self.audio_proj_out = (
                nn.Linear(c.sound_dim, c.hidden_size),
                nn.Linear(c.hidden_size, c.sound_dim),
            )
            self.audio_bias = Cosmos3ModalityBias(c.hidden_size)
            parameter_map["audio_bias.weight"] = "audio_modality_embed"
        if c.action_dim is not None:
            self.action_proj_in = Cosmos3DomainLinear(
                c.action_dim, c.hidden_size, c.num_embodiment_domains
            )
            self.action_proj_out = Cosmos3DomainLinear(
                c.hidden_size, c.action_dim, c.num_embodiment_domains
            )
            self.action_bias = Cosmos3ModalityBias(c.hidden_size)
            parameter_map["action_bias.weight"] = "action_modality_embed"
        if parameter_map:
            register_parameter_codec(self, parameter_map)

    @staticmethod
    def _positions(value, batch, length, device):
        if value is None:
            raise ValueError("Cosmos3 modality positions must be explicitly supplied")
        if value.shape == (batch, length):
            value = value[None].expand(3, -1, -1)
        if (
            value.shape != (3, batch, length)
            or value.device != device
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError("Cosmos3 positions must be finite nonnegative [3,B,S] coordinates")
        return value

    @staticmethod
    def _field_mask(field, b, length, device):
        noisy, valid = field.noisy_frames, field.valid_frames
        if valid is None:
            valid = torch.ones((b, length), dtype=torch.bool, device=device)
        if (
            any(
                x.shape != (b, length) or x.dtype != torch.bool or x.device != device
                for x in (noisy, valid)
            )
            or (noisy & ~valid).any()
        ):
            raise ValueError("Cosmos3 noisy/valid frame masks must align and noisy must be valid")
        times = field.timesteps
        if (
            times.shape != (b, length)
            or times.device != device
            or not torch.isfinite(times).all()
            or ((times < 0) | (times > 1000)).any()
        ):
            raise ValueError("Cosmos3 timesteps use the explicit scheduler range [0,1000]")
        return noisy, valid

    def _vision(self, field, b, device, dtype):
        c = self.config
        p = c.latent_patch_size
        if field is None:
            x = torch.empty(b, 0, c.patch_latent_dim, device=device, dtype=dtype)
            hidden = self.proj_in(x) + self.time_embedder(x.new_empty(b, 0)).to(dtype)
            return (
                hidden,
                x.new_empty(3, b, 0),
                torch.empty(b, 0, dtype=torch.bool, device=device),
                None,
            )
        x = field.sample
        if (
            x.ndim != 5
            or x.shape[:2] != (b, c.latent_channel)
            or min(x.shape[2:]) < 1
            or x.device != device
            or not x.is_floating_point()
        ):
            raise ValueError(
                "Cosmos3 vision must be floating BCTHW latent with configured channels"
            )
        _, _, t, h, w = x.shape
        hp, wp = h + (-h) % p, w + (-w) % p
        noisy, valid = self._field_mask(field, b, t, device)
        patches = F.pad(x, (0, wp - w, 0, hp - h)).reshape(
            b, c.latent_channel, t, hp // p, p, wp // p, p
        )
        patches = patches.permute(0, 2, 3, 5, 4, 6, 1).reshape(
            b, t * (hp // p) * (wp // p), c.patch_latent_dim
        )
        positions = self._positions(field.positions, b, patches.shape[1], device)
        hidden = self.proj_in(patches.to(dtype))
        time = self.time_embedder(field.timesteps * c.timestep_scale).to(hidden)
        area = hp // p * (wp // p)
        hidden = hidden + (time * noisy[..., None]).repeat_interleave(area, 1)
        return hidden, positions, valid.repeat_interleave(area, 1), (t, h, w, hp, wp, noisy)

    def _sequence(self, field, name, b, device, dtype):
        c = self.config
        width = c.sound_dim if name == "sound" else c.action_dim
        if width is None:
            if field is not None:
                raise ValueError(f"Cosmos3 {name} head is disabled by configuration")
            return None
        prefix = "audio" if name == "sound" else "action"
        incoming = getattr(self, prefix + "_proj_in")
        if field is None:
            sample = torch.empty(b, 0, width, device=device, dtype=dtype)
            domains = torch.zeros(b, device=device, dtype=torch.long)
            hidden = incoming(sample) if name == "sound" else incoming(sample, domains)
            hidden = getattr(self, prefix + "_bias")(hidden) + self.time_embedder(
                sample.new_empty(b, 0)
            ).to(dtype)
            return (
                hidden,
                sample.new_empty(3, b, 0),
                torch.empty(b, 0, dtype=torch.bool, device=device),
                None,
                domains,
            )
        sample = field.sample
        if (
            sample.ndim != 3
            or sample.shape[0] != b
            or sample.shape[-1] != width
            or sample.shape[1] < 1
            or sample.device != device
            or not sample.is_floating_point()
        ):
            raise ValueError(f"Cosmos3 {name} sample must be configured floating [B,T,D]")
        noisy, valid = self._field_mask(field, b, sample.shape[1], device)
        positions = self._positions(field.positions, b, sample.shape[1], device)
        domains = field.domain_ids
        if name == "action":
            if (
                domains is None
                or domains.shape != (b,)
                or domains.dtype != torch.long
                or domains.device != device
                or ((domains < 0) | (domains >= c.num_embodiment_domains)).any()
            ):
                raise ValueError("Cosmos3 action requires valid long domain_ids[B]")
            hidden = incoming(sample.to(dtype), domains)
        else:
            if domains is not None:
                raise ValueError("Sound latent cannot select action embodiment domains")
            hidden = incoming(sample.to(dtype))
        hidden = getattr(self, prefix + "_bias")(hidden)
        hidden = (
            hidden
            + self.time_embedder(field.timesteps * c.timestep_scale).to(hidden) * noisy[..., None]
        )
        return hidden, positions, valid, noisy, domains

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        understanding_positions=None,
        understanding_additions=None,
        attention_mask=None,
        vision=None,
        sound=None,
        action=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Supply IDs or understanding embeddings, not both")
        if state is not None and (
            not isinstance(state, Cosmos3State)
            or state.model_key != self.model_key
            or state.kind != "cosmos3_understanding"
        ):
            raise ValueError("Cosmos3 state/config mismatch")
        if state is not None:
            if (
                type(state.seen_tokens) is not int
                or state.seen_tokens < 1
                or state.attention_mask.ndim != 2
                or state.attention_mask.shape[1] != state.seen_tokens
                or state.attention_mask.dtype != torch.bool
            ):
                raise ValueError("Cosmos3 cached state has invalid seen_tokens/mask")
            if len(state.layers) != self.config.num_hidden_layers:
                raise ValueError("Cosmos3 cached state layer count mismatch")
            expected = (
                state.attention_mask.shape[0],
                self.config.num_key_value_heads,
                state.seen_tokens,
                self.config.head_dim,
            )
            for layer in state.layers:
                if len(layer) != 3 or any(
                    not isinstance(value, torch.Tensor)
                    or value.shape != expected
                    or value.device != state.attention_mask.device
                    or not value.is_floating_point()
                    for value in layer
                ):
                    raise ValueError(
                        "Cosmos3 cached state requires three aligned understanding K/Kgen/V tensors per layer"
                    )
        if input_ids is None and inputs_embeds is None:
            if state is None:
                raise ValueError("Cosmos3 requires an understanding prefix or its cached state")
            input_ids = torch.empty(
                state.attention_mask.shape[0],
                0,
                dtype=torch.long,
                device=state.attention_mask.device,
            )
        und = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if und.ndim != 3 or und.shape[-1] != self.config.hidden_size or und.shape[0] < 1:
            raise ValueError("Cosmos3 understanding must be nonempty-batch BSH")
        if understanding_additions is not None:
            if (
                state is not None
                or not isinstance(understanding_additions, (tuple, list))
                or len(understanding_additions) > self.config.num_hidden_layers
            ):
                raise ValueError(
                    "Cosmos3 DeepStack additions belong to an uncached understanding prefill"
                )
            if any(
                value.shape != und.shape or value.device != und.device or value.dtype != und.dtype
                for value in understanding_additions
            ):
                raise ValueError(
                    "Cosmos3 DeepStack additions must align exactly with understanding embeddings"
                )
        b, length, _ = und.shape
        seen = 0 if state is None else state.seen_tokens
        if understanding_positions is None:
            understanding_positions = torch.arange(seen, seen + length, device=und.device)[
                None
            ].expand(b, -1)
        positions_u = self._positions(understanding_positions, b, length, und.device)
        if attention_mask is None:
            current = torch.ones(b, length, dtype=torch.bool, device=und.device)
            attention_mask = (
                current if state is None else torch.cat((state.attention_mask, current), 1)
            )
        if (
            attention_mask.shape != (b, seen + length)
            or attention_mask.dtype != torch.bool
            or attention_mask.device != und.device
        ):
            raise ValueError("Cosmos3 attention_mask is bool[B,all understanding tokens]")
        if state is not None and not torch.equal(attention_mask[:, :seen], state.attention_mask):
            raise ValueError("Cached understanding padding cannot be changed")
        if not attention_mask.any(-1).all():
            raise ValueError("Each Cosmos3 example needs a valid understanding prefix")
        entries = [("vision", self._vision(vision, b, und.device, und.dtype))]
        entries += [
            (name, entry)
            for name in ("sound", "action")
            if (
                entry := self._sequence(
                    sound if name == "sound" else action, name, b, und.device, und.dtype
                )
            )
            is not None
        ]
        gen = torch.cat(tuple(entry[0] for _, entry in entries), 1)
        positions_g = torch.cat(tuple(entry[1] for _, entry in entries), -1)
        valid_g = torch.cat(tuple(entry[2] for _, entry in entries), 1)
        states, hidden = [], [und]
        for index, layer in enumerate(self.layers):
            previous = None if state is None else state.layers[index]
            und, gen, present = layer(
                und,
                gen,
                positions_u,
                positions_g,
                self.rotary_emb,
                attention_mask,
                valid_g,
                previous,
                seen,
            )

            if understanding_additions is not None and index < len(understanding_additions):
                und = und + understanding_additions[index]
            states.append(present)
            if output_hidden_states:
                hidden.append(und)
        und, gen = self.norm(und), self.norm_moe_gen(gen)
        if output_hidden_states:
            hidden[-1] = und
        next_state = (
            Cosmos3State(tuple(states), attention_mask.clone(), seen + length, self.model_key)
            if use_cache
            else None
        )
        result = Cosmos3Output(
            TokenOutput(
                self.lm_head(und), next_state, tuple(hidden) if output_hidden_states else None
            )
        )
        start = 0
        for name, entry in entries:
            value = gen[:, start : start + entry[0].shape[1]]
            start += entry[0].shape[1]
            if name == "vision":
                predicted = self.proj_out(value)
                if entry[3] is None:
                    continue
                t, h, w, hp, wp, noisy = entry[3]
                p = self.config.latent_patch_size
                predicted = predicted.reshape(
                    b, t, hp // p, wp // p, p, p, self.config.latent_channel
                ).permute(0, 6, 1, 2, 4, 3, 5)
                predicted = (
                    predicted.reshape(b, self.config.latent_channel, t, hp, wp)[:, :, :, :h, :w]
                    * noisy[:, None, :, None, None]
                )
            else:
                head = self.audio_proj_out if name == "sound" else self.action_proj_out
                predicted = head(value) if name == "sound" else head(value, entry[4])
                if entry[3] is None:
                    continue
                predicted = predicted * entry[3][..., None]
            setattr(result, name, FieldOutput(predicted, "velocity"))
        return result

    def forward_text(self, *args, **kwargs):

        if any(kwargs.get(name) is not None for name in ("vision", "sound", "action")):
            raise ValueError(
                "forward_text is the understanding role, not a silent multimodal generator"
            )
        return self.forward(*args, **kwargs).text
