"""Diffusion Policy action fields and OpenPI jointly attended language/action experts."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import FieldOutput
from .actions import ActionOutput
from .serialization import LocalModelMixin, configuration_key


def _conv1d(incoming, outgoing, kernel, groups):
    return nn.Sequential(
        nn.Conv1d(incoming, outgoing, kernel, padding=kernel // 2),
        nn.GroupNorm(math.gcd(groups, outgoing), outgoing),
        nn.Mish(),
    )


class ActionResidual(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim, kernel=5, groups=8, predict_scale=True):
        super().__init__()
        self.predict_scale = predict_scale
        self.first, self.second = (
            _conv1d(incoming, outgoing, kernel, groups),
            _conv1d(outgoing, outgoing, kernel, groups),
        )
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, outgoing * (2 if predict_scale else 1))
        )
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv1d(incoming, outgoing, 1)

    def forward(self, x, condition):
        h = self.first(x)
        embedding = self.condition(condition)[..., None]
        if self.predict_scale:
            scale, shift = embedding.chunk(2, 1)
            h = scale * h + shift
        else:
            h = h + embedding
        return self.skip(x) + self.second(h)


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    action_dim: int = 7
    condition_dim: int = 32
    down_dims: tuple[int, ...] = (32, 64, 128)
    time_dim: int = 64
    kernel_size: int = 5
    groups: int = 8
    predict_scale: bool = True
    prediction_type: str = "epsilon"

    def __post_init__(self):
        object.__setattr__(self, "down_dims", tuple(self.down_dims))
        if (
            min(self.action_dim, self.condition_dim, self.time_dim, self.kernel_size, self.groups)
            < 1
            or len(self.down_dims) < 2
            or min(self.down_dims) < 1
            or self.time_dim % 2
            or self.time_dim < 4
            or self.kernel_size % 2 != 1
        ):
            raise ValueError("Invalid diffusion policy dimensions")

    def to_dict(self):
        return {"architecture": "diffusion_policy", **asdict(self)}


class DiffusionPolicy1D(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.Mish(),
            nn.Linear(config.time_dim * 4, config.time_dim),
        )
        dims = (config.action_dim, *config.down_dims)
        condition_dim = config.time_dim + config.condition_dim

        def residual(incoming, outgoing):
            return ActionResidual(
                incoming,
                outgoing,
                condition_dim,
                config.kernel_size,
                config.groups,
                config.predict_scale,
            )

        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        for i, (incoming, outgoing) in enumerate(zip(dims[:-1], dims[1:])):
            self.down.append(
                nn.ModuleList(
                    (
                        residual(incoming, outgoing),
                        residual(outgoing, outgoing),
                        nn.Conv1d(outgoing, outgoing, 3, stride=2, padding=1)
                        if i < len(config.down_dims) - 1
                        else nn.Identity(),
                    )
                )
            )
        self.middle = nn.ModuleList((residual(dims[-1], dims[-1]), residual(dims[-1], dims[-1])))
        for incoming, outgoing in reversed(list(zip(dims[1:-1], dims[2:]))):
            self.up.append(
                nn.ModuleList(
                    (
                        residual(2 * outgoing, incoming),
                        residual(incoming, incoming),
                        nn.ConvTranspose1d(incoming, incoming, 4, stride=2, padding=1),
                    )
                )
            )
        self.output = nn.Sequential(
            _conv1d(dims[1], dims[1], config.kernel_size, config.groups),
            nn.Conv1d(dims[1], config.action_dim, 1),
        )

    def forward(self, sample, time, condition=None):
        if (
            sample.ndim != 3
            or sample.shape[-1] != self.config.action_dim
            or sample.shape[1] % 2 ** (len(self.config.down_dims) - 1)
        ):
            raise ValueError("Action chunk horizon must divide through 1D U-Net downsampling")
        if condition is None or condition.shape != (len(sample), self.config.condition_dim):
            raise ValueError("Diffusion policy needs explicit global observation condition")
        time = torch.as_tensor(time, device=sample.device).expand(len(sample))
        frequencies = torch.exp(
            torch.arange(self.config.time_dim // 2, device=sample.device)
            * (-math.log(10000) / (self.config.time_dim // 2 - 1))
        )
        angles = time[:, None] * frequencies[None]
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), -1).to(sample))
        context = torch.cat((embedding, condition), -1)
        x = sample.transpose(1, 2)
        skips = []
        for first, second, down in self.down:
            x = second(first(x, context), context)
            skips.append(x)
            x = down(x)
        for block in self.middle:
            x = block(x, context)
        for first, second, up in self.up:
            x = up(second(first(torch.cat((x, skips.pop()), 1), context), context))
        return FieldOutput(self.output(x).transpose(1, 2), self.config.prediction_type)


class PiNorm(nn.Module):
    def __init__(self, width, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.modulation = nn.Linear(width, 3 * width)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x, condition=None):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        if self.adaptive:
            if condition is None:
                raise ValueError("pi0.5 adaptive norm requires time conditioning")
            scale, shift, gate = self.modulation(condition)[:, None].chunk(3, -1)
            return (normalized * (1 + scale) + shift).to(x.dtype), gate
        return (normalized * (1 + self.scale)).to(x.dtype), None


class PiBranch(nn.Module):
    def __init__(self, width, intermediate, heads, kv_heads, head_dim, adaptive=False):
        super().__init__()
        self.norm1, self.norm2 = PiNorm(width, adaptive), PiNorm(width, adaptive)
        self.q, self.k, self.v = (
            nn.Linear(width, heads * head_dim, bias=False),
            nn.Linear(width, kv_heads * head_dim, bias=False),
            nn.Linear(width, kv_heads * head_dim, bias=False),
        )
        self.output = nn.Linear(heads * head_dim, width, bias=False)
        self.gate, self.up, self.down = (
            nn.Linear(width, intermediate, bias=False),
            nn.Linear(width, intermediate, bias=False),
            nn.Linear(intermediate, width, bias=False),
        )

    def mlp(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


def _pi_rope(value, positions):
    half = value.shape[-1] // 2
    inverse = 10000 ** (-torch.arange(half, device=value.device).float() / half)
    angles = positions[:, None, :, None] * inverse[None, None, None]
    first, second = value.float().chunk(2, -1)
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        -1,
    ).to(value.dtype)


class PiJointLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.branches = nn.ModuleList(
            (
                PiBranch(
                    config.prefix_width,
                    config.prefix_mlp,
                    config.num_heads,
                    config.kv_heads,
                    config.head_dim,
                ),
                PiBranch(
                    config.action_width,
                    config.action_mlp,
                    config.num_heads,
                    config.kv_heads,
                    config.head_dim,
                    config.pi05,
                ),
            )
        )

    def forward(self, prefix, suffix, positions, mask, time_condition=None, cache=None):
        values = (prefix, suffix)
        normalized = []
        gates = []
        queries = []
        keys = []
        payloads = []
        for branch, x, condition in zip(self.branches, values, (None, time_condition)):
            if x is None:
                normalized.append(None)
                gates.append(None)
                continue
            value, gate = branch.norm1(x, condition)
            normalized.append(value)
            gates.append(gate)
            split = lambda value, heads: value.reshape(
                value.shape[0], value.shape[1], heads, self.config.head_dim
            ).transpose(1, 2)
            queries.append(split(branch.q(value), self.config.num_heads))
            keys.append(split(branch.k(value), self.config.kv_heads))
            payloads.append(split(branch.v(value), self.config.kv_heads))
        query = _pi_rope(torch.cat(queries, 2), positions)
        key = _pi_rope(torch.cat(keys, 2), positions)
        value = torch.cat(payloads, 2)
        if cache is not None:
            key, value = torch.cat((cache[0], key), 2), torch.cat((cache[1], value), 2)
        present = (key, value)
        repeats = self.config.num_heads // self.config.kv_heads
        scores = (
            query.float()
            @ key.repeat_interleave(repeats, 1).float().transpose(-1, -2)
            / math.sqrt(self.config.head_dim)
        )
        scores = scores.masked_fill(~mask[:, None], -2.3819763e38)
        probability = scores.softmax(-1).to(query.dtype)
        attended = (probability @ value.repeat_interleave(repeats, 1)).transpose(1, 2).flatten(2)
        results = []
        offset = 0
        for branch, x, gate, condition in zip(self.branches, values, gates, (None, time_condition)):
            if x is None:
                results.append(None)
                continue
            update = branch.output(attended[:, offset : offset + x.shape[1]])
            offset += x.shape[1]
            x = x + update * (gate if gate is not None else 1)
            normalized, gate = branch.norm2(x, condition)
            results.append(x + branch.mlp(normalized) * (gate if gate is not None else 1))
        return (*results, present)


@dataclass(frozen=True)
class PiConfig:
    action_dim: int = 7
    action_horizon: int = 16
    prefix_width: int = 64
    action_width: int = 32
    prefix_mlp: int = 128
    action_mlp: int = 64
    num_layers: int = 3
    num_heads: int = 4
    kv_heads: int = 1
    head_dim: int = 16
    pi05: bool = False

    def __post_init__(self):
        if (
            min(
                self.action_dim,
                self.action_horizon,
                self.prefix_width,
                self.action_width,
                self.prefix_mlp,
                self.action_mlp,
                self.num_layers,
                self.num_heads,
                self.kv_heads,
                self.head_dim,
            )
            < 1
            or self.head_dim % 2
            or self.action_width % 2
            or self.num_heads % self.kv_heads
        ):
            raise ValueError("Pi experts need common valid head geometry")

    def to_dict(self):
        return {"architecture": "pi_action_expert", **asdict(self)}


@dataclass
class PiPrefixState:
    layers: tuple
    valid: torch.Tensor
    model_key: str


class PiActionExpert(LocalModelMixin, nn.Module):
    """Let different-width prefix/action experts share layer attention while retaining independent weights."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.layers = nn.ModuleList(PiJointLayer(config) for _ in range(config.num_layers))
        self.prefix_norm, self.action_norm = (
            PiNorm(config.prefix_width),
            PiNorm(config.action_width, config.pi05),
        )
        self.action_in, self.action_out = (
            nn.Linear(config.action_dim, config.action_width),
            nn.Linear(config.action_width, config.action_dim),
        )
        if config.pi05:
            self.time_mlp = nn.Sequential(
                nn.Linear(config.action_width, config.action_width),
                nn.SiLU(),
                nn.Linear(config.action_width, config.action_width),
                nn.SiLU(),
            )
        else:
            self.state_projection = nn.Linear(config.action_dim, config.action_width)
            self.action_time = nn.Sequential(
                nn.Linear(2 * config.action_width, config.action_width),
                nn.SiLU(),
                nn.Linear(config.action_width, config.action_width),
            )

    def _suffix(self, sample, time, observation):
        if sample.shape[1:] != (
            self.config.action_horizon,
            self.config.action_dim,
        ) or time.shape != (len(sample),):
            raise ValueError("Pi expects BHA and B times")
        periods = 0.004 * (4 / 0.004) ** torch.linspace(
            0, 1, self.config.action_width // 2, device=sample.device
        )
        angles = time[:, None] * 2 * math.pi / periods[None]
        time_embedding = torch.cat((angles.sin(), angles.cos()), -1).to(sample)
        actions = self.action_in(sample)
        if self.config.pi05:
            return actions, self.time_mlp(time_embedding)
        if observation["proprio"].shape != (len(sample), self.config.action_dim):
            raise ValueError("pi0 proprio padded dimension must equal action_dim")
        actions = self.action_time(
            torch.cat((actions, time_embedding[:, None].expand(-1, len(actions[0]), -1)), -1)
        )
        return torch.cat((self.state_projection(observation["proprio"])[:, None], actions), 1), None

    def _masks(self, prefix_valid, suffix_length):
        b, p = prefix_valid.shape
        suffix_valid = torch.ones(b, suffix_length, device=prefix_valid.device, dtype=torch.bool)
        valid = torch.cat((prefix_valid, suffix_valid), 1)
        groups = torch.zeros(p + suffix_length, device=valid.device, dtype=torch.long)
        groups[p:] = 1
        if not self.config.pi05:
            groups[p + 1 :] = 2
        visible = groups[None] <= groups[:, None]
        mask = visible[None] & valid[:, :, None] & valid[:, None, :]
        positions = valid.long().cumsum(-1) - 1
        return mask, positions

    def forward(self, sample, time, condition=None, *, prefix_state=None):
        if condition is None:
            raise ValueError("Pi requires an observation prefix")
        suffix, modulation = self._suffix(sample, time, condition)
        prefix = condition.get("prefix_embeds") if prefix_state is None else None
        if prefix_state is None:
            if (
                prefix is None
                or prefix.ndim != 3
                or prefix.shape[0] != len(sample)
                or prefix.shape[-1] != self.config.prefix_width
            ):
                raise ValueError("Invalid prefix embeddings")
            valid = condition.get(
                "prefix_mask", torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.bool)
            )
        else:
            if prefix_state.model_key != self.model_key or len(prefix_state.layers) != len(
                self.layers
            ):
                raise ValueError("Pi prefix cache model mismatch")
            valid = prefix_state.valid
        mask, positions = self._masks(valid, suffix.shape[1])
        p = valid.shape[1]
        if prefix_state is not None:
            mask, positions = mask[:, p:], positions[:, p:]
        for index, layer in enumerate(self.layers):
            prefix, suffix, _ = layer(
                prefix,
                suffix,
                positions,
                mask,
                modulation,
                None if prefix_state is None else prefix_state.layers[index],
            )
        suffix, _ = self.action_norm(suffix, modulation)
        return FieldOutput(self.action_out(suffix[:, -self.config.action_horizon :]), "velocity")

    def encode_prefix(self, observation):
        prefix = observation["prefix_embeds"]
        valid = observation.get(
            "prefix_mask", torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.bool)
        )
        mask = valid[:, :, None] & valid[:, None, :]
        positions = valid.long().cumsum(-1) - 1
        caches = []
        for layer in self.layers:
            prefix, _, cache = layer(prefix, None, positions, mask)
            caches.append(cache)
        return PiPrefixState(tuple(caches), valid.clone(), self.model_key)

    @torch.no_grad()
    def sample_actions(self, observation, *, noise=None, steps=10, cache_prefix=True):
        if steps < 1:
            raise ValueError("Positive action denoising steps required")
        prefix = observation["prefix_embeds"]
        sample = (
            torch.randn(
                len(prefix),
                self.config.action_horizon,
                self.config.action_dim,
                device=prefix.device,
                dtype=prefix.dtype,
            )
            if noise is None
            else noise.clone()
        )
        state = self.encode_prefix(observation) if cache_prefix else None
        for index in range(steps):
            time = sample.new_full((len(sample),), 1 - index / steps)
            sample = sample - self(sample, time, observation, prefix_state=state).prediction / steps
        return sample

    @torch.no_grad()
    def predict_chunk(self, observation, state=None):
        if state is not None:
            raise ValueError("Use encode_prefix/forward for explicit Pi cache lifecycle")
        actions = self.sample_actions(observation)
        return ActionOutput(
            actions, torch.full(actions.shape[:2], -torch.inf, device=actions.device)
        )
