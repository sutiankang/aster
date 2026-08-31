"""Sequential Gated DeltaNet with fixed-size recurrent memory."""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import StateCapabilities
from .normalization import RMSNorm


@dataclass(frozen=True)
class HybridState:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    layer_types: tuple[str, ...]
    kind: str = "hybrid_delta"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple(tuple(x.clone() for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
            self.layer_types,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(tuple(x.index_select(0, indices) for x in layer) for layer in self.layers),
            self.seen_tokens,
            self.model_key,
            self.layer_types,
        )

    def truncate(self, length):
        raise ValueError(
            "Delta recurrent memory cannot be truncated: checkpoint+replay is required"
        )


def gated_delta_rule(query, key, value, log_decay, beta, initial=None):
    """Scan [B,S,H,D] inputs with L2-normalized Q/K and FP32 memory [B,H,Dk,Dv]."""
    dtype = query.dtype
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query, key, value = query.float(), key.float(), value.float()
    query = query * query.shape[-1] ** -0.5
    memory = (
        query.new_zeros(query.shape[0], query.shape[2], query.shape[-1], value.shape[-1])
        if initial is None
        else initial.float()
    )
    outputs = []
    for index in range(query.shape[1]):
        k, q, v = key[:, index], query[:, index], value[:, index]
        memory = memory * log_decay[:, index].float().exp()[..., None, None]
        prediction = (memory * k[..., None]).sum(-2)
        correction = (v - prediction) * beta[:, index].float()[..., None]
        memory = memory + k[..., None] * correction[..., None, :]
        outputs.append((memory * q[..., None]).sum(-2))
    return torch.stack(outputs, 1).to(dtype), memory


class GatedRMSNorm(RMSNorm):
    def __init__(self, size, eps=1e-6, *, activation="silu"):
        super().__init__(size, eps)
        if activation not in {"silu", "sigmoid"}:
            raise ValueError("Unsupported Delta output gate")
        self.activation = activation

    def forward(self, hidden, gate):
        gate = F.silu(gate.float()) if self.activation == "silu" else gate.float().sigmoid()
        return (super().forward(hidden) * gate).to(hidden.dtype)


class DeltaDecayGate(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.dt_bias = nn.Parameter(torch.ones(heads))
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(0.01, 16).log())

    def forward(self, alpha):
        return -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)


def _export_delta_decay(module, state, prefix, metadata):

    for internal, public in module._aster_parameter_key_map.items():
        if prefix + internal in state:
            if prefix + public in state:
                raise ValueError("Ambiguous Delta decay parameter names")
            state[prefix + public] = state.pop(prefix + internal)


def _import_delta_decay(module, state, prefix, metadata, strict, missing, unexpected, errors):
    for internal, public in module._aster_parameter_key_map.items():
        if prefix + public in state:
            if prefix + internal in state:
                raise ValueError("Checkpoint contains both internal and public Delta decay names")
            state[prefix + internal] = state.pop(prefix + public)


def delta_public_parameter_name(name):

    for field in ("A_log", "dt_bias"):
        if name.endswith("decay_gate." + field):
            return name[: -len("decay_gate." + field)] + field
    return name


class GatedDeltaNet(nn.Module):
    def __init__(self, c, *, projection_layout="packed_heads", output_gate="silu"):
        super().__init__()
        self.config = c
        if projection_layout not in {"packed_heads", "separate"}:
            raise ValueError("Unknown DeltaNet projection layout")
        self.projection_layout = projection_layout
        key = c.linear_num_key_heads * c.linear_key_head_dim
        value = c.linear_num_value_heads * c.linear_value_head_dim
        channels = 2 * key + value
        self.conv1d = nn.Conv1d(
            channels, channels, c.linear_conv_kernel_dim, groups=channels, bias=False
        )
        if projection_layout == "packed_heads":
            self.in_proj_qkvz = nn.Linear(c.hidden_size, 2 * key + 2 * value, bias=False)
            self.in_proj_ba = nn.Linear(c.hidden_size, 2 * c.linear_num_value_heads, bias=False)
        else:
            self.in_proj_qkv = nn.Linear(c.hidden_size, channels, bias=False)
            self.in_proj_z = nn.Linear(c.hidden_size, value, bias=False)
            self.in_proj_b = nn.Linear(c.hidden_size, c.linear_num_value_heads, bias=False)
            self.in_proj_a = nn.Linear(c.hidden_size, c.linear_num_value_heads, bias=False)
        self.decay_gate = DeltaDecayGate(c.linear_num_value_heads)
        self._aster_parameter_key_map = {
            "decay_gate.A_log": "A_log",
            "decay_gate.dt_bias": "dt_bias",
        }
        self.register_state_dict_post_hook(_export_delta_decay)
        self.register_load_state_dict_pre_hook(_import_delta_decay)

        self.norm = GatedRMSNorm(c.linear_value_head_dim, c.rms_norm_eps, activation=output_gate)
        self.out_proj = nn.Linear(value, c.hidden_size, bias=False)

    @property
    def A_log(self):
        return self.decay_gate.A_log

    @property
    def dt_bias(self):
        return self.decay_gate.dt_bias

    def forward(self, hidden, previous=None, padding=None, *, seen_tokens=0, use_cache=False):
        c = self.config
        batch, length, _ = hidden.shape
        if padding is not None:
            if (
                padding.shape != (batch, seen_tokens + length)
                or not ((padding == 0) | (padding == 1)).all()
            ):
                raise ValueError("Hybrid padding must cover the complete token history")
            hidden = hidden * padding[:, -length:, None].to(hidden.dtype)
        hk, hv, dk, dv = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        ratio = hv // hk
        if self.projection_layout == "packed_heads":
            packed = self.in_proj_qkvz(hidden).reshape(batch, length, hk, 2 * dk + 2 * ratio * dv)
            q, k, v, z = packed.split((dk, dk, ratio * dv, ratio * dv), -1)
            ba = self.in_proj_ba(hidden).reshape(batch, length, hk, 2 * ratio)
            beta, alpha = (x.reshape(batch, length, hv) for x in ba.chunk(2, -1))
            mixed = torch.cat((q.flatten(-2), k.flatten(-2), v.flatten(-2)), -1).transpose(1, 2)
        else:
            mixed = self.in_proj_qkv(hidden).transpose(1, 2)
            z = self.in_proj_z(hidden)
            beta, alpha = self.in_proj_b(hidden), self.in_proj_a(hidden)
        history_length = c.linear_conv_kernel_dim - 1
        history = (
            mixed.new_zeros(batch, mixed.shape[1], history_length)
            if previous is None
            else previous[0]
        )
        memory = None if previous is None else previous[1]
        if (
            history.shape != (batch, mixed.shape[1], history_length)
            or memory is not None
            and memory.shape != (batch, hv, dk, dv)
        ):
            raise ValueError(
                "Delta state needs causal-convolution history plus fixed-size recurrent memory"
            )
        extended = torch.cat((history, mixed), -1)
        mixed = F.silu(self.conv1d(extended)).transpose(1, 2)
        q, k, v = mixed.split((hk * dk, hk * dk, hv * dv), -1)
        q, k = (x.reshape(batch, length, hk, dk).repeat_interleave(ratio, 2) for x in (q, k))
        v = v.reshape(batch, length, hv, dv)
        g = self.decay_gate(alpha)
        output, memory = gated_delta_rule(q, k, v, g, beta.sigmoid(), memory)
        output = self.norm(output, z.reshape(batch, length, hv, dv)).flatten(-2)
        history = extended[..., -history_length:] if history_length else extended[..., :0]
        return self.out_proj(output), (history, memory) if use_cache else None
