"""Differentiable Kimi Delta Attention with channel-wise decay and fixed-size memory."""

import torch
from torch import nn
import torch.nn.functional as F


def situ_glu(gate, up, beta=4.0, linear_beta=25.0):
    """Smoothly limit both SiTU-GLU branches; hard clipping changes the function."""
    activated = beta * torch.tanh(gate.float() / beta) * torch.sigmoid(gate.float())
    linear = (
        up.float() if linear_beta is None else linear_beta * torch.tanh(up.float() / linear_beta)
    )
    return (activated * linear).to(gate.dtype)


def kda_scan(query, key, value, log_decay, beta, initial=None, valid=None):
    """Scan Q/K/V [B,S,H,D] with channel-wise decay [B,S,H,Dk] and fixed
    memory [B,H,Dk,Dv]."""
    if (
        query.ndim != 4
        or key.shape != query.shape
        or value.shape[:3] != query.shape[:3]
        or log_decay.shape != query.shape
        or beta.shape != query.shape[:3]
    ):
        raise ValueError("KDA expects aligned [B,S,H,D] queries/keys/vector gates")
    if query.shape[1] == 0:
        raise ValueError("KDA scan requires a nonempty sequence")
    dtype = value.dtype
    q = query.float() * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6)
    k = key.float() * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)
    q = q * query.shape[-1] ** -0.5
    shape = (query.shape[0], query.shape[2], query.shape[-1], value.shape[-1])
    memory = query.new_zeros(shape, dtype=torch.float32) if initial is None else initial.float()
    if memory.shape != shape:
        raise ValueError("KDA recurrent memory has incompatible head/key/value dimensions")
    if valid is not None and (
        valid.shape != query.shape[:2] or not ((valid == 0) | (valid == 1)).all()
    ):
        raise ValueError("KDA validity must be a binary current-token mask")
    outputs = []
    for t in range(query.shape[1]):
        decayed = memory * log_decay[:, t].float().exp()[..., None]
        innovation = value[:, t].float() - (decayed * k[:, t, ..., None]).sum(-2)
        updated = (
            decayed
            + (beta[:, t].float()[..., None] * k[:, t])[..., None] * innovation[..., None, :]
        )
        if valid is not None:
            updated = torch.where(valid[:, t, None, None, None].bool(), updated, memory)
        memory = updated
        output = (q[:, t, ..., None] * memory).sum(-2)
        if valid is not None:
            output = output * valid[:, t, None, None]
        outputs.append(output)
    return torch.stack(outputs, 1).to(dtype), memory


class KDADecayGate(nn.Module):
    def __init__(self, heads, head_dim, lower_bound=-5.0):
        super().__init__()
        self.heads, self.head_dim, self.lower_bound = heads, head_dim, lower_bound
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads * head_dim))

    def forward(self, values):
        inner = values.float() + self.dt_bias.float().view(self.heads, self.head_dim)
        scale = self.A_log.float().exp()[:, None]
        if self.lower_bound is None:
            return -scale * F.softplus(inner)
        return self.lower_bound * torch.sigmoid(scale * inner)


class SigmoidRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value, gate):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float() * gate.float().sigmoid()).to(value.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, d = config.linear_num_heads, config.linear_head_dim
        width = h * d
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * width, bias=False)
        self.qkv_conv1d = nn.Conv1d(
            3 * width, 3 * width, config.linear_conv_kernel_dim, groups=3 * width, bias=False
        )
        self.f_a_proj = nn.Linear(config.hidden_size, d, bias=False)
        self.f_b_proj = nn.Linear(d, width, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, h, bias=False)
        self.g_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.decay_gate = KDADecayGate(h, d, config.gate_lower_bound)
        self.o_norm = SigmoidRMSNorm(d, config.rms_norm_eps)
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(self, hidden, previous=None, padding=None, *, seen_tokens=0, use_cache=False):
        c = self.config
        b, s, _ = hidden.shape
        h, d, keep = c.linear_num_heads, c.linear_head_dim, c.linear_conv_kernel_dim - 1
        valid = None
        if padding is not None:
            if padding.shape != (b, seen_tokens + s) or not ((padding == 0) | (padding == 1)).all():
                raise ValueError("KDA padding must cover complete physical token history")
            valid = padding[:, -s:].bool()
        mixed = self.qkv_proj(hidden).transpose(1, 2)
        history = mixed.new_zeros(b, 3 * h * d, keep) if previous is None else previous[0]
        memory = None if previous is None else previous[1]
        if history.shape != (b, 3 * h * d, keep):
            raise ValueError("KDA convolution state shape mismatch")

        if valid is None or valid.all():
            extended = torch.cat((history, mixed), -1)
            convolved = F.silu(self.qkv_conv1d(extended)).transpose(1, 2)
            history = extended[..., -keep:] if keep else extended[..., :0]
        else:
            outputs = []
            for t in range(s):
                window = torch.cat((history, mixed[..., t : t + 1]), -1)
                outputs.append(F.silu(self.qkv_conv1d(window)).squeeze(-1) * valid[:, t, None])
                updated = window[..., -keep:] if keep else window[..., :0]
                history = torch.where(valid[:, t, None, None], updated, history)
            convolved = torch.stack(outputs, 1)
        q, k, v = (x.reshape(b, s, h, d) for x in convolved.chunk(3, -1))
        forget = self.f_b_proj(self.f_a_proj(hidden)).reshape(b, s, h, d)
        log_decay, beta = self.decay_gate(forget), self.b_proj(hidden).float().sigmoid()
        value, memory = kda_scan(q, k, v, log_decay, beta, memory, valid)
        value = self.o_norm(value, self.g_proj(hidden).reshape(b, s, h, d))
        return self.o_proj(value.flatten(-2)), (history, memory) if use_cache else None
