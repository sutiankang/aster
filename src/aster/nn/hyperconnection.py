"""Constrained learned mixing across multiple residual streams."""

import torch
from torch import nn
import torch.nn.functional as F


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class HyperConnection(nn.Module):
    def __init__(self, hidden_size, streams=4, iterations=20, eps=1e-6, norm_eps=1e-6):
        super().__init__()
        if min(hidden_size, streams, iterations) < 1 or min(eps, norm_eps) <= 0:
            raise ValueError("Invalid hyperconnection dimensions/normalization")
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = streams, iterations, eps
        self.input_norm = UnweightedRMSNorm(norm_eps)
        count = (2 + streams) * streams
        self.fn = nn.Parameter(torch.empty(count, streams * hidden_size))
        self.base = nn.Parameter(torch.zeros(count))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, std=0.02)

    def forward(self, streams):
        n = self.hc_mult
        if streams.ndim != 4 or streams.shape[2] != n or streams.shape[3] * n != self.fn.shape[1]:
            raise ValueError("mHC expects [B,S,residual_streams,hidden]")
        flat = self.input_norm(streams.flatten(2).float())
        pre, post, residual = F.linear(flat, self.fn.float()).split((n, n, n * n), -1)
        pre_base, post_base, residual_base = self.base.split((n, n, n * n))
        pre = torch.sigmoid(pre * self.scale[0] + pre_base) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.scale[1] + post_base)
        residual = residual.reshape(*residual.shape[:-1], n, n) * self.scale[
            2
        ] + residual_base.reshape(n, n)
        residual = residual.softmax(-1) + self.hc_eps
        residual = residual / (residual.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            residual = residual / (residual.sum(-1, keepdim=True) + self.hc_eps)
            residual = residual / (residual.sum(-2, keepdim=True) + self.hc_eps)
        collapsed = (pre[..., None] * streams).sum(2).to(streams.dtype)
        return post, residual, collapsed

    @staticmethod
    def expand(update, streams, post, residual):

        return (
            post.to(streams.dtype)[..., None] * update.unsqueeze(-2)
            + residual.to(streams.dtype).transpose(-1, -2) @ streams
        )


class HyperHead(nn.Module):
    def __init__(self, hidden_size, streams=4, eps=1e-6, norm_eps=1e-6):
        super().__init__()
        self.hc_mult, self.eps = streams, eps
        self.input_norm = UnweightedRMSNorm(norm_eps)
        self.hc_fn = nn.Parameter(torch.empty(streams, streams * hidden_size))
        self.hc_base = nn.Parameter(torch.zeros(streams))
        self.hc_scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.hc_fn, std=0.02)

    def forward(self, streams):
        flat = self.input_norm(streams.flatten(2).float())
        pre = (
            torch.sigmoid(
                F.linear(flat, self.hc_fn.float()) * self.hc_scale.float() + self.hc_base.float()
            )
            + self.eps
        )
        return (pre[..., None] * streams).sum(2).to(streams.dtype)
