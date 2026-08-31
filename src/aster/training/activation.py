"""Explicit activation recomputation and saved-tensor storage policies."""

from __future__ import annotations

from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_activation(function, *args, **kwargs):

    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)


def activation_storage(mode, device):
    if mode == "none":
        return nullcontext()
    if mode != "cpu":
        raise ValueError("activation offload 仅 none/cpu，磁盘激活未实现")

    return torch.autograd.graph.save_on_cpu(pin_memory=torch.device(device).type == "cuda")
