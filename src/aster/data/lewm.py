"""LeWorldModel windows and action statistics using shared action semantics."""

import torch
from .actions import ActionNormalizer


def fit_lewm_actions(actions, *, spec):

    if not isinstance(actions, torch.Tensor) or actions.shape[-1] != len(spec.names):
        raise ValueError("Action dimension differs from ActionSpec")
    values = actions.reshape(-1, actions.shape[-1]).float()
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("LeWM action statistics need >=2 finite training rows")
    return ActionNormalizer(values.mean(0), values.std(0, correction=1), spec=spec, clip=False)


def lewm_windows(pixels, actions, episode_end, *, history_size):
    """Use pixels [B,T+1,C,H,W] and episode_end [B,T], including termination or truncation."""
    if (
        type(history_size) is not int
        or history_size < 1
        or not isinstance(pixels, torch.Tensor)
        or pixels.ndim != 5
    ):
        raise ValueError("LeWM windows require explicit positive history and BTCHW pixels")
    if (
        not isinstance(actions, torch.Tensor)
        or actions.ndim != 3
        or actions.shape[:2] != (len(pixels), pixels.shape[1] - 1)
    ):
        raise ValueError("LeWM windows need one actual successor observation per action")
    if (
        not isinstance(episode_end, torch.Tensor)
        or episode_end.dtype != torch.bool
        or episode_end.shape != actions.shape[:2]
    ):
        raise ValueError("LeWM windows need explicit terminal/truncation boundaries")
    if any(value.device != pixels.device for value in (actions, episode_end)) or any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in (pixels, actions)
    ):
        raise ValueError("LeWM windows require aligned finite normalized FP32 tensors")
    images, controls = [], []
    for row in range(len(pixels)):
        for start in range(actions.shape[1] - history_size + 1):
            if episode_end[row, : start + history_size - 1].any():
                continue
            images.append(pixels[row, start : start + history_size + 1])
            controls.append(actions[row, start : start + history_size])
    if not images:
        raise ValueError("No non-crossing LeWM windows in this episode batch")
    return dict(pixels=torch.stack(images), actions=torch.stack(controls))
