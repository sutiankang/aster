"""GR00T N1.7 action flow matching through the shared trainer."""

import math
import torch
from torch import nn
from aster.core import LossTerm
from .generation import FlowPath


class GrootFlowObjective(nn.Module):
    def __init__(self, *, noise_beta_alpha=1.5, noise_beta_beta=1.0, noise_s=0.999):
        super().__init__()
        if (
            any(
                not math.isfinite(v) or v <= 0 for v in (noise_beta_alpha, noise_beta_beta, noise_s)
            )
            or noise_s > 1
        ):
            raise ValueError("GR00T needs positive Beta concentrations and noise_s in (0,1]")
        self.alpha, self.beta, self.noise_s = noise_beta_alpha, noise_beta_beta, noise_s
        self.path = FlowPath(direction="noise_to_data")

    def config_dict(self):
        return {
            "type": "groot_n17_flow",
            "alpha": self.alpha,
            "beta": self.beta,
            "noise_s": self.noise_s,
            "direction": self.path.direction,
            "normalization": "global_valid_action_element",
        }

    def sample_time(self, batch, device, dtype):

        distribution = torch.distributions.Beta(
            torch.tensor(self.alpha, device="cpu", dtype=torch.float32),
            torch.tensor(self.beta, device="cpu", dtype=torch.float32),
        )
        return ((1 - distribution.sample((batch,))) * self.noise_s).to(device=device, dtype=dtype)

    def forward(self, model, batch):
        actions = batch["actions"]
        if (
            actions.ndim != 3
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("GR00T targets must be finite normalized [B,horizon,action_dim]")
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(actions)
        if (
            noise.shape != actions.shape
            or noise.dtype != actions.dtype
            or noise.device != actions.device
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("GR00T noise/target shape, device or dtype mismatch")
        time = batch.get("time")
        if time is None:
            time = self.sample_time(len(actions), actions.device, actions.dtype)
        if (
            time.shape != (len(actions),)
            or time.device != actions.device
            or not torch.isfinite(time).all()
        ):
            raise ValueError("GR00T time must be finite [B] on the target device")
        mask = batch.get("action_mask", torch.ones_like(actions, dtype=torch.bool))
        if (
            mask.shape != actions.shape
            or mask.device != actions.device
            or not ((mask == 0) | (mask == 1)).all()
        ):
            raise ValueError(
                "Action padding mask must match every action scalar and contain only 0/1"
            )
        sample, target = self.path.sample(actions, noise, time)
        output = model(sample, time, batch["condition"])
        if output.prediction_type != "velocity" or output.prediction.shape != target.shape:
            raise ValueError("GR00T requires a matching noise-to-data velocity field")
        errors = (output.prediction.float() - target.float()).square().masked_select(mask.bool())

        return LossTerm(
            errors.sum(), errors.new_tensor(errors.numel()), "action_element", "groot_flow"
        )
