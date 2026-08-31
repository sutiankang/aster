"""Gaussian conditional paths connecting native optimal transport, training, and ODE sampling."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..core import FieldOutput, LossTerm
from .flow_transport import transport_pairing
from .generation import expand, mean_flat


@dataclass(frozen=True)
class GaussianPathSample:
    sample: torch.Tensor
    velocity: torch.Tensor
    mean: torch.Tensor
    standard_deviation: torch.Tensor


def _event(value, reference=None):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or min(value.shape) < 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError("Flow event must be a nonempty finite floating tensor with a batch axis")
    if reference is not None and (
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise ValueError("Flow endpoints/perturbation must have identical shape, dtype and device")


@dataclass(frozen=True)
class GaussianFlowPath:
    """Explicit conditional, target-OT, and Schrodinger-bridge Gaussian paths."""

    kind: str = "conditional"
    sigma: float = 0.0
    bridge_epsilon: float = 1e-8

    def __post_init__(self):
        if self.kind not in {"conditional", "target", "schrodinger"}:
            raise ValueError("Unknown Gaussian flow path")
        if (
            not math.isfinite(self.sigma)
            or self.sigma < 0
            or not math.isfinite(self.bridge_epsilon)
            or self.bridge_epsilon < 0
            or self.kind == "schrodinger"
            and self.sigma == 0
        ):
            raise ValueError("Invalid Gaussian path sigma/bridge epsilon")

    def sample(self, source, target, time, perturbation):
        _event(target)
        _event(source, target)
        _event(perturbation, target)
        if (
            not isinstance(time, torch.Tensor)
            or time.shape != (len(target),)
            or time.device != target.device
            or not time.is_floating_point()
            or not torch.isfinite(time).all()
            or ((time < 0) | (time > 1)).any()
        ):
            raise ValueError("Gaussian path time must be finite with shape [B] in [0,1]")
        if self.kind == "schrodinger" and ((time == 0) | (time == 1)).any():
            raise ValueError(
                "Schrodinger bridge conditional velocity requires the open interval (0,1)"
            )

        dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
        x0, x1, epsilon = (value.to(dtype) for value in (source, target, perturbation))
        t = expand(time.to(dtype), x1)
        if self.kind == "target":
            mean = t * x1
            std = 1 - (1 - self.sigma) * t
            sample = mean + std * epsilon

            velocity = x1 - (1 - self.sigma) * epsilon
        else:
            mean = (1 - t) * x0 + t * x1
            std = torch.full_like(t, self.sigma)
            if self.kind == "schrodinger":
                std = self.sigma * (t * (1 - t)).sqrt()
            sample = mean + std * epsilon
            velocity = x1 - x0
            if self.kind == "schrodinger":
                ratio = (1 - 2 * t) / (2 * t * (1 - t) + self.bridge_epsilon)
                velocity = velocity + ratio * (sample - mean)
        return GaussianPathSample(sample, velocity, mean, std)


def _condition_at(condition, indices, count):
    """Reorder labels, embeddings, and masks whenever OT changes target indices."""
    if condition is None:
        return None
    if isinstance(condition, torch.Tensor):
        if condition.ndim == 0 or len(condition) != count or condition.device != indices.device:
            raise ValueError("Transport conditions must have an aligned batch axis and device")
        return condition[indices]
    if isinstance(condition, dict):
        return {key: _condition_at(value, indices, count) for key, value in condition.items()}
    if isinstance(condition, (list, tuple)):
        return type(condition)(_condition_at(value, indices, count) for value in condition)
    raise ValueError("Transport conditions must be tensors or nested tensor containers")


class GaussianFlowObjective(nn.Module):
    """Resumable conditional/target-OT/SB flow matching with explicit path semantics."""

    def __init__(self, path=None, *, coupling=None, regularization=None, time_epsilon=1e-5):
        super().__init__()
        self.path = (
            GaussianFlowPath()
            if path is None
            else (GaussianFlowPath(**path) if isinstance(path, dict) else path)
        )
        if not isinstance(self.path, GaussianFlowPath):
            raise TypeError("path must be GaussianFlowPath or its explicit configuration")

        self.coupling = coupling or ("exact" if self.path.kind == "schrodinger" else "independent")
        self.regularization = (
            (2 * self.path.sigma**2 if self.path.kind == "schrodinger" else 0.05)
            if regularization is None
            else regularization
        )
        self.time_epsilon = time_epsilon
        if self.coupling not in {"independent", "exact", "sinkhorn"}:
            raise ValueError("Unknown flow coupling")
        if self.path.kind == "target" and self.coupling != "independent":
            raise ValueError("Target-conditional path has no source endpoint to OT-pair")
        if (
            not math.isfinite(self.regularization)
            or self.regularization <= 0
            or not math.isfinite(time_epsilon)
            or not 0 < time_epsilon < 0.5
        ):
            raise ValueError("Invalid OT regularization/time epsilon")

    def config_dict(self):
        return dict(
            type="gaussian_flow",
            path=asdict(self.path),
            coupling=self.coupling,
            regularization=self.regularization,
            time_epsilon=self.time_epsilon,
            coupling_scope="rank_local_microbatch",
        )

    def forward(self, model, batch):
        unknown = set(batch) - {"sample", "noise", "perturbation", "time", "condition"}
        if unknown:
            raise ValueError(f"Unsupported Gaussian flow fields: {sorted(unknown)}")
        data = batch["sample"]
        _event(data)
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(data)
        _event(noise, data)
        condition = batch.get("condition")
        if self.coupling != "independent":
            pair = transport_pairing(
                noise, data, method=self.coupling, regularization=self.regularization
            )
            condition = _condition_at(condition, pair["target_indices"], len(data))
            noise, data = pair["source"], pair["target"]
        time = batch.get("time")
        if time is None:
            time = torch.rand(len(data), device=data.device, dtype=torch.float32)
            if self.path.kind == "schrodinger":
                time = self.time_epsilon + (1 - 2 * self.time_epsilon) * time
        if self.path.kind == "target":
            if "perturbation" in batch:
                raise ValueError(
                    "Target path uses noise as epsilon; extra perturbation is ambiguous"
                )
            perturbation = noise
        else:
            perturbation = batch.get("perturbation")
            if perturbation is None:
                perturbation = torch.randn_like(data) if self.path.sigma else torch.zeros_like(data)
        path = self.path.sample(noise, data, time, perturbation)
        output = model(path.sample.to(data.dtype), time, condition)
        if (
            not isinstance(output, FieldOutput)
            or output.prediction_type != "velocity"
            or output.prediction.shape != data.shape
        ):
            raise ValueError("Gaussian flow matching requires an aligned velocity FieldOutput")

        dtype = path.velocity.dtype
        loss = mean_flat((output.prediction.to(dtype) - path.velocity).square())
        return LossTerm(
            loss.sum(),
            torch.tensor(len(data), device=data.device, dtype=torch.int64),
            "sample",
            "gaussian_flow",
        )
