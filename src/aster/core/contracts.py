"""Shared lifecycle contracts with explicit tensor, state, and loss semantics."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Literal
import torch


@dataclass
class TokenOutput:
    logits: torch.Tensor
    state: Any = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    auxiliary: dict[str, Any] | None = None


@dataclass
class FieldOutput:
    prediction: torch.Tensor
    prediction_type: Literal[
        "epsilon",
        "x0",
        "v",
        "score",
        "velocity",
        "average_velocity",
        "edm_residual",
        "consistency_residual",
    ]

    def __post_init__(self):
        if self.prediction_type not in {
            "epsilon",
            "x0",
            "v",
            "score",
            "velocity",
            "average_velocity",
            "edm_residual",
            "consistency_residual",
        }:
            raise ValueError("Unknown field parameterization")


@dataclass
class LossTerm:
    """A loss numerator and a detached count of valid units.

    The trainer sums each term's numerator and denominator over its declared data
    replica group before normalizing. For microbatches with 2 and 20 valid tokens,
    use (sum_a + sum_b) / 22, not the mean of two microbatch means. Terms with
    different units are normalized independently, then combined by weight."""

    numerator: torch.Tensor
    denominator: torch.Tensor
    unit: str
    name: str = "loss"
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.numerator, torch.Tensor) or self.numerator.ndim != 0:
            raise TypeError("Loss numerator must be a scalar tensor")
        if not isinstance(self.denominator, torch.Tensor) or self.denominator.ndim != 0:
            raise TypeError("Loss denominator must be a scalar tensor")
        if self.denominator.requires_grad:
            raise ValueError("A valid-count denominator cannot carry gradients")
        count = float(self.denominator.detach())
        if not math.isfinite(count) or count < 0:
            raise ValueError("A loss count must be finite and nonnegative")
        if not self.unit or not self.name or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Loss term needs a unit, name and finite nonnegative weight")

    @property
    def mean(self):
        return self.numerator / self.denominator.clamp_min(1)


@dataclass
class LossBundle:
    terms: tuple[LossTerm, ...]

    def __post_init__(self):
        self.terms = tuple(self.terms)
        names = [term.name for term in self.terms]
        if not self.terms or len(set(names)) != len(names):
            raise ValueError("A loss bundle needs distinct named terms")


@dataclass(frozen=True)
class StateCapabilities:
    """Declare supported state operations. Recurrent memory generally requires replay
    rather than arbitrary prefix truncation."""

    kind: str
    forkable: bool = False
    truncatable: bool = False
    reorderable: bool = False
    replayable: bool = False


@dataclass(frozen=True)
class KernelSpec:
    op: str
    provider: str
    version: str
    device: str
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    masks: tuple[str, ...] = ()
    backward: bool = False
    workspace_bytes: int = 0
    side_effects: tuple[str, ...] = ()
    atol: float = 1e-5
    rtol: float = 1e-5
    evidence_kind: str = "native_math_reference"

    def __post_init__(self):
        if (
            not all((self.op, self.provider, self.version, self.device))
            or not self.dtypes
            or not self.layouts
        ):
            raise ValueError("Kernel identity and supported domains must be explicit")
        if self.workspace_bytes < 0 or any(
            not math.isfinite(v) or v < 0 for v in (self.atol, self.rtol)
        ):
            raise ValueError("Invalid workspace or numerical tolerance")
        if self.evidence_kind not in {
            "native_math_reference",
            "native_storage_reference",
            "accelerated_kernel",
        }:
            raise ValueError("Unknown kernel evidence class")


@dataclass
class StageResult:
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
