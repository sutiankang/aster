"""Calibrated approximate DiT residual reuse within one fixed sampling session."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import torch

from ..core import FieldOutput
from ..models.generative import DiT, timestep_embedding


@dataclass(frozen=True)
class ResidualCacheCalibration:
    policy_artifact_id: str
    dataset_fingerprint: str
    probe_id: str
    coefficients: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if (
            not all(
                isinstance(x, str) and x
                for x in (self.policy_artifact_id, self.dataset_fingerprint, self.probe_id)
            )
            or not 1 <= len(self.coefficients) <= 7
            or not all(math.isfinite(x) for x in self.coefficients)
        ):
            raise ValueError(
                "Residual cache requires finite, artifact-bound calibration coefficients"
            )

    def estimate(self, distance):
        value = 0.0
        for coefficient in self.coefficients:
            value = value * distance + coefficient
        if not math.isfinite(value):
            raise ValueError("Non-finite residual distance estimate")

        return max(0.0, value)

    @property
    def fingerprint(self):
        value = {
            "policy": self.policy_artifact_id,
            "dataset": self.dataset_fingerprint,
            "probe": self.probe_id,
            "coefficients": self.coefficients,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


def fit_residual_calibration(
    probe_distances,
    residual_distances,
    *,
    policy_artifact_id,
    dataset_fingerprint,
    probe_id="native_dit_first_adaln_v1",
    degree=2,
    ridge=1e-6,
):
    """Fit actual paired distances by least squares; calibration inputs are explicit."""
    x, y = (
        torch.as_tensor(probe_distances, dtype=torch.float64),
        torch.as_tensor(residual_distances, dtype=torch.float64),
    )
    if (
        x.ndim != 1
        or y.shape != x.shape
        or type(degree) is not int
        or not 0 <= degree <= 6
        or len(x) < degree + 1
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
        or (x < 0).any()
        or (y < 0).any()
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError(
            "Calibration needs enough finite nonnegative paired distances and positive regularization"
        )
    design = torch.vander(x, N=degree + 1)
    coefficients = torch.linalg.solve(
        design.T @ design + ridge * torch.eye(degree + 1, dtype=x.dtype), design.T @ y
    )
    return ResidualCacheCalibration(
        policy_artifact_id, dataset_fingerprint, probe_id, tuple(coefficients.tolist())
    )


class DiTStepCacheSession:
    """Approximate residual reuse within one fixed condition/schedule session."""

    probe_id = "native_dit_first_adaln_v1"

    def __init__(
        self,
        model,
        *,
        policy_artifact_id,
        condition,
        schedule,
        calibration,
        threshold=0.1,
        max_skip=2,
        audit_every=2,
        max_relative_error=0.05,
    ):
        if type(model) is not DiT:
            raise ValueError(
                "Residual cache supports the explicit native DiT architecture, not arbitrary Flux/Wan wrappers"
            )
        if (
            not isinstance(calibration, ResidualCacheCalibration)
            or calibration.policy_artifact_id != policy_artifact_id
            or calibration.probe_id != self.probe_id
        ):
            raise ValueError("Calibration policy/probe differs from this native DiT")
        schedule = tuple(float(x) for x in schedule)
        if (
            len(schedule) < 2
            or not all(math.isfinite(x) for x in schedule)
            or not all(math.isfinite(x) and x >= 0 for x in (threshold, max_relative_error))
            or type(max_skip) is not int
            or max_skip < 0
            or type(audit_every) is not int
            or audit_every < 1
        ):
            raise ValueError(
                "Cache needs finite schedule, nonnegative thresholds, bounded skips and a positive audit interval"
            )
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.policy_artifact_id, self.schedule, self.calibration = (
            policy_artifact_id,
            schedule,
            calibration,
        )
        self.threshold, self.max_skip, self.audit_every, self.max_relative_error = (
            threshold,
            max_skip,
            audit_every,
            max_relative_error,
        )
        self._condition = condition.detach().clone() if condition is not None else None
        condition_digest = hashlib.sha256()
        if self._condition is not None:
            condition_digest.update(
                str((tuple(self._condition.shape), self._condition.dtype)).encode()
            )
            condition_digest.update(
                self._condition.contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
        self.condition_fingerprint = condition_digest.hexdigest()
        self._versions = tuple(
            (name, id(t), t._version)
            for name, t in (*self.model.named_parameters(), *self.model.named_buffers())
        )
        self._next_step, self._signature, self._previous_probe, self._residual = 0, None, None, None
        self._accumulated, self._consecutive, self._candidates = 0.0, 0, 0
        self._lock, self._closed = threading.Lock(), False
        self.full_backbone_calls = self.reused_backbone_calls = 0
        self.guard_failed, self.disabled, self.trace = False, False, []

    @staticmethod
    def _distance(a, b):

        axes = tuple(range(1, a.ndim))
        return float(
            (
                (a.float() - b.float()).abs().mean(axes)
                / b.float().abs().mean(axes).clamp_min(1e-8)
            ).max()
        )

    def _check(self, sample, step):
        if (
            self._closed
            or type(step) is not int
            or step != self._next_step
            or step >= len(self.schedule)
        ):
            raise RuntimeError("Cache session is closed or step order differs from fixed schedule")
        versions = tuple(
            (name, id(t), t._version)
            for name, t in (*self.model.named_parameters(), *self.model.named_buffers())
        )
        if versions != self._versions:
            raise RuntimeError("Policy changed after cache session creation")
        if (
            sample.ndim != 4
            or sample.requires_grad
            or not sample.is_floating_point()
            or not torch.isfinite(sample).all()
        ):
            raise ValueError("DiT cache accepts finite inference-only BCHW samples")
        signature = (tuple(sample.shape), sample.dtype, sample.device)
        if self._signature is not None and signature != self._signature:
            raise ValueError("Batch/shape/dtype/device changed inside a residual cache session")
        self._signature = signature

    def predict(self, sample, *, step):
        with self._lock, torch.inference_mode():
            self._check(sample, step)
            try:
                result = self._predict(sample, step)
                self._next_step += 1
                return result
            except Exception:
                self._closed = True
                raise

    def _predict(self, sample, step):
        model = self.model
        b, c, h, w = sample.shape
        p, d = model.config.patch_size, model.config.hidden_size
        if c != model.config.in_channels or h % p or w % p:
            raise ValueError("Native DiT channel/patch geometry mismatch")
        x = model.patch(sample).flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(
            torch.arange(h // p, device=x.device),
            torch.arange(w // p, device=x.device),
            indexing="ij",
        )
        x = (
            x
            + torch.cat(
                (
                    timestep_embedding(xx.flatten(), d // 2),
                    timestep_embedding(yy.flatten(), d // 2),
                ),
                -1,
            ).to(x.dtype)[None]
        )
        times = torch.full((b,), self.schedule[step], device=x.device, dtype=torch.float32)
        t = model.time(timestep_embedding(times, 256).to(x.dtype))
        if model.condition is not None:
            condition = (
                x.new_zeros(b, model.config.condition_dim)
                if self._condition is None
                else self._condition
            )
            t = t + model.condition(condition)
        elif model.classes is not None:
            condition = (
                torch.full((b,), model.config.num_classes, device=x.device, dtype=torch.long)
                if self._condition is None
                else self._condition
            )
            t = t + model.classes(condition)
        elif self._condition is not None:
            raise ValueError("Unconditional native DiT does not accept conditioning")
        first = model.blocks[0]
        shift, scale, *_ = first.ada(t).chunk(6, -1)
        probe = first.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        if not torch.isfinite(probe).all():
            raise ValueError("Non-finite modulation probe")
        if self._previous_probe is not None:
            self._accumulated += self.calibration.estimate(
                self._distance(probe, self._previous_probe)
            )
        candidate = (
            not self.disabled
            and step not in {0, len(self.schedule) - 1}
            and self._residual is not None
            and self._consecutive < self.max_skip
            and self._accumulated < self.threshold
        )
        if candidate:
            self._candidates += 1
        audit = candidate and self._candidates % self.audit_every == 0
        error = None
        if candidate and not audit:
            hidden = x + self._residual
            self.reused_backbone_calls += 1
            self._consecutive += 1
        else:
            hidden = x
            for block in model.blocks:
                hidden = block(hidden, t)
            self.full_backbone_calls += 1
            if self._residual is not None and (audit or self._consecutive > 0):
                approximate = self._finish(x + self._residual, t, b, h, w, c)
                actual = self._finish(hidden, t, b, h, w, c)
                error = self._distance(approximate, actual)
                if not math.isfinite(error) or error > self.max_relative_error:
                    self.guard_failed = self.disabled = True
            self._residual = (hidden - x).detach().clone()
            self._accumulated, self._consecutive = 0.0, 0
        self._previous_probe = probe.detach().clone()
        output = self._finish(hidden, t, b, h, w, c)
        if not torch.isfinite(output).all():
            raise ValueError("Non-finite cached DiT output")
        self.trace.append(
            {
                "step": step,
                "time": self.schedule[step],
                "reused": bool(candidate and not audit),
                "audit": bool(audit),
                "checked_relative_output_l1": error,
                "guard_failed": self.guard_failed,
            }
        )
        return FieldOutput(output, model.config.prediction_type)

    def _finish(self, hidden, t, b, h, w, c):
        p = self.model.config.patch_size
        shift, scale = self.model.ada(t).chunk(2, -1)
        output = self.model.output(self.model.norm(hidden) * (1 + scale[:, None]) + shift[:, None])
        channels = self.model.config.out_channels or c
        return (
            output.reshape(b, h // p, w // p, p, p, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, channels, h, w)
        )

    def observation(self):
        return {
            "policy_artifact_id": self.policy_artifact_id,
            "calibration_fingerprint": self.calibration.fingerprint,
            "condition_fingerprint": self.condition_fingerprint,
            "provider": "native_dit_calibrated_residual_reuse",
            "evidence_kind": "approximate_transform",
            "threshold": self.threshold,
            "max_skip": self.max_skip,
            "audit_every": self.audit_every,
            "max_relative_error": self.max_relative_error,
            "schedule": self.schedule,
            "completed_steps": self._next_step,
            "full_backbone_calls": self.full_backbone_calls,
            "reused_backbone_calls": self.reused_backbone_calls,
            "guard_failed": self.guard_failed,
            "quality_status": "failed_guard"
            if self.guard_failed
            else "requires_end_to_end_evaluation",
            "trace": copy.deepcopy(self.trace),
        }

    def close(self):
        with self._lock:
            self._closed = True
            self._previous_probe = self._residual = None
