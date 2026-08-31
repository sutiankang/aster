"""Wan TeaCache probes, accumulated decisions, and independent guidance-branch residuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import threading

import torch

from ..core import digest_json
from ..models.video_world import WanVideoDiT


TEACACHE_SOURCE = {
    "repository": "https://github.com/ali-vilab/TeaCache",
    "commit": "7c10efc4702c6b619f47805f7abe4a7a08085aa0",
    "path": "TeaCache4Wan2.1/teacache_generate.py",
    "sha256": "97af76136337869152f3d6fe9e049cadc2c480740c492749fcd5efa80d9bf7ee",
    "decision_lines": [519, 568],
    "profile_lines": [880, 894],
}


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def tensor_fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return digest_json(
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    )


def wan_policy_fingerprint(model):

    if type(model) is not WanVideoDiT:
        raise TypeError(
            "TeaCache adapter supports exactly native Wan2.1, not Wan2.2/other DiT aliases"
        )
    return digest_json(
        {
            "config": model.config.to_dict(),
            "parameters": {
                key: tensor_fingerprint(value) for key, value in model.named_parameters()
            },
            "buffers": {key: tensor_fingerprint(value) for key, value in model.named_buffers()},
        }
    )


def _versions(model):
    return tuple(
        (key, id(value), value._version)
        for key, value in (*model.named_parameters(), *model.named_buffers())
    )


def _relative_l1(current, previous):

    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean()
    if not torch.isfinite(numerator) or not torch.isfinite(denominator):
        raise ValueError("Non-finite TeaCache distance")
    if float(denominator) == 0:
        if float(numerator) == 0:
            return 0.0
        raise ValueError("TeaCache relative distance has zero denominator")
    value = float(numerator / denominator)
    if not math.isfinite(value):
        raise ValueError("Non-finite TeaCache ratio")
    return value


@dataclass(frozen=True)
class WanCacheSampler:
    steps: int = 30
    solver: str = "euler"
    shift: float = 5.0
    guidance_scale: float = 1.0

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 2 or self.solver not in {"euler", "heun"}:
            raise ValueError(
                "Native calibrated TeaCache currently supports >=2 Euler/Heun steps only"
            )
        if (
            any(
                type(v) not in {int, float} or not math.isfinite(v)
                for v in (self.shift, self.guidance_scale)
            )
            or self.shift <= 0
            or self.guidance_scale < 0
        ):
            raise ValueError("Invalid native Wan shift/guidance")

    @property
    def branches(self):
        return ("positive",) if self.guidance_scale == 1 else ("positive", "negative")

    def sigmas(self, device):
        values = torch.linspace(1, 0, self.steps + 1, device=device, dtype=torch.float32)
        return self.shift * values / (1 + (self.shift - 1) * values)

    def evaluation_times(self, device):
        values = self.sigmas(device)
        return (
            values[:-1]
            if self.solver == "euler"
            else torch.stack((values[:-1], values[1:]), 1).flatten()
        )


@dataclass(frozen=True)
class WanTeaCacheSettings:
    threshold: float = 0.1
    mode: str = "default"
    audit_every: int = 0
    maximum_relative_output_error: float = 0.05

    def __post_init__(self):
        if (
            self.mode not in {"default", "retention"}
            or type(self.audit_every) is not int
            or self.audit_every < 0
        ):
            raise ValueError("Invalid TeaCache mode/audit cadence")
        if any(
            type(v) not in {int, float} or not math.isfinite(v) or v < 0
            for v in (self.threshold, self.maximum_relative_output_error)
        ):
            raise ValueError("TeaCache thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class WanCacheCalibration:
    policy_artifact_id: str
    policy_fingerprint: str
    dataset_fingerprint: str
    sampler: WanCacheSampler
    latent_shape: tuple[int, ...]
    mode: str
    coefficients: tuple[float, ...]
    measurements: tuple[dict, ...]
    case_ids: tuple[str, ...]
    source: dict
    torch_version: str
    origin: str = "native_wan21_full_trajectory_residual_fit"

    def __post_init__(self):
        for name in ("latent_shape", "coefficients", "measurements", "case_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all(
            _sha(v)
            for v in (self.policy_artifact_id, self.policy_fingerprint, self.dataset_fingerprint)
        ):
            raise ValueError("Calibration must bind immutable policy and dataset identities")
        if (
            not isinstance(self.sampler, WanCacheSampler)
            or self.mode not in {"default", "retention"}
            or len(self.latent_shape) != 5
            or self.latent_shape[0] != 1
            or any(type(v) is not int or v < 1 for v in self.latent_shape)
        ):
            raise ValueError("Invalid calibration sampler/mode/geometry")
        if (
            self.origin != "native_wan21_full_trajectory_residual_fit"
            or self.source != TEACACHE_SOURCE
        ):
            raise ValueError("Unverified official model coefficients/profiles are not accepted")
        if not 1 <= len(self.coefficients) <= 5 or not all(
            math.isfinite(v) for v in self.coefficients
        ):
            raise ValueError("Invalid measured calibration polynomial")
        if (
            not self.case_ids
            or len(set(self.case_ids)) != len(self.case_ids)
            or len(self.measurements) < len(self.coefficients)
        ):
            raise ValueError("Calibration needs complete unique real trajectory cases")
        for row in self.measurements:
            if (
                set(row) != {"case_id", "branch", "round", "probe_distance", "residual_distance"}
                or row["case_id"] not in self.case_ids
                or row["branch"] not in self.sampler.branches
            ):
                raise ValueError("Malformed measured calibration record")
            if (
                type(row["round"]) is not int
                or row["round"] < 1
                or any(
                    type(row[k]) not in {int, float} or not math.isfinite(row[k]) or row[k] < 0
                    for k in ("probe_distance", "residual_distance")
                )
            ):
                raise ValueError("Invalid calibration numeric measurement")
        rounds = len(self.sampler.evaluation_times("cpu"))
        expected = [
            (case, branch, index)
            for case in self.case_ids
            for index in range(1, rounds)
            for branch in self.sampler.branches
        ]
        if [(row["case_id"], row["branch"], row["round"]) for row in self.measurements] != expected:
            raise ValueError("Calibration trajectory population is incomplete/reordered")

    def estimate(self, distance):
        result = 0.0
        for coefficient in self.coefficients:
            result = result * distance + coefficient
        if not math.isfinite(result):
            raise ValueError("Calibration polynomial overflow")

        return result

    def to_dict(self):
        return asdict(self)

    @property
    def id(self):
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        data = copy.deepcopy(value)
        data["sampler"] = WanCacheSampler(**data["sampler"])
        return cls(**data)


class WanTeaCacheSession:
    """Keep guidance branches and clocks request-local; never share class-level cache state."""

    def __init__(
        self,
        model,
        *,
        policy_artifact_id,
        sampler,
        condition,
        negative_condition=None,
        calibration=None,
        settings=None,
        collect=False,
    ):
        if (
            type(model) is not WanVideoDiT
            or model.training
            or any(p.dtype != torch.float32 for p in model.parameters())
        ):
            raise ValueError(
                "Native Wan TeaCache currently requires the exact eval-mode FP32 model"
            )
        if torch.is_autocast_enabled(next(model.parameters()).device.type):
            raise ValueError("Autocast is not part of the verified FP32 TeaCache profile")
        if not _sha(policy_artifact_id) or not isinstance(sampler, WanCacheSampler):
            raise ValueError("Immutable model artifact and typed native sampler are required")
        if (calibration is None) != (settings is None) or (collect and calibration is not None):
            raise ValueError("Provide both calibration/settings, or neither for full reference")
        self.model, self.sampler, self.policy_artifact_id = model, sampler, policy_artifact_id
        self.calibration, self.settings, self.collect = calibration, settings, bool(collect)
        self.policy_fingerprint = wan_policy_fingerprint(model)
        if calibration is not None:
            if (
                calibration.policy_artifact_id != policy_artifact_id
                or calibration.policy_fingerprint != self.policy_fingerprint
                or calibration.sampler != sampler
                or calibration.mode != settings.mode
                or calibration.torch_version != torch.__version__
            ):
                raise ValueError("TeaCache calibration model/sampler/probe/runtime mismatch")
        self._lock = threading.Lock()
        self._versions = _versions(model)
        self._config_id = digest_json(model.config.to_dict())
        self.reset(condition=condition, negative_condition=negative_condition)

    def reset(self, *, condition, negative_condition=None):

        with self._lock:
            if (
                _versions(self.model) != self._versions
                or self.model.training
                or digest_json(self.model.config.to_dict()) != self._config_id
            ):
                raise RuntimeError("Policy changed; create a new artifact/calibration/session")
            conditions = {"positive": condition}
            if "negative" in self.sampler.branches:
                conditions["negative"] = negative_condition
            if any(
                not isinstance(v, dict)
                or not v
                or any(
                    not isinstance(t, torch.Tensor) or not torch.isfinite(t).all()
                    for t in v.values()
                )
                for v in conditions.values()
            ):
                raise ValueError(
                    "Each CFG branch must contain finite real native condition tensors"
                )

            with torch.inference_mode(False):
                self.conditions = {
                    key: {name: value.detach().clone() for name, value in branch.items()}
                    for key, branch in conditions.items()
                }
            self._condition_versions = tuple(
                (branch, name, id(value), value._version)
                for branch, data in self.conditions.items()
                for name, value in data.items()
            )
            self.condition_fingerprint = digest_json(
                {
                    key: {name: tensor_fingerprint(value) for name, value in branch.items()}
                    for key, branch in self.conditions.items()
                }
            )
            self._states = {
                key: {
                    "probe": None,
                    "residual": None,
                    "accumulated": 0.0,
                    "candidates": 0,
                    "skipped": 0,
                }
                for key in self.sampler.branches
            }
            self._cursor, self._signature, self._closed = 0, None, False
            self.field_calls = self.full_backbone_calls = self.reused_backbone_calls = (
                self.head_calls
            ) = self.audit_backbone_calls = 0
            self.guard_failed = self.disabled = False
            self.trace, self.measurements = [], []

    @torch.inference_mode()
    def predict(self, sample, *, round_index, branch):
        with self._lock:
            try:
                conditions = tuple(
                    (branch, name, id(value), value._version)
                    for branch, data in self.conditions.items()
                    for name, value in data.items()
                )
                if (
                    self._closed
                    or self.model.training
                    or _versions(self.model) != self._versions
                    or digest_json(self.model.config.to_dict()) != self._config_id
                    or conditions != self._condition_versions
                ):
                    raise RuntimeError("TeaCache session closed or immutable policy changed")
                if torch.is_autocast_enabled(sample.device.type):
                    raise ValueError("Autocast changed the verified FP32 inference profile")
                count = len(self.sampler.branches)
                if (
                    type(round_index) is not int
                    or round_index != self._cursor // count
                    or branch != self.sampler.branches[self._cursor % count]
                ):
                    raise ValueError(
                        "Out-of-order/incorrect CFG branch; partial rounds cannot be resumed"
                    )
                times = self.sampler.evaluation_times(sample.device)
                if round_index >= len(times):
                    raise ValueError("Sampler evaluation clock exhausted; reset explicitly")
                if (
                    sample.ndim != 5
                    or sample.shape[0] != 1
                    or sample.dtype != torch.float32
                    or sample.requires_grad
                    or not torch.isfinite(sample).all()
                ):
                    raise ValueError(
                        "TeaCache currently accepts finite FP32 batch-one native latent videos"
                    )
                signature = (tuple(sample.shape), sample.dtype, sample.device)
                if self._signature is not None and self._signature != signature:
                    raise ValueError("Request latent layout changed")
                if (
                    self.calibration is not None
                    and tuple(sample.shape) != self.calibration.latent_shape
                ):
                    raise ValueError("Calibration latent geometry mismatch")
                self._signature = signature
                prepared = self.model.prepare(
                    sample, times[round_index].expand(1), self.conditions[branch]
                )
                state = self._states[branch]
                mode = (
                    self.settings.mode
                    if self.settings is not None
                    else getattr(self, "collect_mode", "default")
                )
                probe = prepared.embedding if mode == "default" else prepared.modulation
                if not torch.isfinite(probe).all():
                    raise ValueError("Non-finite Wan time probe")
                forced = round_index < (1 if mode == "default" else 5) or (
                    mode == "default" and round_index == len(times) - 1
                )

                tracking = self.collect or self.calibration is not None
                distance = (
                    _relative_l1(probe, state["probe"])
                    if state["probe"] is not None and (self.collect or not forced)
                    else None
                )
                if forced:
                    state["accumulated"] = 0.0
                elif self.calibration is not None:
                    state["accumulated"] += self.calibration.estimate(distance)
                reuse = bool(
                    self.calibration is not None
                    and not self.disabled
                    and not forced
                    and state["residual"] is not None
                    and state["accumulated"] < self.settings.threshold
                )
                if reuse:
                    state["candidates"] += 1
                audit = bool(
                    reuse
                    and self.settings.audit_every
                    and state["candidates"] % self.settings.audit_every == 0
                )
                error = None
                if reuse and not audit:
                    hidden = prepared.hidden + state["residual"]
                    state["skipped"] += 1
                    self.reused_backbone_calls += 1
                else:
                    hidden = self.model.run_blocks(prepared)
                    self.full_backbone_calls += 1
                    self.audit_backbone_calls += int(audit)
                    residual = hidden - prepared.hidden if tracking else None
                    if self.collect and state["residual"] is not None:
                        self.measurements.append(
                            {
                                "branch": branch,
                                "round": round_index,
                                "probe_distance": distance,
                                "residual_distance": _relative_l1(residual, state["residual"]),
                            }
                        )
                    if (
                        self.settings is not None
                        and state["residual"] is not None
                        and (audit or state["skipped"])
                    ):
                        approximate = self.model.finish(
                            prepared.hidden + state["residual"], prepared
                        ).prediction
                        actual = self.model.finish(hidden, prepared).prediction
                        self.head_calls += 2
                        try:
                            error = _relative_l1(approximate, actual)
                        except ValueError:
                            self.guard_failed = self.disabled = True
                        if (
                            error is not None
                            and error > self.settings.maximum_relative_output_error
                        ):
                            self.guard_failed = self.disabled = True
                    state["residual"] = residual.detach().clone() if residual is not None else None
                    state["accumulated"], state["skipped"] = 0.0, 0
                state["probe"] = probe.detach().clone() if tracking else None
                output = self.model.finish(hidden, prepared)
                self.field_calls += 1
                self.head_calls += 1
                if not torch.isfinite(output.prediction).all():
                    raise ValueError("Non-finite TeaCache output")
                if _versions(self.model) != self._versions:
                    raise RuntimeError("Immutable policy changed during inference")
                self.trace.append(
                    {
                        "round": round_index,
                        "branch": branch,
                        "time": float(times[round_index]),
                        "probe_distance": distance,
                        "accumulated": state["accumulated"],
                        "forced": forced,
                        "reused": reuse and not audit,
                        "audit": audit,
                        "checked_output_relative_l1": error,
                        "guard_failed": self.guard_failed,
                    }
                )
                self._cursor += 1
                return output
            except Exception:
                self._closed = True
                raise

    def observation(self):
        return {
            "provider": "native_wan21_teacache",
            "evidence_kind": "approximate_inference_transform_not_distillation",
            "source": dict(TEACACHE_SOURCE),
            "policy_artifact_id": self.policy_artifact_id,
            "policy_fingerprint": self.policy_fingerprint,
            "condition_fingerprint": self.condition_fingerprint,
            "sampler": asdict(self.sampler),
            "calibration_id": self.calibration.id if self.calibration else None,
            "settings": asdict(self.settings) if self.settings else None,
            "field_calls": self.field_calls,
            "full_backbone_calls": self.full_backbone_calls,
            "reused_backbone_calls": self.reused_backbone_calls,
            "audit_backbone_calls": self.audit_backbone_calls,
            "head_calls": self.head_calls,
            "guard_failed": self.guard_failed,
            "quality_status": "failed_guard"
            if self.guard_failed
            else "requires_end_to_end_evaluation",
            "trace": copy.deepcopy(self.trace),
        }

    def close(self):
        with self._lock:
            self._closed = True
            for state in self._states.values():
                state["probe"] = state["residual"] = None


@torch.inference_mode()
def sample_wan_teacache(
    model,
    noise,
    condition,
    *,
    policy_artifact_id,
    sampler,
    negative_condition=None,
    calibration=None,
    settings=None,
    collect=False,
    collect_mode="default",
):

    session = WanTeaCacheSession(
        model,
        policy_artifact_id=policy_artifact_id,
        sampler=sampler,
        condition=condition,
        negative_condition=negative_condition,
        calibration=calibration,
        settings=settings,
        collect=collect,
    )
    if collect_mode not in {"default", "retention"}:
        raise ValueError("Unknown time probe")
    session.collect_mode = collect_mode
    try:
        value = run_wan_teacache_session(session, noise)
        return value, session.observation(), copy.deepcopy(session.measurements)
    finally:
        session.close()


@torch.inference_mode()
def run_wan_teacache_session(session, noise):

    if not isinstance(session, WanTeaCacheSession) or session._cursor != 0 or session._closed:
        raise ValueError("Run requires an unused verified native session")
    sampler = session.sampler
    sigmas, value, index = sampler.sigmas(noise.device), noise.clone(), 0

    def velocity(state, clock):
        positive = session.predict(state, round_index=clock, branch="positive").prediction
        if sampler.guidance_scale == 1:
            return positive
        negative = session.predict(state, round_index=clock, branch="negative").prediction
        return negative + sampler.guidance_scale * (positive - negative)

    for left, right in zip(sigmas[:-1], sigmas[1:]):
        first = velocity(value, index)
        index += 1
        proposal = value + (right - left) * first
        if sampler.solver == "heun":
            second = velocity(proposal, index)
            index += 1
            value = value + (right - left) * (first + second) / 2
        else:
            value = proposal
    return value


def calibrate_wan_teacache(
    model,
    cases,
    *,
    policy_artifact_id,
    dataset_fingerprint,
    sampler,
    mode="default",
    degree=2,
    ridge=1e-6,
):

    if type(degree) is not int or not 0 <= degree <= 4 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Invalid real trajectory calibration fit")
    initial_versions = _versions(model)
    initial_config = digest_json(model.config.to_dict())
    rows, ids, shape = [], [], None
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) - {"id", "noise", "positive", "negative"}
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in ids
        ):
            raise ValueError(
                "Calibration cases must be explicit, unique native noise/condition records"
            )
        current = tuple(case["noise"].shape)
        if shape is not None and current != shape:
            raise ValueError("Calibration requires one fixed latent bucket")
        shape = current
        ids.append(case["id"])
        _, _, measures = sample_wan_teacache(
            model,
            case["noise"],
            case["positive"],
            policy_artifact_id=policy_artifact_id,
            sampler=sampler,
            negative_condition=case.get("negative"),
            collect=True,
            collect_mode=mode,
        )
        rows.extend({"case_id": case["id"], **row} for row in measures)
    if len(rows) < degree + 1:
        raise ValueError("Not enough real calibration measurements")
    if (
        _versions(model) != initial_versions
        or digest_json(model.config.to_dict()) != initial_config
        or model.training
    ):
        raise ValueError("Policy changed between calibration trajectories")
    x = torch.tensor([row["probe_distance"] for row in rows], dtype=torch.float64)
    y = torch.tensor([row["residual_distance"] for row in rows], dtype=torch.float64)
    design = torch.vander(x, N=degree + 1)
    coefficients = torch.linalg.solve(
        design.T @ design + ridge * torch.eye(degree + 1, dtype=x.dtype), design.T @ y
    )
    return WanCacheCalibration(
        policy_artifact_id,
        wan_policy_fingerprint(model),
        dataset_fingerprint,
        sampler,
        shape,
        mode,
        tuple(coefficients.tolist()),
        tuple(rows),
        tuple(ids),
        dict(TEACACHE_SOURCE),
        torch.__version__,
    )
