"""CT, CD, and improved consistency training with explicit target and sampling EMA roles."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn

from ..core import LossTerm
from .generation import expand


@dataclass(frozen=True)
class ConsistencyConfig:
    mode: str = "ict"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    total_steps: int = 800000
    initial_scales: int | None = None
    final_scales: int | None = None
    curriculum: str | None = None
    target_ema: float | None = None
    target_ema_mode: str | None = None
    sampling: str | None = None
    weighting: str | None = None
    metric: str | None = None
    huber_factor: float = 0.00054
    log_mean: float = -1.1
    log_std: float = 2.0

    time_scale: float = 250.0
    teacher_time_scale: float = 0.25
    sampling_ema: float | None = 0.9999
    seed: int = 0

    def __post_init__(self):
        if self.mode not in {"ct", "cd", "ict"}:
            raise ValueError("Consistency mode must be ct, cd or ict")
        defaults = {
            "ct": (2, 150, "progressive", 0.95, "adaptive", "uniform", "uniform", "mse"),
            "cd": (40, 40, "fixed", 0.0, "fixed", "uniform", "uniform", "mse"),
            "ict": (
                10,
                1280,
                "doubling",
                0.0,
                "fixed",
                "lognormal",
                "inverse_delta",
                "pseudo_huber",
            ),
        }
        for key, value in zip(
            (
                "initial_scales",
                "final_scales",
                "curriculum",
                "target_ema",
                "target_ema_mode",
                "sampling",
                "weighting",
                "metric",
            ),
            defaults[self.mode],
        ):
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        if any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in (
                self.sigma_min,
                self.sigma_max,
                self.sigma_data,
                self.rho,
                self.target_ema,
                self.huber_factor,
                self.log_mean,
                self.log_std,
                self.time_scale,
                self.teacher_time_scale,
            )
        ):
            raise ValueError("Consistency scalar settings must be finite real numbers")
        if (
            not 0 < self.sigma_min < self.sigma_max
            or min(
                self.sigma_data,
                self.rho,
                self.log_std,
                self.huber_factor,
                self.time_scale,
                self.teacher_time_scale,
            )
            <= 0
        ):
            raise ValueError("Invalid consistency scales")
        if any(
            type(value) is not int or value < 1
            for value in (self.total_steps, self.initial_scales, self.final_scales)
        ):
            raise ValueError("Curriculum counts must be positive integers")
        if self.initial_scales < 2 or self.final_scales < self.initial_scales:
            raise ValueError("Invalid scale-count bounds")
        if self.curriculum not in {"fixed", "progressive", "doubling"} or self.sampling not in {
            "uniform",
            "lognormal",
        }:
            raise ValueError("Unknown consistency curriculum or interval distribution")
        if self.curriculum == "fixed" and self.initial_scales != self.final_scales:
            raise ValueError("Fixed scales must have equal endpoints")
        if (
            self.curriculum == "doubling"
            and self.total_steps < math.log2(self.final_scales // self.initial_scales) + 1
        ):
            raise ValueError("Doubling curriculum has a zero-length stage")
        if self.target_ema_mode not in {"fixed", "adaptive"} or not 0 <= self.target_ema < 1:
            raise ValueError("Invalid target EMA")
        if self.target_ema_mode == "adaptive" and (
            self.curriculum != "progressive" or not self.target_ema > 0
        ):
            raise ValueError("Adaptive EMA requires progressive scales and positive starting decay")
        if self.mode == "ict" and (self.target_ema != 0 or self.target_ema_mode != "fixed"):
            raise ValueError("iCT requires stop-gradient current weights, not a lagging EMA target")
        if self.weighting not in {
            "uniform",
            "snr",
            "snr+1",
            "karras",
            "truncated-snr",
            "inverse_delta",
        }:
            raise ValueError("Unknown consistency weighting")
        if self.metric not in {"mse", "l1", "pseudo_huber"}:
            raise ValueError("Only explicit MSE/L1/vector pseudo-Huber metrics are implemented")
        if self.sampling_ema is not None and (
            type(self.sampling_ema) not in {int, float}
            or not math.isfinite(self.sampling_ema)
            or not 0 <= self.sampling_ema < 1
        ):
            raise ValueError("Invalid inference-only EMA decay")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

    def to_dict(self):
        return asdict(self)

    def scales_and_ema(self, step):
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise ValueError("Invalid completed-update curriculum step")
        if self.curriculum == "fixed":
            return self.initial_scales, self.target_ema
        if self.curriculum == "doubling":
            period = math.floor(
                self.total_steps / (math.log2(self.final_scales // self.initial_scales) + 1)
            )

            count = min(self.initial_scales * 2 ** (step // period), self.final_scales) + 1
            return count, self.target_ema
        intervals = max(
            math.ceil(
                math.sqrt(
                    step
                    / self.total_steps
                    * ((self.final_scales + 1) ** 2 - self.initial_scales**2)
                    + self.initial_scales**2
                )
                - 1
            ),
            1,
        )
        decay = (
            math.exp(math.log(self.target_ema) * self.initial_scales / intervals)
            if self.target_ema_mode == "adaptive"
            else self.target_ema
        )
        return intervals + 1, decay

    def levels(self, step):
        count, _ = self.scales_and_ema(step)
        low, high = self.sigma_min ** (1 / self.rho), self.sigma_max ** (1 / self.rho)
        values = (low + torch.linspace(0, 1, count, dtype=torch.float64) * (high - low)).pow(
            self.rho
        )
        values[0], values[-1] = self.sigma_min, self.sigma_max
        return values

    def interval_probabilities(self, step):
        levels = self.levels(step)
        if self.sampling == "uniform":
            return torch.full((len(levels) - 1,), 1 / (len(levels) - 1), dtype=torch.float64)
        cdf = torch.erf((levels.log() - self.log_mean) / (math.sqrt(2) * self.log_std))
        masses = cdf[1:] - cdf[:-1]
        if not (masses >= 0).all() or not masses.sum() > 0:
            raise ValueError("Numerically degenerate lognormal interval distribution")
        return masses / masses.sum()


def consistency_denoise(
    model, sample, sigma, condition=None, *, sigma_min=0.002, sigma_data=0.5, time_scale=250.0
):
    """Apply boundary preconditioning in FP32; outer autocast controls matrix products."""
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or value <= 0
        for value in (sigma_min, sigma_data, time_scale)
    ):
        raise ValueError("Consistency preconditioning requires positive finite scales")
    if (
        sigma.shape != (len(sample),)
        or not torch.isfinite(sigma).all()
        or (sigma < sigma_min).any()
    ):
        raise ValueError("One finite consistency noise level >= sigma_min is required per sample")
    sigma = sigma.float()
    sample = sample.float()
    scale = expand(sigma, sample)
    variance = scale.square() + sigma_data**2
    delta = scale - sigma_min
    output = model(sample / variance.sqrt(), sigma.log() * time_scale, condition)
    if output.prediction_type != "consistency_residual" or output.prediction.shape != sample.shape:
        raise ValueError(
            "Consistency network must return an aligned consistency_residual FieldOutput"
        )
    return (
        sigma_data**2 / (delta.square() + sigma_data**2) * sample
        + sigma_data * delta / variance.sqrt() * output.prediction.float()
    )


def _teacher_denoise(model, sample, sigma, condition, config):
    scale = expand(sigma.float(), sample)
    variance = scale.square() + config.sigma_data**2
    output = model(
        sample.float() / variance.sqrt(), sigma.float().log() * config.teacher_time_scale, condition
    )
    if output.prediction_type != "edm_residual" or output.prediction.shape != sample.shape:
        raise ValueError(
            "CD teacher must return aligned EDM residuals with an explicitly declared time unit"
        )
    return (
        config.sigma_data**2 / variance * sample.float()
        + config.sigma_data * scale / variance.sqrt() * output.prediction.float()
    )


def consistency_metric(predicted, target, metric, huber_factor=0.00054):
    difference = predicted.float() - target.float()
    if metric == "mse":
        return difference.square().flatten(1).mean(1)
    if metric == "l1":
        return difference.abs().flatten(1).mean(1)
    if metric != "pseudo_huber":
        raise ValueError("Unknown consistency metric")

    energy = difference.square().flatten(1).sum(1)
    constant = huber_factor * math.sqrt(difference[0].numel())

    return energy / ((energy + constant**2).sqrt() + constant)


def _weight(high, low, config):
    snr = high.float().reciprocal().square()
    return {
        "uniform": lambda: torch.ones_like(high),
        "snr": lambda: snr,
        "snr+1": lambda: snr + 1,
        "karras": lambda: snr + 1 / config.sigma_data**2,
        "truncated-snr": lambda: snr.clamp_min(1),
        "inverse_delta": lambda: (high - low).float().reciprocal(),
    }[config.weighting]()


def _torch_rng():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _set_torch_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


@contextmanager
def _mode(model, training):
    modes = [(module, module.training) for module in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, previous in modes:
            module.training = previous


class _ConsistencyObjective(nn.Module):
    def __init__(self, config, target, teacher=None):
        super().__init__()
        self.config, self.target, self.teacher = config, target, teacher

    def config_dict(self):
        return {"type": "consistency_lifecycle", "schema_version": 1, **self.config.to_dict()}

    def forward(self, model, batch):
        config = self.config
        sample, high, low, noise = (
            batch["sample"],
            batch["sigma_high"],
            batch["sigma_low"],
            batch["noise"],
        )
        condition = batch.get("condition")
        noisy = sample.float() + expand(high, sample) * noise.float()
        before = _torch_rng()
        predicted = consistency_denoise(
            model,
            noisy,
            high,
            condition,
            sigma_min=config.sigma_min,
            sigma_data=config.sigma_data,
            time_scale=config.time_scale,
        )
        after = _torch_rng()

        try:
            with torch.no_grad():
                if self.teacher is None:
                    next_sample = sample.float() + expand(low, sample) * noise.float()
                else:
                    with _mode(self.teacher, False):
                        slope = (
                            noisy - _teacher_denoise(self.teacher, noisy, high, condition, config)
                        ) / expand(high, noisy)
                        proposal = noisy + expand(low - high, noisy) * slope
                        next_slope = (
                            proposal
                            - _teacher_denoise(self.teacher, proposal, low, condition, config)
                        ) / expand(low, noisy)
                        next_sample = noisy + expand(low - high, noisy) * (slope + next_slope) / 2
                _set_torch_rng(before)
                with _mode(self.target, model.training):
                    target = consistency_denoise(
                        self.target,
                        next_sample,
                        low,
                        condition,
                        sigma_min=config.sigma_min,
                        sigma_data=config.sigma_data,
                        time_scale=config.time_scale,
                    )
        finally:
            _set_torch_rng(after)
        values = consistency_metric(
            predicted, target.detach(), config.metric, config.huber_factor
        ) * _weight(high, low, config)
        return LossTerm(
            values.sum(),
            torch.tensor(len(values), device=values.device, dtype=torch.int64),
            "sample",
            "consistency",
        )


def _fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _model_identity(model):
    if model is None:
        return None
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config": model.config.to_dict(),
    }


class ConsistencyMethod:
    """Own one student update, a frozen target, an optional EDM teacher, and a separate
    sampling EMA. These roles must not be substituted for one another."""

    def __init__(self, engine, *, target_factory, config=None, teacher=None):
        config = config or ConsistencyConfig()
        error, prepared_target, prepared_ema, declared = None, None, None, None
        try:
            if not isinstance(config, ConsistencyConfig):
                raise TypeError("Expected ConsistencyConfig")
            if engine._busy or engine._failed:
                raise RuntimeError(
                    "Consistency construction requires a successful idle Trainer boundary"
                )
            if {"consistency_target", "consistency_ema", "consistency_teacher"} & set(
                engine.roles
            ) or "consistency_method" in engine.states:
                raise ValueError("Consistency role/state names are already owned")
            if any(
                getattr(engine.parallel.config, name) != 1
                for name in (
                    "tensor_parallel",
                    "pipeline_parallel",
                    "context_parallel",
                    "gtp_remat",
                    "expert_parallel",
                    "expert_tensor_parallel",
                )
            ):
                raise ValueError(
                    "Consistency lifecycle currently supports DP x ZeRO only, not model-parallel field providers"
                )
            if (teacher is not None) != (config.mode == "cd"):
                raise ValueError("Exactly CD needs an explicit pretrained EDM teacher")
            if not callable(target_factory):
                raise TypeError("A target structure factory is required")
            for model in (engine.model, teacher):
                if model is not None and any(
                    isinstance(module, nn.modules.batchnorm._BatchNorm)
                    for module in model.modules()
                ):
                    raise ValueError(
                        "Paired consistency calls require replayable stateless normalization, not BatchNorm buffers"
                    )
            if (
                getattr(getattr(engine.model, "config", None), "prediction_type", None)
                != "consistency_residual"
            ):
                raise ValueError("Student config must explicitly declare consistency_residual")
            if (
                teacher is not None
                and getattr(getattr(teacher, "config", None), "prediction_type", None)
                != "edm_residual"
            ):
                raise ValueError("Teacher config must explicitly declare edm_residual")
            if teacher is not None and getattr(teacher, "_aster_training_owned", False):
                raise ValueError(
                    "Teacher must be an independent dense snapshot, not another live Trainer owner"
                )
            prepared_target = target_factory()
            prepared_ema = target_factory() if config.sampling_ema is not None else None
            for candidate in (prepared_target, prepared_ema):
                if candidate is None:
                    continue
                if (
                    type(candidate) is not type(engine.model)
                    or candidate is engine.model
                    or candidate.config.to_dict() != engine.model.config.to_dict()
                ):
                    raise ValueError(
                        "Target factory must construct an independent model with the exact student configuration"
                    )
                if getattr(candidate, "_aster_training_owned", False):
                    raise ValueError(
                        "Target factory must not deepcopy a training-owned shard layout"
                    )
            if prepared_target is prepared_ema:
                raise ValueError(
                    "Target and sampling EMA require independent factories, not one shared instance"
                )
            declared = {
                "method": config.to_dict(),
                "student": _model_identity(engine.model),
                "teacher": _model_identity(teacher),
                "teacher_sha256": _fingerprint(teacher) if teacher is not None else None,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records = engine.parallel.world.gather_objects((error, declared))
        if any(item[0] for item in records) or any(item[1] != records[0][1] for item in records):
            raise ValueError(f"Consistency construction preflight failed: {records}")
        self.engine, self.config, self._contract = engine, config, config.to_dict()
        self.target = engine.clone_target(
            "model", "consistency_target", factory=lambda: prepared_target
        )
        self.sampling_model = (
            engine.clone_target("model", "consistency_ema", factory=lambda: prepared_ema)
            if config.sampling_ema is not None
            else None
        )
        self.teacher = (
            engine.add_role("consistency_teacher", teacher, trainable=False)
            if teacher is not None
            else None
        )
        self._teacher_digest = _fingerprint(self.teacher) if self.teacher is not None else None
        identities = engine.parallel.world.gather_objects(self._teacher_digest)
        if len(set(identities)) != 1:
            raise ValueError("All DP ranks must start from the same pretrained teacher weights")
        self._target_digest = _fingerprint(self.target)
        self._sampling_digest = (
            _fingerprint(self.sampling_model) if self.sampling_model is not None else None
        )
        self._model_contracts = self._model_identities()
        self.generator = torch.Generator(device="cpu").manual_seed(
            config.seed + engine.parallel.rank
        )
        self.objective = _ConsistencyObjective(config, self.target, self.teacher)
        self._initial_role_updates = engine.roles["model"].updates
        self.updates, self._incomplete = 0, False
        engine.register_state("consistency_method", self)

    def _model_identities(self):
        return {
            key: _model_identity(value)
            for key, value in (
                ("student", self.engine.model),
                ("target", self.target),
                ("teacher", self.teacher),
                ("sampling_ema", self.sampling_model),
            )
        }

    def _check_contract(self):
        if (
            self.config.to_dict() != self._contract
            or self.objective.config.to_dict() != self._contract
        ):
            raise ValueError("Consistency method configuration changed")
        if self.objective.target is not self.target or self.objective.teacher is not self.teacher:
            raise ValueError("Consistency objective role identity changed")
        if self.engine.roles["model"].updates != self._initial_role_updates + self.updates:
            raise ValueError(
                "Student optimizer advanced outside the consistency lifecycle; its target is no longer current"
            )
        if self._model_identities() != self._model_contracts:
            raise ValueError("Consistency model/teacher/target configuration changed")
        if _fingerprint(self.target) != self._target_digest:
            raise ValueError("Consistency target weights changed outside the registered lifecycle")
        if self.teacher is not None and _fingerprint(self.teacher) != self._teacher_digest:
            raise ValueError("Frozen consistency teacher weights changed")
        if (
            self.sampling_model is not None
            and _fingerprint(self.sampling_model) != self._sampling_digest
        ):
            raise ValueError(
                "Consistency sampling EMA weights changed outside the registered lifecycle"
            )

    def _validate_batch(self, batch):
        if not isinstance(batch, dict) or set(batch) - {
            "sample",
            "noise",
            "interval_indices",
            "condition",
        }:
            raise ValueError("Unknown consistency batch fields")
        sample = batch["sample"]
        if (
            not isinstance(sample, torch.Tensor)
            or sample.ndim < 2
            or min(sample.shape) < 1
            or not sample.is_floating_point()
            or not torch.isfinite(sample).all()
        ):
            raise ValueError("Consistency samples must be nonempty finite floating [B,...] tensors")
        if sample.device != self.engine.device:
            raise ValueError("Consistency sample device must match Trainer")
        noise = batch.get("noise")
        if noise is not None and (
            noise.shape != sample.shape
            or noise.device != sample.device
            or not noise.is_floating_point()
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("Consistency noise must align with finite samples")
        condition = batch.get("condition")
        if condition is not None and (
            not isinstance(condition, torch.Tensor)
            or condition.ndim < 1
            or len(condition) != len(sample)
            or condition.device != sample.device
            or not torch.isfinite(condition).all()
        ):
            raise ValueError("Consistency condition must be an aligned finite Tensor")

        for model in (self.engine.model, self.teacher):
            if model is None:
                continue
            declared = model.config
            channels = getattr(declared, "in_channels", None)
            if channels is not None and (sample.ndim != 4 or sample.shape[1] != channels):
                raise ValueError("Declared image field channels/layout mismatch")
            if getattr(declared, "out_channels", None) not in {None, channels}:
                raise ValueError("Consistency residual must preserve channel count")
            divisor = (
                2 ** (len(declared.channel_mult) - 1)
                if hasattr(declared, "channel_mult")
                else getattr(declared, "patch_size", 1)
            )
            if channels is not None and any(size % divisor for size in sample.shape[-2:]):
                raise ValueError("Image shape violates field downsampling/patch divisor")
            classes, width = (
                getattr(declared, "num_classes", 0),
                getattr(declared, "condition_dim", 0),
            )
            if condition is not None:
                if width and (
                    condition.shape != (len(sample), width) or not condition.is_floating_point()
                ):
                    raise ValueError("Condition vector shape/dtype differs from model")
                if classes and (
                    condition.shape != (len(sample),)
                    or condition.dtype != torch.long
                    or ((condition < 0) | (condition > classes)).any()
                ):
                    raise ValueError("Class condition outside declared support")
                if not width and not classes:
                    raise ValueError("Unconditional field does not accept a condition")
        indices = batch.get("interval_indices")
        count, _ = self.config.scales_and_ema(self.updates)
        if indices is not None and (
            indices.dtype != torch.long
            or indices.shape != (len(sample),)
            or ((indices < 0) | (indices >= count - 1)).any()
        ):
            raise ValueError(
                "Interval indices must select one adjacent curriculum interval per sample"
            )

    def update(self, microbatches):
        batches, error = None, None
        try:
            if self._incomplete:
                raise RuntimeError(
                    "Consistency round incomplete; restore its last complete checkpoint"
                )
            self._check_contract()
            if self.updates >= self.config.total_steps:
                raise ValueError("Consistency training budget exhausted")
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("Consistency microbatch count differs from Trainer accumulation")
            for batch in batches:
                self._validate_batch(batch)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records = self.engine.parallel.world.gather_objects(
            (error, self.updates, len(batches) if batches is not None else None)
        )
        if any(item[0] for item in records) or any(item[1:] != records[0][1:] for item in records):
            raise ValueError(f"Consistency collective preflight failed: {records}")
        self._incomplete = True
        try:
            return self._run_update(batches)
        except BaseException:
            self.engine._failed = True
            raise

    def _run_update(self, batches):
        levels, probability = (
            self.config.levels(self.updates),
            self.config.interval_probabilities(self.updates),
        )
        prepared = []
        for batch in batches:
            sample = batch["sample"]
            indices = batch.get("interval_indices")
            if indices is None:
                indices = torch.multinomial(
                    probability, len(sample), replacement=True, generator=self.generator
                )
            indices = indices.cpu()
            noise = batch.get("noise")
            if noise is None:
                noise = torch.randn(sample.shape, generator=self.generator, dtype=torch.float32).to(
                    sample.device
                )
            prepared.append(
                {
                    "sample": sample,
                    "noise": noise,
                    "condition": batch.get("condition"),
                    "sigma_low": levels[indices].to(device=sample.device, dtype=torch.float32),
                    "sigma_high": levels[indices + 1].to(device=sample.device, dtype=torch.float32),
                }
            )
        result = self.engine.phase("consistency", objective=self.objective, microbatches=prepared)
        if not result.updated:
            raise RuntimeError(
                "Consistency student update skipped; restore the last complete round"
            )
        _, decay = self.config.scales_and_ema(self.updates)
        self.engine.update_target("model", "consistency_target", decay)
        if self.sampling_model is not None:
            self.engine.update_target("model", "consistency_ema", self.config.sampling_ema)
        self._target_digest = _fingerprint(self.target)
        self._sampling_digest = (
            _fingerprint(self.sampling_model) if self.sampling_model is not None else None
        )
        self.updates += 1
        self._incomplete = False
        return result

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint/export an incomplete consistency round")
        self._check_contract()
        return {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "models": deepcopy(self._model_contracts),
            "updates": self.updates,
            "initial_role_updates": self._initial_role_updates,
            "generator": self.generator.get_state(),
            "teacher_sha256": self._teacher_digest,
            "target_sha256": self._target_digest,
            "sampling_sha256": self._sampling_digest,
        }

    def load_state_dict(self, state):
        if (
            set(state)
            != {
                "schema_version",
                "config",
                "models",
                "updates",
                "initial_role_updates",
                "generator",
                "teacher_sha256",
                "target_sha256",
                "sampling_sha256",
            }
            or state["schema_version"] != 1
        ):
            raise ValueError("Unknown consistency checkpoint state")
        if (
            state["config"] != self._contract
            or self.config.to_dict() != self._contract
            or state["teacher_sha256"] != self._teacher_digest
        ):
            raise ValueError(
                "Consistency checkpoint settings or pretrained teacher identity differ"
            )
        if (
            state["models"] != self._model_contracts
            or self._model_identities() != self._model_contracts
        ):
            raise ValueError("Consistency checkpoint model/teacher configuration differs")
        if (
            type(state["updates"]) is not int
            or not 0 <= state["updates"] <= self.config.total_steps
        ):
            raise ValueError("Invalid completed consistency updates")
        if (
            type(state["initial_role_updates"]) is not int
            or state["initial_role_updates"] < 0
            or self.engine.roles["model"].updates
            != state["initial_role_updates"] + state["updates"]
        ):
            raise ValueError("Consistency checkpoint student optimizer/target update clocks differ")
        if _fingerprint(self.target) != state["target_sha256"] or (
            self.sampling_model is not None
            and _fingerprint(self.sampling_model) != state["sampling_sha256"]
        ):
            raise ValueError(
                "Consistency frozen target/EMA payload does not match its checkpoint identity"
            )
        if self.teacher is not None and _fingerprint(self.teacher) != self._teacher_digest:
            raise ValueError("Teacher checkpoint payload changed")
        self.generator.set_state(state["generator"].cpu())
        self._initial_role_updates = state["initial_role_updates"]
        self.updates = state["updates"]
        self._target_digest = state["target_sha256"]
        self._sampling_digest = state["sampling_sha256"]
        self._incomplete = False

    def export_config(self):
        self.state_dict()
        return {
            "schema_version": 1,
            "method": "consistency",
            "training": self.config.to_dict(),
            "completed_updates": self.updates,
            "teacher_sha256": self._teacher_digest,
            "sampling_role": "consistency_ema" if self.sampling_model is not None else "model",
        }


@torch.no_grad()
def sample_consistency(
    model,
    noise,
    sigmas,
    *,
    condition=None,
    sigma_min=0.002,
    sigma_data=0.5,
    time_scale=250.0,
    generator=None,
    clip_denoised=True,
):
    """Call the model once per sigma and reinject noise with scale
    sqrt(sigma**2 - sigma_min**2) between predictions."""
    if (
        not isinstance(noise, torch.Tensor)
        or noise.ndim < 2
        or min(noise.shape) < 1
        or not noise.is_floating_point()
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("Sampling noise must be nonempty finite floating [B,...]")
    levels = torch.as_tensor(sigmas, dtype=torch.float64, device="cpu")
    if (
        levels.ndim != 1
        or not len(levels)
        or not torch.isfinite(levels).all()
        or (levels < sigma_min).any()
        or not (levels[:-1] > levels[1:]).all()
    ):
        raise ValueError("Sampling sigma calls must be finite, descending and >= sigma_min")
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or value <= 0
        for value in (sigma_min, sigma_data, time_scale)
    ):
        raise ValueError("Invalid consistency sampling preconditioning")
    if type(clip_denoised) is not bool:
        raise TypeError("clip_denoised must be explicit boolean")
    with _mode(model, False):
        sample = noise.float() * float(levels[0])
        for index, level in enumerate(levels):
            sample = consistency_denoise(
                model,
                sample,
                sample.new_full((len(sample),), float(level)),
                condition,
                sigma_min=sigma_min,
                sigma_data=sigma_data,
                time_scale=time_scale,
            )
            if clip_denoised:
                sample = sample.clamp(-1, 1)
            if index + 1 < len(levels):
                perturbation = torch.randn(
                    sample.shape, device=sample.device, dtype=sample.dtype, generator=generator
                )
                sample = (
                    sample + math.sqrt(float(levels[index + 1]) ** 2 - sigma_min**2) * perturbation
                )
        return sample
