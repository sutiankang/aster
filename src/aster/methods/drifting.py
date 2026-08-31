"""Drifting queues, training guidance, global feature statistics, and resumable updates."""

from copy import deepcopy
import hashlib
import math

import torch
from torch import nn

from ..core import FieldOutput, LossBundle, LossTerm
from ..core.serialization import canonical_json
from ..models.drifting import DriftingConfig
from .generative_distillation import drifting_loss


class ClassMemoryBank:
    """Bounded class-wise CPU ring buffers. Sample without replacement when possible;
    empty classes fail explicitly."""

    def __init__(self, num_classes, capacity, sample_shape, *, max_bytes=512 * 1024 * 1024):
        shape = tuple(sample_shape)
        if (
            any(type(x) is not int or x < 1 for x in (num_classes, capacity, *shape))
            or not shape
            or type(max_bytes) is not int
            or max_bytes < 1
        ):
            raise ValueError("Invalid memory-bank geometry/budget")
        if num_classes * capacity * math.prod(shape) * 4 > max_bytes:
            raise ValueError("Requested memory bank exceeds its explicit byte budget")
        self.num_classes, self.capacity, self.sample_shape = num_classes, capacity, shape
        self.values = torch.zeros(num_classes, capacity, *shape, dtype=torch.float32)
        self.cursor = torch.zeros(num_classes, dtype=torch.int64)
        self.count = torch.zeros_like(self.cursor)

    def _labels(self, labels):
        if (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or not len(labels)
            or labels.device.type != "cpu"
            or labels.dtype != torch.int64
            or (labels < 0).any()
            or (labels >= self.num_classes).any()
        ):
            raise ValueError("Memory-bank labels must be nonempty CPU int64 class indices")

    def add(self, samples, labels):
        self._labels(labels)
        if (
            samples.shape != (len(labels), *self.sample_shape)
            or samples.dtype != torch.float32
            or samples.device.type != "cpu"
            or not torch.isfinite(samples).all()
        ):
            raise ValueError("Memory-bank samples must be aligned finite CPU FP32")
        for sample, label in zip(samples.detach(), labels.tolist()):
            position = int(self.cursor[label])
            self.values[label, position].copy_(sample)
            self.cursor[label] = (position + 1) % self.capacity
            self.count[label] = min(int(self.count[label]) + 1, self.capacity)

    def sample(self, labels, number, *, generator):
        self._labels(labels)
        if type(number) is not int or number < 1 or (self.count[labels] == 0).any():
            raise ValueError("Positive sample count and nonempty requested classes are required")
        selected = []
        for label in labels.tolist():
            count = int(self.count[label])
            indices = (
                torch.randperm(count, generator=generator)[:number]
                if count >= number
                else torch.randint(count, (number,), generator=generator)
            )
            selected.append(self.values[label, indices])
        return torch.stack(selected)

    def state_dict(self):
        return dict(
            values=self.values.clone(), cursor=self.cursor.clone(), count=self.count.clone()
        )

    def validate_state(self, state):
        if (
            set(state) != {"values", "cursor", "count"}
            or state["values"].shape != self.values.shape
            or state["values"].dtype != torch.float32
            or not torch.isfinite(state["values"]).all()
            or state["cursor"].shape != self.cursor.shape
            or state["count"].shape != self.count.shape
            or state["cursor"].dtype != torch.int64
            or state["count"].dtype != torch.int64
            or any(x.device.type != "cpu" for x in state.values())
            or (state["cursor"] < 0).any()
            or (state["cursor"] >= self.capacity).any()
            or (state["count"] < 0).any()
            or (state["count"] > self.capacity).any()
            or ((state["count"] < self.capacity) & (state["cursor"] != state["count"])).any()
        ):
            raise ValueError("Corrupt memory-bank geometry, cursor or samples")

    def load_state_dict(self, state):
        self.validate_state(state)
        self.values.copy_(state["values"])
        self.cursor.copy_(state["cursor"])
        self.count.copy_(state["count"])


def sample_training_cfg(
    count, *, minimum=1.0, maximum=4.0, power=1.0, no_cfg_fraction=0.0, generator=None
):
    """Inverse-CDF sampling of a truncated power law; power=1 is the log-uniform limit."""
    if (
        type(count) is not int
        or count < 1
        or not all(math.isfinite(x) for x in (minimum, maximum, power, no_cfg_fraction))
        or not 1 <= minimum <= maximum
        or not 0 <= no_cfg_fraction <= 1
    ):
        raise ValueError("Invalid training CFG distribution")
    fraction = torch.rand(count, generator=generator)
    exponent = 1 - power
    if abs(exponent) < 1e-6:
        values = (math.log(minimum) + fraction * (math.log(maximum) - math.log(minimum))).exp()
    else:
        values = (minimum**exponent + fraction * (maximum**exponent - minimum**exponent)) ** (
            1 / exponent
        )
    reset = torch.rand(count, generator=generator) < no_cfg_fraction
    return values.masked_fill(reset, 1.0)


class SpatialFeatureStatistics(nn.Module):
    """Convert BCHW encoder maps into multiscale [batch, locations, features] statistics."""

    def __init__(
        self,
        encoder=None,
        *,
        patch_sizes=(2, 4),
        use_mean=True,
        use_std=True,
        input_patch_size=1,
        global_feature=True,
    ):
        super().__init__()
        if (
            any(type(x) is not int or x < 1 for x in patch_sizes)
            or len(set(patch_sizes)) != len(patch_sizes)
            or type(input_patch_size) is not int
            or input_patch_size < 1
        ):
            raise ValueError("Feature patch sizes must be distinct positive integers")
        self.encoder, self.patch_sizes, self.use_mean, self.use_std = (
            encoder,
            tuple(patch_sizes),
            use_mean,
            use_std,
        )
        self.input_patch_size, self.global_feature = input_patch_size, global_feature

    def config_dict(self):
        return dict(
            type="spatial_feature_statistics",
            patch_sizes=self.patch_sizes,
            use_mean=self.use_mean,
            use_std=self.use_std,
            input_patch_size=self.input_patch_size,
            global_feature=self.global_feature,
            encoder=(
                "pixels"
                if self.encoder is None
                else self.encoder.config_dict()
                if hasattr(self.encoder, "config_dict")
                else type(self.encoder).__qualname__
            ),
        )

    @staticmethod
    def _std(value, dim):
        work = value.float()
        return ((work - work.mean(dim, keepdim=True)).square().mean(dim).clamp_min(0) + 1e-6).sqrt()

    def forward(self, samples):
        if (
            samples.ndim != 4
            or not samples.is_floating_point()
            or not torch.isfinite(samples).all()
        ):
            raise ValueError("Spatial features require finite BCHW samples")
        raw = samples
        b, c, h, w = samples.shape
        p = self.input_patch_size
        if h % p or w % p:
            raise ValueError("Feature input geometry is not divisible by its patch size")

        samples = (
            samples.reshape(b, c, h // p, p, w // p, p)
            .permute(0, 3, 5, 1, 2, 4)
            .reshape(b, p * p * c, h // p, w // p)
        )
        features = {"pixels": samples} if self.encoder is None else self.encoder(samples)
        result = {"norm_x": (samples.float().square().mean((2, 3)) + 1e-6).sqrt()[:, None]}
        if self.global_feature:
            result["global"] = raw.permute(0, 2, 3, 1).reshape(b, 1, -1)
        if not isinstance(features, dict) or not features:
            raise ValueError("Feature encoder must return a nonempty named BCHW mapping")
        for name, feature in sorted(features.items()):
            if (
                not isinstance(name, str)
                or name in result
                or not isinstance(feature, torch.Tensor)
                or feature.ndim != 4
                or len(feature) != len(samples)
                or not torch.isfinite(feature).all()
            ):
                raise ValueError("Invalid spatial feature schema")
            b, channels, h, w = feature.shape
            flat = feature.flatten(2).transpose(1, 2)
            result[name] = flat
            if self.use_mean:
                result[name + "_mean"] = flat.mean(1, keepdim=True)
            if self.use_std:
                result[name + "_std"] = self._std(flat, 1)[:, None]
            for size in self.patch_sizes:
                if h % size == 0 and w % size == 0:
                    patch = (
                        feature.reshape(b, channels, h // size, size, w // size, size)
                        .permute(0, 2, 4, 3, 5, 1)
                        .reshape(b, h // size * w // size, size * size, channels)
                    )
                    if self.use_mean:
                        result[f"{name}_mean_{size}"] = patch.mean(2)
                    if self.use_std:
                        result[f"{name}_std_{size}"] = self._std(patch, 2)
        return result


class DriftingMethod:
    """Train over a complete logical batch with shared DP statistics and rank-local queues."""

    def __init__(
        self,
        engine,
        features,
        *,
        feature_identity,
        num_classes=None,
        positive_capacity=64,
        negative_capacity=512,
        positive_samples=32,
        negative_samples=16,
        generated_samples=8,
        cfg_min=1.0,
        cfg_max=4.0,
        cfg_power=1.0,
        no_cfg_fraction=0.0,
        radii=(0.02, 0.05, 0.2),
        seed=0,
        max_bank_bytes=512 * 1024 * 1024,
    ):
        self.engine = engine
        config = engine.model.config
        if not isinstance(config, DriftingConfig) or engine.accumulation_steps != 1:
            raise ValueError(
                "DriftingMethod needs its native generator and one complete force batch"
            )
        if any(
            getattr(engine.parallel.config, key, 1) > 1
            for key in (
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
                "gtp_remat",
                "expert_parallel",
                "expert_tensor_parallel",
            )
        ):
            raise ValueError("Drifting currently supports DP/ZeRO, not implicit model sharding")
        classes = config.num_classes if num_classes is None else num_classes
        if (
            not isinstance(feature_identity, str)
            or not feature_identity
            or type(classes) is not int
            or not 1 <= classes <= config.num_classes
            or any(
                type(x) is not int or x < 1
                for x in (
                    positive_capacity,
                    negative_capacity,
                    positive_samples,
                    negative_samples,
                    generated_samples,
                )
            )
            or generated_samples < 2
            or not radii
            or any(not math.isfinite(x) or x <= 0 for x in radii)
        ):
            raise ValueError("Invalid Drifting features/classes/group geometry")

        sample_training_cfg(
            1,
            minimum=cfg_min,
            maximum=cfg_max,
            power=cfg_power,
            no_cfg_fraction=no_cfg_fraction,
            generator=torch.Generator().manual_seed(0),
        )
        digest = hashlib.sha256(feature_identity.encode())
        for name, value in sorted(features.state_dict().items()):
            digest.update(canonical_json([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        self.settings = dict(
            feature_identity=feature_identity,
            feature_weights=digest.hexdigest(),
            feature_config=features.config_dict()
            if hasattr(features, "config_dict")
            else type(features).__qualname__,
            num_classes=classes,
            positive_capacity=positive_capacity,
            negative_capacity=negative_capacity,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            generated_samples=generated_samples,
            cfg_min=cfg_min,
            cfg_max=cfg_max,
            cfg_power=cfg_power,
            no_cfg_fraction=no_cfg_fraction,
            radii=tuple(radii),
            max_bank_bytes=max_bank_bytes,
            model=config.to_dict(),
            statistics="global_dp_batch",
            bank_scope="rank_local",
        )
        declarations = engine.parallel.world.gather_objects(self.settings)
        if any(value != self.settings for value in declarations):
            raise ValueError("Drifting rank configurations must agree")
        shape = (config.out_channels, config.input_size, config.input_size)
        total_bytes = (classes * positive_capacity + negative_capacity) * math.prod(shape) * 4
        if total_bytes > max_bank_bytes:
            raise ValueError("Combined Drifting banks exceed their explicit byte budget")
        self.positive = ClassMemoryBank(classes, positive_capacity, shape, max_bytes=max_bank_bytes)
        self.negative = ClassMemoryBank(1, negative_capacity, shape, max_bytes=max_bank_bytes)
        self.features = engine.add_role("drifting_features", features, trainable=False)
        self.rng = torch.Generator().manual_seed(seed + engine.parallel.world.rank)
        self.updates, self._incomplete = 0, False
        engine.register_state("drifting_method", self)

    def _preflight(self, batches):
        error, batch = None, None
        try:
            if self._incomplete:
                raise RuntimeError("Restore the last complete Drifting checkpoint before retry")
            values = list(batches)
            if len(values) != 1:
                raise ValueError("One complete force batch is required")
            batch = values[0]
            if set(batch) - {"samples", "labels", "cfg", "noise", "noise_labels"}:
                raise ValueError("Unsupported Drifting batch fields")
            samples, labels = batch["samples"], batch["labels"]
            self.positive._labels(labels)
            if (
                samples.shape != (len(labels), *self.positive.sample_shape)
                or samples.dtype != torch.float32
                or samples.device.type != "cpu"
                or not torch.isfinite(samples).all()
            ):
                raise ValueError("Supply aligned finite CPU FP32 observed samples and labels")
            b, s, c = len(labels), self.settings, self.engine.model.config
            if "cfg" in batch and (
                batch["cfg"].shape != (b,)
                or not torch.isfinite(batch["cfg"]).all()
                or (batch["cfg"] < 1).any()
                or batch["cfg"].device.type != "cpu"
            ):
                raise ValueError("Explicit CFG must be finite CPU B values >=1")
            if "noise" in batch and (
                batch["noise"].shape
                != (b * s["generated_samples"], c.in_channels, c.input_size, c.input_size)
                or batch["noise"].dtype != torch.float32
                or batch["noise"].device != self.engine.device
                or not torch.isfinite(batch["noise"]).all()
            ):
                raise ValueError(
                    "Explicit noise must match grouped generator inputs on Trainer device"
                )
            if "noise_labels" in batch and (
                not c.noise_classes
                or batch["noise_labels"].shape != (b * s["generated_samples"], c.noise_coords)
                or batch["noise_labels"].dtype != torch.int64
                or batch["noise_labels"].device != self.engine.device
                or (batch["noise_labels"] < 0).any()
                or (batch["noise_labels"] >= c.noise_classes).any()
            ):
                raise ValueError("Invalid explicit discrete noise")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.engine.parallel.world.gather_objects(error)
        if any(errors):
            raise ValueError("Drifting collective preflight failed: " + str(errors))
        return batch

    def update(self, microbatches):
        batch = self._preflight(microbatches)
        self._incomplete = True
        s, c, device = self.settings, self.engine.model.config, self.engine.device
        labels, samples = batch["labels"], batch["samples"]
        self.positive.add(samples, labels)
        self.negative.add(samples, torch.zeros_like(labels))
        positive = self.positive.sample(labels, s["positive_samples"], generator=self.rng).to(
            device
        )
        negative = self.negative.sample(
            torch.zeros_like(labels), s["negative_samples"], generator=self.rng
        ).to(device)
        cfg = batch.get("cfg")
        if cfg is None:
            cfg = sample_training_cfg(
                len(labels),
                minimum=s["cfg_min"],
                maximum=s["cfg_max"],
                power=s["cfg_power"],
                no_cfg_fraction=s["no_cfg_fraction"],
                generator=self.rng,
            )
        cfg = cfg.to(device)
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn(
                len(labels) * s["generated_samples"],
                c.in_channels,
                c.input_size,
                c.input_size,
                device=device,
            )
        condition = labels.to(device).repeat_interleave(s["generated_samples"])
        if c.noise_classes:
            noise_labels = batch.get("noise_labels")
            if noise_labels is None:
                noise_labels = torch.randint(
                    c.noise_classes, (len(noise), c.noise_coords), device=device
                )
            condition = dict(labels=condition, noise_labels=noise_labels)

        self.features.eval()
        error, fixed = None, None
        try:
            with torch.no_grad(), self.engine._autocast():
                fixed = self.features(torch.cat((positive, negative), 1).flatten(0, 1))
        except Exception as exc:
            error = str(exc)
        self.engine._collective_error(error)
        prepared = dict(
            noise=noise, condition=condition, cfg=cfg, features=fixed, groups=len(labels)
        )

        def objective(model, data):
            output = model(
                data["noise"],
                data["cfg"].repeat_interleave(s["generated_samples"]),
                data["condition"],
            )
            if not isinstance(output, FieldOutput) or output.prediction_type != "x0":
                raise ValueError(
                    "Drifting generator must produce direct samples, not velocity/noise"
                )
            error, generated, schema = None, None, None
            try:
                generated = self.features(output.prediction)
                if (
                    not isinstance(generated, dict)
                    or not generated
                    or generated.keys() != data["features"].keys()
                ):
                    raise ValueError("Generator and fixed feature keys must agree")
                for key, value in generated.items():
                    fixed = data["features"][key]
                    if (
                        value.ndim != 3
                        or fixed.ndim != 3
                        or value.shape[1:] != fixed.shape[1:]
                        or len(value) != data["groups"] * s["generated_samples"]
                        or len(fixed)
                        != data["groups"] * (s["positive_samples"] + s["negative_samples"])
                        or min(value.shape) < 1
                        or not torch.isfinite(value).all()
                        or not torch.isfinite(fixed).all()
                    ):
                        raise ValueError("Features must be finite aligned BFD tensors")
                schema = [(key, tuple(value.shape[1:])) for key, value in sorted(generated.items())]
            except Exception as exc:
                error = str(exc)

            self.engine._collective_error(error)
            schemas = self.engine.parallel.dp.gather_objects(schema)
            if (
                any(value != schema for value in schemas)
                or generated.keys() != data["features"].keys()
            ):
                raise ValueError("All Drifting ranks/branches need the same named feature schema")
            terms = []
            for name in sorted(generated):
                gen, fixed = generated[name], data["features"][name]
                b, g, p, n = (
                    data["groups"],
                    s["generated_samples"],
                    s["positive_samples"],
                    s["negative_samples"],
                )
                positions, width = gen.shape[1:]
                gen = (
                    gen.reshape(b, g, positions, width)
                    .permute(0, 2, 1, 3)
                    .reshape(b * positions, g, width)
                )
                fixed = (
                    fixed.reshape(b, p + n, positions, width)
                    .permute(0, 2, 1, 3)
                    .reshape(b * positions, p + n, width)
                )
                weight = (
                    ((data["cfg"] - 1) * (g - 1) / n)
                    .repeat_interleave(positions)[:, None]
                    .expand(-1, n)
                )
                term, _ = drifting_loss(
                    gen,
                    fixed[:, :p],
                    fixed[:, p:],
                    radii=s["radii"],
                    negative_weights=weight,
                    statistics_group=self.engine.parallel.dp,
                )
                terms.append(LossTerm(term.numerator, term.denominator, "feature_group", name))
            return LossBundle(tuple(terms))

        result = self.engine.phase(
            "drifting",
            objective=objective,
            microbatches=[prepared],
            freeze_roles=("drifting_features",),
        )
        if not result.updated:
            raise RuntimeError(
                "Drifting update failed; restore the last complete checkpoint with its banks"
            )
        self.updates += 1
        self._incomplete = False
        return result

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint incomplete Drifting bank/optimizer state")
        return dict(
            settings=deepcopy(self.settings),
            positive=self.positive.state_dict(),
            negative=self.negative.state_dict(),
            rng=self.rng.get_state().clone(),
            updates=self.updates,
        )

    def load_state_dict(self, state):
        if (
            state["settings"] != self.settings
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("Drifting checkpoint method/feature identity mismatch")
        self.positive.validate_state(state["positive"])
        self.negative.validate_state(state["negative"])
        check = torch.Generator()
        check.set_state(state["rng"])
        self.positive.load_state_dict(state["positive"])
        self.negative.load_state_dict(state["negative"])
        self.rng.set_state(state["rng"])
        self.updates = state["updates"]
        self._incomplete = False
