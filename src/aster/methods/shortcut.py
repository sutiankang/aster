"""Shortcut step stratification, two-half-step bootstrap, flow anchors, and EMA targets."""

from copy import deepcopy
import math
import torch

from ..core import FieldOutput, LossTerm
from ..models.interval_dit import IntervalDiT, IntervalDiTConfig
from .generation import expand, mean_flat


def shortcut_levels(count, levels, bias=0):
    """Construct deterministic step strata once for the full DP/accumulation window,
    then distribute each rank's slice."""
    if (
        type(count) is not int
        or count < 1
        or type(levels) is not int
        or levels < 1
        or bias not in (0, 1)
    ):
        raise ValueError("Invalid Shortcut level stratification")
    if bias == 0:
        result = torch.arange(levels - 1, -1, -1).repeat_interleave(count // levels)
        guided_count = count // levels
    else:
        if levels < 3:
            raise ValueError("Biased Shortcut stratification requires at least 8 base steps")
        repeat = (count // 2) // levels
        result = torch.cat(
            (
                torch.arange(levels - 1, 1, -1).repeat_interleave(repeat),
                torch.ones(count // 4, dtype=torch.int64),
                torch.zeros(count // 4, dtype=torch.int64),
            )
        )
        guided_count = repeat
    return torch.cat((result, torch.zeros(count - len(result), dtype=torch.int64))), guided_count


@torch.no_grad()
def shortcut_bootstrap_target(
    model, x, t, levels, labels, *, guidance_mask=None, guidance_scale=1.0
):
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("Invalid Shortcut bootstrap guidance scale")
    delta = torch.pow(2.0, -levels.float()) / 2
    if (
        x.ndim != 4
        or levels.shape != (len(x),)
        or t.shape != levels.shape
        or (levels < 0).any()
        or not torch.isfinite(t).all()
        or (t < 0).any()
        or (t + 2 * delta > 1 + 1e-7).any()
    ):
        raise ValueError("Shortcut bootstrap interval must lie within [0,1]")

    def velocity(value, time):
        if guidance_mask is not None and (
            guidance_mask.shape != levels.shape or guidance_mask.dtype != torch.bool
        ):
            raise ValueError("Bootstrap guidance mask must align with groups")
        if guidance_mask is None:
            output = model(value, time, levels.float() + 1, labels)
        else:
            null = torch.full_like(labels[guidance_mask], model.config.num_classes)
            output = model(
                torch.cat((value, value[guidance_mask])),
                torch.cat((time, time[guidance_mask])),
                torch.cat((levels.float() + 1, levels[guidance_mask].float() + 1)),
                torch.cat((labels, null)),
            )
        if not isinstance(output, FieldOutput) or output.prediction_type != "average_velocity":
            raise ValueError("Shortcut target requires a step-conditioned average velocity")
        if guidance_mask is None:
            return output.prediction
        conditional, unconditional = (
            output.prediction[: len(value)].clone(),
            output.prediction[len(value) :],
        )
        conditional[guidance_mask] = unconditional + guidance_scale * (
            conditional[guidance_mask] - unconditional
        )
        return conditional

    first = velocity(x, t)
    middle = (x + expand(delta, x) * first).clamp(-4, 4)
    second = velocity(middle, t + delta)
    return ((first + second) / 2).clamp(-4, 4)


class ShortcutMethod:
    def __init__(
        self,
        engine,
        *,
        base_steps=128,
        bootstrap_every=8,
        bootstrap_ema=True,
        ema_decay=0.999,
        bootstrap_cfg=False,
        cfg_scale=1.5,
        class_dropout=0.1,
        dt_bias=0,
    ):
        self.engine = engine
        c = engine.model.config
        if not isinstance(c, IntervalDiTConfig) or c.variant != "shortcut":
            raise ValueError("Shortcut requires its log2-step-conditioned DiT")
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
            raise ValueError("Shortcut currently admits DP/ZeRO only")
        if (
            type(base_steps) is not int
            or base_steps < 2
            or base_steps & (base_steps - 1)
            or type(bootstrap_every) is not int
            or bootstrap_every < 2
            or type(bootstrap_ema) is not bool
            or type(bootstrap_cfg) is not bool
            or not all(math.isfinite(x) for x in (ema_decay, cfg_scale, class_dropout))
            or not 0 <= ema_decay < 1
            or cfg_scale < 0
            or not 0 <= class_dropout <= 1
        ):
            raise ValueError("Invalid Shortcut lifecycle/sampling configuration")
        self.levels = int(math.log2(base_steps))
        shortcut_levels(1, self.levels, dt_bias)
        self.settings = dict(
            base_steps=base_steps,
            bootstrap_every=bootstrap_every,
            bootstrap_ema=bootstrap_ema,
            ema_decay=ema_decay,
            bootstrap_cfg=bootstrap_cfg,
            cfg_scale=cfg_scale,
            class_dropout=class_dropout,
            dt_bias=dt_bias,
            model=c.to_dict(),
            strata_scope="global_dp_accumulation_window",
            data_scope="rank_microbatch_shuffle",
            endpoint_sigma=1e-5,
            target_clip=4.0,
        )
        declarations = engine.parallel.world.gather_objects(self.settings)
        if any(value != self.settings for value in declarations):
            raise ValueError("Shortcut rank configurations must agree")
        self.target = None
        if bootstrap_ema:
            with torch.random.fork_rng(devices=[]):
                target = IntervalDiT(c)
            target.load_state_dict(engine.export_state_dict(only_rank_zero=False))
            self.target = engine.add_role("shortcut_target", target, trainable=False)
        self.updates, self._incomplete = 0, False
        engine.register_state("shortcut_method", self)

    def _preflight(self, microbatches):
        error, batches = None, None
        try:
            if self._incomplete:
                raise RuntimeError("Restore the last complete Shortcut checkpoint")
            batches = list(microbatches)
            if len(batches) != self.engine.accumulation_steps:
                raise ValueError("Shortcut window length must match accumulation_steps")
            c = self.engine.model.config
            for batch in batches:
                if set(batch) != {"sample", "labels"}:
                    raise ValueError("Shortcut batches require exactly sample and labels")
                sample, labels = batch["sample"], batch["labels"]
                if (
                    sample.ndim != 4
                    or tuple(sample.shape[1:]) != (c.in_channels, c.input_size, c.input_size)
                    or sample.device != self.engine.device
                    or sample.dtype != torch.float32
                    or not torch.isfinite(sample).all()
                    or len(sample) < self.settings["bootstrap_every"]
                    or len(sample) % self.settings["bootstrap_every"]
                ):
                    raise ValueError(
                        "Shortcut needs finite FP32 images and microbatch size divisible by bootstrap_every"
                    )
                if (
                    labels.shape != (len(sample),)
                    or labels.dtype != torch.int64
                    or labels.device != sample.device
                    or (labels < 0).any()
                    or (labels >= c.num_classes).any()
                ):
                    raise ValueError("Shortcut needs genuine aligned int64 class labels")
        except Exception as exc:
            error = str(exc)
        self.engine._collective_error(error)
        return batches

    def update(self, microbatches):
        batches = self._preflight(microbatches)
        s, device = self.settings, self.engine.device
        local_counts = [len(batch["sample"]) // s["bootstrap_every"] for batch in batches]
        counts = self.engine.parallel.dp.gather_objects(local_counts)
        all_levels, guided_count = shortcut_levels(
            sum(sum(v) for v in counts), self.levels, s["dt_bias"]
        )
        offset = sum(sum(v) for v in counts[: self.engine.parallel.dp.rank])
        target_model = self.engine.model if self.target is None else self.target
        previous_mode = target_model.training
        self._incomplete = True
        prepared = []
        try:
            target_model.eval()
            for batch, boot_count in zip(batches, local_counts):
                samples, labels = batch["sample"], batch["labels"]
                order = torch.randperm(len(samples), device=device)
                samples, labels = samples[order], labels[order]
                if s["cfg_scale"] == 0:
                    labels = torch.full_like(labels, self.engine.model.config.num_classes)
                levels = all_levels[offset : offset + boot_count].to(device)
                selected_cfg = (
                    torch.arange(offset, offset + boot_count, device=device) < guided_count
                )
                offset += boot_count
                sections = torch.pow(2.0, levels.float())
                time = (torch.rand(boot_count, device=device) * sections).floor() / sections
                noise = torch.randn_like(samples[:boot_count])
                bootstrap_x = (1 - (1 - 1e-5) * expand(time, noise)) * noise + expand(
                    time, noise
                ) * samples[:boot_count]
                with torch.no_grad(), self.engine._autocast():
                    targets = shortcut_bootstrap_target(
                        target_model,
                        bootstrap_x,
                        time,
                        levels,
                        labels[:boot_count],
                        guidance_mask=selected_cfg if s["bootstrap_cfg"] else None,
                        guidance_scale=s["cfg_scale"],
                    )
                flow_time = (
                    torch.randint(s["base_steps"], (len(samples),), device=device).float()
                    / s["base_steps"]
                )
                noise = torch.randn_like(samples)
                flow_x = (1 - (1 - 1e-5) * expand(flow_time, noise)) * noise + expand(
                    flow_time, noise
                ) * samples
                flow_v = samples - (1 - 1e-5) * noise
                flow_labels = labels.masked_fill(
                    torch.rand(len(labels), device=device) < s["class_dropout"],
                    self.engine.model.config.num_classes,
                )
                flow_count = len(samples) - boot_count
                prepared.append(
                    dict(
                        sample=torch.cat((bootstrap_x, flow_x[:flow_count])),
                        target=torch.cat((targets, flow_v[:flow_count])),
                        time=torch.cat((time, flow_time[:flow_count])),
                        labels=torch.cat((labels[:boot_count], flow_labels[:flow_count])),
                        levels=torch.cat(
                            (
                                levels.float(),
                                torch.full((flow_count,), float(self.levels), device=device),
                            )
                        ),
                    )
                )
        finally:
            target_model.train(previous_mode)

        def objective(model, batch):
            predicted = model(batch["sample"], batch["time"], batch["levels"], batch["labels"])
            if predicted.prediction_type != "average_velocity":
                raise ValueError("Shortcut prediction parameterization changed")
            losses = mean_flat((predicted.prediction.float() - batch["target"].float()).square())
            return LossTerm(
                losses.sum(),
                torch.tensor(len(losses), device=device, dtype=torch.int64),
                "sample",
                "shortcut",
            )

        result = self.engine.phase(
            "shortcut",
            objective=objective,
            microbatches=prepared,
            freeze_roles=("shortcut_target",) if self.target is not None else (),
        )
        if not result.updated:
            raise RuntimeError("Shortcut optimizer did not update; restore complete checkpoint")
        if self.target is not None:
            state = self.engine.export_state_dict(only_rank_zero=False)
            with torch.no_grad():
                for name, target in self.target.state_dict().items():
                    if target.is_floating_point():
                        target.lerp_(state[name].to(target), 1 - s["ema_decay"])
                    else:
                        target.copy_(state[name].to(target))
        self.updates += 1
        self._incomplete = False
        return result

    def state_dict(self):
        if self._incomplete:
            raise RuntimeError("Cannot checkpoint an incomplete Shortcut/EMA update")
        return dict(settings=deepcopy(self.settings), updates=self.updates)

    def load_state_dict(self, state):
        if (
            state["settings"] != self.settings
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("Shortcut checkpoint algorithm/model identity mismatch")
        self.updates, self._incomplete = state["updates"], False


@torch.no_grad()
def sample_shortcut(model, noise, *, labels=None, steps=1, guidance_scale=1.0):
    if not isinstance(model.config, IntervalDiTConfig) or model.config.variant != "shortcut":
        raise ValueError("Shortcut sampler needs its log2-step model")
    if (
        type(steps) is not int
        or steps < 1
        or steps & (steps - 1)
        or not math.isfinite(guidance_scale)
        or guidance_scale < 0
    ):
        raise ValueError(
            "Shortcut inference steps must be a power of two; guidance must be finite >=0"
        )
    mode = model.training
    try:
        model.eval()
        current = noise.clone()
        levels = noise.new_full((len(noise),), float(math.log2(steps)))
        for index in range(steps):
            time = noise.new_full((len(noise),), index / steps)

            output = model(current, time, levels, None if guidance_scale == 0 else labels)
            if output.prediction_type != "average_velocity":
                raise ValueError("Shortcut inference requires average_velocity")
            velocity = output.prediction
            if guidance_scale not in (0, 1) and labels is not None:
                output = model(current, time, levels, None)
                if output.prediction_type != "average_velocity":
                    raise ValueError("Shortcut inference requires average_velocity")
                unconditional = output.prediction
                velocity = unconditional + guidance_scale * (velocity - unconditional)
            current = current + velocity / steps
        return current
    finally:
        model.train(mode)
