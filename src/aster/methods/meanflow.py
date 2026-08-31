"""MeanFlow training paths, guidance, exact directional derivatives, and adaptive weighting."""

import math
import torch
from torch import nn
from torch.autograd import forward_ad

from ..core import LossTerm
from ..models.interval_dit import IntervalDiTConfig
from .generation import expand


def sample_meanflow_times(
    size,
    *,
    device="cpu",
    distribution="logit_normal",
    mean=-0.4,
    std=1.0,
    diagonal_fraction=0.75,
    generator=None,
):
    if (
        type(size) is not int
        or size < 1
        or distribution not in {"uniform", "logit_normal"}
        or not all(math.isfinite(x) for x in (mean, std, diagonal_fraction))
        or std <= 0
        or not 0 <= diagonal_fraction <= 1
    ):
        raise ValueError("Invalid MeanFlow time distribution")
    draw = (
        (lambda: torch.rand(size, device=device, generator=generator))
        if distribution == "uniform"
        else (
            lambda: (torch.randn(size, device=device, generator=generator) * std + mean).sigmoid()
        )
    )
    first, second = draw(), draw()
    t, r = torch.maximum(first, second), torch.minimum(first, second)

    r[: int(size * diagonal_fraction)] = t[: int(size * diagonal_fraction)]
    return t, r


def meanflow_directional_derivative(model, noisy, time, duration, velocity, condition):
    """Use exact forward-mode automatic differentiation rather than finite differences.
    no_grad disables reverse-mode graph recording, not the forward directional derivative."""
    with torch.no_grad(), forward_ad.dual_level():
        x = forward_ad.make_dual(noisy, velocity)
        t = forward_ad.make_dual(time, torch.ones_like(time))
        h = forward_ad.make_dual(duration, torch.ones_like(duration))
        result = model(x, t, h, condition)
        if result.prediction_type != "average_velocity":
            raise ValueError("MeanFlow requires the interval average-velocity parameterization")
        _, tangent = forward_ad.unpack_dual(result.prediction)
        if tangent is None:
            tangent = torch.zeros_like(noisy)
        return tangent.detach()


class MeanFlowObjective(nn.Module):
    def __init__(
        self,
        *,
        distribution="logit_normal",
        time_mean=-0.4,
        time_std=1.0,
        diagonal_fraction=0.75,
        guidance=True,
        omega=1.0,
        kappa=0.5,
        guidance_start=0.0,
        guidance_end=1.0,
        class_dropout=0.1,
        norm_power=1.0,
        norm_epsilon=0.01,
    ):
        super().__init__()
        sample_meanflow_times(
            1,
            distribution=distribution,
            mean=time_mean,
            std=time_std,
            diagonal_fraction=diagonal_fraction,
            generator=torch.Generator().manual_seed(0),
        )
        if (
            not all(
                math.isfinite(x)
                for x in (
                    omega,
                    kappa,
                    guidance_start,
                    guidance_end,
                    class_dropout,
                    norm_power,
                    norm_epsilon,
                )
            )
            or type(guidance) is not bool
            or omega < 0
            or kappa < 0
            or not 0 <= guidance_start <= guidance_end <= 1
            or not 0 <= class_dropout <= 1
            or norm_power < 0
            or norm_epsilon <= 0
        ):
            raise ValueError("Invalid MeanFlow guidance/weighting configuration")
        self.options = dict(
            distribution=distribution,
            time_mean=time_mean,
            time_std=time_std,
            diagonal_fraction=diagonal_fraction,
            guidance=guidance,
            omega=omega,
            kappa=kappa,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
            class_dropout=class_dropout,
            norm_power=norm_power,
            norm_epsilon=norm_epsilon,
        )

    def config_dict(self):
        return dict(
            type="meanflow",
            **self.options,
            time_direction="data_to_noise",
            jvp="native_forward_ad_split_primal",
        )

    def _validate(self, model, batch):
        c = model.config
        if not isinstance(c, IntervalDiTConfig) or c.variant != "meanflow":
            raise ValueError("MeanFlow needs its explicit duration-conditioned DiT")
        if set(batch) - {"sample", "labels", "noise", "time", "reference_time", "drop_count"}:
            raise ValueError("Unknown MeanFlow training fields")
        clean, labels = batch["sample"], batch["labels"]
        if (
            not isinstance(clean, torch.Tensor)
            or clean.ndim != 4
            or not len(clean)
            or tuple(clean.shape[1:]) != (c.in_channels, c.input_size, c.input_size)
            or not clean.is_floating_point()
            or not torch.isfinite(clean).all()
        ):
            raise ValueError("MeanFlow samples must be finite BCHW matching the model")
        b = len(clean)
        if (
            not isinstance(labels, torch.Tensor)
            or labels.shape != (b,)
            or labels.dtype != torch.int64
            or labels.device != clean.device
            or (labels < 0).any()
            or (labels >= c.num_classes).any()
        ):
            raise ValueError("MeanFlow requires aligned genuine int64 class labels")
        if ("time" in batch) != ("reference_time" in batch):
            raise ValueError("Supply both time and reference_time, or neither")
        if "time" in batch:
            t, r = batch["time"], batch["reference_time"]
            if (
                any(
                    not isinstance(v, torch.Tensor)
                    or v.shape != (b,)
                    or v.device != clean.device
                    or not v.is_floating_point()
                    or not torch.isfinite(v).all()
                    for v in (t, r)
                )
                or (r < 0).any()
                or (r > t).any()
                or (t > 1).any()
            ):
                raise ValueError("MeanFlow requires explicit 0 <= r <= t <= 1")
        if "noise" in batch:
            noise = batch["noise"]
            if (
                not isinstance(noise, torch.Tensor)
                or noise.shape != clean.shape
                or noise.dtype != clean.dtype
                or noise.device != clean.device
                or not torch.isfinite(noise).all()
            ):
                raise ValueError(
                    "MeanFlow noise must align sample dtype/device/shape and be finite"
                )
        if "drop_count" in batch and (
            type(batch["drop_count"]) is not int or not 0 <= batch["drop_count"] <= b
        ):
            raise ValueError("MeanFlow drop_count must be an integer in [0,B]")

    def preflight_microbatches(self, model, batches):

        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        c, o = model.config, self.options
        clean, labels = batch["sample"], batch["labels"]
        b = len(clean)
        if "time" in batch:
            t, r = batch["time"], batch["reference_time"]
        else:
            t, r = sample_meanflow_times(
                b,
                device=clean.device,
                distribution=o["distribution"],
                mean=o["time_mean"],
                std=o["time_std"],
                diagonal_fraction=o["diagonal_fraction"],
            )
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
        z, velocity = (1 - expand(t, clean)) * clean + expand(t, clean) * noise, noise - clean
        with torch.no_grad():
            guided = velocity
            if o["guidance"]:
                null = torch.full_like(labels, c.num_classes)
                unconditional = model(z, t, torch.zeros_like(t), null).prediction
                mask = (t >= o["guidance_start"]) & (t <= o["guidance_end"])
                omega = expand(torch.where(mask, o["omega"], 1.0), clean)
                kappa = expand(torch.where(mask, o["kappa"], 0.0), clean)
                conditional = (
                    model(z, t, torch.zeros_like(t), labels).prediction
                    if o["kappa"]
                    else torch.zeros_like(velocity)
                )
                guided = (
                    omega * velocity + (1 - omega - kappa) * unconditional + kappa * conditional
                )
            drop = batch.get("drop_count")
            if drop is None:
                drop = int((torch.rand(b, device=clean.device) < o["class_dropout"]).sum())
            active_labels = labels.clone()
            active_labels[:drop] = c.num_classes
            guided = guided.clone()
            guided[:drop] = velocity[:drop]
            tangent = meanflow_directional_derivative(model, z, t, t - r, guided, active_labels)
            target = guided - expand(t - r, guided) * tangent
        prediction = model(z, t, t - r, active_labels).prediction
        squared = (prediction.float() - target.float()).square().flatten(1).sum(1)
        weighted = squared / (squared.detach() + o["norm_epsilon"]).pow(o["norm_power"])
        return LossTerm(
            weighted.sum(),
            torch.tensor(b, device=clean.device, dtype=torch.int64),
            "sample",
            "meanflow",
        )


@torch.no_grad()
def sample_meanflow(model, noise, *, labels=None, timesteps=(1.0, 0.0)):
    c = model.config
    if not isinstance(c, IntervalDiTConfig) or c.variant != "meanflow":
        raise ValueError("MeanFlow sampler requires a duration-conditioned model")
    times = tuple(float(x) for x in timesteps)
    if (
        len(times) < 2
        or times[0] != 1
        or times[-1] != 0
        or any(not math.isfinite(x) for x in times)
        or any(a <= b for a, b in zip(times, times[1:]))
    ):
        raise ValueError("MeanFlow schedule must decrease strictly from noise=1 to data=0")
    mode = model.training
    try:
        model.eval()
        current = noise.clone()
        for t, r in zip(times, times[1:]):
            time = current.new_full((len(current),), t)
            duration = current.new_full((len(current),), t - r)
            output = model(current, time, duration, labels)
            if output.prediction_type != "average_velocity":
                raise ValueError("Wrong MeanFlow prediction parameterization")
            current = current - (t - r) * output.prediction
        return current
    finally:
        model.train(mode)
