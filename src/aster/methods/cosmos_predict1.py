"""Cosmos-Predict1 Kendall/EDM training and its Euler sampling convention."""

import math
import torch
from torch import nn

from aster.core import LossTerm
from .generation import edm_denoise, expand, karras_sigmas


class CosmosPredict1Objective(nn.Module):
    def __init__(
        self,
        *,
        sigma_data=0.5,
        log_mean=0.0,
        log_std=1.0,
        loss_add_logvar=True,
        loss_reduce="sum",
        loss_scale=1.0,
    ):
        super().__init__()
        if (
            any(not math.isfinite(value) for value in (sigma_data, log_mean, log_std, loss_scale))
            or sigma_data <= 0
            or log_std <= 0
            or loss_scale <= 0
        ):
            raise ValueError("Invalid Predict1 noise/loss parameters")
        if type(loss_add_logvar) is not bool or loss_reduce not in {"sum", "mean"}:
            raise ValueError("Invalid Predict1 loss reduction/logvar choice")
        self.sigma_data, self.log_mean, self.log_std = sigma_data, log_mean, log_std
        self.loss_add_logvar, self.loss_reduce, self.loss_scale = (
            loss_add_logvar,
            loss_reduce,
            loss_scale,
        )

    def config_dict(self):
        return dict(
            type="cosmos_predict1",
            sigma_data=self.sigma_data,
            log_mean=self.log_mean,
            log_std=self.log_std,
            loss_add_logvar=self.loss_add_logvar,
            loss_reduce=self.loss_reduce,
            loss_scale=self.loss_scale,
        )

    def forward(self, model, batch):
        clean = batch["sample"]
        b = len(clean)
        sigma = batch.get("sigma")

        if sigma is None:
            sigma = (torch.randn(b, device=clean.device) * self.log_std + self.log_mean).exp()
        if (
            sigma.shape != (b,)
            or sigma.device != clean.device
            or not torch.isfinite(sigma).all()
            or (sigma <= 0).any()
        ):
            raise ValueError("Predict1 sigma must be finite positive [B]")
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape or noise.device != clean.device:
            raise ValueError("Predict1 noise shape/device mismatch")
        predicted = edm_denoise(
            model,
            clean + expand(sigma, clean) * noise,
            sigma,
            batch["condition"],
            sigma_data=self.sigma_data,
        )
        weights = batch.get("sample_weight", torch.ones_like(sigma))
        mask = batch.get("loss_mask", torch.ones_like(clean))
        if (
            weights.shape != (b,)
            or weights.device != clean.device
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("Predict1 sample weights must be finite nonnegative [B]")
        if (
            mask.shape != clean.shape
            or mask.device != clean.device
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError("Predict1 weighted loss mask must exactly match latent shape")
        sigma_weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        errors = (predicted.float() - clean.float()).square() * mask
        errors = (errors * expand(weights * sigma_weight, clean)).flatten(1)
        if self.loss_add_logvar:
            if not hasattr(model, "predict_logvar"):
                raise ValueError("Predict1 Kendall loss requires explicit net+logvar model")
            logvar = model.predict_logvar((sigma.log() / 4).to(clean.dtype)).float()
            if logvar.shape != (b,):
                raise ValueError("Predict1 logvar must return one scalar per sample")

            errors = errors * torch.exp(-logvar[:, None]) + logvar[:, None]
        per_sample = errors.sum(-1) if self.loss_reduce == "sum" else errors.mean(-1)
        return LossTerm(
            per_sample.sum() * self.loss_scale,
            torch.tensor(b, dtype=torch.int64, device=clean.device),
            "sample",
            "cosmos_predict1",
        )


@torch.no_grad()
def sample_cosmos_predict1(
    model,
    noise,
    condition,
    *,
    negative_condition=None,
    steps=35,
    guidance=1.5,
    sigma_data=0.5,
    sigma_min=0.0002,
    sigma_max=80.0,
    rho=7.0,
):

    if (
        type(steps) is not int
        or steps < 2
        or any(not math.isfinite(x) for x in (guidance, sigma_data, sigma_min, sigma_max, rho))
        or sigma_data <= 0
    ):
        raise ValueError("Invalid Predict1 Euler schedule/guidance")
    if not noise.is_floating_point() or noise.shape[0] < 1:
        raise ValueError("Predict1 Euler requires floating unit noise")
    if negative_condition is None and guidance != 0:
        raise ValueError("Predict1 extra guidance requires explicit negative_condition")
    sigmas = karras_sigmas(
        steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=noise.device
    ).float()
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        x = noise * math.sqrt(sigma_max**2 + 1)
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            denominator = sigma.square() + sigma_data**2
            scaled = x / denominator.sqrt()
            time = (sigma.log() / 4).to(x.dtype).expand(len(x))
            positive = model(scaled, time, condition)
            if positive.prediction_type != "edm_residual" or positive.prediction.shape != x.shape:
                raise ValueError("Predict1 Euler expects same-shape edm_residual output")
            prediction = positive.prediction
            if negative_condition is not None:
                negative = model(scaled, time, negative_condition)
                if (
                    negative.prediction_type != "edm_residual"
                    or negative.prediction.shape != x.shape
                ):
                    raise ValueError(
                        "Predict1 CFG residuals must have identical parameterization/shape"
                    )
                prediction = prediction + guidance * (prediction - negative.prediction)

            working = x.float()
            denoised = (
                sigma_data**2 / denominator * working
                + sigma * sigma_data / denominator.sqrt() * prediction
            )
            x = (working + (next_sigma - sigma) * (working - denoised) / sigma).to(prediction.dtype)
        return x
    finally:
        for module, mode in modes.items():
            module.training = mode
