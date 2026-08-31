"""Metric formulas; comparable benchmarks also require identical data and feature protocols."""

import math
import re
import string
import numpy as np
import torch
import torch.nn.functional as F


def perplexity(nll_sum, token_count):
    if token_count <= 0 or not math.isfinite(nll_sum):
        raise ValueError("Perplexity requires finite NLL and positive token count")
    return math.exp(nll_sum / token_count)


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction, references, *, normalize=False):
    if not references:
        raise ValueError("No reference answers")
    transform = normalize_answer if normalize else lambda text: text
    return float(any(transform(prediction) == transform(reference) for reference in references))


def token_f1(prediction, reference):
    from collections import Counter

    predicted, truth = normalize_answer(prediction).split(), normalize_answer(reference).split()
    overlap = sum((Counter(predicted) & Counter(truth)).values())
    if not predicted or not truth:
        return float(predicted == truth)
    return 2 * overlap / (len(predicted) + len(truth))


def pass_at_k(total, correct, k):
    if not 0 <= correct <= total or not 1 <= k <= total:
        raise ValueError("Need 0<=c<=n and 1<=k<=n")
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod((total - correct - i) / (total - i) for i in range(k))


def word_error_rate(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        raise ValueError("WER denominator is undefined for empty references")
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        current = [i]
        for j, predicted in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (token != predicted)))
        row = current
    return row[-1] / len(ref)


def _features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need >=2 finite feature vectors")
    return values


def frechet_distance(real_features, generated_features):
    """Compute the Frechet metric for declared features; different extractors define
    different, non-interchangeable FID/FVD protocols."""
    real, generated = _features(real_features), _features(generated_features)
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Feature dimensions differ")
    mean_r, mean_g = real.mean(0), generated.mean(0)
    cov_r, cov_g = (
        np.atleast_2d(np.cov(real, rowvar=False)),
        np.atleast_2d(np.cov(generated, rowvar=False)),
    )
    values, vectors = np.linalg.eigh((cov_r + cov_r.T) / 2)
    root_r = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    product = root_r @ cov_g @ root_r
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    result = np.square(mean_r - mean_g).sum() + np.trace(cov_r) + np.trace(cov_g) - 2 * trace_root
    if result < -1e-6:
        raise FloatingPointError("Numerically invalid covariance distance")
    return max(float(result), 0.0)


def kernel_inception_distance(real_features, generated_features):
    """Unbiased cubic-polynomial MMD; finite-sample estimates may legitimately be negative."""
    real, generated = _features(real_features), _features(generated_features)
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Feature dimensions differ")
    kernel = lambda left, right: (left @ right.T / real.shape[1] + 1) ** 3
    rr, gg, rg = kernel(real, real), kernel(generated, generated), kernel(real, generated)
    n, m = len(real), len(generated)
    return float(
        (rr.sum() - np.trace(rr)) / (n * (n - 1))
        + (gg.sum() - np.trace(gg)) / (m * (m - 1))
        - 2 * rg.mean()
    )


def psnr(reference, prediction, *, data_range=1.0):
    if reference.shape != prediction.shape or reference.ndim < 2 or data_range <= 0:
        raise ValueError("Invalid PSNR inputs")
    mse = (reference.double() - prediction.double()).square().flatten(1).mean(1)

    return 10 * torch.log10(data_range**2 / mse)


def ssim(reference, prediction, *, data_range=1.0, window=11, sigma=1.5):
    if (
        reference.shape != prediction.shape
        or reference.ndim != 4
        or min(reference.shape[-2:]) < window
        or window % 2 != 1
        or data_range <= 0
    ):
        raise ValueError("SSIM requires BCHW images and an odd valid window")
    coords = torch.arange(window, device=reference.device, dtype=torch.float64) - (window - 1) / 2
    weights = (-coords.square() / (2 * sigma**2)).exp()
    weights /= weights.sum()
    kernel = (weights[:, None] * weights[None]).expand(reference.shape[1], 1, window, window)
    blur = lambda x: F.conv2d(x.double(), kernel, groups=x.shape[1])
    x, y = reference.double(), prediction.double()
    mx, my = blur(x), blur(y)
    vx, vy, cov = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return (
        (
            ((2 * mx * my + c1) * (2 * cov + c2))
            / ((mx.square() + my.square() + c1) * (vx + vy + c2))
        )
        .flatten(1)
        .mean(1)
    )


def interquartile_mean(values):
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite evaluation returns")

    left, right = 0.25 * len(values), 0.75 * len(values)
    weights = np.maximum(
        0, np.minimum(np.arange(len(values)) + 1, right) - np.maximum(np.arange(len(values)), left)
    )
    return float(np.dot(weights, values) / weights.sum())


def serving_window(results, *, start, end, ttft_slo=None, tpot_slo=None):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid measurement window")
    admitted = [result for result in results if start <= result.received_at < end]
    completed = [
        result
        for result in admitted
        if result.error_code is None
        and result.stop_reason not in {"cancelled", "error", "timeout"}
        and result.finished_at <= end
    ]
    good = []
    for result in completed:
        metrics = result.metrics()
        if ttft_slo is not None and (
            metrics["ttft_seconds"] is None or metrics["ttft_seconds"] > ttft_slo
        ):
            continue
        if tpot_slo is not None and (
            metrics["tpot_seconds"] is None or metrics["tpot_seconds"] > tpot_slo
        ):
            continue
        good.append(result)
    duration = end - start
    return {
        "admitted": len(admitted),
        "completed": len(completed),
        "failed_or_incomplete": len(admitted) - len(completed),
        "window_seconds": duration,
        "throughput_requests_per_second": len(completed) / duration,
        "throughput_tokens_per_second": sum(len(result.token_ids) for result in completed)
        / duration,
        "goodput_requests_per_second": len(good) / duration,
    }
