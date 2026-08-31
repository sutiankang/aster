"""Sampling transforms and rejection correction with distinct model and behavior probabilities."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    eos_token_ids: tuple[int, ...] = ()
    min_new_tokens: int = 0
    repetition_penalty: float = 1.0
    logit_bias: tuple[tuple[int, float], ...] = ()

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (self.max_new_tokens, self.top_k, self.seed, self.min_new_tokens)
        ):
            raise TypeError("Generation lengths, top-k and seed must be integers")
        if self.max_new_tokens < 1 or not 0 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("Invalid generation lengths")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        if self.top_k < 0 or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("Invalid top-k/top-p")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("Invalid repetition penalty")
        if any(type(token) is not int or token < 0 for token in self.eos_token_ids):
            raise ValueError("Negative EOS id")
        if any(token < 0 or not math.isfinite(bias) for token, bias in self.logit_bias):
            raise ValueError("Invalid logit bias")
        if len({token for token, _ in self.logit_bias}) != len(self.logit_bias):
            raise ValueError("Duplicate logit-bias keys")

    @property
    def transform_order(self):
        return (
            "logit_bias",
            "repetition_penalty",
            "min_length_eos_mask",
            "grammar_constraint",
            "temperature_or_greedy",
            "top_k",
            "top_p",
            "renormalize",
        )


@dataclass(frozen=True)
class SampledToken:
    token_id: int
    raw_model_logprob: float
    behavior_logprob: float


def distributions(logits, config, *, context_ids=(), generated_count=0, allowed_token_ids=None):

    raw = logits.detach().float().cpu()
    if (
        raw.ndim != 1
        or raw.numel() == 0
        or torch.isnan(raw).any()
        or torch.isposinf(raw).any()
        or not torch.isfinite(raw).any()
    ):
        raise ValueError("Logits must contain a finite categorical support")
    raw_logp = raw.log_softmax(-1)
    scores = raw.clone()
    for token, bias in config.logit_bias:
        if token >= len(scores):
            raise ValueError("Logit bias token outside vocabulary")
        scores[token] += bias
    for token in set(context_ids):
        if not 0 <= token < len(scores):
            raise ValueError("Context token outside vocabulary")
        scores[token] = (
            scores[token] * config.repetition_penalty
            if scores[token] < 0
            else scores[token] / config.repetition_penalty
        )
    if generated_count < config.min_new_tokens:
        for token in config.eos_token_ids:
            if token >= len(scores):
                raise ValueError("EOS outside vocabulary")
            scores[token] = -torch.inf
    if not torch.isfinite(scores).any():
        raise ValueError("Sampling constraints eliminated every token")
    if allowed_token_ids is not None:
        if not allowed_token_ids or any(
            type(token) is not int or not 0 <= token < len(scores) for token in allowed_token_ids
        ):
            raise ValueError("Grammar has invalid or empty next-token support")
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        allowed[list(allowed_token_ids)] = True
        scores[~allowed] = -torch.inf
        if not torch.isfinite(scores).any():
            raise ValueError("Model constraints conflict with grammar support")
    if config.temperature == 0:
        behavior = torch.full_like(scores, -torch.inf)
        behavior[int(scores.argmax())] = 0.0
        return raw_logp, behavior
    scores /= config.temperature
    if config.top_k:
        threshold = scores.topk(min(config.top_k, len(scores))).values[-1]
        scores[scores < threshold] = -torch.inf
    if config.top_p < 1:
        sorted_scores, indices = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(-1).cumsum(-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    return raw_logp, scores.log_softmax(-1)


def sample_token(
    logits, config, generator, *, context_ids=(), generated_count=0, allowed_token_ids=None
):
    raw, behavior = distributions(
        logits,
        config,
        context_ids=context_ids,
        generated_count=generated_count,
        allowed_token_ids=allowed_token_ids,
    )
    token = int(torch.multinomial(behavior.exp(), 1, generator=generator))
    if not torch.isfinite(raw[token]) or not torch.isfinite(behavior[token]):
        raise ValueError("Selected-token probability cannot be represented as finite logprob")
    return SampledToken(token, float(raw[token]), float(behavior[token]))


def speculative_accept(target_probs, draft_probs, draft_token, generator):

    p, q = target_probs.detach().double().cpu(), draft_probs.detach().double().cpu()
    if p.ndim != 1 or p.shape != q.shape or not 0 <= draft_token < p.numel():
        raise ValueError("Mismatched speculative distributions")
    for values in (p, q):
        if (
            not torch.isfinite(values).all()
            or (values < 0).any()
            or not torch.isclose(values.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-7)
        ):
            raise ValueError("Speculative inputs must be probability distributions")
    if q[draft_token] <= 0:
        raise ValueError("Draft token has zero proposal probability")
    if float(torch.rand((), generator=generator)) < min(
        1.0, float(p[draft_token] / q[draft_token])
    ):
        return draft_token, True
    residual = (p - q).clamp_min(0)
    if residual.sum() <= 0:
        raise RuntimeError("Invalid zero rejection mass")
    return int(torch.multinomial(residual / residual.sum(), 1, generator=generator)), False
