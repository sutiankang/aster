"""LLaDA mask-diffusion training and semi-autoregressive remasking."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm


class MaskedDiffusionObjective(nn.Module):
    def __init__(self, mask_token_id, *, epsilon=1e-3, sft=False, random_length_probability=0.01):
        super().__init__()
        if mask_token_id < 0 or not 0 < epsilon < 1 or not 0 <= random_length_probability <= 1:
            raise ValueError("Invalid masked diffusion settings")
        self.mask_token_id, self.epsilon, self.sft = mask_token_id, epsilon, sft
        self.random_length_probability = random_length_probability

    def config_dict(self):
        return {
            "type": "masked_diffusion",
            "mask_token_id": self.mask_token_id,
            "epsilon": self.epsilon,
            "sft": self.sft,
            "random_length_probability": self.random_length_probability,
        }

    def forward(self, model, batch):
        clean = batch["input_ids"]
        if clean.ndim != 2 or min(clean.shape) < 1 or (clean == self.mask_token_id).any():
            raise ValueError("Clean training text cannot contain the reserved diffusion MASK")
        valid = batch.get("attention_mask", torch.ones_like(clean, dtype=torch.bool)).bool()
        if valid.shape != clean.shape:
            raise ValueError("Attention padding must align tokens")
        eligible = valid.clone()
        if self.sft:
            prompt_lengths = batch["prompt_lengths"]
            if (
                prompt_lengths.shape != (len(clean),)
                or (prompt_lengths < 0).any()
                or (prompt_lengths >= clean.shape[1]).any()
            ):
                raise ValueError("SFT needs nonempty responses and explicit prompt boundaries")
            eligible &= (
                torch.arange(clean.shape[1], device=clean.device)[None] >= prompt_lengths[:, None]
            )
        elif self.random_length_probability and torch.rand(()) < self.random_length_probability:
            length = int(torch.randint(1, clean.shape[1] + 1, ()))
            clean, valid, eligible = clean[:, :length], valid[:, :length], eligible[:, :length]
        if (eligible.sum(-1) == 0).any():
            raise ValueError("Every masked-diffusion example needs eligible target positions")
        time = batch.get("time", None)
        if time is None:
            time = torch.rand(len(clean), device=clean.device)
        if time.shape != (len(clean),) or not ((0 <= time) & (time <= 1)).all():
            raise ValueError("Diffusion time must be a B vector in [0,1]")
        probability = self.epsilon + (1 - self.epsilon) * time
        masked = batch.get("masked_indices", None)
        if masked is None:
            masked = (
                torch.rand(clean.shape, device=clean.device) < probability[:, None]
            ) & eligible
        if masked.shape != clean.shape or masked.dtype != torch.bool or (masked & ~eligible).any():
            raise ValueError("Masked positions must be eligible response/text tokens")
        noisy = torch.where(masked, self.mask_token_id, clean)
        logits = model(input_ids=noisy, attention_mask=valid, use_cache=False).logits.float()
        if logits.shape[:2] != clean.shape:
            raise ValueError("Mask predictor must preserve full token positions")
        values = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), clean.reshape(-1), reduction="none"
        ).reshape_as(clean)
        values = values / probability[:, None]
        values = values.masked_fill(~masked, 0)
        if self.sft:
            numerator = (values.sum(-1) / eligible.sum(-1)).sum()
            denominator, unit = values.new_tensor(len(clean)), "sequence"
        else:
            numerator, denominator, unit = values.sum(), eligible.sum().to(values), "token_slot"
        return LossTerm(numerator, denominator, unit, "masked_diffusion")


@torch.no_grad()
def sample_masked_diffusion(
    model,
    prompt,
    *,
    mask_token_id,
    generation_length=128,
    steps=128,
    block_length=None,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    attention_mask=None,
    forbidden_token_ids=(),
    confidence_excluded_ids=(),
    generator=None,
):
    """Move fixed quotas of masked positions to tokens. Bidirectional denoising must
    not reuse an autoregressive KV cache."""
    block_length = generation_length if block_length is None else block_length
    if (
        prompt.ndim != 2
        or generation_length < 1
        or block_length < 1
        or generation_length % block_length
        or steps < 1
        or steps % (generation_length // block_length)
    ):
        raise ValueError("Generation blocks and per-block step counts must divide exactly")
    if (
        temperature < 0
        or cfg_scale < 0
        or remasking not in {"low_confidence", "random"}
        or (prompt == mask_token_id).any()
    ):
        raise ValueError("Invalid sampling settings or MASK in prompt")
    count, prompt_length = prompt.shape
    device = prompt.device
    tokens = torch.full(
        (count, prompt_length + generation_length), mask_token_id, dtype=torch.long, device=device
    )
    tokens[:, :prompt_length] = prompt
    valid = None
    if attention_mask is not None:
        if attention_mask.shape != prompt.shape:
            raise ValueError("Prompt padding mask shape differs")
        valid = torch.cat(
            (attention_mask, attention_mask.new_ones(count, generation_length)), dim=1
        )
    blocks = generation_length // block_length
    per_block_steps = steps // blocks
    was_training = model.training
    model.eval()
    try:
        for block in range(blocks):
            start, end = (
                prompt_length + block * block_length,
                prompt_length + (block + 1) * block_length,
            )
            quota = torch.full(
                (per_block_steps,), block_length // per_block_steps, device=device, dtype=torch.long
            )
            quota[: block_length % per_block_steps] += 1
            for amount in quota.tolist():
                if amount == 0:
                    continue
                masked = tokens == mask_token_id
                if cfg_scale:
                    unconditional = tokens.clone()
                    unconditional[:, :prompt_length] = mask_token_id
                    doubled_mask = None if valid is None else torch.cat((valid, valid))
                    conditioned, unconditioned = model(
                        input_ids=torch.cat((tokens, unconditional)),
                        attention_mask=doubled_mask,
                        use_cache=False,
                    ).logits.chunk(2)
                    logits = unconditioned + (cfg_scale + 1) * (conditioned - unconditioned)
                else:
                    logits = model(input_ids=tokens, attention_mask=valid, use_cache=False).logits
                logits = logits.float().clone()
                if any(
                    i < 0 or i >= logits.shape[-1]
                    for i in (*forbidden_token_ids, *confidence_excluded_ids)
                ):
                    raise ValueError("Special token outside vocabulary")
                logits[..., list(forbidden_token_ids)] = -torch.inf
                scores = logits
                if temperature:
                    uniform = torch.rand(
                        logits.shape, device=device, dtype=torch.float64, generator=generator
                    ).clamp_min(torch.finfo(torch.float64).tiny)
                    scores = logits.double() - temperature * torch.log(-torch.log(uniform))
                prediction = scores.argmax(-1)
                confidence_logits = logits.clone()
                confidence_logits[..., list(confidence_excluded_ids)] = -torch.inf
                if remasking == "low_confidence":
                    confidence = (
                        confidence_logits.softmax(-1).gather(-1, prediction[..., None]).squeeze(-1)
                    )
                else:
                    confidence = torch.rand(prediction.shape, device=device, generator=generator)
                eligible = masked.clone()
                eligible[:, :start] = False
                eligible[:, end:] = False
                confidence = confidence.masked_fill(~eligible, -torch.inf)
                if (eligible.sum(-1) < amount).any() or torch.isnan(confidence).any():
                    raise ValueError("Invalid transfer quota or categorical distribution")
                selected = confidence.topk(amount, dim=-1).indices
                tokens.scatter_(1, selected, prediction.gather(1, selected))
            if (tokens[:, start:end] == mask_token_id).any():
                raise RuntimeError("Mask predictor did not resolve all positions in a block")
        return tokens
    finally:
        model.train(was_training)
