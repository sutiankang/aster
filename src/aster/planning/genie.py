"""Native MaskGIT remasking and latent-action video generation."""

import math
import torch

from ..models.genie import GenieWorldConfig, GenieTokenizerConfig


def maskgit_remask(selected_probabilities, unknown, mask_length, gumbel, temperature):

    if (
        selected_probabilities.shape != unknown.shape
        or gumbel.shape != unknown.shape
        or unknown.ndim != 2
    ):
        raise ValueError("MaskGIT confidence tensor dimensions differ")
    confidence = (
        selected_probabilities.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        + temperature * gumbel
    )
    confidence = confidence.masked_fill(~unknown, torch.inf)
    cutoff = confidence.sort(-1).values.gather(
        -1, mask_length[:, None].clamp(0, confidence.shape[-1] - 1)
    )
    return (confidence < cutoff) & unknown


@torch.no_grad()
def sample_maskgit(
    logits_fn,
    tokens,
    *,
    mask_token_id,
    steps=25,
    token_temperature=1.0,
    choice_temperature=2.0,
    mask_order="confidence",
    generator=None,
):
    """

    References:
    https://github.com/google-research/maskgit/tree/1db23594e1bd328ee78eadcd148a19281cd0f5b8"""
    if (
        tokens.ndim != 2
        or not tokens.numel()
        or tokens.dtype != torch.int64
        or type(mask_token_id) is not int
        or mask_token_id < 1
    ):
        raise ValueError("MaskGIT requires int64 [B,N] tokens and explicit positive mask token")
    if (tokens < 0).any() or (tokens > mask_token_id).any() or type(steps) is not int or steps < 1:
        raise ValueError("Invalid MaskGIT token/step configuration")
    if (
        not math.isfinite(token_temperature)
        or token_temperature <= 0
        or not math.isfinite(choice_temperature)
        or choice_temperature < 0
        or mask_order not in {"confidence", "random"}
    ):
        raise ValueError("Invalid MaskGIT sampling temperatures/order")
    current = tokens.clone()
    initial_unknown = (current == mask_token_id).sum(-1)
    if not initial_unknown.any():
        return current, dict(model_calls=0, unknown_counts=[0], steps=0, mask_order=mask_order)
    unknown_counts = []
    for step in range(steps):
        unknown = current == mask_token_id
        unknown_counts.append(int(unknown.sum()))
        logits = logits_fn(current)
        if logits.shape != (*current.shape, mask_token_id) or not torch.isfinite(logits).all():
            raise ValueError(
                "MaskGIT logits must be finite and exclude mask from output vocabulary"
            )
        probabilities = (logits.float() / token_temperature).softmax(-1)
        sampled = torch.multinomial(
            probabilities.reshape(-1, mask_token_id), 1, generator=generator
        ).reshape_as(current)
        predicted = torch.where(unknown, sampled, current)
        if step == steps - 1:
            return predicted, dict(
                model_calls=steps, unknown_counts=unknown_counts, steps=steps, mask_order=mask_order
            )
        ratio = (step + 1) / steps

        length = torch.floor(initial_unknown * math.cos(math.pi * 0.5 * ratio)).long()
        length = torch.maximum(torch.ones_like(length), torch.minimum(unknown.sum(-1) - 1, length))
        selected = probabilities.gather(-1, predicted[..., None]).squeeze(-1)
        uniform = torch.rand(current.shape, device=current.device, generator=generator).clamp_(
            1e-7, 1 - 1e-7
        )
        gumbel = -(-uniform.log()).log()
        if mask_order == "random":
            selected = torch.ones_like(selected)
            remask = maskgit_remask(selected, unknown, length, gumbel, 1.0)
        else:
            remask = maskgit_remask(
                selected, unknown, length, gumbel, choice_temperature * (1 - ratio)
            )
        current = predicted.masked_fill(remask, mask_token_id)
    raise AssertionError("MaskGIT loop must return a fully predicted final iteration")


@torch.no_grad()
def sample_genie_frame(world, context_tokens, action_ids, **sampling):

    if not isinstance(world.config, GenieWorldConfig):
        raise ValueError("Genie sampling needs native world/action-codebook composite")
    c = world.config.dynamics
    if (
        context_tokens.ndim != 3
        or context_tokens.shape[-1] != c.spatial_tokens
        or not 1 <= context_tokens.shape[1] < c.max_frames
        or context_tokens.dtype != torch.int64
        or (context_tokens < 0).any()
        or (context_tokens >= c.vocab_size).any()
    ):
        raise ValueError("Genie context tokens or maximum frame window invalid")
    if action_ids.shape != context_tokens.shape[:2] or action_ids.device != context_tokens.device:
        raise ValueError("Genie action history must include one action per context frame")
    modes = {module: module.training for module in world.modules()}
    try:
        world.eval()
        actions = world.action_model.quantizer.lookup(action_ids)
        initial = torch.full_like(context_tokens[:, -1], c.mask_token_id)

        def predict(current):
            return world.dynamics(torch.cat((context_tokens, current[:, None]), 1), actions)[:, -1]

        return sample_maskgit(predict, initial, mask_token_id=c.mask_token_id, **sampling)
    finally:
        for module, mode in modes.items():
            module.training = mode


@torch.no_grad()
def generate_genie_video(tokenizer, world, prompt, action_ids, **sampling):

    if type(tokenizer.config) is not GenieTokenizerConfig or not isinstance(
        world.config, GenieWorldConfig
    ):
        raise ValueError("Genie generation requires explicit tokenizer/world configurations")
    tc, wc = tokenizer.config, world.config
    if (tc.num_codes, tc.spatial_tokens, tc.max_frames) != (
        wc.dynamics.vocab_size,
        wc.dynamics.spatial_tokens,
        wc.dynamics.max_frames,
    ):
        raise ValueError("Genie tokenizer and world code/context geometries differ")
    if (tc.image_channels, tc.image_height, tc.image_width) != (
        wc.action.image_channels,
        wc.action.image_height,
        wc.action.image_width,
    ):
        raise ValueError("Genie tokenizer and latent-action pixel geometries differ")
    if (
        prompt.ndim != 5
        or prompt.shape[1] != 1
        or action_ids.ndim != 2
        or action_ids.shape[0] != len(prompt)
        or action_ids.device != prompt.device
        or not 1 <= action_ids.shape[1] < tc.max_frames
    ):
        raise ValueError("Genie prompt/action horizon differs from finite context window")
    modes = {module: module.training for module in tokenizer.modules()}
    try:
        tokenizer.eval()
        tokens = tokenizer.encode(prompt).indices
        diagnostics = []
        for index in range(action_ids.shape[1]):
            frame, result = sample_genie_frame(
                world, tokens, action_ids[:, : index + 1], **sampling
            )
            tokens = torch.cat((tokens, frame[:, None]), 1)
            diagnostics.append(result)
        decoded = tokenizer.decode(tokens)

        decoded[:, :1] = prompt.to(decoded)
        return decoded, dict(
            model_calls=sum(row["model_calls"] for row in diagnostics),
            frames=diagnostics,
            tokenizer_encodes=1,
            tokenizer_decodes=1,
            tokens=tokens,
        )
    finally:
        for module, mode in modes.items():
            module.training = mode
