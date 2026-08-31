"""Paired inferred-action versus random-action PSNR for Genie controllability."""

import math
import torch

from ..planning.genie import generate_genie_video


def paired_delta_psnr(
    reference, inferred_actions_prediction, random_actions_prediction, *, mse_floor=1e-12
):

    predictions = (reference, inferred_actions_prediction, random_actions_prediction)
    if (
        reference.ndim != 4
        or not len(reference)
        or any(
            v.shape != reference.shape
            or v.device != reference.device
            or not v.is_floating_point()
            or not torch.isfinite(v).all()
            or v.min() < 0
            or v.max() > 1
            for v in predictions
        )
    ):
        raise ValueError("Paired PSNR requires aligned finite float [0,1] BCHW tensors")
    if not math.isfinite(mse_floor) or mse_floor <= 0:
        raise ValueError("PSNR MSE floor must be finite positive")
    scores = [
        -10
        * (value.double() - reference.double())
        .square()
        .flatten(1)
        .mean(1)
        .clamp_min(mse_floor)
        .log10()
        for value in predictions[1:]
    ]
    return scores[0] - scores[1], scores[0], scores[1]


@torch.no_grad()
def evaluate_genie_controllability(
    tokenizer,
    world,
    video,
    *,
    time_index=4,
    seed=0,
    mse_floor=1e-12,
    steps=25,
    token_temperature=1.0,
    choice_temperature=2.0,
    mask_order="confidence",
):

    if (
        type(time_index) is not int
        or time_index < 1
        or type(seed) is not int
        or video.ndim != 5
        or video.shape[1] <= time_index
    ):
        raise ValueError("Invalid Genie controllability horizon/seed/video")
    if time_index >= min(
        tokenizer.config.max_frames,
        world.config.action.max_frames,
        world.config.dynamics.max_frames,
    ):
        raise ValueError("Genie metric horizon exceeds trained finite context")
    modes = {module: module.training for module in world.modules()}
    try:
        world.eval()
        inferred = world.action_model.encode(video[:, : time_index + 1]).indices
    finally:
        for module, mode in modes.items():
            module.training = mode
    action_rng = torch.Generator(device=video.device).manual_seed(seed + 1)
    random = torch.randint(
        world.config.action.num_codes, inferred.shape, device=video.device, generator=action_rng
    )
    sampling = dict(
        steps=steps,
        token_temperature=token_temperature,
        choice_temperature=choice_temperature,
        mask_order=mask_order,
    )
    predicted, direct_info = generate_genie_video(
        tokenizer,
        world,
        video[:, :1],
        inferred,
        generator=torch.Generator(device=video.device).manual_seed(seed),
        **sampling,
    )
    randomized, random_info = generate_genie_video(
        tokenizer,
        world,
        video[:, :1],
        random,
        generator=torch.Generator(device=video.device).manual_seed(seed),
        **sampling,
    )
    delta, direct, baseline = paired_delta_psnr(
        video[:, time_index],
        predicted[:, time_index],
        randomized[:, time_index],
        mse_floor=mse_floor,
    )
    return dict(
        metric=f"delta_psnr_t{time_index}",
        per_sample=delta.cpu(),
        inferred_psnr=direct.cpu(),
        random_psnr=baseline.cpu(),
        mean=float(delta.mean()),
        time_index=time_index,
        seed=seed,
        mse_floor=mse_floor,
        data_range=1.0,
        model_calls=direct_info["model_calls"] + random_info["model_calls"],
        sampling=sampling,
        public_quality_evaluated=False,
        inferred_action_ids=inferred.cpu(),
        random_action_ids=random.cpu(),
    )
