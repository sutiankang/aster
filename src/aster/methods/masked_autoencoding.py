"""Masked reconstruction pretraining for native Drifting feature encoders."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossBundle, LossTerm


def sample_patch_mask(
    samples, *, input_patch_size=1, patch_size=4, minimum=0.75, maximum=0.75, generator=None
):
    if (
        samples.ndim != 4
        or not all(math.isfinite(x) for x in (minimum, maximum))
        or not 0 <= minimum <= maximum <= 1
        or type(input_patch_size) is not int
        or input_patch_size < 1
        or type(patch_size) is not int
        or patch_size < 1
    ):
        raise ValueError("Invalid patch-mask geometry/distribution")
    b, _, h, w = samples.shape
    block = input_patch_size * patch_size
    if h % block or w % block:
        raise ValueError("Mask patches must divide the input exactly")
    ratio = minimum + (maximum - minimum) * torch.rand(
        b, device=samples.device, generator=generator
    )
    masked = (
        torch.rand(b, 1, h // block, w // block, device=samples.device, generator=generator)
        < ratio[:, None, None, None]
    )
    return masked.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)


class MaskedAutoencodingObjective(nn.Module):
    def __init__(self, *, classification_weight=0.0, mask_min=0.75, mask_max=0.75):
        super().__init__()
        if (
            not all(math.isfinite(x) for x in (classification_weight, mask_min, mask_max))
            or not 0 <= classification_weight <= 1
            or not 0 <= mask_min <= mask_max <= 1
        ):
            raise ValueError("Invalid MAE objective weights or masking distribution")
        self.classification_weight, self.mask_min, self.mask_max = (
            classification_weight,
            mask_min,
            mask_max,
        )

    def config_dict(self):
        return dict(
            type="drifting_mae",
            classification_weight=self.classification_weight,
            mask_min=self.mask_min,
            mask_max=self.mask_max,
        )

    def forward(self, model, batch):
        if set(batch) - {"samples", "labels", "mask"}:
            raise ValueError("Unknown masked-autoencoding fields")
        samples = batch["samples"]
        mask = batch.get("mask")
        if mask is None:
            mask = sample_patch_mask(
                samples,
                input_patch_size=model.config.input_patch_size,
                patch_size=model.config.patch_size,
                minimum=self.mask_min,
                maximum=self.mask_max,
            )
        output = model(samples, mask)
        per_image = (
            (output.reconstruction.float() - output.patched_target.float()).square() * mask
        ).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) + 1e-8)
        count = torch.tensor(len(samples), dtype=torch.int64, device=samples.device)
        terms = [
            LossTerm(
                per_image.sum(),
                count,
                "image",
                "masked_reconstruction",
                1 - self.classification_weight,
            )
        ]
        if self.classification_weight:
            labels = batch["labels"]
            if labels.shape != (len(samples),) or labels.dtype != torch.int64:
                raise ValueError("MAE classification labels must be aligned int64")
            terms.append(
                LossTerm(
                    F.cross_entropy(output.logits.float(), labels, reduction="sum"),
                    count,
                    "image",
                    "classification",
                    self.classification_weight,
                )
            )
        else:
            terms[0] = LossTerm(
                terms[0].numerator + 0 * output.logits.sum(),
                count,
                "image",
                "masked_reconstruction",
            )
        return LossBundle(tuple(terms))
