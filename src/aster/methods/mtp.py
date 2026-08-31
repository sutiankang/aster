"""Shared-model next-token CE and independently normalized multi-token prediction losses."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossTerm, LossBundle
from .supervised import model_inputs


class MultiTokenPredictionObjective(nn.Module):
    """Normalize each prediction depth by its valid targets before applying
    mtp_weight / depth; do not share a denominator across unequal depth masks."""

    def __init__(
        self, *, depth=1, base_weight=1.0, mtp_weight=0.1, detach_base=False, label_smoothing=0.0
    ):
        super().__init__()
        if type(depth) is not int or depth < 1 or type(detach_base) is not bool:
            raise ValueError("MTP depth must be a positive integer and detach_base a boolean")
        if (
            not all(math.isfinite(x) for x in (base_weight, mtp_weight, label_smoothing))
            or min(base_weight, mtp_weight) < 0
            or base_weight + mtp_weight <= 0
            or not 0 <= label_smoothing < 1
        ):
            raise ValueError("MTP loss needs finite nonnegative weights and valid smoothing")
        self.depth, self.base_weight, self.mtp_weight = depth, base_weight, mtp_weight
        self.detach_base, self.label_smoothing = detach_base, label_smoothing

    def config_dict(self):
        return dict(
            type="sequential_mtp_ce",
            depth=self.depth,
            base_weight=self.base_weight,
            mtp_weight=self.mtp_weight,
            detach_base=self.detach_base,
            label_smoothing=self.label_smoothing,
            padding="right_unpacked",
        )

    def forward(self, model, batch):
        inputs = model_inputs(batch)
        ids = inputs.get("input_ids")
        if (
            ids is None
            or ids.ndim != 2
            or ids.dtype not in (torch.int32, torch.int64)
            or len(ids) < 1
            or ids.shape[1] < self.depth + 2
        ):
            raise ValueError(
                "MTP requires complete B,S integer token sequences of length >= depth+2"
            )
        if set(inputs) - {"input_ids", "attention_mask", "position_ids"} or "segment_ids" in batch:
            raise ValueError(
                "MTP objective currently requires unpacked token sequences; no implicit multimodal/segment reset"
            )
        labels = batch.get("labels", ids)
        if labels.shape != ids.shape or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("MTP labels must align with full token coordinates")
        valid = labels.ne(-100)
        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"]
            if loss_mask.shape != ids.shape or loss_mask.dtype != torch.bool:
                raise ValueError("MTP loss_mask must be an aligned boolean target mask")
            valid = valid & loss_mask
        padding = inputs.get("attention_mask")
        if padding is not None:
            if padding.shape != ids.shape or not ((padding == 0) | (padding == 1)).all():
                raise ValueError("MTP needs an explicit binary B,S padding mask")
            padding = padding.bool()
            if ((~padding[:, :-1]) & padding[:, 1:]).any():
                raise ValueError(
                    "MTP training currently supports right padding, not left/gapped/pseudo-packed sequences"
                )
            valid = valid & padding
        output = model(
            **inputs, use_cache=False, mtp_depth=self.depth, detach_mtp_base=self.detach_base
        )
        auxiliary = output.auxiliary or {}
        offsets, predictions = auxiliary.get("mtp_offsets"), auxiliary.get("mtp_logits")
        if (
            offsets != tuple(range(1, self.depth + 1))
            or not isinstance(predictions, tuple)
            or len(predictions) != self.depth
        ):
            raise ValueError(
                "Model must expose exact sequential mtp_offsets/mtp_logits, not an unrelated auxiliary head"
            )

        def term(logits, offset, weight, name):
            targets, mask = labels[:, offset + 1 :], valid[:, offset + 1 :]
            if logits.shape[:2] != (len(ids), ids.shape[1] - offset):
                raise ValueError("MTP logit sequence length violates its target offset")
            prediction = logits[:, :-1].float()
            targets = targets.masked_fill(~mask, 0).long()
            values = F.cross_entropy(
                prediction.reshape(-1, prediction.shape[-1]),
                targets.reshape(-1),
                reduction="none",
                label_smoothing=self.label_smoothing,
            ).reshape_as(targets)
            return LossTerm(
                values.masked_select(mask).sum(), mask.sum(dtype=torch.int64), "token", name, weight
            )

        terms = [term(output.logits, 0, self.base_weight, "next_token")]
        terms.extend(
            term(logits, offset, self.mtp_weight / self.depth, f"mtp_{offset}")
            for offset, logits in zip(offsets, predictions)
        )
        return LossBundle(tuple(terms))
