"""Distinct DPO, IPO, and SimPO objectives with explicit frozen-reference semantics."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm
from .supervised import (
    sequence_logprobs,
    native_causal_config,
    preflight_causal_microbatches,
    supervision_mask,
)


class PreferenceObjective(nn.Module):
    def __init__(self, reference=None, *, method="dpo", beta=0.1, margin=0.5):
        super().__init__()
        if method not in {"dpo", "ipo", "simpo"} or beta <= 0:
            raise ValueError("Unsupported preference method or beta")
        if method in {"dpo", "ipo"} and reference is None:
            raise ValueError("DPO/IPO require a frozen reference model")
        self.reference = reference.eval().requires_grad_(False) if reference is not None else None
        self.method, self.beta, self.margin = method, beta, margin

    def config_dict(self):
        return {
            "type": "preference",
            "method": self.method,
            "beta": self.beta,
            "margin": self.margin,
        }

    @torch.no_grad()
    def preflight_microbatches(self, model, batches):
        """Validate the complete accumulation window and all model roles used by this
        objective before forward or sharded parameter communication."""
        if (
            self.method not in {"dpo", "ipo", "simpo"}
            or not math.isfinite(self.beta)
            or self.beta <= 0
            or not math.isfinite(self.margin)
        ):
            raise ValueError("Invalid declared preference objective")
        if self.method != "simpo" and self.reference is None:
            raise ValueError("DPO/IPO require their actual reference model")
        if self.reference is not None and {id(p) for p in model.parameters()} & {
            id(p) for p in self.reference.parameters()
        }:
            raise ValueError("Reference cannot alias trainable policy")
        batches = list(batches)
        for batch in batches:
            if not isinstance(batch, dict) or not {"chosen", "rejected"} <= set(batch):
                raise ValueError("Preference microbatches require chosen/rejected pairs")
        paths = [batch[name] for batch in batches for name in ("chosen", "rejected")]
        preflight_causal_microbatches(model, paths)
        if self.method != "simpo":
            preflight_causal_microbatches(self.reference, paths)
        if native_causal_config(model) is not None:
            for batch in batches:
                counts = []
                for name in ("chosen", "rejected"):
                    item = batch[name]
                    labels = item.get("labels", item.get("input_ids"))
                    valid = supervision_mask(item, labels)[:, 1:]
                    if (valid.sum(-1) == 0).any():
                        raise ValueError("Every preference response needs supervised tokens")
                    counts.append(len(valid))
                if counts[0] != counts[1]:
                    raise ValueError("Chosen/rejected pairs must align")
        return batches

    def _scores(self, model, batch):
        logp, mask = sequence_logprobs(model, batch)
        count = mask.sum(-1)
        if (count == 0).any():
            raise ValueError("Every preference response needs supervised tokens")
        return logp.sum(-1) / (count if self.method in {"ipo", "simpo"} else 1)

    def forward(self, model, batch):
        if self.reference is not None and {id(p) for p in model.parameters()} & {
            id(p) for p in self.reference.parameters()
        }:
            raise ValueError("Reference cannot alias trainable policy")
        chosen, rejected = (
            self._scores(model, batch["chosen"]),
            self._scores(model, batch["rejected"]),
        )
        if chosen.shape != rejected.shape:
            raise ValueError("Chosen/rejected pairs must align")
        difference = chosen - rejected
        if self.method != "simpo":
            self.reference.eval()
            with torch.no_grad():
                reference_difference = self._scores(self.reference, batch["chosen"]) - self._scores(
                    self.reference, batch["rejected"]
                )

            difference = difference - reference_difference
        if self.method == "ipo":
            values = (difference - 1 / (2 * self.beta)).square()
        else:
            values = -F.logsigmoid(
                self.beta * difference - (self.margin if self.method == "simpo" else 0)
            )
        return LossTerm(values.sum(), values.new_tensor(values.numel()), "pair", self.method)
