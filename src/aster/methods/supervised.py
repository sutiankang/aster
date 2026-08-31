"""Supervised objectives with explicit valid counts and no optimizer ownership."""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle


MODEL_INPUT_KEYS = {
    "input_ids",
    "inputs_embeds",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "image_grid_thw",
    "video_values",
    "decoder_input_ids",
    "decoder_attention_mask",
}


def model_inputs(batch):

    if "model_inputs" in batch:
        return dict(batch["model_inputs"])
    return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def supervision_mask(batch, labels):
    """Construct valid supervision positions from labels and explicit masks without
    executing the model."""
    mask = labels.ne(-100)
    if "loss_mask" in batch:
        if batch["loss_mask"].shape != labels.shape:
            raise ValueError("loss_mask must align token by token")
        mask &= batch["loss_mask"].bool()

    input_mapping = batch.get("model_inputs", batch)
    mask_key = (
        "decoder_attention_mask" if "decoder_input_ids" in input_mapping else "attention_mask"
    )
    position_mask = input_mapping.get(mask_key)
    if position_mask is not None and position_mask.ndim == 2:
        if position_mask.shape != labels.shape:
            raise ValueError("Padding mask must align with supervised positions")
        mask &= position_mask.bool()
    return mask


def token_targets(batch, logits, *, causal=True):
    labels = batch.get("labels", batch.get("input_ids"))
    if labels is None or labels.shape != logits.shape[:2]:
        raise ValueError("Provide labels aligned to the complete output token sequence")
    mask = supervision_mask(batch, labels)
    if causal:
        return logits[:, :-1].float(), labels[:, 1:], mask[:, 1:]
    return logits.float(), labels, mask


def native_causal_config(model):
    """Return supported native token-model configuration without reading parameters
    or invoking forward; unknown architecture semantics are not inferred."""
    from ..models.decoder import CausalLM
    from ..models.moe import MixtralForCausalLM, DeepSeekV3ForCausalLM
    from ..models.sparse import DeepSeekV32ForCausalLM, DeepSeekV32Config
    from ..models.config import (
        LlamaConfig,
        Qwen2Config,
        Qwen3Config,
        MistralConfig,
        MixtralConfig,
        DeepSeekV3Config,
    )

    supported = {
        CausalLM: {LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig},
        MixtralForCausalLM: {MixtralConfig},
        DeepSeekV3ForCausalLM: {DeepSeekV3Config},
        DeepSeekV32ForCausalLM: {DeepSeekV32Config},
    }
    config = getattr(model, "config", None)
    return config if type(config) in supported.get(type(model), set()) else None


@torch.no_grad()
def preflight_causal_microbatches(model, batches, *, causal=True):
    """Validate the complete accumulation window before model work begins.

    Call separately for each actual student, teacher, or reference forward so that
    vocabulary, position limits, masks, and targets are checked against the right model."""
    batches = list(batches)
    config = native_causal_config(model)
    if not causal or config is None:
        return batches
    allowed = {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}
    cache_fields = {"state", "past_key_values", "use_cache", "cache_position"}
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("Native causal CE requires a dictionary batch")
        if cache_fields & set(batch):
            raise ValueError("Native causal training forbids caller-owned cache/past state")
        if "model_inputs" in batch and not isinstance(batch["model_inputs"], dict):
            raise ValueError("model_inputs must be an explicit dictionary")
        arguments = model_inputs(batch)
        if set(arguments) - allowed:
            raise ValueError(
                "Native causal CE accepts only explicit token inputs; no media or cache"
            )
        if "model_inputs" in batch and any(name in batch for name in allowed):
            raise ValueError("Do not mix nested and top-level model inputs")
        ids, embeds = arguments.get("input_ids"), arguments.get("inputs_embeds")
        if (ids is None) == (embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        if ids is not None:
            if (
                not isinstance(ids, torch.Tensor)
                or ids.dtype != torch.long
                or ids.ndim != 2
                or ((ids < 0) | (ids >= config.vocab_size)).any()
            ):
                raise ValueError("input_ids must be int64 [B,T] inside the native vocabulary")
            shape, device = ids.shape, ids.device
        else:
            if (
                not isinstance(embeds, torch.Tensor)
                or embeds.ndim != 3
                or not embeds.is_floating_point()
                or embeds.shape[-1] != config.hidden_size
                or not torch.isfinite(embeds).all()
            ):
                raise ValueError("inputs_embeds must be finite [B,T,hidden_size]")
            shape, device = embeds.shape[:2], embeds.device
        if shape[0] < 1 or not 1 <= shape[1] <= config.max_position_embeddings:
            raise ValueError(
                "Native causal batch/physical sequence length is outside declared bounds"
            )
        for name in ("attention_mask", "position_ids"):
            value = arguments.get(name)
            if value is None:
                continue
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.device != device
                or value.requires_grad
            ):
                raise ValueError(
                    name + " must be fixed and aligned to the complete input tokens/device"
                )
            if name == "attention_mask" and not ((value == 0) | (value == 1)).all():
                raise ValueError("attention_mask must be binary on the full physical axis")
            if name == "position_ids" and (value.dtype != torch.long or (value < 0).any()):
                raise ValueError("position_ids must be nonnegative int64")

        labels = batch.get("labels", batch.get("input_ids"))
        if (
            not isinstance(labels, torch.Tensor)
            or labels.shape != shape
            or labels.dtype != torch.long
            or labels.device != device
            or ((labels != -100) & ((labels < 0) | (labels >= config.vocab_size))).any()
        ):
            raise ValueError("Labels must be aligned int64 vocabulary IDs or -100")
        mask = batch.get("loss_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor)
            or mask.shape != shape
            or mask.device != device
            or mask.requires_grad
            or not ((mask == 0) | (mask == 1)).all()
        ):
            raise ValueError("loss_mask must be fixed aligned binary data")
    return batches


class CrossEntropyObjective(nn.Module):
    """Next-token CE when causal=True; aligned MLM/seq2seq CE otherwise.

    Only valid supervised positions contribute to the numerator and denominator.
    Optional model auxiliaries must provide their own LossTerm counts."""

    def __init__(self, *, causal=True, label_smoothing=0.0, auxiliary_weights=None):
        super().__init__()
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0,1)")
        self.causal, self.label_smoothing = causal, label_smoothing
        self.auxiliary_weights = dict(auxiliary_weights or {})

    def config_dict(self):
        return {
            "type": "cross_entropy",
            "causal": self.causal,
            "label_smoothing": self.label_smoothing,
            "auxiliary_weights": self.auxiliary_weights,
        }

    def preflight_microbatches(self, model, batches):
        return preflight_causal_microbatches(model, batches, causal=self.causal)

    def forward(self, model, batch):
        output = model(**model_inputs(batch), use_cache=False)
        logits, targets, mask = token_targets(batch, output.logits, causal=self.causal)
        targets = targets.masked_fill(~mask, 0)
        values = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape_as(targets)
        terms = [
            LossTerm(values.masked_select(mask).sum(), mask.sum().to(values.dtype), "token", "ce")
        ]
        for name, weight in self.auxiliary_weights.items():
            auxiliary = (output.auxiliary or {}).get(name)
            if not isinstance(auxiliary, LossTerm):
                raise ValueError(f"Model must expose auxiliary {name} as an explicit LossTerm")
            terms.append(
                LossTerm(auxiliary.numerator, auxiliary.denominator, auxiliary.unit, name, weight)
            )
        return terms[0] if len(terms) == 1 else LossBundle(tuple(terms))


def sequence_logprobs(model, batch):
    output = model(**model_inputs(batch), use_cache=False)
    logits, labels, mask = token_targets(batch, output.logits)
    selected = (
        F.log_softmax(logits, -1).gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    )
    return selected.masked_fill(~mask, 0), mask


class RegressionObjective(nn.Module):
    def __init__(self, *, input_key="inputs", target_key="targets", unit="sample"):
        super().__init__()
        self.input_key, self.target_key, self.unit = input_key, target_key, unit

    def config_dict(self):
        return {
            "type": "regression",
            "input_key": self.input_key,
            "target_key": self.target_key,
            "unit": self.unit,
        }

    def forward(self, model, batch):
        prediction = model(batch[self.input_key])
        target = batch[self.target_key]
        if prediction.shape != target.shape:
            raise ValueError("Regression target shape mismatch")
        errors = (prediction - target).square().flatten(1).mean(1)
        mask = batch.get("valid", torch.ones_like(errors, dtype=torch.bool)).bool()
        if mask.shape != errors.shape:
            raise ValueError("valid mask must select samples")
        return LossTerm(errors.masked_select(mask).sum(), mask.sum().to(errors.dtype), self.unit)


class ContrastiveObjective(nn.Module):
    """Symmetric in-batch CLIP contrastive loss. Global negatives require an explicit
    distributed encoder stage; this objective does not silently gather features."""

    def forward(self, model, batch):
        image_features, text_features, logit_scale = model(batch["images"], batch["tokens"])
        image_features, text_features = (
            F.normalize(image_features.float(), dim=-1),
            F.normalize(text_features.float(), dim=-1),
        )
        if image_features.shape != text_features.shape:
            raise ValueError("Contrastive pairs must align")
        logits = logit_scale.exp() * image_features @ text_features.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        values = (
            F.cross_entropy(logits, labels, reduction="none")
            + F.cross_entropy(logits.T, labels, reduction="none")
        ) / 2
        return LossTerm(values.sum(), values.new_tensor(len(values)), "pair", "contrastive")


class SigmoidContrastiveObjective(nn.Module):
    """Independent pairwise SigLIP binary loss, not a softmax over CLIP candidates."""

    def forward(self, model, batch):
        output = model(**batch["model_inputs"])
        logits = output.logits_per_text.float()
        if logits.ndim != 2 or min(logits.shape) < 1:
            raise ValueError("SigLIP needs a nonempty text-by-image logit matrix")
        positive = batch.get("pair_labels")
        if positive is None:
            if logits.shape[0] != logits.shape[1]:
                raise ValueError("Rectangular comparisons need explicit pair_labels")
            positive = torch.eye(len(logits), device=logits.device, dtype=torch.bool)
        if positive.shape != logits.shape or positive.dtype != torch.bool:
            raise ValueError("pair_labels must be a boolean text-by-image matrix")
        signed_logits = logits * (2 * positive.to(logits.dtype) - 1)
        return LossTerm(
            -F.logsigmoid(signed_logits).sum(), logits.new_tensor(len(logits)), "text", "siglip"
        )
