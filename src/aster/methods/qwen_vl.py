"""Connect raw Qwen media examples to shared objectives without a separate trainer."""

from dataclasses import fields, is_dataclass, replace
import torch
from torch import nn
from ..core import digest_json
from ..data.qwen_vl import Qwen3VLProcessor
from .supervised import CrossEntropyObjective


def _to_device(value, device):

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _to_device(getattr(value, field.name), device)
                for field in fields(value)
                if field.init
            },
        )
    return value


class RawQwenObjective(nn.Module):
    def __init__(self, processor, *, visual_prefill="image", objective=None, generation_fields=()):
        super().__init__()
        from .cosmos3 import Cosmos3VisualFlowObjective

        if not isinstance(processor, Qwen3VLProcessor):
            raise ValueError("Raw Qwen objective needs a native processor")
        if visual_prefill not in {"image", "video", "both", "none"}:
            raise ValueError("Unknown raw Qwen visual_prefill")
        self.processor, self.visual_prefill = processor, visual_prefill
        self.objective = CrossEntropyObjective() if objective is None else objective
        if not isinstance(self.objective, (CrossEntropyObjective, Cosmos3VisualFlowObjective)):
            raise ValueError(
                "Raw bridge audits causal CE or Cosmos3 visual flow, not arbitrary hidden call graphs"
            )
        if isinstance(self.objective, CrossEntropyObjective) and not self.objective.causal:
            raise ValueError("Qwen raw text supervision is next-token causal CE")
        names = tuple(generation_fields)
        if len(set(names)) != len(names) or set(names) - {"vision", "sound", "action"}:
            raise ValueError("Invalid Cosmos generation field graph")
        self.generation_fields = tuple(
            name for name in ("vision", "sound", "action") if name in names
        )
        if isinstance(self.objective, Cosmos3VisualFlowObjective):
            if self.objective.visual_prefill != visual_prefill:
                raise ValueError("Raw and Cosmos visual_prefill declarations must match")
        elif names:
            raise ValueError("Pure Qwen CE cannot consume Cosmos latent generation fields")

    def config_dict(self):
        return dict(
            type="raw_qwen_vl",
            processor=self.processor.config.to_dict(),
            processor_id=self.processor.fingerprint,
            tokenizer_id=self.processor.tokenizer_id,
            visual_prefill=self.visual_prefill,
            generation_fields=self.generation_fields,
            objective=self.objective.config_dict(),
        )

    def _prepare(self, model, batch):
        if not isinstance(batch, dict) or set(batch) - {
            "examples",
            "model_inputs",
            "noise",
            "loss_mask",
        }:
            raise ValueError(
                "Raw Qwen batch accepts only examples, explicit latent inputs/noise and loss_mask"
            )
        prepared = self.processor.prepare(batch["examples"], model.config)
        actual = {item.kind for item in prepared.media}
        expected = (
            {"image", "video"}
            if self.visual_prefill == "both"
            else (set() if self.visual_prefill == "none" else {self.visual_prefill})
        )
        if actual != expected:
            raise ValueError(
                "Raw Qwen visual_prefill must match on every rank/microbatch before gathers"
            )
        from ..models.cosmos3_vlm import Cosmos3VLMConfig
        from .cosmos3 import Cosmos3VisualFlowObjective

        if isinstance(model.config, Cosmos3VLMConfig) != isinstance(
            self.objective, Cosmos3VisualFlowObjective
        ):
            raise ValueError(
                "Cosmos requires its explicit visual-flow objective, not a logits-only CE alias"
            )
        extra = batch.get("model_inputs", {})
        if not isinstance(extra, dict) or set(extra) != set(self.generation_fields):
            raise ValueError(
                "Additional model inputs must match the declared Cosmos generation_fields exactly"
            )
        if "noise" in batch and (
            not isinstance(batch["noise"], dict)
            or set(batch["noise"]) - set(self.generation_fields)
        ):
            raise ValueError("Explicit noise must belong to the declared generation fields")
        result = prepared.training_batch()
        result["model_inputs"].update(extra)
        if "loss_mask" in batch:
            mask = batch["loss_mask"]
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != prepared.labels.shape
            ):
                raise ValueError(
                    "Response-only loss_mask must explicitly align every prepared token"
                )
            result["loss_mask"] = mask
        if "noise" in batch:
            result["noise"] = batch["noise"]
        result = _to_device(result, next(model.parameters()).device)

        result["_raw_qwen_objective"] = digest_json(self.config_dict())
        result["_raw_qwen_model"] = digest_json(model.config.to_dict())
        return result

    def preflight_microbatches(self, model, batches):
        prepared = [self._prepare(model, batch) for batch in batches]
        preflight = getattr(self.objective, "preflight_microbatches", None)
        if preflight is not None:
            prepared = list(preflight(model, prepared))
        return prepared

    def forward(self, model, batch):
        if "_raw_qwen_objective" not in batch:
            batch = self.preflight_microbatches(model, [batch])[0]
        elif batch["_raw_qwen_objective"] != digest_json(self.config_dict()) or batch[
            "_raw_qwen_model"
        ] != digest_json(model.config.to_dict()):
            raise ValueError("Prepared Qwen batch belongs to a different processor/objective/model")
        return self.objective(model, batch)
