"""Llama/Qwen TP-by-PP stage providers using the shared pipeline scheduler."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from aster.core import LossTerm
from aster.methods.supervised import token_targets
from .pipeline import PipelineStage, PipelineObjective, PipelineLossSpec
from .parallel import vocab_parallel_cross_entropy
from .causal_parallel import TensorParallelCrossEntropyObjective, _reject_attention_backend


class _CausalStageBody(nn.Module):
    def __init__(self, source, context):
        super().__init__()
        config = source.config
        self.first, self.last = context.pp.rank == 0, context.pp.rank == context.pp.size - 1
        start = config.num_hidden_layers * context.pp.rank // context.pp.size
        end = config.num_hidden_layers * (context.pp.rank + 1) // context.pp.size
        self.model = nn.Module()
        if self.first:
            self.model.embed_tokens = source.model.embed_tokens

        self.model.layers = nn.ModuleDict(
            {str(index): source.model.layers[index] for index in range(start, end)}
        )
        if self.last:
            self.model.norm, self.lm_head = source.model.norm, source.lm_head
        if config.tie_word_embeddings:
            parameter = self.model.embed_tokens.weight if self.first else self.lm_head.weight
            parameter._aster_extra_gradient_group = context.pp
            parameter._aster_pp_tied_key = "model.embed_tokens.weight"
            parameter._aster_unique_norm_owner = self.first

    def forward(self, inputs, positions, padding):
        hidden = (
            self.model.embed_tokens(inputs) if self.first and inputs.dtype == torch.long else inputs
        )
        for layer in self.model.layers.values():
            hidden, _, _ = layer(hidden, positions, padding, None, 0, False)
        return self.lm_head(self.model.norm(hidden)) if self.last else hidden


class CausalPipelineStage(PipelineStage):
    def __init__(self, source, context, *, schedule):
        _reject_attention_backend(source)
        body = _CausalStageBody(source, context)
        names = {
            f"module.{name}": name for name, _ in body.named_parameters(remove_duplicate=False)
        }
        names.update(
            {f"module.{name}": name for name, _ in body.named_buffers(remove_duplicate=False)}
        )
        super().__init__(body, context.pp, schedule=schedule, parameter_names=names)
        self.config, self.context = source.config, context
        self._batch_metadata = None
        self._aster_shared_runtime_handles = (context, context.tp, context.pp, context.dp)

    @contextmanager
    def batch_context(self, arguments):
        if self._batch_metadata is not None:
            raise RuntimeError("Pipeline stage metadata is not reentrant")
        source = arguments.get("input_ids", arguments.get("inputs_embeds"))
        positions = arguments.get("position_ids")
        if positions is None:
            positions = torch.arange(source.shape[1], device=source.device)[None].expand(
                source.shape[0], -1
            )
        self._batch_metadata = (positions, arguments.get("attention_mask"))
        try:
            yield source
        finally:
            self._batch_metadata = None

    def _run_module(self, inputs):
        if self._batch_metadata is None:
            raise RuntimeError("Causal pipeline requires an explicitly prepared batch context")
        return self.module(inputs, *self._batch_metadata)


@dataclass(frozen=True)
class _PreparedBatch:
    arguments: dict
    targets: dict


class CausalPipelineCrossEntropyObjective(PipelineObjective):
    def __init__(self, context):
        self.context = context
        self.validator = TensorParallelCrossEntropyObjective(context)
        super().__init__(self._criterion, specs=(PipelineLossSpec("ce", "token"),))

    def config_dict(self):
        return {
            **self.validator.config_dict(),
            "type": "native_causal_pipeline_ce",
            "batch_preflight": "whole_window_before_pipeline_schedule",
            "boundary": "hidden_tensor",
        }

    def preflight_microbatches(self, model, microbatches):
        if not isinstance(model, CausalPipelineStage):
            raise TypeError("Pipeline objective requires a causal pipeline provider")
        result = []
        for batch in microbatches:
            arguments = self.validator._preflight(model, batch)
            result.append(
                _PreparedBatch(
                    arguments, {**batch, "labels": batch.get("labels", arguments.get("input_ids"))}
                )
            )
        return result

    def _criterion(self, logits, batch):
        logits, labels, mask = token_targets(batch, logits)
        values = vocab_parallel_cross_entropy(
            logits, labels.masked_fill(~mask, -100), self.context.tp
        )
        return LossTerm(values.masked_select(mask).sum(), mask.sum().to(torch.int64), "token", "ce")

    def forward(self, model, batch):
        if not isinstance(batch, _PreparedBatch):
            raise TypeError(
                "Use Trainer.step: pipeline window must be preflighted before scheduling"
            )
        with model.batch_context(batch.arguments) as inputs:
            return super().forward(model, (inputs, batch.targets))
