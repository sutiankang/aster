"""Native TinyBERT feature matching and MiniLM relation distillation."""

from contextlib import contextmanager
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossTerm, LossBundle, TokenOutput
from ..models import build_model, BertConfig
from .supervised import model_inputs


@contextmanager
def capture_bert_projections(model, layers):
    """Capture projections from the actual forward without rerunning the network or
    consuming another dropout/random-number sequence."""
    captured, handles = {}, []
    try:
        for layer in sorted(set(layers)):
            attention = model.get_submodule(f"bert.encoder.layer.{layer}.attention.self")
            for name in ("query", "key", "value"):

                def remember(module, inputs, output, key=(layer, name)):
                    captured[key] = output

                handles.append(getattr(attention, name).register_forward_hook(remember))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def split_relations(value, heads):
    if value.ndim != 3 or heads < 1 or value.shape[-1] % heads:
        raise ValueError("Projection width must be divisible by relation-head count")
    return value.reshape(*value.shape[:2], heads, value.shape[-1] // heads).transpose(1, 2)


def relation_kl(student_left, student_right, teacher_left, teacher_right, valid, *, heads, name):
    """Normalize each query distribution over valid keys and reduce over valid queries
    and relation heads."""
    if valid.ndim != 2 or valid.dtype != torch.bool or not valid.any(-1).all():
        raise ValueError("Every example needs at least one valid token")

    def scores(left, right):
        left, right = split_relations(left.float(), heads), split_relations(right.float(), heads)
        if (
            left.shape != right.shape
            or left.shape[0] != len(valid)
            or left.shape[2] != valid.shape[1]
        ):
            raise ValueError("Aligned token projections are required")
        result = left @ right.transpose(-1, -2) / math.sqrt(left.shape[-1])

        return result.masked_fill(~valid[:, None, None, :], torch.finfo(result.dtype).min)

    student_logp = F.log_softmax(scores(student_left, student_right), -1)
    teacher_logp = F.log_softmax(scores(teacher_left.detach(), teacher_right.detach()), -1)
    values = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    queries = valid[:, None, :].expand_as(values)
    return LossTerm(
        values.masked_select(queries).sum(), queries.sum().to(values), "query_head", name
    )


class TinyBERTStudent(nn.Module):
    """Keep fit_dense projections in the student role so optimizer, ZeRO, and checkpoint
    state include every trainable projection."""

    def __init__(self, student, teacher_hidden_size):
        super().__init__()
        if not isinstance(student.config, BertConfig) or teacher_hidden_size < 1:
            raise ValueError("TinyBERT wrapper requires a native BERT student")
        self.student, self.config = student, student.config
        self.fit_dense = nn.Linear(student.config.hidden_size, teacher_hidden_size)

    def forward(self, *args, **kwargs):
        output = self.student(*args, **kwargs)
        states = (
            None
            if output.hidden_states is None
            else tuple(self.fit_dense(value) for value in output.hidden_states)
        )
        return TokenOutput(output.logits, hidden_states=states)


class TinyBERTObjective(nn.Module):
    def __init__(
        self, teacher, *, attention_pairs, hidden_pairs, padding_reduction="official_slots"
    ):
        super().__init__()
        if (
            not attention_pairs
            or not hidden_pairs
            or padding_reduction not in {"official_slots", "valid"}
        ):
            raise ValueError("Explicit layer mapping and padding reduction are required")
        self.teacher = teacher.eval().requires_grad_(False)
        self.attention_pairs, self.hidden_pairs = tuple(attention_pairs), tuple(hidden_pairs)
        self.padding_reduction = padding_reduction

    def config_dict(self):
        return {
            "type": "tinybert",
            "attention_pairs": self.attention_pairs,
            "hidden_pairs": self.hidden_pairs,
            "padding_reduction": self.padding_reduction,
        }

    def forward(self, model, batch):
        if not isinstance(model, TinyBERTStudent):
            raise ValueError("TinyBERT fit_dense must be part of the trainable student role")
        if model.config.num_attention_heads != self.teacher.config.num_attention_heads:
            raise ValueError("TinyBERT raw attention MSE requires matched heads; MiniLMv2 does not")
        inputs = model_inputs(batch)
        inputs.update(
            {
                key: batch[key]
                for key in ("token_type_ids",)
                if key in batch and "model_inputs" not in batch
            }
        )
        valid = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
        if valid.ndim != 2:
            raise ValueError("Encoder distillation requires a 2D valid-token mask")
        with capture_bert_projections(
            model.student, [pair[0] for pair in self.attention_pairs]
        ) as student_qkv:
            output = model(**inputs, output_hidden_states=True)
        self.teacher.eval()
        with (
            torch.no_grad(),
            capture_bert_projections(
                self.teacher, [pair[1] for pair in self.attention_pairs]
            ) as teacher_qkv,
        ):
            target = self.teacher(**inputs, output_hidden_states=True)
        heads = model.config.num_attention_heads

        def attention(qkv, layer):
            q, k = [split_relations(qkv[layer, name].float(), heads) for name in ("query", "key")]
            score = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            score = score.masked_fill(~valid[:, None, None, :], -10000.0)

            return torch.where(score <= -100, 0.0, score)

        terms = []
        for index, (student_layer, teacher_layer) in enumerate(self.attention_pairs):
            errors = (
                attention(student_qkv, student_layer) - attention(teacher_qkv, teacher_layer)
            ).square()
            mask = (valid[:, None, :, None] & valid[:, None, None, :]).expand_as(errors)
            selected = (
                errors.reshape(-1)
                if self.padding_reduction == "official_slots"
                else errors.masked_select(mask)
            )
            terms.append(
                LossTerm(
                    selected.sum(),
                    selected.new_tensor(selected.numel()),
                    "attention_slot",
                    f"attention_{index}",
                )
            )
        for index, (student_layer, teacher_layer) in enumerate(self.hidden_pairs):
            errors = (
                output.hidden_states[student_layer].float()
                - target.hidden_states[teacher_layer].float()
            ).square()
            selected = (
                errors.reshape(-1)
                if self.padding_reduction == "official_slots"
                else errors.masked_select(valid[..., None].expand_as(errors))
            )
            terms.append(
                LossTerm(
                    selected.sum(),
                    selected.new_tensor(selected.numel()),
                    "hidden_slot",
                    f"hidden_{index}",
                )
            )

        return LossBundle(tuple(terms))


class MiniLMObjective(nn.Module):
    def __init__(self, teacher, *, student_layer, teacher_layer, version=2, relation_heads=12):
        super().__init__()
        if version not in {1, 2} or relation_heads < 1:
            raise ValueError("MiniLM version must be 1 or 2")
        self.teacher = teacher.eval().requires_grad_(False)
        self.student_layer, self.teacher_layer = student_layer, teacher_layer
        self.version, self.relation_heads = version, relation_heads

    def config_dict(self):
        return {
            "type": "minilm",
            "student_layer": self.student_layer,
            "teacher_layer": self.teacher_layer,
            "version": self.version,
            "relation_heads": self.relation_heads,
        }

    def forward(self, model, batch):
        if self.version == 1 and (
            model.config.num_attention_heads != self.teacher.config.num_attention_heads
            or self.relation_heads != model.config.num_attention_heads
        ):
            raise ValueError("MiniLMv1 transfers original attention heads; use v2 to repartition")
        inputs = model_inputs(batch)
        if "token_type_ids" in batch and "model_inputs" not in batch:
            inputs["token_type_ids"] = batch["token_type_ids"]
        valid = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
        with capture_bert_projections(model, [self.student_layer]) as student:
            model(**inputs)
        self.teacher.eval()
        with (
            torch.no_grad(),
            capture_bert_projections(self.teacher, [self.teacher_layer]) as teacher,
        ):
            self.teacher(**inputs)
        relations = (
            (("query", "key"), ("value", "value"))
            if self.version == 1
            else tuple((name, name) for name in ("query", "key", "value"))
        )
        return LossBundle(
            tuple(
                relation_kl(
                    student[self.student_layer, left],
                    student[self.student_layer, right],
                    teacher[self.teacher_layer, left],
                    teacher[self.teacher_layer, right],
                    valid,
                    heads=self.relation_heads,
                    name=f"{left}_{right}",
                )
                for left, right in relations
            )
        )


class EncoderDistillationMethod:
    def __init__(
        self, engine, teacher, *, tokenizer_fingerprints, kind="minilm", **objective_settings
    ):
        if (
            len(tokenizer_fingerprints) != 2
            or not tokenizer_fingerprints[0]
            or tokenizer_fingerprints[0] != tokenizer_fingerprints[1]
        ):
            raise ValueError("Encoder token positions must have identical tokenization semantics")
        if kind not in {"tinybert", "minilm"}:
            raise ValueError("Unknown encoder distillation method")
        self.engine, self.kind = engine, kind
        self.teacher = engine.add_role("encoder_teacher", teacher, trainable=False)
        self.objective = (TinyBERTObjective if kind == "tinybert" else MiniLMObjective)(
            self.teacher, **objective_settings
        )
        self.fingerprint, self.updates = tokenizer_fingerprints[0], 0
        engine.register_state("encoder_distillation", self)

    def update(self, microbatches):
        result = self.engine.phase(
            "encoder_" + self.kind, objective=self.objective, microbatches=microbatches
        )
        if result.updated:
            self.updates += 1
        return result

    def export_student(self):

        state = self.engine.export_state_dict()
        if self.engine.parallel.rank != 0:
            return None
        config = self.engine.model.config
        if self.kind == "tinybert":
            state = {
                name.removeprefix("student."): value
                for name, value in state.items()
                if name.startswith("student.")
            }
        student = build_model(config)
        student.load_state_dict(state, strict=True)
        return student.eval()

    def state_dict(self):
        return {
            "objective": self.objective.config_dict(),
            "tokenizer": self.fingerprint,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (
            state.get("objective") != expected["objective"]
            or state.get("tokenizer") != self.fingerprint
        ):
            raise ValueError("Encoder distillation supervision changed")
        self.updates = state["updates"]
