"""Semantically aligned continuous-output distillation across modalities."""

from dataclasses import dataclass, asdict

import torch
from torch import nn

from ..core import FieldOutput, LossTerm
from ..models.actions import ActionOutput


@dataclass(frozen=True)
class PredictionAlignment:
    domain: str
    output_space_fingerprint: str
    student_processor_fingerprint: str
    teacher_processor_fingerprint: str
    teacher_artifact_id: str
    field_parameterization: str | None = None
    feature_key: str | None = None

    def __post_init__(self):
        if self.domain not in {"action", "field", "feature"} or not all(
            (
                self.output_space_fingerprint,
                self.student_processor_fingerprint,
                self.teacher_processor_fingerprint,
                self.teacher_artifact_id,
            )
        ):
            raise ValueError("Output space, processors and teacher artifact must be pinned")
        if (self.domain == "field") != (self.field_parameterization is not None):
            raise ValueError("Only field distillation declares a field parameterization")
        if self.domain == "feature" and not self.feature_key:
            raise ValueError("Feature output requires an explicit output field name")


class PredictionDistillationObjective(nn.Module):
    def __init__(self, teacher, alignment, *, distance="mse"):
        super().__init__()
        if distance not in {"mse", "l1"}:
            raise ValueError("Continuous output distillation supports explicit L1 or MSE")
        self.teacher = teacher.eval().requires_grad_(False)
        self.alignment, self.distance = alignment, distance

    def config_dict(self):
        return {
            "type": "aligned_prediction_distillation",
            "alignment": asdict(self.alignment),
            "distance": self.distance,
        }

    def _prediction(self, output):
        c = self.alignment
        if c.domain == "action":
            if not isinstance(output, ActionOutput):
                raise TypeError("Action distillation requires the typed ActionOutput")
            return output.actions
        if c.domain == "field":
            if (
                not isinstance(output, FieldOutput)
                or output.prediction_type != c.field_parameterization
            ):
                raise ValueError("Teacher/student field parameterizations differ")
            return output.prediction
        value = (
            output.get(c.feature_key)
            if isinstance(output, dict)
            else getattr(output, c.feature_key, None)
        )
        if not isinstance(value, torch.Tensor):
            raise ValueError("Declared feature field is absent or not a Tensor")
        return value

    def forward(self, model, batch):
        if batch.get("alignment") != asdict(self.alignment):
            raise ValueError(
                "Batch processor/output space/teacher provenance differs from KD contract"
            )
        student_inputs, teacher_inputs = batch["student_inputs"], batch["teacher_inputs"]
        if self.alignment.domain == "action" and (
            "actions" in student_inputs or "actions" in teacher_inputs
        ):
            raise ValueError(
                "Policy distillation uses executable priors, not a posterior that reads target actions"
            )
        if self.alignment.domain == "field":
            for key in ("sample", "time"):
                if (
                    key not in student_inputs
                    or key not in teacher_inputs
                    or not torch.equal(student_inputs[key], teacher_inputs[key])
                ):
                    raise ValueError("Field distillation must use exactly the same sample and time")
        if {id(value) for value in model.parameters()} & {
            id(value) for value in self.teacher.parameters()
        }:
            raise ValueError("Teacher and student must not share parameter ownership")
        self.teacher.eval()
        prediction = self._prediction(model(**student_inputs))
        with torch.no_grad():
            target = self._prediction(self.teacher(**teacher_inputs))
        if prediction.shape != target.shape or prediction.ndim < 2:
            raise ValueError(
                "Output axes must align; learned projection belongs explicitly to the student"
            )
        valid = batch.get("valid")
        if valid is None:
            valid = torch.ones_like(prediction, dtype=torch.bool)
        if valid.shape != prediction.shape or valid.dtype != torch.bool:
            raise ValueError(
                "KD validity mask must explicitly align every supervised output element"
            )
        error = prediction.float() - target.float()
        values = error.square() if self.distance == "mse" else error.abs()
        return LossTerm(
            values.masked_select(valid).sum(),
            valid.sum().to(values),
            self.alignment.domain + "_element",
            "prediction_distillation",
        )


class MultimodalDistillationMethod:
    def __init__(self, engine, teacher, alignment, *, distance="mse"):
        self.engine = engine
        self.teacher = engine.add_role("prediction_teacher", teacher, trainable=False)
        self.objective = PredictionDistillationObjective(self.teacher, alignment, distance=distance)
        self.updates = 0
        engine.register_state("multimodal_distillation", self)

    def update(self, microbatches):
        result = self.engine.phase(
            "multimodal_distillation", objective=self.objective, microbatches=microbatches
        )
        if result.updated:
            self.updates += 1
        return result

    def state_dict(self):
        return {"objective": self.objective.config_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        if state["objective"] != self.objective.config_dict():
            raise ValueError("Continuous KD alignment/teacher changed")
        self.updates = state["updates"]
