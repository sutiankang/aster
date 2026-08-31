"""JEPA prediction objectives with jointly trained predictors and explicit EMA targets."""

import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle
from ..models.jepa import JEPAEncoder, select_patches


class JEPAObjective(nn.Module):
    def __init__(self, target_encoder, *, loss_exponent=1.0, regularization_weight=0.0):
        super().__init__()
        if loss_exponent <= 0 or regularization_weight < 0:
            raise ValueError("Invalid JEPA loss settings")
        self.target_encoder = target_encoder.eval().requires_grad_(False)
        self.loss_exponent, self.regularization_weight = loss_exponent, regularization_weight

    def config_dict(self):
        return {
            "type": "jepa",
            "loss_exponent": self.loss_exponent,
            "regularization_weight": self.regularization_weight,
        }

    def forward(self, model, batch):
        pixels = batch["pixel_values"]
        contexts, targets = batch["context_indices"], batch["target_indices"]
        contexts = [contexts] if isinstance(contexts, torch.Tensor) else contexts
        targets = [targets] if isinstance(targets, torch.Tensor) else targets
        if not contexts or len(contexts) != len(targets):
            raise ValueError("JEPA context and target mask pairs must align")
        self.target_encoder.eval()
        with torch.no_grad():
            encoded = self.target_encoder(pixels)
            encoded = F.layer_norm(encoded, (encoded.shape[-1],))
        prediction_values, deviations = [], []
        for index, (context, target) in enumerate(zip(contexts, targets)):
            prediction = model(
                pixels, context, target, mask_index=index % model.config.num_mask_tokens
            )
            desired = select_patches(encoded, target)
            prediction_values.append(
                (prediction.float() - desired.float()).abs().pow(self.loss_exponent).mean((1, 2))
                / self.loss_exponent
            )
            if self.regularization_weight:
                if prediction.shape[1] < 2:
                    raise ValueError(
                        "JEPA variance regularization needs at least two target patches"
                    )
                deviations.append((prediction.float().var(1, correction=1) + 1e-4).sqrt())

        values = torch.stack(prediction_values).mean(0)
        count = values.new_tensor(len(values))
        terms = [LossTerm(values.sum(), count, "sample", "jepa_prediction")]
        if deviations:
            regularizer = F.relu(1 - torch.stack(deviations).mean(0)).mean(-1)
            terms.append(
                LossTerm(
                    regularizer.sum(), count, "sample", "jepa_variance", self.regularization_weight
                )
            )
        return terms[0] if len(terms) == 1 else LossBundle(tuple(terms))


class JEPAMethod:
    def __init__(
        self,
        engine,
        *,
        ema_start=0.996,
        ema_end=1.0,
        total_updates=100000,
        loss_exponent=1.0,
        regularization_weight=0.0,
    ):
        if not 0 <= ema_start <= ema_end <= 1 or total_updates < 1:
            raise ValueError("Invalid JEPA momentum schedule")
        self.engine, self.ema_start, self.ema_end, self.total_updates = (
            engine,
            ema_start,
            ema_end,
            total_updates,
        )
        target = engine.clone_target(
            "model",
            "target_encoder",
            source_path="encoder",
            factory=lambda: JEPAEncoder(engine.model.config.encoder),
        )
        self.objective = JEPAObjective(
            target, loss_exponent=loss_exponent, regularization_weight=regularization_weight
        )
        self.updates = 0
        engine.register_state("jepa_method", self)

    def update(self, microbatches):
        result = self.engine.phase("jepa", objective=self.objective, microbatches=microbatches)
        if result.updated:
            momentum = self.ema_start + (self.ema_end - self.ema_start) * min(
                self.updates / self.total_updates, 1.0
            )
            self.engine.update_target("model", "target_encoder", momentum, source_path="encoder")
            self.updates += 1
        return result

    def state_dict(self):
        return {
            "ema_start": self.ema_start,
            "ema_end": self.ema_end,
            "total_updates": self.total_updates,
            "objective": self.objective.config_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if any(state[key] != value for key, value in self.state_dict().items() if key != "updates"):
            raise ValueError("JEPA method configuration differs")
        self.updates = state["updates"]
