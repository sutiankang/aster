import torch
from torch import nn
import torch.nn.functional as F
from ..core import LossTerm, LossBundle


class ACTObjective(nn.Module):
    def __init__(self, *, kl_weight=10.0, padding_weight=0.0, normalization="padded"):
        super().__init__()
        if kl_weight < 0 or padding_weight < 0:
            raise ValueError("Negative ACT loss weight")
        if normalization not in {"padded", "valid"}:
            raise ValueError("Unknown ACT normalization")
        self.kl_weight, self.padding_weight, self.normalization = (
            kl_weight,
            padding_weight,
            normalization,
        )

    def config_dict(self):
        return {
            "type": "act_cvae",
            "kl_weight": self.kl_weight,
            "padding_weight": self.padding_weight,
            "normalization": self.normalization,
        }

    def forward(self, model, batch):
        padding = batch.get(
            "action_padding",
            torch.zeros(
                batch["actions"].shape[:2], dtype=torch.bool, device=batch["actions"].device
            ),
        )
        output = model(
            batch["proprio"],
            batch["vision_tokens"],
            actions=batch["actions"],
            action_padding=padding,
            vision_padding=batch.get("vision_padding"),
            vision_positions=batch.get("vision_positions"),
        )
        valid = (~padding)[..., None].expand_as(batch["actions"])

        reconstruction = (output.actions - batch["actions"]).abs().masked_select(valid)
        kl = -0.5 * (1 + output.logvar - output.mean.square() - output.logvar.exp()).sum(-1)
        count = (
            kl.new_tensor(valid.numel()) if self.normalization == "padded" else valid.sum().to(kl)
        )
        terms = [
            LossTerm(
                reconstruction.sum(),
                count,
                "action_slot_element" if self.normalization == "padded" else "action_element",
                "action_reconstruction",
            ),
            LossTerm(kl.sum(), kl.new_tensor(len(kl)), "sequence", "action_kl", self.kl_weight),
        ]
        if self.padding_weight:
            values = F.binary_cross_entropy_with_logits(
                output.pad_logits, padding.to(output.pad_logits), reduction="sum"
            )
            terms.append(
                LossTerm(
                    values,
                    values.new_tensor(padding.numel()),
                    "action_slot",
                    "action_padding",
                    self.padding_weight,
                )
            )
        return LossBundle(tuple(terms))


class PiActionObjective(nn.Module):
    """Use the OpenPI direction: t=1 is noise, t=0 is action.
    Training times follow Beta(1.5, 1) * 0.999 + 0.001."""

    def forward(self, model, batch):
        actions = batch["actions"]
        noise = batch.get("noise")
        time = batch.get("time")
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.rand(len(actions), device=actions.device).pow(1 / 1.5) * 0.999 + 0.001
        if time.shape != (len(actions),) or ((time < 0) | (time > 1)).any():
            raise ValueError("Pi time must be B in [0,1]")
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        result = model(noisy, time, batch["observation"])
        if result.prediction_type != "velocity":
            raise ValueError("Pi action expert must predict time derivative")
        errors = (result.prediction - (noise - actions)).square()
        return LossTerm(
            errors.sum(), errors.new_tensor(errors.numel()), "action_element", "pi_flow"
        )
