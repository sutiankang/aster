"""Selective state-space recurrence with sequence-length-independent memory."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from aster.core.contracts import StateCapabilities


@dataclass(frozen=True)
class MambaState:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seen_tokens: int
    model_key: str
    kind: str = "mamba_ssm"

    @property
    def capabilities(self):
        return StateCapabilities(self.kind, forkable=True, reorderable=True, replayable=True)

    def fork(self):
        return type(self)(
            tuple((conv.clone(), memory.clone()) for conv, memory in self.layers),
            self.seen_tokens,
            self.model_key,
        )

    def reorder(self, indices):
        return type(self)(
            tuple(
                (conv.index_select(0, indices), memory.index_select(0, indices))
                for conv, memory in self.layers
            ),
            self.seen_tokens,
            self.model_key,
        )

    def truncate(self, length):
        raise ValueError(
            "Selective memory has no token axis to truncate; restore checkpoint and replay"
        )


def selective_scan(inputs, dt, a_log, b, c, skip, gate, initial=None):
    """Use input/step/gate [B,S,I], B/C [B,S,N], A_log [I,N], and memory [B,I,N]."""
    batch, length, width = inputs.shape
    if (
        dt.shape != inputs.shape
        or gate.shape != inputs.shape
        or b.shape != c.shape
        or b.shape[:2] != (batch, length)
        or a_log.shape != (width, b.shape[-1])
    ):
        raise ValueError("Selective scan dimensions disagree")
    memory = (
        torch.zeros(batch, width, b.shape[-1], device=inputs.device, dtype=torch.float32)
        if initial is None
        else initial.float()
    )
    a = -a_log.float().exp()
    outputs = []
    for index in range(length):
        step = dt[:, index].float()
        decay = torch.exp(step[..., None] * a[None])
        drive = step[..., None] * b[:, index, None].float() * inputs[:, index, :, None].float()
        memory = decay * memory + drive
        projected = (memory.to(inputs.dtype) @ c[:, index, :, None]).squeeze(-1)
        output = (projected + inputs[:, index] * skip) * F.silu(gate[:, index])
        outputs.append(output)
    return torch.stack(outputs, 1), memory
