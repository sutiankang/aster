"""Action normalization, tokenization, and temporal ensembling without hardware side effects."""

from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class ActionSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]
    coordinate_frame: str
    representation: str
    control_hz: float
    execution_horizon: int

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", tuple(self.units))
        if (
            not self.names
            or len(self.names) != len(self.units)
            or len(set(self.names)) != len(self.names)
            or not all(self.units)
            or not self.coordinate_frame
            or self.representation not in {"absolute", "delta", "velocity"}
            or not math.isfinite(self.control_hz)
            or self.control_hz <= 0
            or self.execution_horizon < 1
        ):
            raise ValueError("ActionSpec needs explicit physical semantics")

    def to_dict(self):
        return asdict(self)


class ActionNormalizer:
    def __init__(self, center, scale, *, spec, mode="standard", clip=False):
        self.center, self.scale = torch.as_tensor(center).float(), torch.as_tensor(scale).float()
        if (
            self.center.shape != (len(spec.names),)
            or self.scale.shape != self.center.shape
            or not torch.isfinite(self.center).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
            or mode not in {"standard", "quantile"}
        ):
            raise ValueError("Invalid action statistics")
        self.spec, self.mode, self.clip = spec, mode, clip

    @classmethod
    def fit(cls, actions, *, spec, valid=None, mode="standard", clip=False):
        flat = actions.reshape(-1, actions.shape[-1])
        if valid is not None:
            flat = flat[valid.reshape(-1)]
        if not len(flat) or not torch.isfinite(flat).all():
            raise ValueError("Need finite training-only action statistics")
        if mode == "standard":
            center, scale = flat.mean(0), flat.std(0, correction=0).clamp_min(1e-6)
        elif mode == "quantile":
            low, high = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device), dim=0)
            center, scale = (low + high) / 2, ((high - low) / 2).clamp_min(1e-6)
        else:
            raise ValueError("Unknown action normalization")
        return cls(center, scale, spec=spec, mode=mode, clip=clip)

    def normalize(self, actions):
        result = (actions - self.center.to(actions)) / self.scale.to(actions)
        return result.clamp(-1, 1) if self.clip else result

    def denormalize(self, actions):
        return actions * self.scale.to(actions) + self.center.to(actions)

    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "mode": self.mode,
            "clip": self.clip,
        }


class TemporalEnsembler:
    """Align action chunks by absolute control tick; a valid zero action is not padding."""

    def __init__(self, horizon, *, decay=0.01, prefer_recent=False):
        if horizon < 1 or decay < 0 or not math.isfinite(decay):
            raise ValueError("Invalid ensemble configuration")
        self.horizon, self.decay, self.prefer_recent = horizon, decay, prefer_recent
        self.entries = []
        self.last_tick = -1

    def add(self, start_tick, actions, valid=None):
        if (
            start_tick < 0
            or actions.ndim != 2
            or len(actions) != self.horizon
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Invalid action chunk")
        if self.entries and start_tick <= self.entries[-1][0]:
            raise ValueError("Chunk timestamps must strictly increase")
        valid = (
            torch.ones(self.horizon, device=actions.device, dtype=torch.bool)
            if valid is None
            else valid
        )
        if valid.shape != (self.horizon,) or valid.dtype != torch.bool:
            raise ValueError("Invalid action mask")
        self.entries.append((start_tick, actions.detach().clone(), valid.clone()))

    def action(self, tick):
        if tick <= self.last_tick:
            raise ValueError("Controller ticks must strictly increase")
        self.entries = [entry for entry in self.entries if entry[0] + self.horizon > tick]
        selected = [
            actions[tick - start]
            for start, actions, valid in self.entries
            if 0 <= tick - start < self.horizon and valid[tick - start]
        ]
        if not selected:
            raise RuntimeError(
                "No valid action for tick; request a new chunk, do not reuse stale control"
            )
        values = torch.stack(selected)
        order = torch.arange(len(values), device=values.device, dtype=values.dtype)
        weights = ((order if self.prefer_recent else -order) * self.decay).softmax(0)
        self.last_tick = tick
        return (values * weights[:, None]).sum(0)

    def reset(self):
        self.entries.clear()
        self.last_tick = -1

    def state_dict(self):
        return {
            "horizon": self.horizon,
            "decay": self.decay,
            "prefer_recent": self.prefer_recent,
            "last_tick": self.last_tick,
            "entries": self.entries,
        }

    def load_state_dict(self, state):
        if (state["horizon"], state["decay"], state["prefer_recent"]) != (
            self.horizon,
            self.decay,
            self.prefer_recent,
        ):
            raise ValueError("Controller configuration changed")
        self.last_tick = state["last_tick"]
        self.entries = [(s, a.clone(), v.clone()) for s, a, v in state["entries"]]


class UniformActionTokenizer:
    """Use OpenVLA linspace boundaries, digitize, and reverse tail-vocabulary mapping."""

    def __init__(self, vocab_size, *, bins=256, low=-1.0, high=1.0):
        if vocab_size <= bins + 1 or bins < 2 or low >= high:
            raise ValueError("Invalid action vocabulary interval")
        self.vocab_size, self.bins, self.low, self.high = vocab_size, bins, low, high
        self.edges = torch.linspace(low, high, bins, dtype=torch.float64)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2

    def encode(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Cannot tokenize non-finite actions")
        values = actions.double().clamp(self.low, self.high)
        digitized = torch.searchsorted(
            self.edges.to(values.device), values.contiguous(), right=True
        )
        return self.vocab_size - digitized

    def decode(self, ids):
        if (
            ids.dtype not in {torch.int32, torch.int64}
            or ((ids < self.vocab_size - self.bins) | (ids >= self.vocab_size)).any()
        ):
            raise ValueError("Generated token is outside declared action vocabulary")
        indices = (self.vocab_size - ids - 1).clamp(0, self.bins - 2)
        return self.centers.to(ids.device)[indices].float()

    def to_dict(self):
        return {
            "type": "openvla_uniform",
            "vocab_size": self.vocab_size,
            "bins": self.bins,
            "low": self.low,
            "high": self.high,
        }
