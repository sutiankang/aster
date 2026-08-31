"""LeWorldModel latent-goal CEM/MPC with explicit window and control-tick semantics.

Planning reference: stable-worldmodel 0.0.5, MIT,
revision 3a85ac6888c39db90af648993fd0b23ac4c0a51d.
Copyright(c)2025 AI.QED Group @ Brown.
Use unbiased elite std, candidate zero equal to mean, no clipping, and return
the final mean. Environment action bounds remain an explicit caller contract."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import torch
from ..models.lewm import LeWorldModel
from ..data.actions import ActionNormalizer


@dataclass(frozen=True)
class LeWMCEMConfig:
    horizon: int = 5
    num_samples: int = 300
    topk: int = 30
    n_steps: int = 30
    batch_size: int = 1
    initial_std: float = 1.0

    def __post_init__(self):
        for key in ("horizon", "num_samples", "topk", "n_steps", "batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Invalid LeWM CEM dimension")
        if (
            not 2 <= self.topk <= self.num_samples
            or not math.isfinite(self.initial_std)
            or self.initial_std <= 0
        ):
            raise ValueError(
                "LeWM unbiased elite std needs 2<=topk<=samples and positive initial std"
            )


@dataclass(frozen=True)
class LeWMPlan:
    actions: torch.Tensor
    std: torch.Tensor
    elite_cost: torch.Tensor


class LeWMCEM:
    def __init__(self, model, config=LeWMCEMConfig(), *, seed=0):
        if not isinstance(model, LeWorldModel) or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("LeWM planner needs native model and explicit seed")
        self.model, self.config = model, config
        self.device = next(model.parameters()).device
        if any(type(module).__name__ == "Zero3Unit" for module in model.modules()):
            raise ValueError(
                "Export a complete inference model before planning, not live collective shards"
            )
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

    def state_dict(self):
        return dict(
            schema_version=1,
            model=self.model.config.to_dict(),
            config=asdict(self.config),
            device=str(self.device),
            rng=self.rng.get_state(),
        )

    def load_state_dict(self, state):
        identity = self.state_dict()
        if set(state) != set(identity) or any(
            state[key] != identity[key] for key in identity if key != "rng"
        ):
            raise ValueError("LeWM CEM checkpoint identity differs")
        probe = torch.Generator(device=self.device)
        probe.set_state(state["rng"])
        self.rng.set_state(state["rng"])

    @torch.no_grad()
    def solve(self, pixels, goal_pixels, *, init_action=None):
        c = self.config
        m = self.model.config
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 5
            or pixels.shape[1] > m.history_size
            or min(pixels.shape) < 1
        ):
            raise ValueError("LeWM planning pixels must be nonempty B,H,C,W,W history")
        if (
            not isinstance(goal_pixels, torch.Tensor)
            or goal_pixels.ndim != 5
            or len(goal_pixels) != len(pixels)
            or min(goal_pixels.shape) < 1
        ):
            raise ValueError("LeWM planning needs aligned B,T,C,H,W goal pixels")
        if (
            pixels.shape[1] > c.horizon
            or pixels.shape[2] != m.encoder.num_channels
            or goal_pixels.shape[2] != m.encoder.num_channels
        ):
            raise ValueError("LeWM planning horizon/channel mismatch")
        if any(
            value.dtype != torch.float32
            or value.device != self.device
            or not torch.isfinite(value).all()
            for value in (pixels, goal_pixels)
        ):
            raise ValueError(
                "LeWM planner requires finite preprocessed FP32 pixels on model device"
            )
        b = len(pixels)
        mean = torch.zeros(b, c.horizon, m.action_dim, device=self.device)
        if init_action is not None:
            if (
                init_action.ndim != 3
                or init_action.shape[0] != b
                or init_action.shape[1] > c.horizon
                or init_action.shape[2] != m.action_dim
                or init_action.dtype != torch.float32
                or init_action.device != self.device
                or not torch.isfinite(init_action).all()
            ):
                raise ValueError("Invalid explicit warm-start action prefix")
            mean[:, : init_action.shape[1]] = init_action
        std = torch.full_like(mean, c.initial_std)
        elite_cost = torch.empty(b, device=self.device)

        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            for start in range(0, b, c.batch_size):
                end = min(b, start + c.batch_size)
                size = end - start
                history = self.model.encode(pixels[start:end])
                goal = self.model.encode(goal_pixels[start:end])[:, -1]
                local_mean, local_std = mean[start:end], std[start:end]
                for _ in range(c.n_steps):
                    noise = torch.randn(
                        size,
                        c.num_samples,
                        c.horizon,
                        m.action_dim,
                        generator=self.rng,
                        device=self.device,
                    )
                    candidates = noise * local_std[:, None] + local_mean[:, None]
                    candidates[:, 0] = local_mean
                    prediction = self.model.rollout_latents(history, candidates)[:, :, -1]
                    costs = (prediction - goal[:, None]).square().sum(-1)
                    if not torch.isfinite(costs).all():
                        raise ValueError("Non-finite LeWM candidate cost")
                    values, indices = costs.topk(c.topk, dim=1, largest=False)
                    selected = candidates[torch.arange(size, device=self.device)[:, None], indices]
                    local_mean, local_std = selected.mean(1), selected.std(1, correction=1)
                mean[start:end], std[start:end] = local_mean, local_std
                elite_cost[start:end] = values.mean(1)
        finally:
            for module, training in modes:
                module.training = training
        return LeWMPlan(mean.detach(), std.detach(), elite_cost.detach())


class LeWMMPC:
    def __init__(self, planner, *, normalizer, action_block=1, receding_horizon=1, warm_start=True):
        if (
            not isinstance(planner, LeWMCEM)
            or not isinstance(normalizer, ActionNormalizer)
            or normalizer.clip
        ):
            raise ValueError("LeWM MPC needs CEM and explicit non-clipping action normalization")
        if (
            type(action_block) is not int
            or action_block < 1
            or type(receding_horizon) is not int
            or not 1 <= receding_horizon <= planner.config.horizon
        ):
            raise ValueError("Invalid LeWM receding/action-block horizon")
        if len(normalizer.spec.names) * action_block != planner.model.config.action_dim:
            raise ValueError("Action block does not match trained LeWM action dimension")
        self.planner, self.normalizer = planner, normalizer
        self.action_block, self.receding_horizon, self.warm_start = (
            action_block,
            receding_horizon,
            bool(warm_start),
        )
        self.pending, self.next_init = [], None

    def reset(self):
        self.pending, self.next_init = [], None

    def act(self, pixels, goal_pixels):
        if self.pending and len(self.pending[0]) != len(pixels):
            raise ValueError("Reset MPC before changing environment batch size")
        if not self.pending:
            result = self.planner.solve(pixels, goal_pixels, init_action=self.next_init)
            self.next_init = (
                result.actions[:, self.receding_horizon :].clone() if self.warm_start else None
            )
            plan = result.actions[:, : self.receding_horizon].reshape(
                len(pixels), self.receding_horizon * self.action_block, -1
            )
            self.pending = list(plan.transpose(0, 1).unbind(0))
        return self.normalizer.denormalize(self.pending.pop(0)).clone()

    def state_dict(self):
        return dict(
            schema_version=1,
            planner=self.planner.state_dict(),
            normalizer=self.normalizer.to_dict(),
            action_block=self.action_block,
            receding_horizon=self.receding_horizon,
            warm_start=self.warm_start,
            pending=[item.clone() for item in self.pending],
            next_init=None if self.next_init is None else self.next_init.clone(),
        )

    def load_state_dict(self, state):
        own = self.state_dict()
        if set(state) != set(own) or any(
            state[key] != own[key]
            for key in (
                "schema_version",
                "normalizer",
                "action_block",
                "receding_horizon",
                "warm_start",
            )
        ):
            raise ValueError("LeWM MPC state identity differs")
        pending, initial = state["pending"], state["next_init"]
        if (
            not isinstance(pending, list)
            or len(pending) > self.action_block * self.receding_horizon
        ):
            raise ValueError("Invalid LeWM pending queue")
        if any(
            not isinstance(item, torch.Tensor)
            or item.ndim != 2
            or item.shape[-1] != len(self.normalizer.spec.names)
            or item.device != self.planner.device
            or not torch.isfinite(item).all()
            for item in pending
        ):
            raise ValueError("Invalid LeWM pending action tensor")
        if initial is not None and (
            not isinstance(initial, torch.Tensor)
            or initial.ndim != 3
            or initial.shape[1:]
            != (
                self.planner.config.horizon - self.receding_horizon,
                self.planner.model.config.action_dim,
            )
            or initial.device != self.planner.device
            or not torch.isfinite(initial).all()
        ):
            raise ValueError("Invalid LeWM warm-start tensor")
        self.planner.load_state_dict(state["planner"])
        self.pending = [item.clone() for item in pending]
        self.next_init = None if initial is None else initial.clone()
