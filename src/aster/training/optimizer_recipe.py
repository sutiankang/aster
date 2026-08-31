"""Serializable optimizer configuration using logical parameter layouts."""

from dataclasses import asdict, dataclass
import math
from .muon import MuonFactory, SOURCES


@dataclass(frozen=True)
class MuonSettings:
    profile: str
    matrix_learning_rate: float
    auxiliary_modules: tuple[str, ...] = ("lm_head",)
    matrix_weight_decay: float | None = None
    auxiliary_weight_decay: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    normalization_epsilon: float = 1e-7
    auxiliary_betas: tuple[float, float] = (0.9, 0.95)
    auxiliary_epsilon: float | None = None
    missing_grad: str = "skip"
    type: str = "muon"

    def __post_init__(self):
        if self.type != "muon" or self.profile not in SOURCES:
            raise ValueError("Muon recipe requires type=muon and explicit keller/moonlight profile")
        if not isinstance(self.auxiliary_modules, (tuple, list)) or not self.auxiliary_modules:
            raise ValueError(
                "Muon auxiliary_modules must be explicit module FQNs including lm_head"
            )
        modules = tuple(self.auxiliary_modules)
        if (
            any(
                not isinstance(name, str)
                or not name
                or any(not (part.isidentifier() or part.isdigit()) for part in name.split("."))
                for name in modules
            )
            or len(set(modules)) != len(modules)
            or "lm_head" not in modules
        ):
            raise ValueError(
                "Muon auxiliary module FQNs must be distinct, exact, and include lm_head; no glob guessing"
            )
        object.__setattr__(self, "auxiliary_modules", modules)
        moon = self.profile == "moonlight"
        for key, value in (
            ("matrix_weight_decay", 0.1 if moon else 0.0),
            ("auxiliary_weight_decay", 0.1 if moon else 0.0),
            ("auxiliary_epsilon", 1e-8 if moon else 1e-10),
        ):
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        positive = ("matrix_learning_rate", "normalization_epsilon", "auxiliary_epsilon")
        nonnegative = ("matrix_weight_decay", "auxiliary_weight_decay")
        for key in positive + nonnegative + ("momentum",):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (key in positive and value == 0)
            ):
                raise ValueError(f"Muon {key} must be a finite valid numeric value")
            object.__setattr__(self, key, float(value))
        if (
            self.momentum >= 1
            or type(self.nesterov) is not bool
            or type(self.ns_steps) is not int
            or not 1 <= self.ns_steps < 100
        ):
            raise ValueError("Invalid Muon momentum/Nesterov/Newton-Schulz controls")
        if (
            not isinstance(self.auxiliary_betas, (tuple, list))
            or len(self.auxiliary_betas) != 2
            or any(
                type(beta) not in (int, float) or not math.isfinite(beta) or not 0 <= beta < 1
                for beta in self.auxiliary_betas
            )
        ):
            raise ValueError("Muon auxiliary_betas needs two finite values in [0,1)")
        object.__setattr__(
            self, "auxiliary_betas", tuple(float(beta) for beta in self.auxiliary_betas)
        )
        if self.missing_grad not in {"skip", "zero"}:
            raise ValueError("Muon missing_grad must explicitly be skip or zero")

    def to_dict(self):
        values = asdict(self)
        values["auxiliary_modules"], values["auxiliary_betas"] = (
            list(self.auxiliary_modules),
            list(self.auxiliary_betas),
        )
        return values


def parse_optimizer_settings(value):
    """Preserve Trainer AdamW defaults for omitted/null/explicit AdamW settings."""
    if value is None or type(value) is MuonSettings:
        return value
    if not isinstance(value, dict):
        raise ValueError("Training optimizer must be a JSON object or null")
    if value.get("type") == "adamw":
        if set(value) != {"type"}:
            raise ValueError(
                "Default AdamW recipe accepts only type; learning_rate remains training.learning_rate"
            )
        return None
    if value.get("type") != "muon":
        raise ValueError("Recipe optimizer type must explicitly be adamw or muon")
    try:
        return MuonSettings(**value)
    except TypeError as error:
        raise ValueError(f"Unknown or missing Muon optimizer fields: {error}") from error


def validate_optimizer_recipe(settings, context, provider, model_config):

    if settings.optimizer is None:
        return
    if provider not in {"dense", "native_tp"}:
        raise ValueError(
            "Muon recipe supports only certified dense/native_tp language providers; no PP/EP"
        )
    if any(
        getattr(context.config, axis, 1) != 1
        for axis in (
            "pipeline_parallel",
            "context_parallel",
            "expert_parallel",
            "expert_tensor_parallel",
            "gtp_remat",
        )
    ):
        raise ValueError(
            "Muon recipe supports only TP x DP x ZeRO; PP/CP/EP/ETP/GTP are not certified"
        )
    if model_config.get("architecture") not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("Muon language recipe currently certifies only native Llama/Qwen2/Qwen3")
    if (
        type(settings.learning_rate) not in (int, float)
        or not math.isfinite(settings.learning_rate)
        or settings.learning_rate <= 0
    ):
        raise ValueError("Muon auxiliary Adam learning_rate must be finite and positive")


def build_recipe_optimizer(settings, model):
    """Fix logical parameter names on the complete model, then bind the sharded owners."""
    if settings.optimizer is None:
        return None, None
    from ..models import CausalLM, LlamaConfig, Qwen2Config, Qwen3Config

    if type(model) is not CausalLM or type(model.config) not in {
        LlamaConfig,
        Qwen2Config,
        Qwen3Config,
    }:
        raise ValueError(
            "Muon recipe FQN selection requires the original certified dense causal model"
        )
    c = settings.optimizer
    source = SOURCES[c.profile]
    common = dict(
        missing_grad=c.missing_grad, source_commit=source["commit"], source_sha256=source["sha256"]
    )
    muon = dict(
        lr=c.matrix_learning_rate,
        weight_decay=c.matrix_weight_decay,
        momentum=c.momentum,
        nesterov=c.nesterov,
        ns_steps=c.ns_steps,
        normalization_epsilon=c.normalization_epsilon,
        matrix_kind="matrix",
        **common,
    )
    auxiliary = dict(
        lr=settings.learning_rate,
        weight_decay=c.auxiliary_weight_decay,
        betas=c.auxiliary_betas,
        eps=c.auxiliary_epsilon,
        **common,
    )
    factory = MuonFactory.from_model(
        model,
        auxiliary_modules=c.auxiliary_modules,
        profile=c.profile,
        muon_options=muon,
        auxiliary_options=auxiliary,
    )

    groups = [
        {key: list(value) if isinstance(value, tuple) else value for key, value in group.items()}
        for group in factory.groups
    ]
    identity = dict(
        type="native_muon_with_aux_adam",
        settings=c.to_dict(),
        auxiliary_learning_rate=settings.learning_rate,
        source=dict(source),
        groups=groups,
        selection="full_dense_fqn_embeddings_heads_vectors_aux_v1",
        matrix_execution="full_logical_matrix_gather_then_reshard_reference",
    )
    return factory, identity
