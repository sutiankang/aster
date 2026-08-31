"""Collective recipe lifecycle for shared data, model, and objective execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import copy
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist

from ..core import StageResult, atomic_json, digest_json, file_digest, read_json
from ..core.serialization import RunLock
from ..data import StatefulSampler
from .parallel import ParallelContext


def recipe_context(
    parallel=None,
    *,
    allow_tensor_parallel=False,
    allow_pipeline_parallel=False,
    allow_expert_parallel=False,
):

    if parallel is None and (
        (dist.is_initialized() and dist.get_world_size() > 1)
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    ):
        raise ValueError(
            "Multi-rank recipes require explicit parallel=...; use aster distributed-train, not train/run"
        )
    context = parallel if parallel is not None else ParallelContext()
    axes = (
        ("context_parallel", "gtp_remat")
        + (() if allow_tensor_parallel else ("tensor_parallel",))
        + (() if allow_pipeline_parallel else ("pipeline_parallel",))
        + (() if allow_expert_parallel else ("expert_parallel", "expert_tensor_parallel"))
    )
    if any(getattr(context.config, name) != 1 for name in axes):
        raise ValueError(
            "Built-in recipes support DP/ZeRO only; model-parallel providers must explicitly construct a Trainer with topology-aware model, batches and objective"
        )
    return context


def collective_local(context, function, label):
    """Collectively propagate local-operation errors. The callback must not hide its own collectives."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = context.world.gather_objects(error)
    if any(errors):
        message = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item)
        raise RuntimeError(f"{label}: {message}")
    return value


def leader_call(context, function, label):
    value, error = None, None
    if context.rank == 0:
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    payload = context.world.gather_objects({"value": value, "error": error})[0]
    if payload["error"]:
        raise RuntimeError(f"{label}: {payload['error']}")
    return payload["value"]


def agree(context, value, label):
    identities = context.world.gather_objects(digest_json(value))
    if len(set(identities)) != 1:
        raise ValueError(f"All ranks must agree on {label}")


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings, context):
    device = torch.device(settings.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but this installed torch runtime has no available CUDA"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if context.world.size > 1 and device.index is not None and device.index != local_rank:
            raise ValueError(
                "Distributed training.device must be cuda or the matching LOCAL_RANK; never place every worker on cuda:0"
            )
        device = torch.device(
            "cuda",
            local_rank
            if context.world.size > 1
            else (device.index if device.index is not None else torch.cuda.current_device()),
        )
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Built-in training recipes support explicit cpu/cuda devices")
    if (
        context.world.size > 1
        and dist.get_backend(context.world.handle) == "nccl"
        and device.type != "cuda"
    ):
        raise ValueError("NCCL recipes require training.device=cuda")
    return device


class RecipeSampler(StatefulSampler):
    """Shuffle globally, then form equal-length strided replica slices with synchronized epochs."""

    def __init__(self, dataset, *, seed, context, tail="drop"):
        if tail not in {"drop", "error"}:
            raise ValueError("replica_tail must be drop/error")
        self.tail = tail
        self.dropped_per_epoch = len(dataset) % context.dp.size
        if len(dataset) < context.dp.size:
            raise ValueError(
                "Dataset has fewer records than DP replicas; reduce DP or prepare more records"
            )
        if self.dropped_per_epoch and tail == "error":
            raise ValueError("Dataset length must divide DP replicas when replica_tail=error")
        super().__init__(dataset, seed=seed, rank=context.dp.rank, world_size=context.dp.size)

    def _indices(self):
        if self._cached_epoch == self.epoch:
            return self._cached_indices
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        usable = len(indices) - self.dropped_per_epoch
        self._cached_epoch, self._cached_indices = (
            self.epoch,
            indices[:usable][self.rank :: self.world_size],
        )
        return self._cached_indices

    def state_dict(self):
        return {
            **super().state_dict(),
            "tail": self.tail,
            "dropped_per_epoch": self.dropped_per_epoch,
        }


class RecipeState:
    """Bind objective, preprocessing, and training settings to exact-resume identity."""

    def __init__(self, identity):
        self.identity = digest_json(identity)
        self.history, self.consecutive_skips = [], 0

    def state_dict(self):
        return {
            "identity": self.identity,
            "history": copy.deepcopy(self.history),
            "consecutive_skips": self.consecutive_skips,
        }

    def load_state_dict(self, state):
        if (
            set(state) != {"identity", "history", "consecutive_skips"}
            or state["identity"] != self.identity
        ):
            raise ValueError(
                "Recipe objective/data/training identity changed; resume is not a new training run"
            )
        if (
            not isinstance(state["history"], list)
            or type(state["consecutive_skips"]) is not int
            or state["consecutive_skips"] < 0
        ):
            raise ValueError("Invalid recipe history/skip state")
        self.history, self.consecutive_skips = (
            copy.deepcopy(state["history"]),
            state["consecutive_skips"],
        )


def trainer_kwargs(settings, context, device, directory, *, optimizer_factory=None):

    options = {
        "lr": settings.learning_rate,
        "device": device,
        "accumulation_steps": settings.accumulation_steps,
        "max_grad_norm": settings.max_grad_norm,
        "max_grad_value": settings.max_grad_value,
        "precision": settings.precision,
        "parallel": context,
        "zero_stage": settings.zero_stage,
        "ema_decay": settings.ema_decay,
        "offload_optimizer": settings.offload_optimizer,
        "offload_parameters": settings.offload_parameters,
        "activation_offload": settings.activation_offload,
        "communication_overlap": settings.communication_overlap,
        "bucket_bytes": settings.bucket_bytes,
    }
    if settings.offload_optimizer == "nvme":
        options["offload_directory"] = Path(directory) / f"optimizer-rank-{context.rank}"
    if getattr(settings, "optimizer", None) is not None:
        from .muon import MuonFactory

        if type(optimizer_factory) is not MuonFactory:
            raise ValueError(
                "Muon recipe requires explicit certified language FQN selection; tensor recipes are not admitted"
            )
        options["optimizer_factory"] = optimizer_factory
    elif optimizer_factory is not None:
        raise ValueError(
            "Optimizer factory cannot silently override the declared default AdamW recipe"
        )
    return options


def _training_rng_policy(engine):
    group = getattr(engine.model, "_aster_replicated_rng_group", None)
    if group is None:
        return engine.parallel.rank, "seed+global_rank; checkpoint restores exact rank RNG"
    if group is not engine.parallel.tp:
        raise ValueError("Replicated recipe RNG requires the explicit attention TP group")

    return (
        group.ranks[0],
        "seed+attention_TP_leader_rank; shared router jitter inside TP, independent data replicas; checkpoint restores exact rank RNG",
    )


def fit_engine(
    engine,
    *,
    config,
    settings,
    dataset,
    sampler,
    microbatch,
    directory,
    parents=(),
    optimizer_identity=None,
):
    directory = Path(directory)
    context = engine.parallel
    stable_settings = asdict(settings)
    for field in ("steps", "checkpoint_every", "max_consecutive_skips"):
        stable_settings.pop(field)

    if stable_settings.get("optimizer") is None:
        stable_settings.pop("optimizer", None)
    stable_config = {
        key: value for key, value in config.items() if key not in {"resume", "training"}
    }
    identity = {
        "config": stable_config,
        "training": stable_settings,
        "data": dataset.fingerprint,
        "parents": list(parents),
        "parallel": context.to_dict(),
    }
    if optimizer_identity is not None:
        identity["optimizer"] = optimizer_identity
    state = RecipeState(identity)
    agree(
        context, {"identity": state.identity, "data_length": len(dataset)}, "recipe/data identity"
    )
    engine.register_state("sampler", sampler)
    engine.register_state("recipe", state)

    offset, _ = _training_rng_policy(engine)
    seed_training(settings.seed + offset)
    if config.get("resume"):
        engine.load_checkpoint(config["resume"])
    if engine.steps > settings.steps:
        raise ValueError("Requested total steps are below the resumed checkpoint step")
    while engine.steps < settings.steps:
        batches = collective_local(
            context,
            lambda: [microbatch() for _ in range(settings.accumulation_steps)],
            "Prepare microbatches",
        )
        result = engine.step(batches)
        state.history.append(
            {
                "step": result.step,
                "loss": result.loss,
                "updated": result.updated,
                "overflow": result.overflow,
            }
        )
        state.consecutive_skips = 0 if result.updated else state.consecutive_skips + 1
        if state.consecutive_skips > settings.max_consecutive_skips:
            raise RuntimeError(
                "Too many skipped updates (overflow or zero valid loss count); inspect data/precision instead of looping forever"
            )
        if (
            result.updated
            and settings.checkpoint_every
            and engine.steps % settings.checkpoint_every == 0
        ):
            engine.save_checkpoint(directory / f"checkpoint-{engine.steps}")
    engine.save_checkpoint(directory / "checkpoint-final")
    return state


def publish_model(
    engine,
    *,
    config,
    model_config,
    settings,
    dataset,
    sampler,
    state,
    directory,
    store,
    kind,
    metadata,
    extra_files,
    parents=(),
    optimizer_identity=None,
):

    from ..models import build_model

    directory = Path(directory)
    tensors = engine.export_state_dict()
    runtime = engine.export_runtime_state()
    evidence = {
        "parallel": engine.parallel.to_dict(),
        "zero_stage": settings.zero_stage,
        "batch_size_per_replica": settings.batch_size,
        "global_batch_size": settings.batch_size
        * settings.accumulation_steps
        * engine.parallel.dp.size,
        "replica_tail": settings.replica_tail,
        "dropped_records_per_epoch": sampler.dropped_per_epoch,
        "training_rng": _training_rng_policy(engine)[1],
        "backend": "native_torch_collectives",
    }
    if optimizer_identity is not None:
        evidence["optimizer"] = optimizer_identity

    def publish():
        export = directory / "export"
        complete = build_model(model_config)
        complete.load_state_dict(tensors, strict=True)
        from .runtime_state import apply_runtime_state

        apply_runtime_state(complete, runtime)
        complete.save_pretrained(export / "model")
        extra_files(export)
        atomic_json(
            export / "recipe.json",
            {
                "config": config,
                "data_fingerprint": dataset.fingerprint,
                "training": asdict(settings),
                "parents": parents,
                "execution": evidence,
            },
        )
        artifact = store.publish(
            export, kind=kind, metadata={**metadata, "execution": evidence}, parents=parents
        )
        atomic_json(directory / "history.json", state.history)
        metrics = (
            {"final_loss": state.history[-1]["loss"]}
            if state.history and state.history[-1]["loss"] is not None
            else {}
        )
        return asdict(
            StageResult(
                {"model": artifact.id},
                metrics,
                {
                    "steps": engine.steps,
                    "checkpoint": str(directory / "checkpoint-final"),
                    **evidence,
                },
            )
        )

    return StageResult(**leader_call(engine.parallel, publish, "Publish trained model"))


@contextmanager
def collective_run_directory(context, directory, signature):

    directory = Path(directory)
    lock = RunLock(directory / "run.lock")
    leader_call(context, lambda: (lock.__enter__(), None)[1], "Acquire distributed run lock")
    try:
        probe = {
            "signature": signature,
            "nonce": leader_call(
                context, lambda: uuid.uuid4().hex, "Create shared filesystem probe"
            ),
        }
        leader_call(
            context,
            lambda: atomic_json(directory / "distributed-input.json", probe),
            "Write distributed run identity",
        )
        identity = collective_local(
            context,
            lambda: read_json(directory / "distributed-input.json"),
            "Verify shared output filesystem",
        )
        errors = context.world.gather_objects(identity != probe)
        if any(errors):
            raise ValueError("Distributed output directory is not a coherent shared filesystem")
        yield directory
    finally:
        leader_call(
            context, lambda: lock.__exit__(None, None, None), "Release distributed run lock"
        )


def run_distributed_recipe(config, *, kind, directory, store, parallel):

    from ..recipes import fit_language
    from ..tensor_recipes import fit_tensors

    if kind not in {"language", "tensor"}:
        raise ValueError("Distributed recipe kind must be language/tensor")
    context = recipe_context(
        parallel,
        allow_tensor_parallel=kind == "language",
        allow_pipeline_parallel=kind == "language",
        allow_expert_parallel=kind == "language",
    )
    source = Path(__file__).resolve().parents[1]
    package_signature = collective_local(
        context,
        lambda: digest_json(
            {
                str(path.relative_to(source)): file_digest(path)
                for path in sorted(source.rglob("*.py"))
            }
        ),
        "Fingerprint training implementation",
    )
    signature = digest_json(
        {"kind": kind, "config": config, "parallel": context.to_dict(), "code": package_signature}
    )
    agree(
        context,
        {
            "signature": signature,
            "output": str(Path(directory).absolute()),
            "store": str(store.root),
        },
        "distributed run configuration",
    )
    with collective_run_directory(context, directory, signature) as path:
        manifest = path / "stage.json"
        prior = leader_call(
            context,
            lambda: read_json(manifest) if manifest.exists() else None,
            "Inspect distributed stage",
        )
        if prior is not None:
            if prior.get("signature") != signature or prior.get("status") != "complete":
                raise ValueError(
                    "Existing distributed run differs or is incomplete; use a new output directory and explicit resume"
                )
            leader_call(
                context,
                lambda: [store.get(value).id for value in prior["result"]["artifacts"].values()],
                "Verify completed artifacts",
            )
            return StageResult(**prior["result"])
        leader_call(
            context,
            lambda: atomic_json(manifest, {"signature": signature, "status": "started"}),
            "Start distributed stage",
        )
        try:
            fit = fit_language if kind == "language" else fit_tensors
            result = fit(config, {}, path, store, parallel=context)
            leader_call(
                context,
                lambda: atomic_json(
                    manifest,
                    {"signature": signature, "status": "complete", "result": asdict(result)},
                ),
                "Commit distributed stage",
            )
            return result
        except Exception as exc:
            if context.rank == 0:
                atomic_json(
                    manifest,
                    {
                        "signature": signature,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise
