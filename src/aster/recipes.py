"""Composable data-to-training-to-artifact-to-evaluation workflows."""

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from .core import ArtifactStore, StageResult, atomic_json, digest_json, file_digest, read_json
from .data import ByteTokenizer, load_tokenizer, JsonlDataset, causal_collate
from .models import build_model, load_model
from .models.config import config_from_dict
from .methods import CrossEntropyObjective, DistillationObjective
from .training import Trainer
from .training.optimizer_recipe import MuonSettings, parse_optimizer_settings
from .evaluation import (
    ComparisonProtocol,
    EvaluationRecord,
    EvaluationRun,
    perplexity,
    quality_gate,
)


@dataclass(frozen=True)
class TrainSettings:
    steps: int = 100
    batch_size: int = 4
    accumulation_steps: int = 1
    max_length: int = 128
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    max_grad_value: float | None = None
    precision: str = "fp32"
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 0
    zero_stage: int = 0
    ema_decay: float | None = None
    offload_optimizer: str = "none"
    offload_parameters: str = "none"
    activation_offload: str = "none"
    communication_overlap: bool = False
    bucket_bytes: int = 25 * 1024 * 1024
    replica_tail: str = "drop"
    max_consecutive_skips: int = 16
    optimizer: MuonSettings | dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "optimizer", parse_optimizer_settings(self.optimizer))
        if any(
            type(value) is not int or value < 1
            for value in (
                self.steps,
                self.batch_size,
                self.accumulation_steps,
                self.max_length,
                self.bucket_bytes,
            )
        ):
            raise ValueError(
                "Training steps/batch/accumulation/length/bucket must be positive integers"
            )
        if any(
            type(value) is not int or value < 0
            for value in (self.seed, self.checkpoint_every, self.max_consecutive_skips)
        ):
            raise ValueError("Training seed/checkpoint/skip limit must be nonnegative integers")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("Learning rate must be finite and positive")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive or null")
        if self.max_grad_value is not None and (
            type(self.max_grad_value) not in {int, float}
            or not math.isfinite(self.max_grad_value)
            or self.max_grad_value <= 0
        ):
            raise ValueError("max_grad_value must be positive or null")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(
                "Unsupported recipe precision; FP8 needs an explicit kernel/model provider"
            )
        if type(self.zero_stage) is not int or self.zero_stage not in {0, 1, 2, 3}:
            raise ValueError("zero_stage must be 0/1/2/3")
        if self.ema_decay is not None and (
            not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1
        ):
            raise ValueError("ema_decay must be in [0,1)")
        if self.offload_optimizer not in {"none", "cpu", "nvme"} or self.activation_offload not in {
            "none",
            "cpu",
        }:
            raise ValueError("Unsupported optimizer/activation offload")
        if self.offload_parameters not in {"none", "cpu"} or (
            self.offload_parameters != "none" and self.zero_stage != 3
        ):
            raise ValueError("Parameter offload requires ZeRO3 and supports CPU storage only")
        if type(self.communication_overlap) is not bool or (
            self.communication_overlap and self.zero_stage != 0
        ):
            raise ValueError("Communication overlap currently requires ZeRO0")
        if self.replica_tail not in {"drop", "error"}:
            raise ValueError("replica_tail must explicitly be drop/error")


class LanguageData:
    def __init__(self, path, tokenizer, max_length):
        self.records = JsonlDataset(path)
        self.tokenizer, self.max_length = tokenizer, max_length
        self.fingerprint = digest_json(
            {
                "file": self.records.fingerprint,
                "tokenizer": tokenizer.to_dict(),
                "max_length": max_length,
            }
        )

    def __len__(self):
        return len(self.records)

    def verify(self):
        self.records.verify()

    def __getitem__(self, index):
        record = self.records[index]
        if "text" in record:
            ids = self.tokenizer.encode(record["text"]) + [self.tokenizer.eos_token_id]
            labels = list(ids)
        elif "input_ids" in record:
            ids = list(record["input_ids"])
            labels = list(record.get("labels", ids))
        else:
            raise ValueError("Language JSONL needs text or explicit input_ids/labels")
        if not 2 <= len(ids) <= self.max_length or len(labels) != len(ids):
            raise ValueError(
                "Prepare document chunks fitting max_length; recipe never silently truncates supervision"
            )
        if any(type(i) is not int or not 0 <= i < self.tokenizer.vocab_size for i in ids):
            raise ValueError("Input token outside artifact vocabulary")
        return {"input_ids": ids, "labels": labels}


def _batch(dataset, sampler, size, pad_id):
    records = []
    while len(records) < size:
        chunk = sampler.take(size - len(records))
        records.extend(chunk)
        if not chunk:
            if not len(dataset):
                raise ValueError("Empty training dataset")
            sampler.next_epoch()
    return causal_collate(records, pad_token_id=pad_id)


def _device_batch(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _device_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_device_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_device_batch(value, device) for value in batch]
    return batch


def load_predictor_artifact(artifact, *, device="cpu"):

    if (artifact.path / "model" / "optimization.json").is_file():
        from .inference.optimization import load_optimized_model

        model = load_optimized_model(artifact.path / "model")
    else:
        model = load_model(artifact.path / "model")
    return model.to(device), load_tokenizer(artifact.path / "tokenizer")


def _input_artifact(config, inputs, key="model"):
    artifact_id = config.get("artifact")
    if artifact_id is None:
        if len(inputs) != 1:
            raise ValueError("Provide one unambiguous input artifact or explicit artifact ID")
        artifact_id = next(iter(inputs.values()))["artifacts"][key]
    return artifact_id


def fit_language(config, inputs, directory, store, *, parallel=None):
    from .training.recipes import (
        recipe_context,
        agree,
        collective_local,
        seed_training,
        resolve_device,
        RecipeSampler,
        trainer_kwargs,
        fit_engine,
        publish_model,
    )

    context = recipe_context(
        parallel,
        allow_tensor_parallel=True,
        allow_pipeline_parallel=True,
        allow_expert_parallel=True,
    )
    agree(context, {"config": config, "inputs": inputs}, "language recipe inputs")
    allowed = {
        "model",
        "data",
        "tokenizer",
        "training",
        "distillation",
        "resume",
        "training_provider",
        "pipeline_schedule",
        "router_aux_coefficient",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown language recipe fields: {set(config) - allowed}")
    provider = config.get("training_provider", "dense")
    if provider not in {"dense", "native_tp", "native_pipeline", "native_moe"}:
        raise ValueError(
            "training_provider must explicitly be dense/native_tp/native_pipeline/native_moe"
        )
    if (context.ep.size > 1 or context.etp.size > 1) and provider != "native_moe":
        raise ValueError("EP/ETP recipes require training_provider=native_moe")
    if "router_aux_coefficient" in config and provider != "native_moe":
        raise ValueError("router_aux_coefficient belongs to native_moe")
    if context.tp.size > 1 and provider == "dense":
        raise ValueError("TP language recipes require training_provider=native_tp/native_pipeline")
    if (context.pp.size > 1) != (provider == "native_pipeline"):
        raise ValueError(
            "PP language recipes require training_provider=native_pipeline and a nontrivial pipeline grid"
        )
    if "pipeline_schedule" in config and provider != "native_pipeline":
        raise ValueError("pipeline_schedule only belongs to native_pipeline")
    if provider != "dense" and config.get("distillation"):
        raise ValueError(
            "Native TP recipe currently supports supervised CE; TP teacher/KD needs its own declared objective"
        )
    settings = TrainSettings(**config.get("training", {}))
    directory = Path(directory)
    from .training.optimizer_recipe import validate_optimizer_recipe, build_recipe_optimizer

    validate_optimizer_recipe(settings, context, provider, config["model"])
    device = collective_local(
        context, lambda: resolve_device(settings, context), "Resolve training device"
    )

    def prepare():
        seed_training(settings.seed)
        tokenizer = (
            load_tokenizer(config["tokenizer"]) if config.get("tokenizer") else ByteTokenizer()
        )
        model_config = config_from_dict(config["model"])
        if model_config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Model vocabulary must match the saved tokenizer, not just tensor shape"
            )
        dataset = LanguageData(config["data"], tokenizer, settings.max_length)
        sampler = RecipeSampler(
            dataset, seed=settings.seed, context=context, tail=settings.replica_tail
        )
        model = build_model(model_config).to(device)
        objective = CrossEntropyObjective()
        parents = ()
        if config.get("distillation"):
            distillation = config["distillation"]
            if set(distillation) - {"teacher_artifact", "kind", "temperature", "weight"}:
                raise ValueError("Unknown distillation recipe field")
            teacher_id = distillation.get("teacher_artifact")
            if teacher_id is None:
                if len(inputs) != 1:
                    raise ValueError("KD needs one unambiguous teacher artifact")
                teacher_id = next(iter(inputs.values()))["artifacts"]["model"]
            artifact = store.get(teacher_id)
            teacher = load_model(artifact.path / "model").to(device)
            teacher_tokenizer = load_tokenizer(artifact.path / "tokenizer")
            objective = DistillationObjective(
                teacher,
                kind=distillation.get("kind", "forward_kl"),
                temperature=distillation.get("temperature", 1.0),
                kd_weight=distillation.get("weight", 0.5),
                tokenizer_fingerprints=(
                    digest_json(tokenizer.to_dict()),
                    digest_json(teacher_tokenizer.to_dict()),
                ),
            )
            parents = (teacher_id,)
        return tokenizer, model_config, dataset, sampler, model, objective, parents

    tokenizer, model_config, dataset, sampler, model, objective, parents = collective_local(
        context, prepare, "Prepare language training"
    )
    optimizer_factory, optimizer_identity = None, None
    if settings.optimizer is not None:
        optimizer_factory, optimizer_identity = collective_local(
            context,
            lambda: build_recipe_optimizer(settings, model),
            "Select Muon logical parameter groups",
        )
        agree(context, optimizer_identity, "Muon recipe parameter ownership")
    if provider == "native_moe":
        if context.etp.size > 1 or context.tp.size > 1:
            from .training.moe_tensor_parallel import (
                parallelize_mixtral_tensor,
                ExpertTensorParallelCrossEntropyObjective,
            )

            model = parallelize_mixtral_tensor(model, context)
            objective = ExpertTensorParallelCrossEntropyObjective(
                context, router_aux_coefficient=config.get("router_aux_coefficient", 0.0)
            )
        else:
            from .training.moe_parallel import (
                parallelize_mixtral,
                ExpertParallelCrossEntropyObjective,
            )

            model = parallelize_mixtral(model, context)
            objective = ExpertParallelCrossEntropyObjective(
                context, router_aux_coefficient=config.get("router_aux_coefficient", 0.0)
            )
    elif provider != "dense":
        from .training.causal_parallel import (
            parallelize_causal_lm,
            TensorParallelCrossEntropyObjective,
        )
        from .training.causal_pipeline import CausalPipelineCrossEntropyObjective

        model = parallelize_causal_lm(
            model, context, pipeline_schedule=config.get("pipeline_schedule", "1f1b")
        )
        objective = (
            CausalPipelineCrossEntropyObjective(context)
            if provider == "native_pipeline"
            else TensorParallelCrossEntropyObjective(context)
        )
    trainer = Trainer(
        model,
        objective,
        **trainer_kwargs(settings, context, device, directory, optimizer_factory=optimizer_factory),
    )
    if isinstance(objective, DistillationObjective):
        objective.teacher = trainer.add_role("teacher", objective.teacher, trainable=False)
    state = fit_engine(
        trainer,
        config=config,
        settings=settings,
        dataset=dataset,
        sampler=sampler,
        microbatch=lambda: _device_batch(
            _batch(dataset, sampler, settings.batch_size, tokenizer.pad_token_id), device
        ),
        directory=directory,
        parents=parents,
        optimizer_identity=optimizer_identity,
    )
    return publish_model(
        trainer,
        config=config,
        model_config=model_config,
        settings=settings,
        dataset=dataset,
        sampler=sampler,
        state=state,
        directory=directory,
        store=store,
        kind="token_predictor",
        metadata={
            "architecture": model_config.to_dict()["architecture"],
            "tokenizer_fingerprint": digest_json(tokenizer.to_dict()),
            "training_data_fingerprint": dataset.fingerprint,
        },
        extra_files=lambda export: tokenizer.save_pretrained(export / "tokenizer"),
        parents=parents,
        optimizer_identity=optimizer_identity,
    )


def evaluate_language(config, inputs, directory, store):
    if set(config) - {"artifact", "data", "max_length", "device"}:
        raise ValueError("Unknown language evaluation field")
    artifact_id = config.get("artifact")
    if artifact_id is None:
        if len(inputs) != 1:
            raise ValueError("Evaluation needs one candidate input")
        artifact_id = next(iter(inputs.values()))["artifacts"]["model"]
    artifact = store.get(artifact_id)
    model, tokenizer = load_predictor_artifact(artifact, device=config.get("device", "cpu"))
    model.eval()
    dataset = LanguageData(config["data"], tokenizer, config.get("max_length", 128))
    protocol = ComparisonProtocol(
        "language-modeling",
        dataset.fingerprint,
        "aster.teacher_forced_nll",
        "1",
        {
            "tokenizer": digest_json(tokenizer.to_dict()),
            "max_length": dataset.max_length,
            "bos_eos": "artifact",
            "reduction": "per_document_negative_nll",
        },
        tuple(map(str, range(len(dataset)))),
        "negative_nll",
        failure_score=-1e6,
    )
    run = EvaluationRun(
        protocol,
        artifact_id,
        environment={"device": str(next(model.parameters()).device), "torch": torch.__version__},
    )
    objective = CrossEntropyObjective()
    nll, tokens = 0.0, 0
    with torch.no_grad():
        for index in range(len(dataset)):
            try:
                batch = _device_batch(
                    causal_collate([dataset[index]], pad_token_id=tokenizer.pad_token_id),
                    next(model.parameters()).device,
                )
                term = objective(model, batch)
                nll += float(term.numerator)
                tokens += int(term.denominator)
                run.add(
                    EvaluationRecord(
                        str(index),
                        "ok",
                        {"negative_nll": -float(term.mean)},
                        details={"nll_sum": float(term.numerator), "tokens": int(term.denominator)},
                    )
                )
            except (ValueError, RuntimeError) as error:
                run.add(EvaluationRecord(str(index), "error", error=str(error)))
    run.finalize()
    report = run.save(Path(directory) / "evaluation")
    metrics = {"success_rate": run.summary()["statuses"]["ok"] / len(dataset)}
    if tokens and metrics["success_rate"] == 1.0:
        metrics["perplexity"] = perplexity(nll, tokens)
    evidence = store.publish(
        report.parent,
        kind="evaluation",
        metadata={"protocol_id": protocol.id, "candidate_artifact_id": artifact_id},
        parents=(artifact_id,),
    )
    return StageResult(
        {"evaluation": evidence.id},
        metrics,
        {
            "report": str(evidence.path / "report.json"),
            "protocol_id": protocol.id,
            "candidate_artifact_id": artifact_id,
        },
    )


def quantize_language(config, inputs, directory, store):

    from .inference.optimization import collect_calibration, quantize_model, save_optimized_model

    allowed = {
        "artifact",
        "data",
        "max_length",
        "targets",
        "bits",
        "group_size",
        "algorithm",
        "max_rows",
        "options",
    }
    if set(config) - allowed:
        raise ValueError("Unknown language quantization field")
    artifact_id = _input_artifact(config, inputs)
    parent = store.get(artifact_id)
    model, tokenizer = load_predictor_artifact(parent)
    model.eval()
    targets = tuple(config["targets"])
    algorithm = config.get("algorithm", "rtn")
    calibration = None
    fingerprint = None
    if algorithm != "rtn":
        if not config.get("data"):
            raise ValueError("Activation-aware quantization requires explicit calibration data")
        dataset = LanguageData(config["data"], tokenizer, config.get("max_length", 128))
        fingerprint = dataset.fingerprint
        batches = (
            causal_collate([dataset[index]], pad_token_id=tokenizer.pad_token_id)
            for index in range(len(dataset))
        )
        calibration = collect_calibration(
            model,
            batches,
            targets=targets,
            dataset_fingerprint=fingerprint,
            max_rows=config.get("max_rows", 2048),
        )
    transformed = quantize_model(
        model,
        targets=targets,
        bits=config.get("bits", 4),
        group_size=config.get("group_size", 128),
        algorithm=algorithm,
        calibration=calibration,
        **config.get("options", {}),
    )
    export = Path(directory) / "export"

    save_optimized_model(
        transformed,
        export / "model",
        base_artifact_id=artifact_id,
        transformation_metadata={"recipe": config, "calibration_fingerprint": fingerprint},
    )
    tokenizer.save_pretrained(export / "tokenizer")
    result = store.publish(
        export,
        kind="token_predictor",
        metadata={
            "transformation": algorithm,
            "calibration_fingerprint": fingerprint,
            "compute_provider": "torch_float_dequant_reference",
        },
        parents=(artifact_id,),
    )
    return StageResult(
        {"model": result.id},
        {},
        {"quality_verified": False, "compute_provider": "torch_float_dequant_reference"},
    )


def gate_candidate(config, inputs, directory, store):

    if set(config) - {"baseline", "candidate", "max_regression", "max_failure_rate", "confidence"}:
        raise ValueError("Unknown quality gate field")
    reports = []
    for key in ("baseline", "candidate"):
        evidence = store.get(inputs[config[key]]["artifacts"]["evaluation"])
        reports.append(evidence.path / "report.json")
    result = quality_gate(
        *reports,
        max_regression=config.get("max_regression", 0.0),
        max_failure_rate=config.get("max_failure_rate", 0.0),
        confidence=config.get("confidence", 0.95),
    )
    atomic_json(Path(directory) / "quality-gate.json", result)
    if not result["passed"]:
        raise RuntimeError(
            "Candidate failed quality gate; evidence saved, no deployment-eligible artifact emitted"
        )
    run = EvaluationRun.load(reports[1])
    store.get(run.candidate_artifact_id)
    return StageResult(
        {"model": run.candidate_artifact_id},
        {"improvement": result["comparison"]["improvement"]},
        {"quality_gate": result, "protocol_id": run.protocol.id},
    )


BUILTIN_STAGES = {
    "language_fit": fit_language,
    "language_evaluate": evaluate_language,
    "language_quantize": quantize_language,
    "quality_gate": gate_candidate,
}
from .tensor_recipes import fit_tensors

BUILTIN_STAGES["tensor_fit"] = fit_tensors
