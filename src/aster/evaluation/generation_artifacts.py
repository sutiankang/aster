"""Bind trained generator weights to their exact parameterization and sampling semantics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import torch

from ..core import atomic_json, digest_json, file_digest, read_json
from ..core.update_provenance import validate_successful_update_record
from ..methods.generation import DiffusionSchedule


def verified_training_update(engine, objective, *, role="model"):

    from ..training.trainer import _objective_configuration

    if engine._busy or engine._failed:
        raise ValueError("Successful objective provenance requires an idle valid Trainer")
    return validate_successful_update_record(
        engine.last_successful_update(role=role),
        _objective_configuration(objective),
        role=role,
        role_updates=engine.roles[role].updates,
    )


def load_native_artifact_model(artifact):

    from ..models import load_model

    root = Path(artifact.path)
    candidates = [path for path in (root, root / "model") if (path / "config.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("Native artifact requires exactly one root or model/ model layout")
    target = candidates[0]
    return load_model(target), target.relative_to(root).as_posix()


def resolve_image_sampling(artifact, model, model_relative_path, plan):

    root = Path(artifact.path)
    objective_path, contract_path = root / "objective.json", root / "generation_contract.json"
    objective = read_json(objective_path) if objective_path.is_file() else None
    contract = read_json(contract_path) if contract_path.is_file() else None
    prediction_type = model.config.prediction_type
    if objective is not None and not isinstance(objective, dict):
        raise ValueError("Training objective must be a JSON object")
    binding = {
        "schema_version": 1,
        "policy_artifact_id": artifact.id,
        "model_relative_path": model_relative_path,
        "prediction_type": prediction_type,
        "objective_file_sha256": file_digest(objective_path) if objective is not None else None,
        "training_objective": objective,
        "generation_contract": contract,
        "generation_contract_sha256": file_digest(contract_path) if contract is not None else None,
        "sampling_mode": plan.sampler,
    }

    update_path = root / "successful_update.json"
    update_record = read_json(update_path) if update_path.is_file() else None
    if update_record is not None:
        names = {"diffusion": "DiffusionObjective", "flow_matching": "FlowObjective"}
        if objective is None or objective.get("type") not in names:
            raise ValueError(
                "Successful update metadata does not describe this image sampler objective"
            )
        descriptor = {
            "class": "aster.methods.generation." + names[objective["type"]],
            "codec": "config_dict",
            "configuration": objective,
        }
        validate_successful_update_record(
            update_record, descriptor, role_updates=update_record.get("role_updates")
        )
    binding.update(
        successful_update=update_record,
        successful_update_file_sha256=file_digest(update_path)
        if update_record is not None
        else None,
        actual_successful_objective_bound=update_record is not None,
    )
    if plan.sampler.startswith("flow_"):
        if prediction_type != "velocity" or contract is not None:
            raise ValueError(
                "Flow sampling requires an instantaneous velocity model, not a direct generator"
            )
        if objective is not None and (
            objective.get("type") != "flow_matching"
            or objective.get("direction") != plan.flow_direction
        ):
            raise ValueError("Flow sampling direction/objective differs from the trained artifact")

        binding["training_semantics_bound"] = objective is not None
        return None, binding
    if plan.sampler == "direct_x0":
        required = {"schema_version", "method", "prediction_type", "generator_time", "training"}
        if prediction_type != "x0" or not isinstance(contract, dict) or set(contract) != required:
            raise ValueError("Direct x0 sampling requires an artifact-bound generation contract")
        if (
            contract["schema_version"] != 1
            or contract["method"] != "dmd"
            or contract["prediction_type"] != "x0"
        ):
            raise ValueError(
                "Unsupported direct generator contract; no implicit DMD2/consistency conversion"
            )
        time = contract["generator_time"]
        if type(time) not in {float, int} or not math.isfinite(time):
            raise ValueError("Direct generator time must be finite and artifact-bound")
        training = contract["training"]
        if not isinstance(training, dict) or any(
            type(training.get(name)) is not int or training[name] < 1
            for name in (
                "generator_updates",
                "method_updates",
                "fake_score_updates",
                "fake_updates_per_method",
            )
        ):
            raise ValueError("DMD direct artifact lacks completed generator update evidence")
        identities = training.get("role_weight_fingerprints")
        if (
            not isinstance(identities, dict)
            or set(identities) != {"model", "real_score", "fake_score"}
            or any(
                not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
                for value in identities.values()
            )
        ):
            raise ValueError(
                "DMD contract must identify the actual generator and both score weight snapshots"
            )
        sigma_data = training.get("sigma_data")
        if type(sigma_data) not in {int, float} or not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError("DMD contract must retain the positive training score scale")
        if identities["model"] != _role_fingerprint(model.config.to_dict(), model.state_dict()):
            raise ValueError(
                "DMD model weights differ from the snapshot bound by the generation contract"
            )
        if objective is not None:
            raise ValueError("Direct generator artifact cannot also declare a VP/flow objective")
        binding["training_semantics_bound"] = True
        return None, binding
    if contract is not None or prediction_type not in {"epsilon", "x0", "v", "score"}:
        raise ValueError("Discrete diffusion cannot reinterpret this generator parameterization")
    if (
        objective is None
        or objective.get("type") != "diffusion"
        or not {"betas", "timestep_map", "learned_variance"} <= set(objective)
    ):
        raise ValueError(
            "Diffusion sampling requires the artifact original betas/timestep_map in objective.json"
        )
    betas, mapping = objective["betas"], objective["timestep_map"]
    if not isinstance(betas, list) or any(
        type(x) not in {int, float} or not math.isfinite(x) for x in betas
    ):
        raise ValueError("Training betas must be finite numeric values")
    if not isinstance(mapping, list) or any(type(x) is not int or x < 0 for x in mapping):
        raise ValueError("Training timestep map must contain nonnegative integer model times")
    if (
        type(objective["learned_variance"]) is not bool
        or objective["learned_variance"] != plan.learned_variance
    ):
        raise ValueError("Sampling learned variance differs from the trained objective")
    original = DiffusionSchedule(betas, timestep_map=mapping)
    if not 2 <= plan.steps <= len(original):
        raise ValueError("Diffusion sampling needs 2..training_steps distinct training marginals")
    indices = (
        tuple(plan.respacing_indices)
        if plan.respacing_indices is not None
        else tuple(i * (len(original) - 1) // (plan.steps - 1) for i in range(plan.steps))
    )
    if len(indices) != plan.steps or indices[-1] >= len(original):
        raise ValueError("Respacing indices must match steps and stay inside the training chain")

    selected = original if indices == tuple(range(len(original))) else original.respaced(indices)
    binding.update(
        training_semantics_bound=True,
        diffusion={
            "original_schedule_id": digest_json({"betas": betas, "timestep_map": mapping}),
            "training_steps": len(original),
            "selected_training_indices": list(indices),
            "effective_betas": selected.betas.tolist(),
            "effective_model_times": selected.timestep_map.tolist(),
            "effective_alpha_bar": selected.alpha_bar.tolist(),
        },
    )
    return selected, binding


def _role_fingerprint(configuration, tensors):

    values = {}
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        values[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return digest_json({"configuration": configuration, "tensors": values})


def publish_dmd_generator(method, store, directory, *, parents=()):

    from ..methods.generative_distillation import DMDMethod
    from ..models import build_model
    from ..models.generative import UNet2D, DiT
    from ..training.sharding import Zero3Unit

    if type(method) is not DMDMethod:
        raise ValueError("DMD exporter requires the native DMDMethod lifecycle")
    engine = method.engine
    method.state_dict()
    if engine.parallel.world.size != 1:
        raise ValueError("This DMD deployment exporter is currently single-rank only")
    if (
        engine.states.get("dmd_method") is not method
        or method.updates < 1
        or engine.roles["model"].updates < 1
    ):
        raise ValueError("Cannot publish an untrained or unregistered DMD generator")
    modules, states, fingerprints = {}, {}, {}
    for role in ("model", "real_score", "fake_score"):
        module = engine.roles[role].model
        module = module.module if isinstance(module, Zero3Unit) else module
        if type(module) not in {UNet2D, DiT}:
            raise ValueError("DMD exporter only implements native UNet2D/DiT roles")
        modules[role] = module
        states[role] = engine.export_state_dict(role=role)
        fingerprints[role] = _role_fingerprint(module.config.to_dict(), states[role])
    if modules["model"].config.prediction_type != "x0" or any(
        modules[name].config.prediction_type != "edm_residual"
        for name in ("real_score", "fake_score")
    ):
        raise ValueError("DMD exporter requires direct x0 generator and EDM residual scores")
    if (
        method.generator_objective.generator_time != method.generator_time
        or method.generator_objective.sigma_data != method.fake_objective.sigma_data
    ):
        raise ValueError("DMD objective controls disagree with the method deployment contract")
    generator_time, sigma_data = method.generator_time, method.generator_objective.sigma_data
    if (
        any(
            type(v) not in {int, float} or not math.isfinite(v)
            for v in (generator_time, sigma_data)
        )
        or sigma_data <= 0
    ):
        raise ValueError("Invalid DMD time/scale")
    parents = tuple(parents)
    for parent in parents:
        store.get(parent, verify=True)
    target = Path(directory).absolute()
    target.mkdir(parents=True, exist_ok=False)

    with torch.random.fork_rng(devices=[]):
        model = build_model(modules["model"].config)
    model.load_state_dict(states["model"], strict=True, assign=True)
    model.save_pretrained(target / "model")
    contract = {
        "schema_version": 1,
        "method": "dmd",
        "prediction_type": "x0",
        "generator_time": generator_time,
        "training": {
            "method_updates": method.updates,
            "generator_updates": engine.roles["model"].updates,
            "fake_score_updates": engine.roles["fake_score"].updates,
            "fake_updates_per_method": method.fake_updates,
            "sigma_data": sigma_data,
            "fake_score_objective": method.fake_objective.config_dict(),
            "role_weight_fingerprints": fingerprints,
            "parents_semantics": "caller_declared_initial_sources_not_automatic_teacher_binding",
            "deployment_only": True,
        },
    }
    atomic_json(target / "generation_contract.json", contract)
    return store.publish(
        target,
        kind="native_direct_generator",
        metadata={
            "method": "dmd",
            "generation_contract_id": digest_json(contract),
            "training_checkpoint_included": False,
        },
        parents=parents,
    )
