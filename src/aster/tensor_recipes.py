"""Shared tensor-dataset training workflows across model domains."""

from pathlib import Path
from torch.utils.data import default_collate
from .core import atomic_json
from .data.tensors import TensorTreeDataset
from .models import build_model
from .models.config import config_from_dict
from .methods.factory import build_objective
from .training import Trainer


def fit_tensors(config, inputs, directory, store, *, parallel=None):
    from .recipes import TrainSettings, _device_batch
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

    context = recipe_context(parallel)
    agree(context, {"config": config, "inputs": inputs}, "tensor recipe inputs")
    allowed = {
        "model",
        "objective",
        "data",
        "preprocessing",
        "training",
        "resume",
        "initial_artifact",
    }
    if set(config) - allowed:
        raise ValueError("Unknown tensor recipe fields")
    settings = TrainSettings(**config.get("training", {}))
    directory = Path(directory)
    device = collective_local(
        context, lambda: resolve_device(settings, context), "Resolve training device"
    )

    def prepare():
        seed_training(settings.seed)
        dataset = TensorTreeDataset(config["data"], preprocessing=config["preprocessing"])
        sampler = RecipeSampler(
            dataset, seed=settings.seed, context=context, tail=settings.replica_tail
        )
        model_config = config_from_dict(config["model"])
        model = build_model(model_config).to(device)
        parents = ()
        initial_id = config.get("initial_artifact")
        if initial_id:
            from .models import load_model

            initial = store.get(initial_id)
            loaded = load_model(initial.path / "model")
            if loaded.config.to_dict() != model_config.to_dict():
                raise ValueError("Initial artifact architecture differs from train configuration")
            model.load_state_dict(loaded.state_dict(), strict=True)
            parents = (initial_id,)
        elif inputs:
            raise ValueError(
                "Choose initial_artifact explicitly when tensor training has upstream artifacts"
            )
        return dataset, sampler, model_config, model, build_objective(config["objective"]), parents

    dataset, sampler, model_config, model, objective, parents = collective_local(
        context, prepare, "Prepare tensor training"
    )
    engine = Trainer(model, objective, **trainer_kwargs(settings, context, device, directory))

    def microbatch():
        records = []
        while len(records) < settings.batch_size:
            selected = sampler.take(settings.batch_size - len(records))
            records.extend(selected)
            if not selected:
                sampler.next_epoch()
        return _device_batch(default_collate(records), device)

    state = fit_engine(
        engine,
        config=config,
        settings=settings,
        dataset=dataset,
        sampler=sampler,
        microbatch=microbatch,
        directory=directory,
        parents=parents,
    )

    def export_preprocessing_and_objective(export):
        atomic_json(export / "preprocessing.json", config["preprocessing"])

        codec = getattr(objective, "config_dict", None)
        if callable(codec):
            from .evaluation.generation_artifacts import verified_training_update

            actual_update = verified_training_update(engine, objective)
            atomic_json(export / "objective.json", codec())
            atomic_json(export / "successful_update.json", actual_update)

    return publish_model(
        engine,
        config=config,
        model_config=model_config,
        settings=settings,
        dataset=dataset,
        sampler=sampler,
        state=state,
        directory=directory,
        store=store,
        kind="model",
        metadata={
            "architecture": model_config.to_dict()["architecture"],
            "data_fingerprint": dataset.fingerprint,
            "objective": config["objective"],
        },
        extra_files=export_preprocessing_and_objective,
        parents=parents,
    )
