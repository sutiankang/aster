from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from aster.core import ArtifactStore, atomic_json, read_json, digest_json
from aster.models import Qwen3Config, build_model, load_model
from aster.recipes import TrainSettings, fit_language
from aster.training import ParallelContext, ParallelConfig, Trainer
from aster.training.optimizer_recipe import (
    MuonSettings,
    build_recipe_optimizer,
    validate_optimizer_recipe,
)
from aster.training.recipes import trainer_kwargs
from aster.training.state import read_payload


def configuration(path, *, profile="keller", stage=0):
    data = path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"input_ids": [3 + i, 5 + i, 7 + i] + [9 + i] * (i % 3)}) for i in range(6)
        ),
        encoding="utf-8",
    )
    return {
        "model": {
            "architecture": "qwen3",
            "vocab_size": 259,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "tie_word_embeddings": True,
        },
        "data": str(data),
        "training": {
            "steps": 1,
            "batch_size": 2,
            "accumulation_steps": 2,
            "max_length": 16,
            "seed": 829,
            "learning_rate": 0.0007,
            "zero_stage": stage,
            "checkpoint_every": 1,
            "optimizer": {
                "type": "muon",
                "profile": profile,
                "matrix_learning_rate": 0.002,
                "auxiliary_modules": ["lm_head"],
            },
        },
    }


@pytest.mark.parametrize(
    "options",
    [
        {"type": "sgd"},
        {"type": "adamw", "eps": 1e-8},
        {"type": "muon"},
        {"type": "muon", "profile": "auto", "matrix_learning_rate": 0.1},
        {"type": "muon", "profile": "keller", "matrix_learning_rate": True},
        {"type": "muon", "profile": "keller", "matrix_learning_rate": float("nan")},
        {"type": "muon", "profile": "keller", "matrix_learning_rate": 0.1, "auxiliary_lr": 0.001},
        {"type": "muon", "profile": "keller", "matrix_learning_rate": 0.1, "auxiliary_modules": []},
        {
            "type": "muon",
            "profile": "keller",
            "matrix_learning_rate": 0.1,
            "auxiliary_modules": [[]],
        },
        {
            "type": "muon",
            "profile": "keller",
            "matrix_learning_rate": 0.1,
            "auxiliary_modules": ["model.*", "lm_head"],
        },
        {"type": "muon", "profile": "keller", "matrix_learning_rate": 0.1, "ns_steps": True},
        {
            "type": "muon",
            "profile": "moonlight",
            "matrix_learning_rate": 0.1,
            "auxiliary_betas": [0.9, 1.0],
        },
    ],
)
def test_muon_recipe_rejects_unknown_ambiguous_and_invalid_settings(options):
    with pytest.raises(ValueError):
        TrainSettings(optimizer=options)


def test_muon_recipe_default_adamw_unchanged_and_exact_fqn_ownership():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    context = ParallelContext()
    default, explicit = TrainSettings(), TrainSettings(optimizer={"type": "adamw"})
    assert asdict(default) == asdict(explicit)
    options = trainer_kwargs(default, context, "cpu", ".")
    assert "optimizer_factory" not in options
    model = build_model(
        Qwen3Config(
            vocab_size=19,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
    )
    reference = deepcopy(model)
    engine = Trainer(model, **options)
    ordinary = torch.optim.AdamW(reference.parameters(), lr=default.learning_rate)
    assert {
        k: v for k, v in engine.roles["model"].optimizer.param_groups[0].items() if k != "params"
    } == {k: v for k, v in ordinary.param_groups[0].items() if k != "params"}
    settings = TrainSettings(
        learning_rate=0.0007,
        optimizer={"type": "muon", "profile": "moonlight", "matrix_learning_rate": 0.02},
    )
    factory, identity = build_recipe_optimizer(settings, reference)
    groups = {group["use_muon"]: group for group in factory.groups}
    named = dict(reference.named_parameters())
    assert groups[True]["lr"] == 0.02 and groups[False]["lr"] == 0.0007
    assert groups[True]["weight_decay"] == groups[False]["weight_decay"] == 0.1
    assert (
        "model.embed_tokens.weight" in groups[False]["names"]
        and "lm_head.weight" not in groups[True]["names"]
    )
    assert all(named[name].ndim == 2 for name in groups[True]["names"])
    selected = groups[True]["names"] + groups[False]["names"]
    assert len(selected) == len(set(selected)) == len(named) and set(selected) == set(named)
    assert json.loads(json.dumps(identity)) == identity
    assert MuonSettings(**settings.optimizer.to_dict()) == settings.optimizer
    with pytest.raises(ValueError, match="not admitted"):
        trainer_kwargs(settings, context, "cpu", ".")


@pytest.mark.parametrize(
    "axis",
    [
        "pipeline_parallel",
        "context_parallel",
        "expert_parallel",
        "expert_tensor_parallel",
        "gtp_remat",
    ],
)
def test_muon_recipe_rejects_uncertified_parallel_before_provider_construction(axis):
    settings = TrainSettings(
        optimizer={"type": "muon", "profile": "keller", "matrix_learning_rate": 0.002}
    )
    axes = asdict(ParallelConfig())
    axes[axis] = 2
    context = SimpleNamespace(config=SimpleNamespace(**axes))
    with pytest.raises(ValueError, match="not certified"):
        validate_optimizer_recipe(settings, context, "dense", {"architecture": "qwen3"})
    with pytest.raises(ValueError, match="dense/native_tp"):
        validate_optimizer_recipe(
            settings, ParallelContext(), "native_moe", {"architecture": "qwen3"}
        )
    with pytest.raises(ValueError, match="Llama/Qwen2/Qwen3"):
        validate_optimizer_recipe(settings, ParallelContext(), "dense", {"architecture": "mamba"})


@pytest.mark.parametrize("profile", ["keller", "moonlight"])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_muon_json_recipe_all_zero_fresh_resume_and_artifact_identity(profile, stage, tmp_path):
    torch.set_num_threads(1)
    config = configuration(tmp_path, profile=profile, stage=stage)
    atomic_json(tmp_path / "recipe.json", config)
    config = read_json(tmp_path / "recipe.json")
    store = ArtifactStore(tmp_path / "store")
    first = fit_language(config, {}, tmp_path / "first", store)
    resumed_config = {
        **config,
        "resume": first.details["checkpoint"],
        "training": {**config["training"], "steps": 3},
    }
    resumed = fit_language(resumed_config, {}, tmp_path / "resumed", store)
    full = fit_language(
        {**config, "training": {**config["training"], "steps": 3}}, {}, tmp_path / "full", store
    )
    actual = load_model(store.get(resumed.artifacts["model"]).path / "model")
    expected = load_model(store.get(full.artifacts["model"]).path / "model")
    torch.testing.assert_close(actual.state_dict(), expected.state_dict(), atol=0, rtol=0)
    assert read_json(tmp_path / "resumed" / "history.json") == read_json(
        tmp_path / "full" / "history.json"
    )
    artifact = store.get(resumed.artifacts["model"])
    recipe = read_json(artifact.path / "recipe.json")
    identity = recipe["execution"]["optimizer"]
    assert identity == artifact.metadata["execution"]["optimizer"] == resumed.details["optimizer"]
    assert (
        identity["settings"]["profile"] == profile and identity["auxiliary_learning_rate"] == 0.0007
    )
    assert identity["source"]["commit"] and identity["groups"][0]["lr"] == 0.002
    bad = deepcopy(resumed_config)
    bad["training"]["optimizer"]["matrix_learning_rate"] = 0.003
    with pytest.raises((ValueError, RuntimeError)):
        fit_language(bad, {}, tmp_path / "bad", store)
    assert not (tmp_path / "bad" / "export").exists()


def test_muon_recipe_resume_rejects_profile_aux_rate_or_matrix_assignment_change(tmp_path):
    torch.set_num_threads(1)
    config = configuration(tmp_path, stage=3)
    store = ArtifactStore(tmp_path / "store")
    first = fit_language(config, {}, tmp_path / "first", store)
    for index, change in enumerate(
        (
            {"profile": "moonlight"},
            {"auxiliary_modules": ["lm_head", "model.layers.0.self_attn.q_proj"]},
            {"momentum": 0.9},
            {"ns_steps": 4},
            {"missing_grad": "zero"},
        )
    ):
        bad = deepcopy(config)
        bad["resume"] = first.details["checkpoint"]
        bad["training"]["steps"] = 2
        bad["training"]["optimizer"].update(change)
        with pytest.raises((ValueError, RuntimeError)):
            fit_language(bad, {}, tmp_path / f"bad-{index}", store)
    bad = deepcopy(config)
    bad["resume"] = first.details["checkpoint"]
    bad["training"]["steps"] = 2
    bad["training"]["learning_rate"] = 0.001
    with pytest.raises((ValueError, RuntimeError)):
        fit_language(bad, {}, tmp_path / "bad-aux", store)


def test_default_adamw_retains_legacy_recipe_identity_and_explicit_default_resume(tmp_path):

    torch.set_num_threads(1)
    config = configuration(tmp_path, stage=3)
    config["training"].pop("optimizer")
    store = ArtifactStore(tmp_path / "store")
    first = fit_language(config, {}, tmp_path / "first", store)
    checkpoint = read_json(first.details["checkpoint"])
    payload = read_payload(tmp_path / "first", checkpoint["entries"][0], trusted=False)
    settings = asdict(TrainSettings(**config["training"]))
    for key in ("steps", "checkpoint_every", "max_consecutive_skips", "optimizer"):
        settings.pop(key)
    artifact = store.get(first.artifacts["model"])
    expected = {
        "config": {
            key: value for key, value in config.items() if key not in {"resume", "training"}
        },
        "training": settings,
        "data": artifact.metadata["training_data_fingerprint"],
        "parents": [],
        "parallel": ParallelContext().to_dict(),
    }
    assert payload["states"]["recipe"]["identity"] == digest_json(expected)
    assert "optimizer" not in first.details and "optimizer" not in artifact.metadata["execution"]
    resumed_config = {
        **config,
        "resume": first.details["checkpoint"],
        "training": {**config["training"], "steps": 2, "optimizer": {"type": "adamw"}},
    }
    resumed = fit_language(resumed_config, {}, tmp_path / "resumed", store)
    full = fit_language(
        {**config, "training": {**config["training"], "steps": 2}}, {}, tmp_path / "full", store
    )
    actual = load_model(store.get(resumed.artifacts["model"]).path / "model")
    expected = load_model(store.get(full.artifacts["model"]).path / "model")
    torch.testing.assert_close(actual.state_dict(), expected.state_dict(), atol=0, rtol=0)
    assert read_json(tmp_path / "resumed" / "history.json") == read_json(
        tmp_path / "full" / "history.json"
    )


@pytest.mark.parametrize(
    "command,filename", [("train", "language_muon.json"), ("run", "language_chain.json")]
)
def test_checked_in_muon_and_readme_default_examples_execute_actual_cli(
    command, filename, tmp_path, monkeypatch, capsys
):

    from aster.cli import main

    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(project)
    arguments = [
        command,
        str(project / "examples" / "recipes" / filename),
        "--output",
        str(tmp_path / "run"),
        "--store",
        str(tmp_path / "store"),
    ]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    if command == "train":
        stage = result["train"]
        assert stage["details"]["optimizer"]["settings"]["missing_grad"] == "zero"
        assert stage["details"]["steps"] == 20
    else:
        stage = result["student"]
        assert stage["details"]["steps"] == 20 and result["teacher"]["details"]["steps"] == 30
        assert "optimizer" not in stage["details"]
        assert result["student_eval"]["metrics"]["perplexity"] > 0
    artifact = ArtifactStore(tmp_path / "store").get(stage["artifacts"]["model"])
    model = load_model(artifact.path / "model")
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[3, 4, 5]])).logits).all()
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == result
