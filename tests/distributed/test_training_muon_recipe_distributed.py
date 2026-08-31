from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import socket
import subprocess

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import ArtifactStore, atomic_json, read_json
from aster.models import load_model
from aster.training import ParallelConfig, ParallelContext
from aster.training.launch import LaunchConfig, launch
from aster.training.recipes import run_distributed_recipe
from aster.training.state import read_payload


def configuration(directory, profile):
    data = directory / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"input_ids": [3 + i, 5 + i, 7 + i] + [9 + i] * (i % 3)}) for i in range(8)
        ),
        encoding="utf-8",
    )
    return {
        "training_provider": "native_tp",
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
            "zero_stage": 3,
            "checkpoint_every": 1,
            "optimizer": {
                "type": "muon",
                "profile": profile,
                "matrix_learning_rate": 0.002,
                "auxiliary_modules": ["lm_head"],
            },
        },
    }


def unused_port():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


class LeaderStore(ArtifactStore):
    def __init__(self, path, rank):
        super().__init__(path)
        self.rank, self.publications = rank, 0

    def publish(self, *args, **kwargs):
        assert self.rank == 0
        self.publications += 1
        return super().publish(*args, **kwargs)


def worker(rank, port, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=150),
    )
    try:
        context = ParallelContext(ParallelConfig(tensor_parallel=2, data_parallel=1))
        for profile in ("keller", "moonlight"):
            directory = Path(root) / profile
            config = read_json(directory / "config.json")
            store = LeaderStore(directory / "store", rank)
            first = run_distributed_recipe(
                config,
                kind="language",
                directory=directory / "first",
                store=store,
                parallel=context,
            )
            assert first.details["parallel"]["tensor_parallel"] == 2
            assert first.details["global_batch_size"] == 4
            assert all(
                item == asdict(first) for item in context.world.gather_objects(asdict(first))
            )
            checkpoint = read_json(directory / "first" / "checkpoint-final")
            assert len(checkpoint["entries"]) == 2
            payload = read_payload(directory / "first", checkpoint["entries"][rank], trusted=False)
            assert payload["states"]["sampler"]["world_size"] == 1
            assert payload["states"]["sampler"]["cursor"] == 4
            artifact = store.get(first.artifacts["model"])
            source = load_model(artifact.path / "model")
            assert source.config.vocab_size == 259 and source.lm_head.weight.shape == (259, 8)
            assert source.lm_head.weight is source.model.embed_tokens.weight
            identity = first.details["optimizer"]
            assert (
                identity
                == artifact.metadata["execution"]["optimizer"]
                == read_json(artifact.path / "recipe.json")["execution"]["optimizer"]
            )
            groups = {g["use_muon"]: g for g in identity["groups"]}
            assert identity["settings"]["profile"] == profile
            assert groups[True]["lr"] == 0.002 and groups[False]["lr"] == 0.0007
            assert "model.layers.0.self_attn.q_proj.weight" in groups[True]["names"]
            assert "model.embed_tokens.weight" in groups[False]["names"]
            names = groups[True]["names"] + groups[False]["names"]
            assert len(names) == len(set(names)) and set(names) == set(
                dict(source.named_parameters())
            )
            assert all("shards." not in name for name in names)
            resumed_config = {
                **config,
                "resume": str(directory / "first" / "checkpoint-final"),
                "training": {**config["training"], "steps": 3},
            }
            resumed = run_distributed_recipe(
                resumed_config,
                kind="language",
                directory=directory / "resumed",
                store=store,
                parallel=context,
            )
            full = run_distributed_recipe(
                {**config, "training": {**config["training"], "steps": 3}},
                kind="language",
                directory=directory / "full",
                store=store,
                parallel=context,
            )
            actual = load_model(store.get(resumed.artifacts["model"]).path / "model")
            expected = load_model(store.get(full.artifacts["model"]).path / "model")
            torch.testing.assert_close(actual.state_dict(), expected.state_dict(), atol=0, rtol=0)
            assert read_json(directory / "resumed" / "history.json") == read_json(
                directory / "full" / "history.json"
            )

            bad = deepcopy(resumed_config)
            bad["training"]["optimizer"]["matrix_learning_rate"] = 0.003
            with pytest.raises((ValueError, RuntimeError)):
                run_distributed_recipe(
                    bad, kind="language", directory=directory / "bad", store=store, parallel=context
                )
            assert store.publications == (3 if rank == 0 else 0)
            assert not (directory / "first" / "run.lock").exists()
    finally:
        dist.destroy_process_group()


def test_muon_json_profiles_tp2_zero3_fresh_resume_and_export(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    for profile in ("keller", "moonlight"):
        directory = tmp_path / profile
        directory.mkdir()
        atomic_json(directory / "config.json", configuration(directory, profile))
    mp.spawn(worker, args=(unused_port(), str(tmp_path)), nprocs=2, join=True)


def test_muon_actual_cli_native_tp2_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    atomic_json(tmp_path / "config.json", configuration(tmp_path, "moonlight"))
    import aster.training.recipe_worker as worker_module

    arguments = [
        "distributed-train",
        str(tmp_path / "config.json"),
        "--output",
        str(tmp_path / "run"),
        "--store",
        str(tmp_path / "store"),
        "--tensor-parallel",
        "2",
        "--backend",
        "gloo",
        "--timeout-seconds",
        "120",
    ]
    try:
        result = launch(
            LaunchConfig(
                nproc_per_node=2, master_port=unused_port(), launcher="native", timeout_seconds=120
            ),
            worker_module.__file__,
            arguments,
            execute=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Muon TP2 CLI failed. stdout={error.stdout}\nstderr={error.stderr}")
    assert result.returncode == 0
    stage = read_json(tmp_path / "run" / "stage.json")
    assert stage["status"] == "complete"
    evidence = stage["result"]["details"]
    assert evidence["parallel"]["tensor_parallel"] == 2 and evidence["zero_stage"] == 3
    assert evidence["optimizer"]["settings"]["profile"] == "moonlight"
    artifact = ArtifactStore(tmp_path / "store").get(stage["result"]["artifacts"]["model"])
    assert artifact.metadata["execution"]["optimizer"] == evidence["optimizer"]
    model = load_model(artifact.path / "model")
    assert (
        model.lm_head.weight.shape == (259, 8)
        and model.lm_head.weight is model.model.embed_tokens.weight
    )
