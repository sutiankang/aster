from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import ArtifactStore, atomic_json, read_json
from aster.models import load_model
from aster.training import ParallelContext
from aster.training.launch import LaunchConfig, launch
from aster.training.recipes import collective_local, leader_call, run_distributed_recipe
from aster.training.state import read_payload


def _configuration(directory, kind, stage):
    if kind == "language":
        path = directory / "data.jsonl"

        path.write_text(
            "\n".join(
                json.dumps({"input_ids": [3 + i, 5 + i, 7 + i] + [9 + i] * (i % 3)})
                for i in range(8)
            ),
            encoding="utf-8",
        )
        model = {
            "architecture": "llama",
            "vocab_size": 259,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
        }
        config = {"model": model, "data": str(path)}
    else:
        path = directory / "data.pt"
        torch.save(
            {"sample": torch.randn(8, 1, 4, 4, generator=torch.Generator().manual_seed(61))}, path
        )
        config = {
            "model": {
                "architecture": "unet2d",
                "in_channels": 1,
                "model_channels": 8,
                "channel_mult": [1],
                "attention_levels": [],
                "num_heads": 2,
                "num_res_blocks": 1,
                "prediction_type": "velocity",
            },
            "objective": {"name": "flow"},
            "data": str(path),
            "preprocessing": {"type": "synthetic_tensor_fixture", "version": "1"},
        }
    config["training"] = {
        "steps": 1,
        "batch_size": 2,
        "max_length": 16,
        "seed": 37,
        "zero_stage": stage,
        "learning_rate": 0.002,
        "checkpoint_every": 1,
    }
    return config


class CountingStore(ArtifactStore):
    def __init__(self, path, rank):
        super().__init__(path)
        self.rank = rank
        self.publications = 0

    def publish(self, *args, **kwargs):
        assert self.rank == 0, "Only leader may publish a trained artifact"
        self.publications += 1
        return super().publish(*args, **kwargs)


def _worker(rank, rendezvous, directory, kind, stage):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        context = ParallelContext()
        directory = Path(directory)
        config = read_json(directory / "config.json")
        store = CountingStore(directory / "store", rank)
        first = run_distributed_recipe(
            config, kind=kind, directory=directory / "first", store=store, parallel=context
        )
        assert all(value == asdict(first) for value in context.world.gather_objects(asdict(first)))
        assert (
            first.details["global_batch_size"] == 4
            and first.details["parallel"]["data_parallel"] == 2
        )
        payload = read_payload(
            directory / "first",
            read_json(directory / "first" / "checkpoint-final")["entries"][rank],
            trusted=False,
        )
        sampler = payload["states"]["sampler"]
        assert sampler["rank"] == rank and sampler["world_size"] == 2 and sampler["cursor"] == 2

        random_states = context.world.gather_objects(payload["rng"]["torch"])
        assert not torch.equal(random_states[0], random_states[1])
        if kind == "language":
            expected = torch.load(directory / "dense-oracle.pt", weights_only=True)
            actual = load_model(store.get(first.artifacts["model"]).path / "model").state_dict()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=4e-5)
        resumed_config = {
            **config,
            "resume": str(directory / "first" / "checkpoint-final"),
            "training": {**config["training"], "steps": 3},
        }
        resumed = run_distributed_recipe(
            resumed_config,
            kind=kind,
            directory=directory / "resumed",
            store=store,
            parallel=context,
        )
        full_config = {**config, "training": {**config["training"], "steps": 3}}
        full = run_distributed_recipe(
            full_config, kind=kind, directory=directory / "full", store=store, parallel=context
        )
        expected = load_model(store.get(full.artifacts["model"]).path / "model").state_dict()
        actual = load_model(store.get(resumed.artifacts["model"]).path / "model").state_dict()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        assert read_json(directory / "full" / "history.json") == read_json(
            directory / "resumed" / "history.json"
        )
        assert store.publications == (3 if rank == 0 else 0)
        assert not (directory / "first" / "run.lock").exists()
        with pytest.raises(RuntimeError, match="rank 1"):
            collective_local(
                context,
                lambda: (_ for _ in ()).throw(ValueError("missing local data")) if rank else None,
                "read data",
            )
        with pytest.raises(RuntimeError, match="publish denied"):
            leader_call(
                context, lambda: (_ for _ in ()).throw(OSError("publish denied")), "publish"
            )
        atomic_json(
            directory / f"rank-{rank}.json", {"success": True, "publications": store.publications}
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("kind", "stage"), [("language", 0), ("language", 3), ("tensor", 0), ("tensor", 3)]
)
def test_distributed_recipe_matches_global_batch_and_exact_resume(tmp_path, kind, stage):
    torch.set_num_threads(1)
    config = _configuration(tmp_path, kind, stage)
    atomic_json(tmp_path / "config.json", config)
    if kind == "language":
        from aster.recipes import fit_language

        dense = {**config, "training": {**config["training"], "zero_stage": 0, "batch_size": 4}}
        result = fit_language(
            dense, {}, tmp_path / "oracle", ArtifactStore(tmp_path / "oracle-store")
        )
        model = load_model(
            ArtifactStore(tmp_path / "oracle-store").get(result.artifacts["model"]).path / "model"
        )
        torch.save(model.state_dict(), tmp_path / "dense-oracle.pt")
    temp_root = Path(tempfile.gettempdir())
    if not str(temp_root).isascii():
        temp_root = Path("C:/Temp")
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="aster_recipe_", dir=temp_root))
    try:
        mp.spawn(
            _worker,
            args=(str(rendezvous_dir / "store"), str(tmp_path), kind, stage),
            nprocs=2,
            join=True,
        )
    finally:
        if rendezvous_dir.parent == temp_root and rendezvous_dir.name.startswith("aster_recipe_"):
            shutil.rmtree(rendezvous_dir)
    assert all(read_json(tmp_path / f"rank-{rank}.json")["success"] for rank in range(2))


def test_native_launcher_runs_distributed_train_cli(tmp_path):
    config = _configuration(tmp_path, "language", 3)
    atomic_json(tmp_path / "config.json", config)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    import aster.training.recipe_worker as worker

    arguments = [
        "distributed-train",
        str(tmp_path / "config.json"),
        "--output",
        str(tmp_path / "run"),
        "--store",
        str(tmp_path / "store"),
        "--backend",
        "gloo",
        "--timeout-seconds",
        "90",
    ]

    def execute():
        try:
            return launch(
                LaunchConfig(
                    nproc_per_node=2, master_port=port, launcher="native", timeout_seconds=90
                ),
                worker.__file__,
                arguments,
                execute=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            pytest.fail(
                f"Native distributed CLI failed. stdout={error.stdout}\nstderr={error.stderr}"
            )

    result = execute()
    assert result.returncode == 0
    manifest = read_json(tmp_path / "run" / "stage.json")
    assert manifest["status"] == "complete"
    assert manifest["result"]["details"]["global_batch_size"] == 4
    assert len(read_json(tmp_path / "run" / "checkpoint-final")["entries"]) == 2
    assert len(list((tmp_path / "store").iterdir())) == 1
    artifact = ArtifactStore(tmp_path / "store").get(manifest["result"]["artifacts"]["model"])
    loaded = load_model(artifact.path / "model")
    assert all(parameter.numel() > 0 for parameter in loaded.parameters())

    again = execute()
    assert again.returncode == 0 and read_json(tmp_path / "run" / "stage.json") == manifest
