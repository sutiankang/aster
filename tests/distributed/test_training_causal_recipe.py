from dataclasses import asdict
from datetime import timedelta
import json
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
from aster.training import ParallelContext, ParallelConfig
from aster.training.launch import LaunchConfig, launch
from aster.training.recipes import run_distributed_recipe
from aster.training.state import read_payload


def _configuration(directory):
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
            "max_length": 16,
            "seed": 37,
            "zero_stage": 3,
            "learning_rate": 0.002,
            "checkpoint_every": 1,
        },
    }


class LeaderStore(ArtifactStore):
    def __init__(self, path, rank):
        super().__init__(path)
        self.rank, self.publications = rank, 0

    def publish(self, *args, **kwargs):
        assert self.rank == 0
        self.publications += 1
        return super().publish(*args, **kwargs)


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=150),
    )
    try:
        context = ParallelContext(ParallelConfig(tensor_parallel=2, data_parallel=2))
        directory = Path(directory)
        config = read_json(directory / "config.json")
        store = LeaderStore(directory / "store", rank)
        first = run_distributed_recipe(
            config, kind="language", directory=directory / "first", store=store, parallel=context
        )
        assert first.details["parallel"]["tensor_parallel"] == 2
        assert first.details["global_batch_size"] == 4
        assert all(item == asdict(first) for item in context.world.gather_objects(asdict(first)))
        checkpoint = read_json(directory / "first" / "checkpoint-final")
        payload = read_payload(directory / "first", checkpoint["entries"][rank], trusted=False)
        sampler = payload["states"]["sampler"]
        assert (
            sampler["rank"] == context.dp.rank
            and sampler["world_size"] == 2
            and sampler["cursor"] == 2
        )
        source = load_model(store.get(first.artifacts["model"]).path / "model")
        assert source.config.vocab_size == 259 and source.lm_head.weight.shape == (259, 8)
        assert source.lm_head.weight is source.model.embed_tokens.weight
        expected_dense = torch.load(directory / "dense-one-step.pt", weights_only=True)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, expected_dense[name], rtol=2e-4, atol=2e-6)
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
        full_config = {**config, "training": {**config["training"], "steps": 3}}
        full = run_distributed_recipe(
            full_config,
            kind="language",
            directory=directory / "full",
            store=store,
            parallel=context,
        )
        actual = load_model(store.get(resumed.artifacts["model"]).path / "model").state_dict()
        expected = load_model(store.get(full.artifacts["model"]).path / "model").state_dict()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert read_json(directory / "resumed" / "history.json") == read_json(
            directory / "full" / "history.json"
        )
        assert store.publications == (3 if rank == 0 else 0)
        assert not (directory / "first" / "run.lock").exists()
    finally:
        dist.destroy_process_group()


def test_full_qwen_tp_dp_recipe_data_resume_artifact(tmp_path):
    torch.set_num_threads(1)
    config = _configuration(tmp_path)
    atomic_json(tmp_path / "config.json", config)
    from aster.recipes import fit_language

    dense = {
        **config,
        "training_provider": "dense",
        "training": {**config["training"], "zero_stage": 0, "batch_size": 4},
    }
    store = ArtifactStore(tmp_path / "dense-store")
    result = fit_language(dense, {}, tmp_path / "dense-run", store)
    torch.save(
        load_model(store.get(result.artifacts["model"]).path / "model").state_dict(),
        tmp_path / "dense-one-step.pt",
    )
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    rendezvous = Path(tempfile.mkdtemp(prefix="aster_causal_recipe_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(rendezvous / "rdzv"), str(tmp_path)), nprocs=4, join=True)
    finally:
        assert rendezvous.parent == root.resolve() and rendezvous.name.startswith(
            "aster_causal_recipe_"
        )
        shutil.rmtree(rendezvous)


@pytest.mark.parametrize("pipeline", [False, True])
def test_native_cli_runs_real_qwen_tp2_dp2_zero3(tmp_path, monkeypatch, pipeline):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    config = _configuration(tmp_path)
    if pipeline:
        config["training_provider"] = "native_pipeline"
        config["model"].update(num_hidden_layers=2, tie_word_embeddings=True)
        config["training"]["accumulation_steps"] = 2
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
        "--tensor-parallel",
        "2",
        "--backend",
        "gloo",
        "--timeout-seconds",
        "120",
    ]
    if pipeline:
        arguments += ["--pipeline-parallel", "2"]
    try:
        result = launch(
            LaunchConfig(
                nproc_per_node=4, master_port=port, launcher="native", timeout_seconds=120
            ),
            worker.__file__,
            arguments,
            execute=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(f"Full-model TP CLI failed. stdout={error.stdout}\nstderr={error.stderr}")
    assert result.returncode == 0
    stage = read_json(tmp_path / "run" / "stage.json")
    assert stage["status"] == "complete"
    assert stage["result"]["details"]["parallel"] == {
        "tensor_parallel": 2,
        "pipeline_parallel": 2 if pipeline else 1,
        "context_parallel": 1,
        "data_parallel": 1 if pipeline else 2,
        "gtp_remat": 1,
        "expert_parallel": 1,
        "expert_tensor_parallel": 1,
    }
    assert stage["result"]["details"]["global_batch_size"] == 4
    assert len(read_json(tmp_path / "run" / "checkpoint-final")["entries"]) == 4
    store = ArtifactStore(tmp_path / "store")
    artifact = store.get(stage["result"]["artifacts"]["model"])
    model = load_model(artifact.path / "model")
    assert model.lm_head.weight.shape == (259, 8)
    assert model.lm_head.weight is model.model.embed_tokens.weight
