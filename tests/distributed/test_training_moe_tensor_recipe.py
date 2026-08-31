import json
import socket
import subprocess

import pytest
import torch

from aster.core import ArtifactStore, atomic_json, read_json
from aster.models import load_model
from aster.training.launch import LaunchConfig, launch
from aster.training.state import read_payload


def test_cli_mixtral_tp_ep_etp_resume_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"input_ids": [3 + i, 5 + i, 7 + i] + [9 + i] * (i % 3)}) for i in range(8)
        ),
        encoding="utf-8",
    )
    config = {
        "training_provider": "native_moe",
        "router_aux_coefficient": 0.02,
        "model": {
            "architecture": "mixtral",
            "vocab_size": 259,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "sliding_window": 3,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": True,
            "router_jitter_noise": 0.05,
        },
        "data": str(data),
        "training": {
            "steps": 1,
            "batch_size": 1,
            "accumulation_steps": 2,
            "max_length": 16,
            "seed": 87,
            "zero_stage": 3,
            "learning_rate": 0.002,
            "checkpoint_every": 1,
        },
    }
    import aster.training.recipe_worker as worker

    def execute(value, name):
        path = tmp_path / f"{name}.json"
        atomic_json(path, value)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        arguments = [
            "distributed-train",
            str(path),
            "--output",
            str(tmp_path / name),
            "--store",
            str(tmp_path / "store"),
            "--tensor-parallel",
            "2",
            "--expert-parallel",
            "2",
            "--expert-tensor-parallel",
            "2",
            "--backend",
            "gloo",
            "--timeout-seconds",
            "120",
        ]
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
            pytest.fail(f"ETP CLI failed: stdout={error.stdout}\nstderr={error.stderr}")
        assert result.returncode == 0
        stage = read_json(tmp_path / name / "stage.json")
        assert stage["status"] == "complete"
        detail = stage["result"]["details"]
        assert (
            detail["global_batch_size"] == 4
            and "attention_TP_leader_rank" in detail["training_rng"]
        )
        assert {
            axis: detail["parallel"][axis]
            for axis in (
                "tensor_parallel",
                "expert_parallel",
                "expert_tensor_parallel",
                "data_parallel",
            )
        } == {
            "tensor_parallel": 2,
            "expert_parallel": 2,
            "expert_tensor_parallel": 2,
            "data_parallel": 2,
        }
        checkpoint = read_json(tmp_path / name / "checkpoint-final")
        assert len(checkpoint["entries"]) == 4
        for rank, entry in enumerate(checkpoint["entries"]):
            state = read_payload(tmp_path / name, entry, trusted=False)["states"]["sampler"]
            assert state["rank"] == rank // 2 and state["world_size"] == 2
        artifact = ArtifactStore(tmp_path / "store").get(stage["result"]["artifacts"]["model"])
        model = load_model(artifact.path / "model")
        assert model.model.layers[0].mlp.experts.gate_up_proj.shape == (4, 24, 8)
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert torch.isfinite(model(torch.tensor([[1, 5, 2]])).logits).all()
        assert not (tmp_path / name / "run.lock").exists()
        return model.state_dict()

    execute(config, "first")
    expected = execute({**config, "training": {**config["training"], "steps": 2}}, "full")
    actual = execute(
        {
            **config,
            "resume": str(tmp_path / "first" / "checkpoint-final"),
            "training": {**config["training"], "steps": 2},
        },
        "resumed",
    )
    for key in expected:
        assert torch.equal(actual[key], expected[key])
    assert read_json(tmp_path / "full" / "history.json") == read_json(
        tmp_path / "resumed" / "history.json"
    )
