from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import build_model, LlamaConfig
from aster.methods import DistillationObjective, PreferenceObjective
from aster.methods.reinforcement import GRPOObjective
from aster.methods.supervised import sequence_logprobs
from aster.training import Trainer, ParallelConfig, ParallelContext


def _tokens(length, rows):
    ids = torch.arange(1, length + 1)[None].expand(rows, -1).clone()
    labels = ids.clone()
    labels[:, 0] = -100
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(length)[None].expand(rows, -1).clone(),
    }


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=2))
        config = LlamaConfig(
            vocab_size=23,
            hidden_size=16,
            intermediate_size=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=32,
        )

        def make(algorithm, zero):
            torch.manual_seed(271)
            policy = build_model(config)
            reference = deepcopy(policy)
            if algorithm == "kd":
                objective = DistillationObjective(
                    reference, feature_weight=0.1, layer_pairs=((1, 1),)
                )
            elif algorithm in {"dpo", "ipo"}:
                objective = PreferenceObjective(reference, method=algorithm)
            else:
                objective = GRPOObjective(kl_weight=0.03)
            batches = []
            for index in range(2):
                data = _tokens(4 + index + rank, 1 + rank)
                if algorithm in {"dpo", "ipo"}:
                    rejected = _tokens(5 + index + rank, 1 + rank)
                    rejected["input_ids"][:, 1:3] = torch.tensor([3, 2])
                    rejected["labels"][:, 1:3] = torch.tensor([3, 2])
                    data = {"chosen": data, "rejected": rejected}
                elif algorithm == "grpo":
                    with torch.no_grad():
                        logp, _ = sequence_logprobs(policy, data)
                    data.update(
                        old_behavior_log_probs=logp.clone(),
                        reference_log_probs=logp.clone(),
                        advantages=torch.ones(1 + rank),
                    )
                batches.append(data)
            engine = Trainer(
                policy,
                objective,
                zero_stage=zero,
                parallel=context,
                accumulation_steps=2,
                lr=0.0005,
            )
            if algorithm != "grpo":
                engine.add_role("reference", reference, trainable=False)
            return engine, policy, reference, batches

        for zero in (0, 3):
            for algorithm in ("kd", "dpo", "ipo", "grpo"):
                engine, policy, reference, good = make(algorithm, zero)
                calls = []
                policy.register_forward_pre_hook(lambda *_: calls.append("policy"))
                reference.register_forward_pre_hook(lambda *_: calls.append("reference"))
                errors = ["ids", "mask", "positions", "labels"]
                if algorithm == "grpo":
                    errors += ["behavior_shape", "reference_nan", "advantages"]
                for corruption in errors:
                    before_calls, before_steps = len(calls), engine.steps
                    before_receipt = engine.last_successful_update()

                    before_parameters = [p.detach().clone() for p in policy.parameters()]
                    bad = deepcopy(good)
                    item = bad[1]["rejected"] if algorithm in {"dpo", "ipo"} else bad[1]
                    if rank == 1:
                        if corruption == "ids":
                            item["input_ids"][0, -1] = config.vocab_size
                        elif corruption == "mask":
                            item["attention_mask"][0, -1] = 2
                        elif corruption == "positions":
                            item["position_ids"][0, -1] = -1
                        elif corruption == "labels":
                            item["labels"][0, -1] = config.vocab_size
                        elif corruption == "behavior_shape":
                            item["old_behavior_log_probs"] = torch.ones(2, 1)
                        elif corruption == "reference_nan":
                            item["reference_log_probs"][0, -1] = float("nan")
                        else:
                            item["advantages"] = torch.ones(2, 1)

                    with pytest.raises(ValueError):
                        engine.step(bad)
                    assert len(calls) == before_calls and engine.steps == before_steps
                    assert engine.last_successful_update() == before_receipt
                    for current, previous in zip(policy.parameters(), before_parameters):
                        torch.testing.assert_close(current, previous, atol=0, rtol=0)
                    outcome = engine.step(good)
                    assert (
                        outcome.updated
                        and engine.steps == before_steps + 1
                        and len(calls) > before_calls
                    )
                    assert engine.last_successful_update()["role_updates"] == engine.steps
                if zero == 3:
                    path = engine.save_checkpoint(Path(output) / f"{algorithm}.json")
                    restored, _, _, restored_good = make(algorithm, zero)
                    restored.load_checkpoint(path)
                    engine.step(good)
                    restored.step(restored_good)
                    expected = engine.export_state_dict(only_rank_zero=False)
                    actual = restored.export_state_dict(only_rank_zero=False)
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_real_dp2_kd_dpo_ipo_grpo_later_bad_batch_and_recover(tmp_path):
    root = Path(tempfile.gettempdir()).resolve()
    if not str(root).isascii():
        root = Path("C:/Temp").resolve()
    directory = Path(tempfile.mkdtemp(prefix="aster-token-preflight-", dir=root)).resolve()
    assert directory.parent == root and directory.name.startswith("aster-token-preflight-")
    processes = None
    try:
        processes = mp.spawn(
            _worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=False
        )
        deadline = time.monotonic() + 150
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Token preflight DP regression exceeded bounded collective time")
    finally:
        if processes is not None:
            for process in processes.processes:
                if process.is_alive():
                    process.terminate()
            for process in processes.processes:
                process.join(timeout=5)

        shutil.rmtree(directory)
