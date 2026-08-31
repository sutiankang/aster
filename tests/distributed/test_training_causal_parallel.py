from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import LlamaConfig, Qwen2Config, Qwen3Config, build_model
from aster.methods import CrossEntropyObjective
from aster.training import (
    Trainer,
    ParallelConfig,
    ParallelContext,
    parallelize_causal_lm,
    TensorParallelCrossEntropyObjective,
)
from aster.training.sharding import Zero3Unit
from aster.training.portable import logical_tensors, gather_tensor, optimizer_mapping
from aster.core import atomic_json


def _config(kind, kv, tied, dropout=0.0):
    return kind(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv,
        max_position_embeddings=32,
        tie_word_embeddings=tied,
        attention_dropout=dropout,
    )


def _data(replica, masked=False):
    ids = torch.tensor([[1, 3, 5, 7, 9, 2], [1, 4, 6, 8, 0, 0], [1, 9, 11, 13, 15, 2]])[
        : 2 + replica
    ]
    ids = (ids + replica).remainder(17)
    padding = torch.ones_like(ids, dtype=torch.bool)
    padding[1, -2:] = False
    labels = ids.clone()
    labels[:, :2] = -100
    mask = padding.clone()
    mask[:, 3] = False
    if masked:
        mask.zero_()
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": padding,
        "loss_mask": mask,
        "position_ids": torch.arange(2, 8)[None].expand(len(ids), -1),
    }


def _reference(model, optimizer, masked=False):
    objective = CrossEntropyObjective()
    terms = [objective(model, _data(i, masked and i == 0)) for i in range(2)]
    numerator = sum(term.numerator for term in terms)
    count = sum(term.denominator for term in terms)
    optimizer.zero_grad(set_to_none=True)
    (numerator / count).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
    optimizer.step()
    return float(numerator.detach() / count), float(norm)


def _gradients(engine):
    role = engine.roles["model"]
    _, owners, sharded = optimizer_mapping(role)
    return {
        entry.name: gather_tensor(
            owners[id(entry.tensor)].grad, entry, engine.parallel, optimizer_sharded=sharded
        )
        for entry in logical_tensors(role.model, engine.parallel)
        if entry.parameter
    }


def _worker(rank, rendezvous, output):
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
        diagnostics = []
        for kind in (LlamaConfig, Qwen2Config, Qwen3Config):
            torch.manual_seed(71)
            original = build_model(_config(kind, 2, False))
            oracle = deepcopy(original)
            expected_optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.002)
            model = parallelize_causal_lm(original, context)
            engine = Trainer(
                model,
                TensorParallelCrossEntropyObjective(context),
                parallel=context,
                lr=0.002,
                max_grad_norm=0.7,
            )
            _reference(oracle, expected_optimizer)
            engine.step([_data(context.dp.rank)])
            gradients = _gradients(engine)
            actual = engine.export_state_dict(only_rank_zero=False)
            emulated = deepcopy(original)
            same_gradient_optimizer = torch.optim.AdamW(emulated.parameters(), lr=0.002)
            gradient_error = 0.0
            weight_error = 0.0
            same_gradient_error = 0.0
            worst = None
            for name, parameter in oracle.named_parameters():
                torch.testing.assert_close(gradients[name], parameter.grad, rtol=5e-5, atol=1e-7)
                gradient_error = max(
                    gradient_error, float((gradients[name] - parameter.grad).abs().max())
                )
                difference = (actual[name] - parameter.detach()).abs()
                if float(difference.max()) > weight_error:
                    flat = int(difference.argmax())
                    weight_error = float(difference.max())
                    worst = {
                        "parameter": name,
                        "flat_index": flat,
                        "dense_gradient": float(parameter.grad.flatten()[flat]),
                        "tp_gradient": float(gradients[name].flatten()[flat]),
                    }
            for name, parameter in emulated.named_parameters():
                parameter.grad = gradients[name].clone()
            same_gradient_optimizer.step()
            for name, value in emulated.state_dict().items():
                torch.testing.assert_close(actual[name], value, rtol=2e-5, atol=2e-7)
                same_gradient_error = max(
                    same_gradient_error, float((actual[name] - value).abs().max())
                )
            diagnostics.append(
                {
                    "family": kind.architecture,
                    "max_gradient_absolute_error": gradient_error,
                    "dense_update_absolute_error": weight_error,
                    "same_gradient_update_error": same_gradient_error,
                    "worst_update": worst,
                }
            )
            for stage, kv, tied in ((0, 2, False), (1, 1, False), (2, 2, True), (3, 1, True)):
                torch.manual_seed(71)
                dense = build_model(_config(kind, kv, tied))
                oracle = deepcopy(dense)

                factory = lambda parameters: torch.optim.SGD(
                    parameters, lr=0.02, momentum=0.8, weight_decay=0.01
                )
                optimizer = factory(oracle.parameters())
                model = parallelize_causal_lm(dense, context)
                local_elements = sum(parameter.numel() for parameter in model.parameters())
                objective = TensorParallelCrossEntropyObjective(context)
                engine = Trainer(
                    model,
                    objective,
                    zero_stage=stage,
                    parallel=context,
                    optimizer_factory=factory,
                    lr=0.02,
                    max_grad_norm=0.7,
                )
                for step in range(2):
                    loss, norm = _reference(oracle, optimizer, masked=step == 1)
                    result = engine.step(
                        [_data(context.dp.rank, step == 1 and context.dp.rank == 0)]
                    )
                    assert result.updated
                    assert result.loss == pytest.approx(loss, rel=2e-6, abs=2e-7)
                    assert result.grad_norm == pytest.approx(norm, rel=3e-5, abs=5e-7)
                    actual = engine.export_state_dict(only_rank_zero=False)
                    gradients = _gradients(engine)
                    for name, parameter in oracle.named_parameters(remove_duplicate=False):
                        torch.testing.assert_close(
                            gradients[name],
                            parameter.grad,
                            rtol=8e-5,
                            atol=2e-7,
                            msg=lambda message: (
                                f"{kind.architecture}/zero{stage}/{name} gradient: " + message
                            ),
                        )
                    assert set(actual) == set(oracle.state_dict())
                    for name, tensor in oracle.state_dict().items():
                        torch.testing.assert_close(
                            actual[name],
                            tensor,
                            rtol=8e-5,
                            atol=4e-7,
                            msg=lambda message: (
                                f"{kind.architecture}/zero{stage}/step{step}/{name}: " + message
                            ),
                        )
                if stage == 3:
                    units = [module for module in model.modules() if isinstance(module, Zero3Unit)]
                    assert all(
                        parameter.numel() == 0
                        for unit in units
                        for parameter in unit.module.parameters()
                    )
                    assert (
                        sum(parameter.numel() for parameter in model.parameters()) < local_elements
                    )
                    checkpoint = Path(output) / f"{kind.architecture}-checkpoint"
                    engine.save_checkpoint(checkpoint)
                    engine.step([_data(context.dp.rank)])
                    expected = engine.export_state_dict(only_rank_zero=False)
                    engine.load_checkpoint(checkpoint, trusted=True)
                    engine.step([_data(context.dp.rank)])
                    for key, value in engine.export_state_dict(only_rank_zero=False).items():
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                    reloaded = build_model(dense.config)
                    reloaded.load_state_dict(expected, strict=True)
                    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
                    with torch.no_grad():
                        model.eval()
                        reloaded.eval()
                        ids = _data(context.dp.rank)["input_ids"]
                        torch.testing.assert_close(
                            model(ids).logits, reloaded(ids).logits, rtol=2e-5, atol=3e-7
                        )
                    if kind is Qwen3Config:
                        engine.save_portable_checkpoint(Path(output) / "portable-padded-qwen3")
                        engine.step([_data(context.dp.rank)])
                        expected_migration = engine.export_state_dict()
                        if rank == 0:
                            torch.save(
                                expected_migration, Path(output) / "expected-after-migration.pt"
                            )
                dist.barrier()
        if rank == 0:
            atomic_json(Path(output) / "adam-roundoff-diagnostics.json", diagnostics)

        torch.manual_seed(88)
        model = parallelize_causal_lm(
            build_model(_config(Qwen3Config, 1, True, dropout=0.2)), context
        )
        objective = TensorParallelCrossEntropyObjective(context)
        engine = Trainer(
            model, objective, zero_stage=3, parallel=context, lr=0.002, max_grad_norm=0.7
        )
        torch.manual_seed(200 + rank)
        engine.step([_data(context.dp.rank)])
        engine.save_checkpoint(Path(output) / "adam-dropout-checkpoint")
        engine.step([_data(context.dp.rank)])
        expected = engine.export_state_dict(only_rank_zero=False)
        engine.load_checkpoint(Path(output) / "adam-dropout-checkpoint", trusted=True)
        engine.step([_data(context.dp.rank)])
        for key, value in engine.export_state_dict(only_rank_zero=False).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        target = engine.clone_target("model", "target", factory=lambda: build_model(model.config))
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=1e-7)

        data = _data(context.dp.rank)
        if rank == 1:
            data["input_ids"][0, 1] = 16
        with pytest.raises(ValueError, match="identical"):
            objective(model, data)
        data = _data(context.dp.rank)
        if rank == 3:
            data["labels"][0, 2] = 17
        with pytest.raises(ValueError, match="preflight"):
            objective(model, data)

        data = _data(context.dp.rank)
        if context.dp.rank == 1:
            ids = data.pop("input_ids")
            data["inputs_embeds"] = torch.nn.functional.embedding(
                ids, expected["model.embed_tokens.weight"]
            )
        with pytest.raises(ValueError, match="same embedding execution path"):
            objective(model, data)
    finally:
        dist.destroy_process_group()


def test_complete_causal_models_tp2_dp2_all_zero_and_standard_export(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path(os.environ.get("SystemDrive", "C:")) / "Temp"
    rendezvous = Path(tempfile.mkdtemp(prefix="aster_causal_tp_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(rendezvous / "rdzv"), str(tmp_path)), nprocs=4, join=True)
    finally:
        assert rendezvous.parent == root.resolve() and rendezvous.name.startswith(
            "aster_causal_tp_"
        )
        shutil.rmtree(rendezvous)

    model = build_model(_config(Qwen3Config, 1, True))
    factory = lambda parameters: torch.optim.SGD(
        parameters, lr=0.02, momentum=0.8, weight_decay=0.01
    )
    engine = Trainer(
        model,
        CrossEntropyObjective(),
        optimizer_factory=factory,
        lr=0.02,
        max_grad_norm=0.7,
        accumulation_steps=2,
    )
    engine.load_portable_checkpoint(tmp_path / "portable-padded-qwen3", seed=59)
    engine.step([_data(0), _data(1)])
    expected = torch.load(tmp_path / "expected-after-migration.pt", weights_only=True)
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=8e-5, atol=4e-7)
