from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import faulthandler
import math
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import MixtralConfig, build_model
from aster.training import Trainer, ParallelContext, ParallelConfig
from aster.training.parallel import vocab_parallel_cross_entropy
from aster.training.moe_parallel import ExpertParallelCrossEntropyObjective
from aster.training.moe_tensor_parallel import (
    parallelize_mixtral_tensor,
    ExpertTensorParallelCrossEntropyObjective,
)
from aster.training.portable import logical_tensors, optimizer_mapping, gather_tensor, local_tensor
from aster.training.sharding import zero3_units
from aster.nn.parameter_codec import public_parameter_names


def _config(**kwargs):
    return MixtralConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        sliding_window=3,
        num_local_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=True,
        **kwargs,
    )


def _batches(rank, size):
    generator = torch.Generator().manual_seed(991)
    result = []
    allocations = ((2, 1, 3, 0), (1, 2, 0, 3)) if size == 4 else ((2, 4), (3, 3))
    for sizes in allocations:
        ids = torch.randint(1, 19, (6, 5), generator=generator)
        mask = torch.ones_like(ids)
        mask[1, -2:] = 0
        mask[3, -1:] = 0
        labels = ids.clone()
        labels[2, 3] = -100
        loss_mask = torch.ones_like(ids)
        loss_mask[0, 2] = 0
        full = dict(input_ids=ids, attention_mask=mask, labels=labels, loss_mask=loss_mask)
        start = sum(sizes[:rank])
        local = {key: value[start : start + sizes[rank]] for key, value in full.items()}
        if not sizes[rank]:
            local = dict(
                input_ids=torch.zeros(1, 5, dtype=torch.long),
                labels=torch.full((1, 5), -100),
                attention_mask=torch.zeros(1, 5, dtype=torch.long),
                loss_mask=torch.zeros(1, 5, dtype=torch.long),
            )
        result.append((local, full))
    return result


def _worker(rank, rendezvous, directory, tp, edp):
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(120, repeat=False)
    world = 4 * edp
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        context = ParallelContext(
            ParallelConfig(
                tensor_parallel=tp,
                data_parallel=world // tp,
                expert_parallel=2,
                expert_tensor_parallel=2,
            )
        )
        assert context.etp.ranks == (rank // 2 * 2, rank // 2 * 2 + 1)
        assert context.ep.ranks == (rank // 4 * 4 + rank % 2, rank // 4 * 4 + rank % 2 + 2)
        assert context.edp.ranks == tuple(range(rank % 4, world, 4))
        with pytest.raises(ValueError, match="attention TP="):
            ParallelContext(
                ParallelConfig(
                    tensor_parallel=2, data_parallel=world // 2, expert_tensor_parallel=4
                )
            )
        with pytest.raises(ValueError, match="expert-layout provider"):
            Trainer(
                build_model(_config()), lambda model, batch: None, parallel=context, zero_stage=3
            )
        if tp > 1:
            with pytest.raises(ValueError, match="attention_dropout=0"):
                parallelize_mixtral_tensor(build_model(_config(attention_dropout=0.1)), context)
            for dtype in (torch.float16, torch.bfloat16):
                complete_logits = torch.tensor(
                    [
                        [12.0, 11.7, -19.0, 0.2, -0.3, -torch.inf],
                        [1.0, -2.0, 3.0, 2.0, 7.0, -torch.inf],
                    ],
                    dtype=dtype,
                )
                local_logits = (
                    complete_logits.chunk(tp, -1)[context.tp.rank].clone().requires_grad_(True)
                )
                reference_logits = complete_logits.clone().requires_grad_(True)
                labels = torch.tensor([1, 4])
                actual = vocab_parallel_cross_entropy(local_logits, labels, context.tp)
                expected = torch.nn.functional.cross_entropy(
                    reference_logits.float(), labels, reduction="none"
                )
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(
                    local_logits.grad,
                    reference_logits.grad.chunk(tp, -1)[context.tp.rank],
                    atol=2e-6,
                    rtol=1e-6,
                )
        for zero in (0, 1, 2, 3):
            torch.manual_seed(907)
            source = build_model(_config())
            oracle = deepcopy(source)

            model = parallelize_mixtral_tensor(source, context)
            objective = ExpertTensorParallelCrossEntropyObjective(
                context, router_aux_coefficient=0.03
            )
            reference_objective = ExpertParallelCrossEntropyObjective(
                context, router_aux_coefficient=0.03
            )
            factory = lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.6, weight_decay=0.03)
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=zero,
                optimizer_factory=factory,
                accumulation_steps=2,
                max_grad_norm=0.4,
            )
            reference_optimizer = factory(oracle.parameters())
            batches = _batches(context.dp.rank, context.dp.size)
            for entry in logical_tensors(model, context):
                full = gather_tensor(entry.tensor, entry, context)
                if entry.persistent:
                    assert torch.equal(full, source.state_dict()[entry.name])
                assert torch.equal(local_tensor(full, entry, context), entry.tensor.detach())
                if ".experts." in entry.name:
                    assert entry.ep_group is context.ep and entry.tp_group is context.etp
                    assert entry.tp_stripes == (2 if "gate_up" in entry.name else 1)
                    shape = list(source.state_dict()[entry.name].shape)
                    shape[0] //= context.ep.size
                    shape[entry.tp_dimension] //= context.etp.size
                    assert entry.shape == tuple(shape)
                    expected_numel = (
                        math.ceil(math.prod(shape) / context.edp.size)
                        if zero == 3
                        else math.prod(shape)
                    )
                    assert entry.tensor.numel() == expected_numel
            for step in range(2):
                reference_optimizer.zero_grad(set_to_none=True)
                bundles = [reference_objective(oracle, full) for _, full in batches]
                expected_loss = sum(
                    sum(bundle.terms[i].numerator for bundle in bundles)
                    / sum(bundle.terms[i].denominator for bundle in bundles)
                    * term.weight
                    for i, term in enumerate(bundles[0].terms)
                )
                expected_loss.backward()
                expected_norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.4)
                reference_optimizer.step()
                result = engine.step([local for local, _ in batches])
                assert result.updated
                assert result.loss == pytest.approx(
                    float(expected_loss.detach()), abs=2e-6, rel=2e-5
                )
                assert result.grad_norm == pytest.approx(float(expected_norm), abs=3e-6, rel=3e-5)
                _, owners, sharded = optimizer_mapping(engine.roles["model"])
                actual_gradients = {
                    entry.name: gather_tensor(
                        owners[id(entry.tensor)].grad, entry, context, optimizer_sharded=sharded
                    )
                    for entry in logical_tensors(engine.model, context)
                    if entry.parameter
                }
                names = public_parameter_names(oracle)
                for name, parameter in oracle.named_parameters():
                    torch.testing.assert_close(
                        actual_gradients[names[name]],
                        parameter.grad,
                        atol=3e-6,
                        rtol=5e-5,
                        msg=f"TP{tp}/EDP{edp}/ZeRO{zero}/{step}/{name}/gradient",
                    )
                exported = engine.export_state_dict(only_rank_zero=False)
                for name, value in oracle.state_dict().items():
                    torch.testing.assert_close(
                        exported[name],
                        value,
                        atol=3e-6,
                        rtol=5e-5,
                        msg=f"TP{tp}/EDP{edp}/ZeRO{zero}/{step}/{name}/weight",
                    )
            if zero == 3:
                assert all(
                    unit.gathers and unit.gathers == unit.releases
                    for unit in zero3_units(engine.model)
                )
                assert all(
                    parameter.numel() == 0
                    for unit in zero3_units(engine.model)
                    for parameter in unit.module.parameters()
                )
            engine.save_portable_checkpoint(Path(directory) / f"portable-{zero}.json")
            dense = build_model(_config())
            dense.load_state_dict(exported, strict=True)
            assert dense.lm_head.weight is dense.model.embed_tokens.weight
            engine.clone_target("model", "target", factory=lambda: build_model(source.config))
            for name, value in engine.export_state_dict(
                role="target", only_rank_zero=False
            ).items():
                assert torch.equal(value, exported[name])
            path = engine.save_checkpoint(Path(directory) / f"native-{zero}.json")
            first = engine.step([local for local, _ in batches])
            expected = engine.export_state_dict(only_rank_zero=False)
            engine.load_checkpoint(path)
            assert engine.step([local for local, _ in batches]) == first
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, expected[name])
            if rank == 0:
                torch.save(expected, Path(directory) / f"next-{zero}.pt")
            before = [unit.gathers for unit in zero3_units(engine.model)]
            invalid = [dict(local) for local, _ in batches]
            if rank == 1:
                invalid[0]["labels"] = invalid[0]["labels"].float()
            with pytest.raises(ValueError, match="Labels"):
                engine.step(invalid)
            assert [
                unit.gathers for unit in zero3_units(engine.model)
            ] == before and not engine._failed
            if tp > 1:
                invalid = [dict(local) for local, _ in batches]
                if rank == 1:
                    invalid[0]["input_ids"] = (invalid[0]["input_ids"] + 1) % 19
                with pytest.raises(ValueError, match="identical"):
                    engine.step(invalid)
                assert [
                    unit.gathers for unit in zero3_units(engine.model)
                ] == before and not engine._failed

        for precision, offload in (("fp32", "cpu"), ("bf16", "nvme")):
            torch.manual_seed(974)
            source = build_model(_config(router_jitter_noise=0.1))

            for layer in source.model.layers:
                layer.mlp.gate.weight.data.zero_()
            model = parallelize_mixtral_tensor(source, context)
            engine = Trainer(
                model,
                ExpertTensorParallelCrossEntropyObjective(context, router_aux_coefficient=0.02),
                parallel=context,
                zero_stage=3,
                precision=precision,
                ema_decay=0.9,
                offload_optimizer=offload,
                offload_parameters="cpu",
                offload_directory=Path(directory) / "disk" if offload == "nvme" else None,
            )
            torch.manual_seed(773 + context.dp.rank)
            batch = _batches(context.dp.rank, context.dp.size)[0][0]
            assert engine.step([batch]).updated
            selected_owner = set((torch.zeros(1, 4).topk(2, -1).indices.flatten() // 2).tolist())
            assert len(selected_owner) == 1
            if context.ep.rank not in selected_owner:
                assert sum(model.model.layers[0].mlp.last_receive_counts) == 0
                assert max(model.model.layers[0].mlp.last_etp_lengths) == 0
            path = engine.save_checkpoint(Path(directory) / f"adam-{precision}.json")
            first = engine.step([batch])
            expected = engine.export_state_dict(only_rank_zero=False)
            ema = engine.export_state_dict(ema=True, only_rank_zero=False)
            engine.load_checkpoint(path)
            assert engine.step([batch]) == first
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, expected[name])
            for name, value in engine.export_state_dict(ema=True, only_rank_zero=False).items():
                assert torch.equal(value, ema[name])
            if tp > 1:
                before = [unit.gathers for unit in zero3_units(engine.model)]
                if rank == 1:
                    torch.rand(1)
                with pytest.raises(ValueError, match="identical"):
                    engine.step([batch])
                assert [
                    unit.gathers for unit in zero3_units(engine.model)
                ] == before and not engine._failed
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp", "edp"), [(1, 1), (2, 1), (2, 2)])
def test_complete_mixtral_ep_etp_edp(tmp_path, tp, edp):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-moe-etp-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-moe-etp-")
    try:
        mp.spawn(
            _worker,
            args=(str(directory / "store"), str(tmp_path), tp, edp),
            nprocs=4 * edp,
            join=True,
        )
    finally:
        shutil.rmtree(directory)
    torch.set_num_threads(1)
    for zero in (0, 1, 2, 3):

        class DenseObjective(ExpertParallelCrossEntropyObjective):
            def preflight_microbatches(self, model, batches):
                return batches

        engine = Trainer(
            build_model(_config()),
            DenseObjective(ParallelContext(), router_aux_coefficient=0.03),
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=0.02, momentum=0.6, weight_decay=0.03
            ),
            accumulation_steps=2,
            max_grad_norm=0.4,
        )
        engine.load_portable_checkpoint(tmp_path / f"portable-{zero}.json", seed=7)
        assert engine.steps == 2
        engine.step([full for _, full in _batches(0, 4 * edp // tp)])
        expected = torch.load(tmp_path / f"next-{zero}.pt", weights_only=True)
        for name, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=3e-6, rtol=5e-5)
