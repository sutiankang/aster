from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import faulthandler
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import MixtralConfig, build_model
from aster.training import (
    Trainer,
    ParallelContext,
    ParallelConfig,
    parallelize_mixtral,
    ExpertParallelCrossEntropyObjective,
)
from aster.training.portable import logical_tensors, optimizer_mapping, gather_tensor
from aster.training.sharding import zero3_units
from aster.nn.parameter_codec import public_parameter_names


def _configuration(**kwargs):
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
        **kwargs,
    )


def _batches(rank):
    generator = torch.Generator().manual_seed(998)
    result = []
    for sizes in ((2, 1, 3, 0), (1, 2, 0, 3)):
        ids = torch.randint(1, 19, (6, 6), generator=generator)
        mask = torch.ones_like(ids)
        mask[1, -2:] = 0
        mask[3, -1:] = 0
        labels = ids.clone()
        labels[2, 3] = -100
        loss_mask = torch.ones_like(ids)
        loss_mask[0, 2] = 0
        full = {"input_ids": ids, "attention_mask": mask, "labels": labels, "loss_mask": loss_mask}
        start = sum(sizes[:rank])
        local = {key: value[start : start + sizes[rank]] for key, value in full.items()}
        if sizes[rank] == 0:
            local = {
                "input_ids": torch.zeros(1, 6, dtype=torch.long),
                "labels": torch.full((1, 6), -100),
                "attention_mask": torch.zeros(1, 6, dtype=torch.long),
                "loss_mask": torch.zeros(1, 6, dtype=torch.long),
            }
        result.append((local, full))
    return result


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    faulthandler.dump_traceback_later(75, repeat=False)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=160),
    )
    try:
        context = ParallelContext(ParallelConfig(data_parallel=4, expert_parallel=2))
        assert context.ep.ranks == ((0, 1) if rank < 2 else (2, 3))
        assert context.edp.ranks == ((0, 2) if rank % 2 == 0 else (1, 3))
        for zero in (0, 1, 2, 3):
            torch.manual_seed(501)
            original = build_model(_configuration(tie_word_embeddings=True))
            oracle = deepcopy(original)

            for layer in original.model.layers:
                layer.mlp.gate.weight.data.zero_()
            oracle.load_state_dict(original.state_dict())
            model = parallelize_mixtral(original, context)
            objective = ExpertParallelCrossEntropyObjective(context, router_aux_coefficient=0.02)
            factory = lambda parameters: torch.optim.SGD(
                parameters, lr=0.02, momentum=0.6, weight_decay=0.03
            )
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
            batches = _batches(rank)
            for step in range(2):
                reference_optimizer.zero_grad(set_to_none=True)
                terms = [objective(oracle, full) for _, full in batches]
                expected_loss = sum(
                    sum(bundle.terms[i].numerator for bundle in terms)
                    / sum(bundle.terms[i].denominator for bundle in terms)
                    * term.weight
                    for i, term in enumerate(terms[0].terms)
                )
                expected_loss.backward()
                expected_norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.4)
                reference_optimizer.step()
                result = engine.step([local for local, _ in batches])
                assert result.updated and result.loss == pytest.approx(
                    float(expected_loss.detach()), abs=2e-6, rel=2e-5
                )
                assert result.grad_norm == pytest.approx(float(expected_norm), abs=2e-6, rel=2e-5)
                actual = engine.export_state_dict(only_rank_zero=False)
                _, owners, sharded = optimizer_mapping(engine.roles["model"])
                gradients = {
                    entry.name: gather_tensor(
                        owners[id(entry.tensor)].grad, entry, context, optimizer_sharded=sharded
                    )
                    for entry in logical_tensors(engine.model, context)
                    if entry.parameter
                }
                names = public_parameter_names(oracle)
                for name, parameter in oracle.named_parameters():
                    torch.testing.assert_close(
                        gradients[names[name]],
                        parameter.grad,
                        atol=2e-6,
                        rtol=3e-5,
                        msg=f"zero{zero}/{step}/{name}/grad",
                    )
                for name, value in oracle.state_dict().items():
                    torch.testing.assert_close(
                        actual[name],
                        value,
                        atol=2e-6,
                        rtol=3e-5,
                        msg=f"zero{zero}/{step}/{name}/weight",
                    )
            if zero == 3:
                assert all(
                    unit.gathers > 0 and unit.gathers == unit.releases
                    for unit in zero3_units(engine.model)
                )
                assert all(
                    parameter.numel() == 0
                    for unit in zero3_units(engine.model)
                    for parameter in unit.module.parameters()
                )

            portable = engine.save_portable_checkpoint(Path(directory) / f"portable-{zero}.json")
            complete = engine.export_state_dict(only_rank_zero=False)
            reloaded = build_model(oracle.config)
            reloaded.load_state_dict(complete, strict=True)
            assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight

            engine.clone_target("model", "target", factory=lambda: build_model(original.config))
            for name, value in engine.export_state_dict(
                role="target", only_rank_zero=False
            ).items():
                assert torch.equal(value, complete[name])
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

        for precision in ("fp32", "bf16"):
            torch.manual_seed(674)
            model = parallelize_mixtral(
                build_model(_configuration(router_jitter_noise=0.1)), context
            )
            engine = Trainer(
                model,
                ExpertParallelCrossEntropyObjective(context),
                parallel=context,
                zero_stage=3,
                precision=precision,
                offload_optimizer="cpu",
                offload_parameters="cpu",
            )
            batch = _batches(rank)[0][0]
            engine.step([batch])
            path = engine.save_checkpoint(Path(directory) / f"adam-jitter-{precision}.json")
            first = engine.step([batch])
            expected = engine.export_state_dict(only_rank_zero=False)
            engine.load_checkpoint(path)
            assert engine.step([batch]) == first
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, expected[name])
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def test_complete_mixtral_ep2_edp2_zero_all_stages(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-moe-model-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-moe-model-")
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=4, join=True)
    finally:
        shutil.rmtree(directory)

    torch.set_num_threads(1)
    for zero in (0, 1, 2, 3):
        context = ParallelContext()
        model = build_model(_configuration(tie_word_embeddings=True))
        objective = ExpertParallelCrossEntropyObjective(context, router_aux_coefficient=0.02)

        class DenseObjective(ExpertParallelCrossEntropyObjective):
            def preflight_microbatches(self, model, batches):
                return batches

        objective = DenseObjective(context, router_aux_coefficient=0.02)
        engine = Trainer(
            model,
            objective,
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=0.02, momentum=0.6, weight_decay=0.03
            ),
            accumulation_steps=2,
            max_grad_norm=0.4,
        )
        engine.load_portable_checkpoint(tmp_path / f"portable-{zero}.json", seed=7)
        assert engine.steps == 2
        assert engine.step([full for _, full in _batches(0)]).updated
        expected = torch.load(tmp_path / f"next-{zero}.pt", weights_only=True)
        for name, value in engine.export_state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=2e-6, rtol=3e-5)
