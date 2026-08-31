from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from aster.models.vmc import MDNRNN, MDNRNNConfig
from aster.methods.vmc import MDNRNNObjective
from aster.methods.vmc_stream import VMCSequenceStream, MDNStreamMethod
from aster.training import Trainer, ParallelContext
from aster.training.sharding import zero3_units
from aster.training.portable import logical_tensors, optimizer_mapping, gather_tensor


def _episodes(distribution=False):
    generator = torch.Generator().manual_seed(491)
    result = []
    for count in (21, 31, 44):
        latent = torch.randn(count, 3, generator=generator) * 0.2
        row = dict(actions=torch.randn(count, 1, generator=generator) * 0.3)
        if distribution:
            row.update(mean=latent, logvar=torch.full_like(latent, -2.0))
        else:
            row["latents"] = latent
        result.append(row)
    return result


def _reference(model, batches, state):
    nll, bce, count = 0.0, 0.0, 0
    for batch in batches:
        output = model(
            batch["latents"][:, :-1],
            batch["actions"][:, :-1],
            batch["restart"][:, :-1],
            state=state,
        )
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=output.logmix),
            torch.distributions.Normal(output.mean, output.logstd.exp()),
        )
        nll = nll - distribution.log_prob(batch["latents"][:, 1:]).sum()
        target = batch["restart"][:, 1:].float()
        bce = (
            bce
            + (
                F.binary_cross_entropy_with_logits(output.restart_logits, target, reduction="none")
                * (1 + 9 * target)
            ).sum()
        )
        count += target.numel()
        state = output.state.detach()
    return nll / (count * 3) + bce / count, state


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
        context = ParallelContext()
        c = MDNRNNConfig(latent_size=3, hidden_size=8, mixtures=2)
        for zero in range(4):
            torch.manual_seed(373)
            model = MDNRNN(c)
            dense = deepcopy(model)
            factory = lambda p: torch.optim.SGD(p, lr=0.003, momentum=0.7)
            engine = Trainer(
                model,
                MDNRNNObjective(sequence_length=4),
                zero_stage=zero,
                parallel=context,
                accumulation_steps=2,
                max_grad_norm=None,
                max_grad_value=1.0,
                optimizer_factory=factory,
                offload_optimizer="cpu",
                offload_parameters="cpu" if zero == 3 else "none",
            )
            stream = VMCSequenceStream(
                _episodes(), batch_size=3, sequence_length=4, rank=rank, world_size=2, seed=29
            )
            method = MDNStreamMethod(engine, stream)
            reference_stream = VMCSequenceStream(
                _episodes(), batch_size=3, sequence_length=4, seed=29
            )
            optimizer, state = factory(dense.parameters()), None
            for update in range(2):
                full, rng = reference_stream.preview(2)
                optimizer.zero_grad(set_to_none=True)
                loss, state = _reference(dense, full, state)
                loss.backward()
                norm = torch.stack([p.grad.square().sum() for p in dense.parameters()]).sum().sqrt()
                torch.nn.utils.clip_grad_value_(dense.parameters(), 1.0)
                optimizer.step()
                reference_stream._commit(2, rng)
                result = method.step()
                assert result.loss == pytest.approx(float(loss.detach()), abs=1e-6)
                assert result.grad_norm == pytest.approx(float(norm), abs=2e-6)
                assert result.terms["restart"]["denominator"] == 18
                selection = slice(stream.lane_start, stream.lane_end)
                torch.testing.assert_close(
                    method.state.cell, state.cell[selection], atol=3e-7, rtol=3e-5
                )
                _, owners, sharded = optimizer_mapping(engine.roles["model"])
                gradients = {
                    entry.name: gather_tensor(
                        owners[id(entry.tensor)].grad, entry, context, optimizer_sharded=sharded
                    )
                    for entry in logical_tensors(engine.model, context)
                    if entry.parameter
                }
                for name, parameter in dense.named_parameters():
                    torch.testing.assert_close(
                        gradients[name], parameter.grad, atol=3e-7, rtol=3e-5
                    )
                for name, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(
                        value, dense.state_dict()[name], atol=3e-7, rtol=3e-5
                    )
            path = engine.save_checkpoint(Path(output) / f"zero-{zero}")
            expected = method.step()
            weights = engine.export_state_dict(only_rank_zero=False)
            hidden, rng = method.state.cell.clone(), stream.state_dict()["latent_rng"]
            engine.load_checkpoint(path)
            assert method.step() == expected
            assert torch.equal(method.state.cell, hidden) and torch.equal(
                stream.state_dict()["latent_rng"], rng
            )
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, weights[name])
            before = [unit.gathers for unit in zero3_units(engine.model)]
            saved_cursor = stream.cursor
            if rank == 1:
                stream.cursor = stream.num_chunks + 1
            with pytest.raises(ValueError, match="preflight"):
                method.step()
            stream.cursor = saved_cursor
            assert [
                unit.gathers for unit in zero3_units(engine.model)
            ] == before and not engine._failed
            if rank == 1:
                method.objective.restart_factor = 3.0
            with pytest.raises(ValueError, match="settings changed"):
                method.step()
            method.objective.restart_factor = 10.0
            assert [unit.gathers for unit in zero3_units(engine.model)] == before
            previous = method.state
            if rank == 1:
                method.state = deepcopy(previous)
                method.state.cell = method.state.cell[:, :-1]
            with pytest.raises(ValueError, match="carry"):
                method.step()
            method.state = previous
            assert [
                unit.gathers for unit in zero3_units(engine.model)
            ] == before and not engine._failed
            assert method.step().updated
            method.advance_epoch()
            assert stream.epoch == 1 and method.state is None

        torch.manual_seed(639)
        c = MDNRNNConfig(
            latent_size=3,
            hidden_size=8,
            mixtures=2,
            input_dropout=0.1,
            output_dropout=0.1,
            recurrent_dropout=0.1,
        )
        engine = Trainer(
            MDNRNN(c),
            MDNRNNObjective(sequence_length=4),
            zero_stage=3,
            parallel=context,
            precision="bf16",
            accumulation_steps=2,
            max_grad_norm=None,
            max_grad_value=1.0,
            offload_optimizer="cpu",
            offload_parameters="cpu",
            optimizer_factory=lambda p: torch.optim.Adam(p, lr=0.001, eps=1e-4),
        )
        stream = VMCSequenceStream(
            _episodes(True), batch_size=3, sequence_length=4, rank=rank, world_size=2, seed=41
        )
        method = MDNStreamMethod(engine, stream)
        method.step()
        path = engine.save_checkpoint(Path(output) / "bf16")
        expected = method.step()
        weights = engine.export_state_dict(only_rank_zero=False)
        hidden = method.state.cell.clone()
        engine.load_checkpoint(path)
        assert method.step() == expected
        assert torch.equal(method.state.cell, hidden)
        for name, value in engine.export_state_dict(only_rank_zero=False).items():
            assert torch.equal(value, weights[name])

        other_engine = Trainer(
            MDNRNN(c), MDNRNNObjective(sequence_length=4), zero_stage=3, parallel=context
        )
        other_stream = VMCSequenceStream(
            _episodes(True), batch_size=3, sequence_length=4 + rank, rank=rank, world_size=2
        )
        with pytest.raises(ValueError, match="preflight"):
            MDNStreamMethod(other_engine, other_stream)
        assert all(unit.gathers == 0 for unit in zero3_units(other_engine.model))
    finally:
        dist.destroy_process_group()


def test_vmc_stream_real_dp2_zero_0_to_3_global_normalization_and_exact_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-vmc-stream-", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster-vmc-stream-"):
            shutil.rmtree(directory)
