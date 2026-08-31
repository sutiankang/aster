from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import LossTerm
from aster.training import Trainer, MuonFactory, MuonWithAuxAdam, ParallelConfig, ParallelContext
from aster.training.portable import optimizer_mapping, logical_tensors, gather_tensor


def objective(model, batch):
    x, target = batch
    output = model(x).float()
    return LossTerm(
        (output - target).square().sum(), torch.tensor(output.numel(), dtype=torch.int64), "element"
    )


def worker(rank, rendezvous, directory):
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

        def data(replica):
            generator = torch.Generator().manual_seed(90 + replica)
            return [
                (torch.randn(n, 3, generator=generator), torch.randn(n, 2, generator=generator))
                for n in ((1, 2) if replica == 0 else (3, 1))
            ]

        for zero, profile, precision, offload in [
            (z, p, v, o)
            for z in range(4)
            for p in ("keller", "moonlight")
            for v, o in (("fp32", "none"), ("bf16", "cpu"), ("bf16", "nvme"))
        ]:

            def make():
                torch.manual_seed(481)
                model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
                factory = MuonFactory.from_model(
                    model,
                    auxiliary_modules=("2",),
                    profile=profile,
                    muon_options={"lr": 0.001},
                    auxiliary_options={"lr": 0.0007},
                )
                return Trainer(
                    model,
                    objective,
                    parallel=context,
                    precision=precision,
                    optimizer_factory=factory,
                    zero_stage=zero,
                    accumulation_steps=2,
                    max_grad_norm=0.5,
                    offload_optimizer=offload,
                    offload_directory=Path(directory) / f"disk-{rank}"
                    if offload == "nvme"
                    else None,
                )

            torch.manual_seed(481)
            reference = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
            engine = make()
            parameters = dict(reference.named_parameters())
            native, owners, sharded = optimizer_mapping(engine.roles["model"])
            entries = logical_tensors(engine.model, context)
            source_groups = [
                {**{key: value for key, value in group.items() if key != "params"}, "params": []}
                for group in native.param_groups
            ]
            mapping = {
                id(owners[id(entry.tensor)]): entry.name
                for entry in entries
                if entry.parameter and id(entry.tensor) in owners
            }
            for source, group in zip(source_groups, native.param_groups):
                source["params"] = [parameters[mapping[id(p)]] for p in group["params"]]
            optimizer = MuonWithAuxAdam(source_groups)
            actual_gradients = {}
            original = native.step

            def capture(*args, **kwargs):
                for entry in entries:
                    if not entry.parameter or id(entry.tensor) not in owners:
                        continue
                    owner = owners[id(entry.tensor)]
                    if owner.grad is not None:
                        actual_gradients[entry.name] = gather_tensor(
                            owner.grad, entry, context, optimizer_sharded=sharded
                        )
                return original(*args, **kwargs)

            native.step = capture
            for step in range(2):
                if precision == "fp32":
                    terms = [
                        objective(reference, batch) for replica in (0, 1) for batch in data(replica)
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    (sum(t.numerator for t in terms) / sum(t.denominator for t in terms)).backward()
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
                    global_gradients = {
                        name: p.grad.clone() for name, p in reference.named_parameters()
                    }
                result = engine.step(data(rank))
                assert result.updated
                if precision == "fp32":
                    for name, gradient in actual_gradients.items():
                        torch.testing.assert_close(
                            gradient, global_gradients[name], atol=5e-8, rtol=4e-6
                        )
                        parameters[name].grad = gradient.clone()
                    optimizer.step()
                    for name, value in engine.export_state_dict(only_rank_zero=False).items():
                        assert torch.equal(value, reference.state_dict()[name])
                for group in native.param_groups:
                    if group["use_muon"]:
                        for owner in group["params"]:
                            assert owner._aster_muon_layout.shape == (5, 3)
                            assert owner.numel() == (8 if zero else 15)
                            state = (
                                native._aster_state_loader(owner)
                                if hasattr(native, "_aster_state_loader")
                                else native.state[owner]
                            )
                            assert state["momentum_buffer"].numel() == owner.numel()
                if step == 0:
                    checkpoint = engine.save_checkpoint(
                        Path(directory) / f"{zero}-{profile}-{precision}-{offload}"
                    )
                    engine.save_portable_checkpoint(
                        Path(directory) / f"portable-{zero}-{profile}-{precision}-{offload}"
                    )
            expected = engine.export_state_dict(only_rank_zero=False)
            fresh = make()
            fresh.load_checkpoint(checkpoint)
            fresh.step(data(rank))
            for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, expected[name])
        torch.manual_seed(481)
        model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
        factory = MuonFactory.from_model(
            model, auxiliary_modules=("1",), profile="keller", muon_options={"lr": 0.01 + rank}
        )
        with pytest.raises(ValueError, match="options differ"):
            Trainer(model, objective, parallel=context, optimizer_factory=factory)
        torch.manual_seed(481)
        model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
        factory = MuonFactory.from_model(model, auxiliary_modules=(), profile="keller")
        if rank == 1:
            factory.groups[0]["names"].reverse()
        with pytest.raises(ValueError, match="collective order"):
            Trainer(model, objective, parallel=context, optimizer_factory=factory)
        torch.manual_seed(481)
        model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
        selected = (
            MuonFactory.from_model(model, auxiliary_modules=(), profile="keller")
            if rank == 0
            else lambda params: torch.optim.SGD(params, lr=0.01)
        )
        with pytest.raises(ValueError, match="factory protocol"):
            Trainer(model, objective, parallel=context, optimizer_factory=selected, zero_stage=3)
        for mutation in ("order", "learning_rate", "missing_field", "missing_params"):
            torch.manual_seed(481)
            model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=3,
                optimizer_factory=MuonFactory.from_model(
                    model, auxiliary_modules=(), profile="keller"
                ),
            )
            native, _, _ = optimizer_mapping(engine.roles["model"])
            group = native.param_groups[0]
            before = [p.detach().clone() for p in engine.roles["model"].parameters]
            forwards = []
            hook = engine.model.register_forward_pre_hook(lambda *args: forwards.append(True))
            if rank == 1:
                if mutation == "order":
                    group["params"].reverse()
                elif mutation == "learning_rate":
                    group["lr"] *= 2
                elif mutation == "missing_field":
                    del group["ns_steps"]
                else:
                    del group["params"]
            with pytest.raises(ValueError, match="optimizer|Muon"):
                engine.step([data(rank)[0]])
            hook.remove()
            assert not forwards and engine.steps == 0 and not engine._failed
            for parameter, value in zip(engine.roles["model"].parameters, before):
                assert torch.equal(parameter, value)
    finally:
        dist.destroy_process_group()


def test_muon_dp2_profiles_global_gradients_offload_and_exact_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-muon-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-muon-")
    try:
        mp.spawn(worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
