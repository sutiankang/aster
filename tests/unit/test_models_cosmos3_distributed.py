from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import Cosmos3Config, Cosmos3Vision, Cosmos3Sequence, build_model
from aster.methods.cosmos3 import Cosmos3FlowObjective
from aster.training import ParallelContext, Trainer


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(548)
        c = Cosmos3Config()
        initial = build_model(c).state_dict()
        ids = torch.tensor([[1, 3, 2, 0], [1, 5, 7, 2]])

        def positions(n):
            return torch.arange(n)[None, None].expand(3, 2, -1) + 15004

        fields = dict(
            vision=Cosmos3Vision(
                torch.randn(2, 2, 2, 3, 4),
                positions(8),
                torch.full((2, 2), 540.0),
                torch.tensor([[False, True], [True, True]]),
            ),
            sound=Cosmos3Sequence(
                torch.randn(2, 3, 4),
                positions(3),
                torch.full((2, 3), 630.0),
                torch.tensor([[True, False, False], [True, True, True]]),
                torch.tensor([[True, False, False], [True, True, True]]),
            ),
            action=Cosmos3Sequence(
                torch.randn(2, 2, 3),
                positions(2),
                torch.full((2, 2), 710.0),
                torch.tensor([[False, False], [True, True]]),
                domain_ids=torch.tensor([1, 3]),
            ),
        )
        noises = {name: torch.randn_like(field.sample) for name, field in fields.items()}
        labels = ids.masked_fill(ids.eq(0), -100)
        full = dict(
            model_inputs=dict(input_ids=ids, attention_mask=ids.ne(0), **fields),
            labels=labels,
            noise=noises,
        )
        local_fields = {}
        for name, field in fields.items():
            changes = {
                key: value[rank : rank + 1]
                for key, value in vars(field).items()
                if isinstance(value, torch.Tensor) and key != "positions"
            }
            local_fields[name] = replace(
                field, **changes, positions=field.positions[:, rank : rank + 1]
            )
        local = dict(
            model_inputs=dict(
                input_ids=ids[rank : rank + 1],
                attention_mask=ids[rank : rank + 1].ne(0),
                **local_fields,
            ),
            labels=labels[rank : rank + 1],
            noise={name: value[rank : rank + 1] for name, value in noises.items()},
        )
        objective = Cosmos3FlowObjective(text_weight=0.2, time_distribution="provided")
        for stage in range(4):
            dense, native = build_model(c), build_model(c)
            dense.load_state_dict(initial)
            native.load_state_dict(initial)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.0001, momentum=0.9)
            engine = Trainer(
                native,
                objective,
                parallel=ParallelContext(),
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda parameters: torch.optim.SGD(
                    parameters, lr=0.0001, momentum=0.9
                ),
            )
            bundle = objective(dense, full)
            loss = sum(term.weight * term.mean for term in bundle.terms)
            loss.backward()
            optimizer.step()
            result = engine.step([local])
            assert result.updated and abs(result.loss - loss.item()) < 3e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=4e-7, rtol=5e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"cosmos3_zero{stage}")
            expected = engine.step([local])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([local])
            assert expected.loss == actual.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_models_cosmos3_real_dp2_all_zero_independent_global_counts_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_cosmos3_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_cosmos3_"):
            shutil.rmtree(directory)
