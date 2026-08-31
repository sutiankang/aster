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

from aster.core import LossBundle, LossTerm
from aster.training import Trainer, ParallelConfig, ParallelContext


class Objective:
    def config_dict(self):
        return {"test": "global_ratio_unequal_valid_counts", "version": 1}

    def __call__(self, model, batch):
        x, y, mask = batch
        prediction = model(x).float()
        count = mask.sum().to(torch.int64)
        return LossBundle(
            (
                LossTerm(
                    (prediction - y).square().masked_select(mask).sum(),
                    count,
                    "sample",
                    "reference",
                    0.2,
                ),
                LossTerm(prediction.masked_select(mask).sum(), count * 2, "pair", "target", 0.4),
            )
        )


def batches(replica):
    generator = torch.Generator().manual_seed(123 + replica)
    output = []
    for count in [1, 2] if replica == 0 else [3, 1]:
        output.append(
            (
                torch.randn(count, 3, generator=generator),
                torch.randn(count, 1, generator=generator),
                torch.ones(count, 1, dtype=torch.bool),
            )
        )
    if replica:
        output[-1][2].zero_()
    return output


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
        objective = Objective()
        for zero, precision, offload, overlap in (
            [(z, "fp32", "cpu", False) for z in (0, 1, 2, 3)]
            + [(0, "fp32", "cpu", True)]
            + [(z, "bf16", "nvme", False) for z in (0, 1, 2, 3)]
        ):
            torch.manual_seed(731)
            reference = nn.Linear(3, 1, bias=False)
            source = deepcopy(reference)
            factory = lambda p: torch.optim.SGD(p, lr=0.015, momentum=0.7, weight_decay=0.02)

            def make(model):
                engine = Trainer(
                    model,
                    objective,
                    parallel=context,
                    zero_stage=zero,
                    optimizer_factory=factory,
                    accumulation_steps=2,
                    max_grad_norm=0.5,
                    precision=precision,
                    offload_optimizer=offload,
                    offload_directory=Path(directory) / f"disk-{rank}-{zero}"
                    if offload == "nvme"
                    else None,
                    communication_overlap=overlap,
                    bucket_bytes=4,
                )
                engine.register_gradient_ratio(
                    "ratio",
                    reference_term="reference",
                    target_term="target",
                    parameter="weight",
                    multiplier=0.8,
                )
                return engine

            engine = make(source)
            optimizer = factory(reference.parameters())
            for update in range(2):
                if precision == "fp32":
                    terms = [
                        objective(reference, batch).terms
                        for replica in (0, 1)
                        for batch in batches(replica)
                    ]
                    means = [
                        sum(group[i].numerator for group in terms)
                        / sum(group[i].denominator for group in terms)
                        for i in range(2)
                    ]
                    first, second = [
                        torch.autograd.grad(value, reference.weight, retain_graph=True)[0]
                        for value in means
                    ]
                    ratio = (first.double().norm() / (second.double().norm() + 1e-4)).clamp(0, 1e4)
                    loss = 0.2 * means[0] + 0.4 * 0.8 * ratio * means[1]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
                    optimizer.step()
                result = engine.step(batches(rank))
                assert result.updated
                record = engine.last_gradient_ratio("ratio")
                copies = context.world.gather_objects(record)
                assert all(copy == record for copy in copies)
                if precision == "fp32":
                    assert record["ratio"] == pytest.approx(float(ratio), rel=2e-6)
                    assert result.grad_norm == pytest.approx(float(norm), rel=2e-6)
                    torch.testing.assert_close(
                        engine.export_state_dict(only_rank_zero=False)["weight"],
                        reference.weight,
                        atol=2e-7,
                        rtol=2e-6,
                    )
                if update == 0:
                    checkpoint = engine.save_checkpoint(
                        Path(directory) / f"checkpoint-{zero}-{precision}-{overlap}"
                    )
            expected, receipt = (
                engine.export_state_dict(only_rank_zero=False),
                engine.last_gradient_ratio("ratio"),
            )
            fresh = make(nn.Linear(3, 1, bias=False))
            fresh.load_checkpoint(checkpoint)
            fresh.step(batches(rank))
            assert fresh.last_gradient_ratio("ratio") == receipt
            for name, value in fresh.export_state_dict(only_rank_zero=False).items():
                assert torch.equal(value, expected[name])

            fresh._gradient_ratio.policies["ratio"]["eps"] = 1e-4 + rank
            with pytest.raises(ValueError, match="Phase declaration"):
                fresh.step(batches(rank))
            assert not fresh._failed
        engine = Trainer(nn.Linear(3, 1), objective, parallel=context, zero_stage=3)
        with pytest.raises(ValueError, match="policy differs"):
            engine.register_gradient_ratio(
                "bad",
                reference_term="reference",
                target_term="target",
                parameter="weight",
                eps=1e-4 + rank,
            )
    finally:
        dist.destroy_process_group()


def test_global_adaptive_ratio_dp_zero_overlap_offload_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster-gradient-ratio-", dir=root)).resolve()
    assert directory.parent == root.resolve() and directory.name.startswith("aster-gradient-ratio-")
    try:
        mp.spawn(worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        shutil.rmtree(directory)
