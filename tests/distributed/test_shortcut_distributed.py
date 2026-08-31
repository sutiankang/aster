from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.interval_dit import IntervalDiT, IntervalDiTConfig
from aster.methods.shortcut import ShortcutMethod
from aster.training import Trainer, ParallelContext


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
        context = ParallelContext()
        for ema in (False, True):
            reference = None
            for stage in range(4):
                torch.manual_seed(449)
                c = IntervalDiTConfig(
                    variant="shortcut",
                    input_size=4,
                    in_channels=1,
                    hidden_size=16,
                    num_layers=1,
                    num_heads=2,
                    num_classes=2,
                )
                engine = Trainer(
                    IntervalDiT(c),
                    parallel=context,
                    zero_stage=stage,
                    accumulation_steps=2,
                    max_grad_norm=None,
                    optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
                )
                method = ShortcutMethod(
                    engine,
                    base_steps=8,
                    bootstrap_every=2,
                    bootstrap_ema=ema,
                    bootstrap_cfg=True,
                    ema_decay=0.8,
                )
                torch.manual_seed(345 + rank)
                size = 4 + 2 * rank
                batches = [
                    dict(sample=torch.randn(size, 1, 4, 4), labels=torch.arange(size) % 2)
                    for _ in range(2)
                ]
                invalid = deepcopy(batches)
                if rank == 1:
                    invalid[1]["labels"][0] = 10
                with pytest.raises(ValueError, match="labels"):
                    method.update(invalid)
                assert not method._incomplete
                assert method.update(batches).updated
                path = engine.save_checkpoint(Path(output) / f"ema{ema}-zero{stage}")
                expected = method.update(batches)
                states = {
                    role: deepcopy(engine.export_state_dict(role=role, only_rank_zero=False))
                    for role in engine.roles
                }
                if reference is None:
                    reference = states
                else:
                    for role in reference:
                        for key, value in states[role].items():
                            torch.testing.assert_close(
                                value, reference[role][key], atol=5e-7, rtol=4e-5
                            )
                engine.load_checkpoint(path, trusted=True)
                actual = method.update(batches)
                assert actual.loss == expected.loss
                for role in states:
                    for key, value in engine.export_state_dict(
                        role=role, only_rank_zero=False
                    ).items():
                        torch.testing.assert_close(value, states[role][key], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_shortcut_real_dp2_all_zero_global_strata_ema_and_restore(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_shortcut_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_shortcut_"):
            shutil.rmtree(directory)
