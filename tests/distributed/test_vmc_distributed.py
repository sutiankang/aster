from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.vmc import MDNRNNConfig, MDNRNN
from aster.methods.vmc import MDNRNNObjective
from aster.training import ParallelContext, Trainer


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
        torch.manual_seed(249)
        c = MDNRNNConfig(latent_size=3, hidden_size=8, mixtures=2)
        initial = MDNRNN(c).state_dict()
        full = dict(
            latents=torch.randn(3, 4, 3),
            actions=torch.rand(3, 4, 1),
            restart=torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool),
            valid=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        )
        selection = slice(0, 1) if rank == 0 else slice(1, 3)
        local = {k: v[selection] for k, v in full.items()}
        for stage in range(4):
            model, dense = MDNRNN(c), MDNRNN(c)
            model.load_state_dict(initial)
            dense.load_state_dict(initial)
            objective = MDNRNNObjective(sequence_length=4)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.001, momentum=0.9),
            )
            for _ in range(2):
                optimizer.zero_grad()
                terms = objective(dense, full).terms
                loss = sum(term.mean for term in terms)
                loss.backward()
                optimizer.step()
                result = engine.step([local])
                assert result.updated and abs(result.loss - loss.item()) < 1e-6
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, dense.state_dict()[key], atol=3e-7, rtol=3e-5)
            checkpoint = engine.save_checkpoint(Path(output) / f"zero{stage}")
            expected = engine.step([local])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([local])
            assert actual.loss == expected.loss
            for key, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_vmc_mdn_real_dp2_zero_0_to_3_unequal_valid_counts_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_vmc_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_vmc_"):
            shutil.rmtree(directory)
