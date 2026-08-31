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
from aster.training import Trainer, ParallelContext


class AliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictions = nn.Module()
        self.predictions.decoder = nn.Linear(3, 2)
        self.predictions.bias = self.predictions.decoder.bias

    def forward(self, x):
        return self.predictions.decoder(x)


def _objective(model, inputs):
    loss = (model(inputs) - 0.4).square()
    return LossTerm(loss.sum(), torch.tensor(loss.numel()), "elements")


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(843)
        model = AliasModel()
        dense = deepcopy(model)
        optimizer = torch.optim.AdamW(dense.parameters(), lr=0.02)
        context = ParallelContext()
        engine = Trainer(
            model,
            _objective,
            parallel=context,
            zero_stage=3,
            lr=0.02,
            max_grad_norm=None,
            ema_decay=0.9,
        )
        assert len(engine.roles["model"].parameters) == 2
        assert sum(parameter.numel() for parameter in engine.roles["model"].parameters) == 4
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            global_inputs = torch.cat((torch.ones(2, 3), torch.ones(3, 3) * 2))
            _objective(dense, global_inputs).mean.backward()
            optimizer.step()
            engine.step([torch.ones(2 + rank, 3) * (1 + rank)])
            exported = engine.export_state_dict(only_rank_zero=False)
            for name, expected in dense.state_dict().items():
                torch.testing.assert_close(exported[name], expected, atol=2e-6, rtol=2e-5)
            assert torch.equal(exported["predictions.bias"], exported["predictions.decoder.bias"])
        path = engine.save_checkpoint(Path(output) / "aliases")
        engine.step([torch.ones(2 + rank, 3) * (1 + rank)])
        expected = engine.export_state_dict(only_rank_zero=False)
        engine.load_checkpoint(path)
        engine.step([torch.ones(2 + rank, 3) * (1 + rank)])
        actual = engine.export_state_dict(only_rank_zero=False)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

        for projected in (True, rank == 0):
            embedding = nn.Embedding(3, 2, max_norm=1.0 if projected else None)
            original = embedding.weight.detach().clone()
            with pytest.raises(ValueError, match="max_norm"):
                Trainer(embedding, parallel=context)
            torch.testing.assert_close(embedding.weight, original)
    finally:
        dist.destroy_process_group()


def test_dp2_zero3_container_alias_single_storage_update_and_exact_resume(tmp_path):
    temp_root = Path(tempfile.gettempdir())
    if not str(temp_root).isascii():
        temp_root = Path("C:/Temp")
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="aster_alias_", dir=temp_root))
    try:
        mp.spawn(_worker, args=(str(rendezvous_dir / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if rendezvous_dir.parent == temp_root and rendezvous_dir.name.startswith("aster_alias_"):
            shutil.rmtree(rendezvous_dir)
