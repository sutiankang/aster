from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.planet import PlaNetConfig, PlaNetWorldModel
from aster.methods.planet import PlaNetObjective
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
        torch.manual_seed(384)
        c = PlaNetConfig(
            observation_dim=4,
            action_dim=2,
            state_size=3,
            belief_size=8,
            hidden_size=8,
            reward_hidden_size=8,
            reward_layers=1,
        )
        initial = PlaNetWorldModel(c).state_dict()
        full = dict(
            observations=torch.randn(3, 4, 4),
            previous_actions=torch.randn(3, 4, 2),
            is_first=torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool),
            valid=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
            rewards=torch.randn(3, 4),
            prior_noise=torch.randn(3, 4, 3),
            posterior_noise=torch.randn(3, 4, 3),
            overshooting_noise=torch.randn(3, 4, 3, 3),
        )
        selection = slice(0, 1) if rank == 0 else slice(1, 3)
        local = {key: value[selection] for key, value in full.items()}
        for stage in range(4):
            model, dense = PlaNetWorldModel(c), PlaNetWorldModel(c)
            model.load_state_dict(initial)
            dense.load_state_dict(initial)
            objective = PlaNetObjective(
                sequence_length=4, free_nats=0.0, overshooting_distance=3, overshooting_weight=0.3
            )
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.0001, momentum=0.9)
            engine = Trainer(
                model,
                objective,
                parallel=context,
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0001, momentum=0.9),
            )
            bad = dict(local)
            if rank == 1:
                bad["observations"] = bad["observations"][:, :3]
            with pytest.raises(ValueError, match="sequence length"):
                engine.step([bad])
            for _ in range(2):
                optimizer.zero_grad()
                terms = objective(dense, full).terms
                loss = sum(term.mean * term.weight for term in terms)
                loss.backward()
                norm = torch.linalg.vector_norm(
                    torch.stack([parameter.grad.norm() for parameter in dense.parameters()])
                ).item()
                optimizer.step()
                result = engine.step([local])
                assert result.updated and abs(result.loss - loss.item()) < 6e-6
                assert abs(result.grad_norm - norm) < 2e-5
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, dense.state_dict()[key], atol=3e-7, rtol=3e-5)
            checkpoint = engine.save_checkpoint(Path(output) / f"zero{stage}")
            expected = engine.step([local])
            state = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([local])
            assert actual.loss == expected.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_planet_dp2_unequal_valid_sequences_zero_0_to_3_full_batch_update_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_planet_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_planet_"):
            shutil.rmtree(directory)
