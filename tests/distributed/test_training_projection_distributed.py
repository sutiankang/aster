from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.core import LossTerm
from aster.training import Trainer, ParallelConfig, ParallelContext
from aster.training.portable import optimizer_mapping


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0, 0.0],
                        [3.0, 4.0, 0.0],
                        [0.1, 0.2, 0.2],
                        [0.0, 0.0, 5.0],
                        [-4.0, 0.0, 0.0],
                    ]
                )
            )

    def forward(self, indices):
        return self.embedding(indices)


def _objective(model, indices):
    values = model(indices).square()
    return LossTerm(values.sum(), torch.tensor(values.numel()), "elements")


class DeclaredObjective:
    def __init__(self, temperature):
        self.temperature = temperature

    def config_dict(self):
        return {"type": "declared_objective", "temperature": self.temperature}

    def __call__(self, model, indices):
        return _objective(model, indices)


def _moments(engine):
    optimizer, _, _ = optimizer_mapping(engine.roles["model"])
    return [
        deepcopy(
            optimizer._aster_state_loader(p)
            if hasattr(optimizer, "_aster_state_loader")
            else optimizer.state.get(p, {})
        )
        for group in optimizer.param_groups
        for p in group["params"]
    ]


def _equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_tree(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_tree(a, b)
    else:
        assert left == right


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
        local_indices = torch.tensor([0, 2, 4] if rank == 0 else [1, 3])
        for stage in range(4):
            for offload in ("none", "cpu", "nvme"):
                destination = Path(output) / f"zero{stage}_{offload}"
                model = EmbeddingModel()
                dense = deepcopy(model)
                optimizer = torch.optim.AdamW(dense.parameters(), lr=0.02)
                engine = Trainer(
                    model,
                    _objective,
                    parallel=context,
                    zero_stage=stage,
                    lr=0.02,
                    max_grad_norm=None,
                    offload_optimizer=offload,
                    offload_parameters="cpu" if stage == 3 else "none",
                    offload_directory=destination / f"disk_{rank}" if offload == "nvme" else None,
                )
                engine.register_embedding_projection("model", "embedding", max_norm=1.0)

                assert len(engine.roles["model"].parameters) == 1
                assert sum(p.numel() for p in engine.roles["model"].parameters) == (
                    8 if stage == 3 else 15
                )
                optimizer.zero_grad(set_to_none=True)
                _objective(dense, torch.arange(5)).mean.backward()
                optimizer.step()
                engine.step([local_indices])
                before = engine.export_state_dict(only_rank_zero=False)["embedding.weight"]
                moments = _moments(engine)
                F.embedding(torch.tensor([0, 2]), dense.embedding.weight, max_norm=1.0)
                assert engine.project_embedding(
                    "model", "embedding", torch.tensor([0] if rank == 0 else [2])
                ) == {"rows": 2, "changed_rows": 1, "event": 1}
                projected = engine.export_state_dict(only_rank_zero=False)
                torch.testing.assert_close(
                    projected["embedding.weight"], dense.embedding.weight, atol=3e-7, rtol=2e-6
                )
                torch.testing.assert_close(
                    projected["head.weight"], projected["embedding.weight"], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    projected["embedding.weight"][[1, 2, 3, 4]],
                    before[[1, 2, 3, 4]],
                    atol=0,
                    rtol=0,
                )
                _equal_tree(_moments(engine), moments)
                optimizer.zero_grad(set_to_none=True)
                _objective(dense, torch.arange(5)).mean.backward()
                optimizer.step()
                engine.step([local_indices])
                torch.testing.assert_close(
                    engine.export_state_dict(only_rank_zero=False)["embedding.weight"],
                    dense.embedding.weight,
                    atol=3e-7,
                    rtol=2e-6,
                )
                checkpoint = engine.save_checkpoint(destination / "checkpoint")
                selected = torch.tensor([], dtype=torch.long) if rank == 0 else torch.tensor([4])
                event = engine.project_embedding("model", "embedding", selected)
                assert event == {"rows": 1, "changed_rows": 1, "event": 2}
                engine.step([local_indices])
                expected = engine.export_state_dict(only_rank_zero=False)
                engine.load_checkpoint(checkpoint)
                assert engine.project_embedding("model", "embedding", selected) == event
                engine.step([local_indices])
                _equal_tree(engine.export_state_dict(only_rank_zero=False), expected)

                with pytest.raises(ValueError, match="collectively"):
                    engine.project_embedding(
                        "model", "embedding", torch.tensor([-1] if rank == 0 else [1])
                    )
                _equal_tree(engine.export_state_dict(only_rank_zero=False), expected)
                assert engine.project_embedding(
                    "model", "embedding", torch.empty(0, dtype=torch.long)
                ) == {"rows": 0, "changed_rows": 0, "event": 3}

        checkpoint = engine.save_checkpoint(Path(output) / "failure_checkpoint")
        original_copy = engine._embedding_projection._copy_masked

        def failing_copy(*args, **kwargs):
            if rank == 1:
                raise RuntimeError("injected owner write failure")
            return original_copy(*args, **kwargs)

        engine._embedding_projection._copy_masked = failing_copy
        with pytest.raises(ValueError, match="injected owner write failure"):
            engine.project_embedding("model", "embedding", torch.tensor([1]))
        assert engine._failed
        engine._embedding_projection._copy_masked = original_copy
        engine.load_checkpoint(checkpoint)
        assert not engine._failed
        _equal_tree(engine.export_state_dict(only_rank_zero=False), expected)

        for configuration in (float("nan") if rank == 1 else 1.0, float(rank + 1)):
            untouched = EmbeddingModel()
            before = deepcopy(untouched.state_dict())
            with pytest.raises(ValueError):
                Trainer(untouched, DeclaredObjective(configuration), parallel=context, zero_stage=3)
            _equal_tree(untouched.state_dict(), before)
        declared = Trainer(EmbeddingModel(), DeclaredObjective(1.0), parallel=context, zero_stage=3)
        checkpoint = declared.save_checkpoint(Path(output) / "objective_identity")

        declared.objective.temperature = 2.0 if rank == 1 else 1.0
        with pytest.raises(ValueError, match="checkpoint"):
            declared.load_checkpoint(checkpoint)

        tp = ParallelContext(ParallelConfig(tensor_parallel=2))
        engine = Trainer(EmbeddingModel(), parallel=tp)
        with pytest.raises(ValueError, match="DP only"):
            engine.register_embedding_projection("model", "embedding", max_norm=1.0)
    finally:
        dist.destroy_process_group()


def test_dp2_projection_owner_moments_global_rows_resume_all_zero_and_offload(tmp_path):
    temp_root = Path(tempfile.gettempdir())
    if not str(temp_root).isascii():
        temp_root = Path("C:/Temp")
    rendezvous_dir = Path(tempfile.mkdtemp(prefix="aster_projection_", dir=temp_root))
    try:
        mp.spawn(_worker, args=(str(rendezvous_dir / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if rendezvous_dir.parent == temp_root and rendezvous_dir.name.startswith(
            "aster_projection_"
        ):
            shutil.rmtree(rendezvous_dir)
