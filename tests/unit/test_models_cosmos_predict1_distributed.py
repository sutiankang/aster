from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.models import (
    CosmosPredict1Config,
    CosmosPredict1Condition,
    CosmosPredict1ModelConfig,
    build_model,
)
from aster.methods.cosmos_predict1 import CosmosPredict1Objective
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
        torch.manual_seed(554)
        config = CosmosPredict1ModelConfig(
            net=CosmosPredict1Config(extra_per_block_abs_pos_emb=True)
        )
        initial = build_model(config).state_dict()
        batches = []
        for index in range(2):
            b, t, w = index + 1, index + 1, 4 + index * 2
            sample = torch.randn(b, 2, t, 4, w)
            batches.append(
                dict(
                    sample=sample,
                    noise=torch.randn_like(sample),
                    sigma=torch.linspace(0.3, 0.8, b),
                    condition=CosmosPredict1Condition(
                        torch.randn(b, index + 2, config.net.crossattn_emb_channels),
                        None if index == 0 else torch.full((b,), 24.0),
                        torch.zeros(b, 2, 3),
                    ),
                )
            )
        objective = CosmosPredict1Objective(loss_scale=0.01)
        for stage in range(4):
            dense, native = build_model(config), build_model(config)
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
            first, second = objective(dense, batches[0]), objective(dense, batches[1])
            loss = (first.numerator + second.numerator) / (first.denominator + second.denominator)
            loss.backward()
            optimizer.step()
            result = engine.step([batches[rank]])
            assert result.updated and abs(result.loss - loss.item()) < 2e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=3e-7, rtol=4e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"predict1_zero{stage}")
            expected = engine.step([batches[rank]])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([batches[rank]])
            assert expected.loss == actual.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_models_cosmos_predict1_real_dp2_ragged_shapes_all_zero_global_count_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_predict1_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_predict1_"):
            shutil.rmtree(directory)
