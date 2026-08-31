from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models.genie import (
    GenieTokenizerConfig,
    GenieTokenizer,
    GenieActionConfig,
    GenieDynamicsConfig,
    GenieWorldConfig,
    GenieWorld,
)
from aster.methods.genie import GenieVQObjective, GenieWorldObjective
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
        torch.manual_seed(874)
        common = dict(
            image_height=8,
            image_width=8,
            image_channels=1,
            hidden_size=8,
            num_heads=2,
            head_dim=4,
            encoder_layers=1,
            decoder_hidden_size=8,
            decoder_num_heads=2,
            decoder_head_dim=4,
            decoder_layers=1,
            latent_dim=3,
            max_frames=4,
            intermediate_ratio=2,
        )
        tc = GenieTokenizerConfig(**common, patch_size=4, num_codes=5)
        ac = GenieActionConfig(**common, patch_size=8, num_codes=3)
        dc = GenieDynamicsConfig(
            spatial_tokens=4,
            vocab_size=5,
            action_dim=3,
            hidden_size=8,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            intermediate_ratio=2,
            max_frames=4,
        )
        wc = GenieWorldConfig(ac, dc)
        video = torch.rand(3, 4, 1, 8, 8)
        valid = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        mask = valid[:, :, None].expand(3, 4, 4).clone()
        mask[:, 0] = False
        selection = slice(0, 1) if rank == 0 else slice(1, 3)
        for constructor, config, objective, batch, name in (
            (
                GenieTokenizer,
                tc,
                GenieVQObjective(sequence_length=4),
                dict(video=video, valid=valid),
                "tokenizer",
            ),
            (
                GenieWorld,
                wc,
                GenieWorldObjective(sequence_length=4),
                dict(video=video, valid=valid, mask=mask, tokens=torch.randint(5, (3, 4, 4))),
                "world",
            ),
        ):
            initial = constructor(config).state_dict()
            local = {key: value[selection] for key, value in batch.items()}
            for stage in range(4):
                model, dense = constructor(config), constructor(config)
                model.load_state_dict(initial)
                dense.load_state_dict(initial)
                optimizer = torch.optim.SGD(dense.parameters(), lr=0.002, momentum=0.9)
                engine = Trainer(
                    model,
                    objective,
                    parallel=context,
                    zero_stage=stage,
                    max_grad_norm=None,
                    optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.002, momentum=0.9),
                )
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = sum(term.mean * term.weight for term in objective(dense, batch).terms)
                    loss.backward()
                    norm = (
                        sum(
                            p.grad.double().square().sum()
                            for p in dense.parameters()
                            if p.grad is not None
                        )
                        .sqrt()
                        .item()
                    )
                    optimizer.step()
                    result = engine.step([local])
                    assert result.updated and abs(result.loss - loss.item()) < 3e-6
                    assert abs(result.grad_norm - norm) < 2e-5
                    for key, value in engine.export_state_dict(only_rank_zero=False).items():
                        torch.testing.assert_close(
                            value, dense.state_dict()[key], atol=7e-7, rtol=5e-5
                        )
                checkpoint = engine.save_checkpoint(Path(output) / f"{name}_zero{stage}")
                expected = engine.step([local])
                weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
                engine.load_checkpoint(checkpoint, trusted=True)
                actual = engine.step([local])
                assert actual.loss == expected.loss
                for key, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_genie_true_dp2_tokenizer_and_world_zero0_to3_unequal_counts(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_genie_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_genie_"):
            shutil.rmtree(directory)
