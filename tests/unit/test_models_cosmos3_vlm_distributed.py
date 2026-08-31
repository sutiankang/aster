from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.models.cosmos3_vlm import Cosmos3VLM, Cosmos3VLMConfig
from aster.models.cosmos3 import Cosmos3Vision, cosmos3_positions
from aster.models.qwen_vl import pack_qwen_pixels
from aster.methods.cosmos3 import Cosmos3VisualFlowObjective
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
        torch.manual_seed(569)
        c = Cosmos3VLMConfig()
        initial = Cosmos3VLM(c).state_dict()
        media = [
            pack_qwen_pixels(torch.randn(1, 3, 8, width), c.vision_config) for width in (8, 12)
        ]
        rows = [torch.tensor([[1, 26] + [28] * count + [27, 3, 2]]) for count in (4, 6)]
        ids = torch.cat((F.pad(rows[0], (0, 2)), rows[1]))
        mask = ids.ne(0)
        vision = Cosmos3Vision(
            torch.randn(2, 2, 2, 2, 2),
            cosmos3_positions((2, 1, 1), batch_size=2),
            torch.full((2, 2), 650.0),
            torch.tensor([[False, True], [True, True]]),
        )
        noise = torch.randn_like(vision.sample)
        full = dict(
            model_inputs=dict(
                input_ids=ids,
                attention_mask=mask,
                pixel_values=torch.cat([x[0] for x in media]),
                image_grid_thw=torch.cat([x[1] for x in media]),
                vision=vision,
            ),
            labels=ids.masked_fill(~mask, -100),
            noise={"vision": noise},
        )
        local = dict(
            model_inputs=dict(
                input_ids=rows[rank],
                pixel_values=media[rank][0],
                image_grid_thw=media[rank][1],
                vision=replace(
                    vision,
                    sample=vision.sample[rank : rank + 1],
                    positions=vision.positions[:, rank : rank + 1],
                    timesteps=vision.timesteps[rank : rank + 1],
                    noisy_frames=vision.noisy_frames[rank : rank + 1],
                ),
            ),
            labels=rows[rank],
            noise={"vision": noise[rank : rank + 1]},
        )
        objective = Cosmos3VisualFlowObjective(text_weight=0.2, time_distribution="provided")
        for stage in range(4):
            dense, native = Cosmos3VLM(c), Cosmos3VLM(c)
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
            loss = sum(term.weight * term.mean for term in objective(dense, full).terms)
            loss.backward()
            optimizer.step()
            result = engine.step([local])
            assert result.updated and abs(result.loss - loss.item()) < 3e-5
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=5e-7, rtol=5e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"cosmos3_vlm_zero{stage}")
            expected = engine.step([local])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([local])
            assert actual.loss == expected.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
        bad = deepcopy(local)
        if rank:
            bad["model_inputs"].pop("pixel_values")
            bad["model_inputs"].pop("image_grid_thw")
        before = engine.steps
        try:
            engine.step([bad])
            raise AssertionError("visual/pure-text rank mismatch must fail before leaf gathers")
        except ValueError as error:
            assert "visual_prefill" in str(error)
        assert engine.steps == before
    finally:
        dist.destroy_process_group()


def test_models_cosmos3_visual_real_dp2_grid_padding_all_zero_resume_and_media_preflight(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_cosmos3vlm_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_cosmos3vlm_"):
            shutil.rmtree(directory)
