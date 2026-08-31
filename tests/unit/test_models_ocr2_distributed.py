from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aster.models import OCR2Config, build_model
from aster.methods import CrossEntropyObjective
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
        torch.manual_seed(538)
        c = OCR2Config()
        initial = build_model(c).state_dict()
        counts = [
            c.vision_config.global_queries + 1,
            c.vision_config.global_queries + 1 + 2 * c.vision_config.local_queries,
        ]
        sequences = [
            torch.tensor([1] + [c.image_token_id] * count + [3, 5, 7, 2]) for count in counts
        ]
        ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        valid = (
            torch.arange(ids.shape[1])[None] < torch.tensor([len(x) for x in sequences])[:, None]
        )
        inputs = dict(
            input_ids=ids,
            attention_mask=valid,
            images_seq_mask=ids.eq(c.image_token_id),
            pixel_values=torch.randn(2, 3, 32, 32),
            pixel_values_local=(None, torch.randn(2, 3, 24, 24)),
            images_spatial_crop=torch.tensor([[1, 1], [2, 1]]),
        )
        labels = ids.masked_fill(~valid | ids.eq(c.image_token_id), -100)
        complete = dict(model_inputs=inputs, labels=labels)
        local_inputs = {
            name: (value[rank : rank + 1] if isinstance(value, torch.Tensor) else (value[rank],))
            for name, value in inputs.items()
        }
        local = dict(model_inputs=local_inputs, labels=labels[rank : rank + 1])
        objective = CrossEntropyObjective(auxiliary_weights={"router_aux": 0.001})
        for stage in range(4):
            dense = build_model(c)
            dense.load_state_dict(initial)
            native = build_model(c)
            native.load_state_dict(initial)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.001, momentum=0.9)
            engine = Trainer(
                native,
                objective,
                parallel=ParallelContext(),
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda parameters: torch.optim.SGD(
                    parameters, lr=0.001, momentum=0.9
                ),
            )
            reference = objective(dense, complete)
            reference_loss = sum(term.weight * term.mean for term in reference.terms)
            reference_loss.backward()
            optimizer.step()
            result = engine.step([local])
            assert result.updated and abs(result.loss - reference_loss.item()) < 2e-6
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=3e-7, rtol=4e-5, msg=name
                )
            checkpoint = engine.save_checkpoint(Path(output) / f"ocr_zero{stage}")
            expected = engine.step([local])
            weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
            engine.load_checkpoint(checkpoint, trusted=True)
            actual = engine.step([local])
            assert expected.loss == actual.loss
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_models_ocr2_real_dp2_variable_crops_all_zero_and_resume(tmp_path):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_ocr2_", dir=root)).resolve()
    try:
        mp.spawn(_worker, args=(str(directory / "store"), str(tmp_path)), nprocs=2, join=True)
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_ocr2_"):
            shutil.rmtree(directory)
