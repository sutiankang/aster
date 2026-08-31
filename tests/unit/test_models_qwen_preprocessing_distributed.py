from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from aster.data.qwen_vl import QwenMediaConfig, Qwen3VLProcessor, RawImage, RawVideo, VideoMetadata
from aster.data.tokenization import ByteTokenizer
from aster.methods.qwen_vl import RawQwenObjective
from aster.models import Qwen3VLConfig, Qwen3VLTextConfig, build_model
from aster.training import ParallelContext, Trainer


def _worker(rank, rendezvous, output, video):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=100),
    )
    try:
        torch.manual_seed(710)
        c = Qwen3VLConfig(
            text_config=Qwen3VLTextConfig(vocab_size=300),
            image_token_id=280,
            video_token_id=281,
            vision_start_token_id=282,
            vision_end_token_id=283,
        )
        initial = build_model(c).state_dict()
        tokenizer = ByteTokenizer()
        processor = Qwen3VLProcessor(
            QwenMediaConfig(
                patch_size=2,
                image_min_pixels=64,
                image_max_pixels=256,
                video_min_pixels=64,
                video_max_pixels=768,
                max_sequence_length=512,
            ),
            encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
            tokenizer_id=tokenizer.fingerprint,
        )
        examples = []
        for count, width, answer in ((3, 8, " short."), (5, 12, " this answer has more tokens.")):
            frames = torch.randint(256, (count, 3, 8, width), dtype=torch.uint8)
            media = (
                RawVideo(frames, VideoMetadata(10.0, count), False)
                if video
                else RawImage(frames[0])
            )
            examples.append([(1,), "Inspect ", media, answer])
        objective = RawQwenObjective(processor, visual_prefill="video" if video else "image")
        for stage in range(4):
            dense, model = build_model(c), build_model(c)
            dense.load_state_dict(initial)
            model.load_state_dict(initial)
            optimizer = torch.optim.SGD(dense.parameters(), lr=0.0003, momentum=0.9)
            engine = Trainer(
                model,
                objective,
                parallel=ParallelContext(),
                zero_stage=stage,
                max_grad_norm=None,
                optimizer_factory=lambda parameters: torch.optim.SGD(
                    parameters, lr=0.0003, momentum=0.9
                ),
            )
            loss = objective(dense, {"examples": examples}).mean
            loss.backward()
            optimizer.step()
            result = engine.step([{"examples": [examples[rank]]}])
            assert result.updated and abs(result.loss - loss.item()) < 3e-6
            for name, value in engine.export_state_dict(only_rank_zero=False).items():
                torch.testing.assert_close(
                    value, dense.state_dict()[name], atol=5e-7, rtol=5e-5, msg=name
                )
            if stage == 3:
                checkpoint = engine.save_checkpoint(Path(output) / "raw_qwen_zero3")
                expected = engine.step([{"examples": [examples[rank]]}])
                weights = deepcopy(engine.export_state_dict(only_rank_zero=False))
                engine.load_checkpoint(checkpoint, trusted=True)
                actual = engine.step([{"examples": [examples[rank]]}])
                assert actual.loss == expected.loss
                for name, value in engine.export_state_dict(only_rank_zero=False).items():
                    torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
        forwards = []
        handle = engine.model.register_forward_pre_hook(lambda *args: forwards.append(True))
        try:
            bad = list(examples[rank])
            if rank == 1:
                if video:
                    bad[2] = RawVideo(torch.zeros(3, 3, 8, 8), VideoMetadata(10.0, 3), False)
                else:
                    bad[2] = RawImage(torch.zeros(3, 8, 8))
            steps = engine.steps
            with pytest.raises(ValueError, match="uint8"):
                engine.step([{"examples": [bad]}])
            assert not forwards and engine.steps == steps

            mismatch = [(1,), "Pure text."] if rank else examples[0]
            with pytest.raises(ValueError, match="visual_prefill"):
                engine.step([{"examples": [mismatch]}])
            assert not forwards and engine.steps == steps
        finally:
            handle.remove()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("video", [False, True])
def test_models_qwen_raw_real_dp2_unequal_media_tokens_all_zero_resume_and_symmetric_preflight(
    tmp_path, video
):
    root = Path(tempfile.gettempdir())
    if not str(root).isascii():
        root = Path("C:/Temp")
    directory = Path(tempfile.mkdtemp(prefix="aster_rawqwen_", dir=root)).resolve()
    try:
        mp.spawn(
            _worker, args=(str(directory / "store"), str(tmp_path), video), nprocs=2, join=True
        )
    finally:
        if directory.parent == root.resolve() and directory.name.startswith("aster_rawqwen_"):
            shutil.rmtree(directory)
