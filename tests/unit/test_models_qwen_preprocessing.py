from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from aster.core import atomic_json, read_json
from aster.inference.state import StateError
from aster.data.qwen_vl import (
    QwenMediaConfig,
    Qwen3VLProcessor,
    RawImage,
    RawVideo,
    VideoMetadata,
    prepare_image,
    prepare_video,
)
from aster.data.tokenization import ByteTokenizer
from aster.models import Qwen3VLConfig, Qwen3VLTextConfig, build_model, load_model
from aster.methods.qwen_vl import RawQwenObjective
from aster.training import Trainer


def setup(*, backend="torch", cosmos=False):
    torch.set_num_threads(1)
    c = Qwen3VLConfig(
        text_config=Qwen3VLTextConfig(vocab_size=300),
        image_token_id=280,
        video_token_id=281,
        vision_start_token_id=282,
        vision_end_token_id=283,
    )
    if cosmos:
        from aster.models.cosmos3 import Cosmos3Config
        from aster.models.cosmos3_vlm import Cosmos3VLMConfig

        c = Cosmos3VLMConfig(
            mot=Cosmos3Config(vocab_size=300),
            image_token_id=280,
            video_token_id=281,
            vision_start_token_id=282,
            vision_end_token_id=283,
        )
    tokenizer = ByteTokenizer()
    processor = Qwen3VLProcessor(
        QwenMediaConfig(
            patch_size=2,
            image_min_pixels=64,
            image_max_pixels=256,
            video_min_pixels=64,
            video_max_pixels=768,
            image_backend=backend,
            max_sequence_length=512,
        ),
        encode_text=lambda text: tokenizer.encode(text, add_special_tokens=False),
        tokenizer_id=tokenizer.fingerprint,
    )
    return c, tokenizer, processor


def example(video=False, width=11):
    raw = torch.randint(256, (5, 3, 7, width), dtype=torch.uint8)
    medium = RawVideo(raw, VideoMetadata(10.0, 5), False) if video else RawImage(raw[0])
    return [(1,), "Describe ", medium, " answer is blue."]


@pytest.mark.parametrize(
    "raw",
    [
        torch.zeros(3, 8, 8),
        torch.zeros(8, 8, dtype=torch.uint8),
        torch.zeros(4, 8, 8, dtype=torch.uint8),
        "https://example.invalid/image.png",
        torch.zeros(3, 1, 201, dtype=torch.uint8),
    ],
)
def test_models_qwen_raw_image_rejects_bad_or_implicit_input(raw):
    _, _, processor = setup()
    with pytest.raises(ValueError):
        prepare_image(raw, processor.config)


def test_models_qwen_raw_video_explicit_metadata_sampling_and_budgets():
    _, _, p = setup()
    frames = torch.randint(256, (3, 3, 8, 8), dtype=torch.uint8)
    for fps in (None, 0, -1, float("nan"), True):
        with pytest.raises(ValueError, match="FPS"):
            VideoMetadata(fps, 10)
    sampled = RawVideo(frames, VideoMetadata(10.0, 20, (0, 7, 19)), False)
    result = prepare_video(sampled, p.config)
    assert result.frame_indices == (0, 7, 19) and result.timestamps == (0.35, 1.9)
    assert result.grid.tolist() == [[2, 4, 4]] and result.placeholder_count == 8
    with pytest.raises(ValueError, match="resample"):
        prepare_video(replace(sampled, do_sample_frames=True), p.config)
    with pytest.raises(ValueError, match="original frame"):
        prepare_video(RawVideo(frames, VideoMetadata(10.0, 20), False), p.config)
    with pytest.raises(ValueError, match="mutually"):
        prepare_video(replace(sampled, sample_fps=2.0, num_frames=3), p.config)
    with pytest.raises(ValueError, match="pixel budget"):
        prepare_video(sampled, replace(p.config, max_input_pixels=20))
    with pytest.raises(ValueError, match="temporal_patch"):
        prepare_video(RawVideo(frames[:1], VideoMetadata(10.0, 1), False), p.config)


def test_models_qwen_raw_processor_video_exact_timestamp_spans_and_identity_roundtrip(tmp_path):
    c, tokenizer, processor = setup()
    frames = torch.randint(256, (3, 3, 8, 8), dtype=torch.uint8)
    video = RawVideo(frames, VideoMetadata(10.0, 20, (0, 7, 19)), False)
    prepared = processor.prepare([[(1,), "Watch ", video, " answer"]], c)
    first = tokenizer.encode("Watch <0.3 seconds>", add_special_tokens=False)
    second = tokenizer.encode("<1.9 seconds>", add_special_tokens=False)
    expected = (
        [1]
        + first
        + [282]
        + [281] * 4
        + [283]
        + second
        + [282]
        + [281] * 4
        + [283]
        + tokenizer.encode(" answer", add_special_tokens=False)
    )
    assert prepared.model_inputs["input_ids"].tolist() == [expected]
    assert prepared.model_inputs["mm_token_type_ids"].eq(2).sum() == 8
    assert prepared.labels[prepared.model_inputs["mm_token_type_ids"].ne(0)].eq(-100).all()

    positions = prepared.position_ids
    for index in torch.nonzero(prepared.model_inputs["input_ids"][0] == 282).flatten():
        assert len(positions[0, 0, index + 1 : index + 5].unique()) == 1
        assert (
            positions[1, 0, index + 1 : index + 5].max()
            - positions[1, 0, index + 1 : index + 5].min()
            == 1
        )
        assert (
            positions[2, 0, index + 1 : index + 5].max()
            - positions[2, 0, index + 1 : index + 5].min()
            == 1
        )
    processor.save_pretrained(tmp_path / "processor")
    restored = Qwen3VLProcessor.from_pretrained(
        tmp_path / "processor",
        encode_text=processor.encode_text,
        tokenizer_id=processor.tokenizer_id,
    )
    assert restored.fingerprint == processor.fingerprint
    assert (
        restored.prepare([[(1,), "Watch ", video, " answer"]], c).media_fingerprint
        == prepared.media_fingerprint
    )
    with pytest.raises(ValueError, match="tokenizer identity"):
        Qwen3VLProcessor.from_pretrained(
            tmp_path / "processor", encode_text=processor.encode_text, tokenizer_id="foreign"
        )
    data = read_json(tmp_path / "processor" / "qwen_processor.json")
    data["media"]["video_cap_pixels_per_frame"] = True
    atomic_json(tmp_path / "processor" / "qwen_processor.json", data)
    with pytest.raises(ValueError, match="fingerprint"):
        Qwen3VLProcessor.from_pretrained(
            tmp_path / "processor",
            encode_text=processor.encode_text,
            tokenizer_id=processor.tokenizer_id,
        )
    assert (
        replace(processor.config, image_backend="pil").fingerprint != processor.config.fingerprint
    )


def test_models_qwen_raw_patch_contract_and_token_injection_rejected():
    c, _, p = setup()
    with pytest.raises(ValueError, match="patch/temporal"):
        p.prepare([example()], replace(c, vision_config=replace(c.vision_config, patch_size=4)))
    with pytest.raises(ValueError, match="injected"):
        p.prepare([[(282, 280, 283)]], c)
    with pytest.raises(ValueError, match="visual_prefill"):
        RawQwenObjective(p, visual_prefill="video").preflight_microbatches(
            build_model(c), [{"examples": [example()]}]
        )
    with pytest.raises(ValueError, match="generation_fields"):
        RawQwenObjective(p).preflight_microbatches(
            build_model(c),
            [{"examples": [example()], "model_inputs": {"input_ids": torch.ones(1, 1)}}],
        )


@pytest.mark.parametrize("stage,video", [(0, False), (3, True)])
def test_models_qwen_raw_training_safe_export_exact_resume_and_real_runner(tmp_path, stage, video):
    from aster.inference.task_runners import StatefulTokenRunner

    torch.manual_seed(708)
    c, _, p = setup()
    raw = {"examples": [example(video)]}
    model = build_model(c)
    objective = RawQwenObjective(p, visual_prefill="video" if video else "image")
    trainer = Trainer(model, objective, zero_stage=stage, lr=0.003)
    first = trainer.step([raw]).loss
    for _ in range(11):
        last = trainer.step([raw]).loss
    assert last < first * 0.8
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([raw])
    expected_weights = deepcopy(trainer.export_state_dict())
    trainer.load_checkpoint(checkpoint, trusted=True)
    actual = trainer.step([raw])
    assert actual.loss == expected.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
    dense = build_model(c)
    dense.load_state_dict(trainer.export_state_dict())
    dense.save_pretrained(tmp_path / "model")
    restored = load_model(tmp_path / "model")
    prepared = p.prepare(raw["examples"], c)
    runner = StatefulTokenRunner(
        restored, policy_artifact_id="sha256-local", processor_id=p.fingerprint
    )
    inputs = dict(prepared.model_inputs)
    ids = inputs.pop("input_ids")
    prefix = runner.forward(ids, modality_inputs=inputs)
    reference = restored(**prepared.model_inputs)
    torch.testing.assert_close(prefix.logits, reference.logits, atol=0, rtol=0)
    fork = runner.fork(prefix.state)
    token = torch.tensor([[4]])
    cached = runner.forward(token, state=fork)
    replay_inputs = dict(
        inputs,
        attention_mask=torch.cat(
            (inputs["attention_mask"], torch.ones(1, 1, dtype=torch.bool)), -1
        ),
        mm_token_type_ids=torch.cat(
            (inputs["mm_token_type_ids"], torch.zeros(1, 1, dtype=torch.long)), -1
        ),
    )
    replay = runner.replay(torch.cat((ids, token), -1), modality_inputs=replay_inputs)
    torch.testing.assert_close(cached.logits, replay.logits[:, -1:], atol=3e-6, rtol=4e-5)
    foreign = replace(prefix.state, processor_id="another-resize-backend")
    with pytest.raises(StateError, match="processor"):
        runner.forward(token, state=foreign)


def test_models_qwen_raw_unequal_microbatch_normalization_and_saved_objective_identity(tmp_path):
    torch.manual_seed(711)
    c, _, p = setup()
    raw = [example(width=8), example(width=12)]
    raw[1].append(" A longer supervised suffix.")
    initial = build_model(c).state_dict()

    def engine(processor, accumulation_steps=1):
        model = build_model(c)
        model.load_state_dict(initial)
        return Trainer(
            model,
            RawQwenObjective(processor),
            max_grad_norm=None,
            accumulation_steps=accumulation_steps,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.0003),
        )

    full, accumulated = engine(p), engine(p, 2)
    expected = full.step([{"examples": raw}])
    actual = accumulated.step([{"examples": [row]} for row in raw])
    assert abs(expected.loss - actual.loss) < 3e-6
    for name, value in full.export_state_dict().items():
        torch.testing.assert_close(
            value, accumulated.export_state_dict()[name], atol=4e-7, rtol=4e-5, msg=name
        )
    checkpoint = accumulated.save_checkpoint(tmp_path / "resume")
    foreign_processor = Qwen3VLProcessor(
        replace(p.config, image_backend="pil"),
        encode_text=p.encode_text,
        tokenizer_id=p.tokenizer_id,
    )
    with pytest.raises(ValueError, match="identity|配置|configuration|manifest|schema|不同"):
        engine(foreign_processor, 2).load_checkpoint(checkpoint, trusted=True)


def test_models_qwen_raw_cosmos_shared_visual_flow_train_and_video_codec():
    from aster.models.wan22_vae import Wan22VAEConfig, Wan22VideoVAE
    from aster.methods.cosmos3 import Cosmos3VisualFlowObjective, Cosmos3VideoPipeline

    torch.manual_seed(709)
    c, _, p = setup(cosmos=True)
    model = build_model(c)
    vae = Wan22VideoVAE(Wan22VAEConfig())
    pipeline = Cosmos3VideoPipeline(model, vae)
    examples = [example()]
    inputs = p.prepare(examples, c).model_inputs
    assert "mm_token_type_ids" not in inputs
    video = torch.randn(1, 3, 5, 16, 16).tanh()
    noise = torch.randn(1, 2, 2, 1, 1)
    prepared = pipeline.training_batch(
        video,
        inputs,
        timesteps=torch.full((1, 2), 700.0),
        noisy_frames=torch.tensor([[False, True]]),
        noise={"vision": noise},
    )
    raw = dict(
        examples=examples,
        model_inputs={"vision": prepared["model_inputs"]["vision"]},
        noise={"vision": noise},
    )
    objective = RawQwenObjective(
        p,
        objective=Cosmos3VisualFlowObjective(text_weight=0.2, time_distribution="provided"),
        generation_fields=("vision",),
    )
    engine = Trainer(model, objective, lr=0.002)
    initial = engine.step([raw]).loss
    for _ in range(7):
        final = engine.step([raw]).loss
    assert final < initial * 0.75 and all(parameter.grad is None for parameter in vae.parameters())
    generated = pipeline.generate(
        inputs, noise, condition_video=video[:, :, :1], steps=2, solver="euler"
    )
    assert generated.video.shape == video.shape and torch.isfinite(generated.video).all()
    torch.testing.assert_close(
        generated.latents["vision"][:, :, :1],
        pipeline.encode_video(video[:, :, :1]),
        atol=0,
        rtol=0,
    )
