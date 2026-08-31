from dataclasses import asdict
import copy
import pytest
import torch

from aster.models import ACTConfig, DiTConfig, LlavaConfig, build_model
from aster.training import Trainer
from aster.methods.distillation import DistillationObjective
from aster.methods.multimodal_distillation import PredictionAlignment, MultimodalDistillationMethod


@pytest.mark.parametrize("domain", ["action", "field"])
def test_real_continuous_kd_alignment_training_and_restore(tmp_path, domain):
    torch.manual_seed(55)
    torch.set_num_threads(1)
    if domain == "action":
        config = ACTConfig(
            proprio_dim=3,
            action_dim=2,
            vision_dim=8,
            hidden_size=16,
            latent_dim=4,
            horizon=3,
            num_heads=2,
            posterior_layers=1,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_size=24,
        )
        inputs = {"proprio": torch.randn(2, 3), "vision_tokens": torch.randn(2, 4, 8)}
        alignment = PredictionAlignment(
            "action",
            "units-frame-rate-horizon-and-normalizer-sha",
            "encoder-a",
            "encoder-a",
            "teacher-checkpoint-sha",
        )
    else:
        config = DiTConfig(
            in_channels=2,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            patch_size=2,
            prediction_type="velocity",
        )
        inputs = {"sample": torch.randn(2, 2, 4, 4), "time": torch.rand(2)}
        alignment = PredictionAlignment(
            "field",
            "same-latent-encoder-scale-shift",
            "vae-a",
            "vae-a",
            "teacher-field-sha",
            field_parameterization="velocity",
        )
    model, teacher = build_model(config), build_model(config)
    if domain == "field":
        torch.nn.init.normal_(teacher.output.weight, std=0.1)
    before, teacher_before = copy.deepcopy(model.state_dict()), copy.deepcopy(teacher.state_dict())
    engine = Trainer(model, lr=0.002)
    method = MultimodalDistillationMethod(engine, teacher, alignment)
    batch = {"student_inputs": inputs, "teacher_inputs": inputs, "alignment": asdict(alignment)}
    assert method.update([batch]).updated
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    for key, value in teacher_before.items():
        torch.testing.assert_close(value, teacher.state_dict()[key], rtol=0, atol=0)
    engine.save_checkpoint(tmp_path / "checkpoint")
    method.update([batch])
    expected = copy.deepcopy(model.state_dict())
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    method.update([batch])
    for key, value in expected.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="provenance"):
        method.objective(
            model,
            {
                **batch,
                "alignment": {**batch["alignment"], "output_space_fingerprint": "different-units"},
            },
        )
    if domain == "field":
        with pytest.raises(ValueError, match="same sample"):
            method.objective(
                model, {**batch, "teacher_inputs": {**inputs, "time": inputs["time"] + 0.1}}
            )


def test_vlm_logit_and_hidden_kd_updates_real_visual_and_language_paths():
    torch.manual_seed(56)
    torch.set_num_threads(1)
    student, teacher = build_model(LlavaConfig()), build_model(LlavaConfig())
    teacher_before = copy.deepcopy(teacher.state_dict())
    objective = DistillationObjective(
        teacher,
        tokenizer_fingerprints=("paired-vocab", "paired-vocab"),
        kd_weight=1.0,
        feature_weight=0.1,
        layer_pairs=((0, 0), (1, 1)),
    )
    engine = Trainer(student, objective, lr=0.001)
    engine.add_role("teacher", teacher, trainable=False)
    tokens = torch.tensor([[1] + [31] * 16 + [2, 4], [1] + [31] * 16 + [3, 5]])
    labels = tokens.clone()
    labels[:, :17] = -100
    pixels = torch.randn(2, 3, 16, 16)
    vision_before = student.model.vision_tower.embeddings.patch_embedding.weight.detach().clone()
    assert engine.step([{"input_ids": tokens, "pixel_values": pixels, "labels": labels}]).updated
    assert not torch.equal(
        vision_before, student.model.vision_tower.embeddings.patch_embedding.weight
    )
    for key, value in teacher_before.items():
        torch.testing.assert_close(value, teacher.state_dict()[key], rtol=0, atol=0)
