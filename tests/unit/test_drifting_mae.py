from copy import deepcopy
import pytest
import torch
import torch.nn.functional as F

from aster.models.drifting_features import MAEResNetConfig, MAEResNet, patch_input
from aster.methods.masked_autoencoding import MaskedAutoencodingObjective, sample_patch_mask
from aster.methods.drifting import SpatialFeatureStatistics
from aster.training import Trainer


def _config():
    return MAEResNetConfig(
        in_channels=2, num_classes=3, base_channels=32, layers=(1, 1, 1, 1), patch_size=2
    )


def test_patch_and_bernoulli_mask_follow_author_layout():
    x = torch.arange(2 * 4 * 4.0).reshape(1, 2, 4, 4)
    expected = x[0, :, :2, :2].permute(1, 2, 0).flatten()
    torch.testing.assert_close(patch_input(x, 2)[0, :, 0, 0], expected, atol=0, rtol=0)
    generator, oracle = torch.Generator().manual_seed(51), torch.Generator().manual_seed(51)
    ratio = 0.2 + 0.6 * torch.rand(1, generator=oracle)
    mask = (
        (torch.rand(1, 1, 2, 2, generator=oracle) < ratio[:, None, None, None])
        .repeat_interleave(2, -2)
        .repeat_interleave(2, -1)
    )
    torch.testing.assert_close(
        sample_patch_mask(x, patch_size=2, minimum=0.2, maximum=0.8, generator=generator), mask
    )


@pytest.mark.parametrize("stage", [0, 3])
def test_mae_train_exact_resume_and_frozen_drifting_features(stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(684)
    objective = MaskedAutoencodingObjective(classification_weight=0.2, mask_min=0.5, mask_max=0.9)
    model = MAEResNet(_config())
    engine = Trainer(model, objective, lr=0.0001, zero_stage=stage)
    batch = dict(samples=torch.randn(2, 2, 8, 8), labels=torch.tensor([0, 2]))
    assert engine.step([batch]).updated
    path = engine.save_checkpoint(tmp_path / "checkpoint")
    expected = engine.step([batch])
    weights = deepcopy(engine.export_state_dict())
    engine.load_checkpoint(path, trusted=True)
    actual = engine.step([batch])
    assert actual.loss == expected.loss
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    frozen = MAEResNet(_config())
    frozen.load_state_dict(weights)
    extractor = (
        SpatialFeatureStatistics(frozen.encoder, patch_sizes=(2, 4)).eval().requires_grad_(False)
    )
    samples = batch["samples"].clone().requires_grad_()
    features = extractor(samples)
    assert features["layer4"].shape == (2, 1, 256) and "global" in features
    sum(value.square().mean() for value in features.values()).backward()
    assert samples.grad.abs().sum() > 0 and all(p.grad is None for p in extractor.parameters())


def test_reconstruction_loss_sums_channels_and_preserves_zero_mask():
    torch.set_num_threads(1)
    torch.manual_seed(52)
    model = MAEResNet(_config())
    samples = torch.randn(2, 2, 8, 8)
    mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    mask[0, :, :4] = True
    output = model(samples, mask)
    objective = MaskedAutoencodingObjective(classification_weight=0.3)
    terms = objective(model, dict(samples=samples, mask=mask, labels=torch.tensor([1, 2]))).terms
    expected = (output.reconstruction[0, :, :4] - samples[0, :, :4]).square().sum() / 32
    torch.testing.assert_close(terms[0].numerator, expected)
    torch.testing.assert_close(
        terms[1].numerator, F.cross_entropy(output.logits, torch.tensor([1, 2]), reduction="sum")
    )
    assert terms[0].denominator.dtype == torch.int64 and terms[0].weight == 0.7
