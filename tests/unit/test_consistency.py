from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from aster.core import FieldOutput, ArtifactStore, atomic_json
from aster.models.generative import UNet2D, UNetConfig
from aster.methods.consistency import (
    ConsistencyConfig,
    ConsistencyMethod,
    consistency_denoise,
    consistency_metric,
    sample_consistency,
    _ConsistencyObjective,
)
from aster.training import Trainer


def _config(mode="ict", **kwargs):
    return ConsistencyConfig(
        mode=mode,
        total_steps=20,
        initial_scales=4,
        final_scales=4,
        curriculum="fixed",
        target_ema_mode="fixed",
        sampling_ema=0.8,
        **kwargs,
    )


def _model(*, teacher=False, dropout=0.0):
    return UNet2D(
        UNetConfig(
            in_channels=1,
            model_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            num_heads=1,
            attention_levels=(),
            dropout=dropout,
            prediction_type="edm_residual" if teacher else "consistency_residual",
        )
    )


def _batch(count=2):
    generator = torch.Generator().manual_seed(461 + count)
    return {
        "sample": torch.randn(count, 1, 4, 4, generator=generator),
        "noise": torch.randn(count, 1, 4, 4, generator=generator),
        "interval_indices": torch.arange(count) % 3,
    }


class Probe(nn.Module):
    def __init__(self, kind="consistency_residual"):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(0.2))
        self.kind = kind
        self.calls = []

    def forward(self, sample, time, condition=None):
        self.calls.append(time.detach().clone())
        return FieldOutput(self.coefficient * sample + time[:, None] * 0.001, self.kind)


def test_consistency_curricula_logprob_boundary_metric_and_sampler():
    ict = ConsistencyConfig(total_steps=80)
    assert [ict.scales_and_ema(step)[0] for step in (0, 9, 10, 20, 70, 80)] == [
        11,
        11,
        21,
        41,
        1281,
        1281,
    ]
    ct = ConsistencyConfig(mode="ct", total_steps=100)
    assert ct.scales_and_ema(0) == pytest.approx((2, 0.95**2))
    levels = ict.levels(0)
    cdf = 0.5 * (1 + torch.erf((levels.log() + 1.1) / (math.sqrt(2) * 2.0)))
    torch.testing.assert_close(
        ict.interval_probabilities(0), (cdf[1:] - cdf[:-1]) / (cdf[-1] - cdf[0])
    )
    model = Probe()
    sample = torch.randn(3, 4)
    sigma = torch.full((3,), 0.002)
    torch.testing.assert_close(consistency_denoise(model, sample, sigma), sample, rtol=0, atol=0)
    torch.testing.assert_close(model.calls[-1], sigma.log() * 250)
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    value = consistency_metric(x, torch.zeros_like(x), "pseudo_huber")
    c = 0.00054 * math.sqrt(3)
    expected = (x.square().sum(-1) + c * c).sqrt() - c
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), x)[0],
        x.detach() / (x.detach().square().sum() + c * c).sqrt(),
    )
    model.calls.clear()
    model.train()
    noise = torch.randn(3, 4)
    generator = torch.Generator().manual_seed(19)
    actual = sample_consistency(model, noise, [5.0, 0.4], generator=generator, clip_denoised=False)
    assert len(model.calls) == 2 and model.training
    first = consistency_denoise(model, noise * 5, noise.new_full((3,), 5.0))
    perturbation = torch.randn(noise.shape, generator=torch.Generator().manual_seed(19))
    second = consistency_denoise(
        model, first + math.sqrt(0.4**2 - 0.002**2) * perturbation, noise.new_full((3,), 0.4)
    )
    torch.testing.assert_close(actual, second)
    torch.testing.assert_close(sample_consistency(model, noise, [5.0], clip_denoised=False), first)


@pytest.mark.parametrize("mode", ["ct", "cd", "ict"])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_actual_unet_consistency_lifecycle_radam_resume_and_export(mode, stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(263)
    model = _model(dropout=0.2)
    teacher = _model(teacher=True) if mode == "cd" else None
    initial_teacher = deepcopy(teacher.state_dict()) if teacher is not None else None
    engine = Trainer(
        model,
        zero_stage=stage,
        optimizer_factory=lambda parameters: torch.optim.RAdam(parameters, lr=0.003),
        max_grad_norm=1.0,
    )
    method = ConsistencyMethod(
        engine, config=_config(mode), target_factory=lambda: _model(dropout=0.2), teacher=teacher
    )
    initial = engine.export_state_dict()
    target_initial = deepcopy(method.target.state_dict())

    for _ in range(7):
        assert method.update([_batch()]).updated
    current = engine.export_state_dict()
    assert any(not torch.equal(value, initial[key]) for key, value in current.items())
    assert all(not p.requires_grad and p.grad is None for p in method.target.parameters())
    assert any(
        not torch.equal(value, target_initial[key])
        for key, value in method.target.state_dict().items()
    )
    if mode in {"cd", "ict"}:
        for key, value in current.items():
            torch.testing.assert_close(method.target.state_dict()[key], value, rtol=0, atol=0)
    if teacher is not None:
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, initial_teacher[key], rtol=0, atol=0)
    path = engine.save_checkpoint(tmp_path / "native")
    stochastic = {"sample": _batch()["sample"]}
    expected_result = method.update([stochastic])
    expected = engine.export_state_dict()
    expected_target = deepcopy(method.target.state_dict())
    expected_rng = method.generator.get_state()
    engine.load_checkpoint(path)
    actual_result = method.update([stochastic])
    assert actual_result == expected_result and method.updates == 8
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    for key, value in method.target.state_dict().items():
        torch.testing.assert_close(value, expected_target[key], rtol=0, atol=0)
    assert torch.equal(method.generator.get_state(), expected_rng)
    if mode == "cd" and stage == 3:
        fresh_teacher = _model(teacher=True)
        fresh_teacher.load_state_dict(initial_teacher)
        fresh_engine = Trainer(
            _model(dropout=0.2),
            zero_stage=stage,
            optimizer_factory=lambda parameters: torch.optim.RAdam(parameters, lr=0.003),
            max_grad_norm=1.0,
        )
        fresh_method = ConsistencyMethod(
            fresh_engine,
            config=_config(mode),
            target_factory=lambda: _model(dropout=0.2),
            teacher=fresh_teacher,
        )
        fresh_engine.load_checkpoint(path)
        assert fresh_method.update([stochastic]) == expected_result
        for key, value in fresh_engine.export_state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    deploy = _model(dropout=0.2)
    deploy.load_state_dict(engine.export_state_dict(role="consistency_ema"), strict=True)
    deploy.save_pretrained(tmp_path / "export")
    atomic_json(tmp_path / "export" / "consistency.json", method.export_config())
    store = ArtifactStore(tmp_path / "store")
    artifact = store.publish(
        tmp_path / "export", kind="aster_model", metadata={"method": "consistency"}
    )
    reloaded = UNet2D.from_pretrained(store.get(artifact.id).path)
    generator = torch.Generator().manual_seed(17)
    noise = torch.randn(2, 1, 4, 4, generator=generator)
    torch.testing.assert_close(
        sample_consistency(reloaded, noise, [80.0]),
        sample_consistency(deploy, noise, [80.0]),
        rtol=0,
        atol=0,
    )


def test_consistency_rejects_before_update_and_incomplete_target_commit(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(122)
    model = _model()
    engine = Trainer(model, zero_stage=3)
    method = ConsistencyMethod(engine, config=_config(), target_factory=_model)
    before = engine.export_state_dict()
    state = method.generator.get_state().clone()
    malformed = _batch()
    malformed["interval_indices"][0] = 9
    with pytest.raises(ValueError, match="preflight"):
        method.update([malformed])
    assert not method._incomplete and torch.equal(state, method.generator.get_state())
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    checkpoint = engine.save_checkpoint(tmp_path / "complete")
    original = engine.update_target

    def failed_commit(*args, **kwargs):
        raise RuntimeError("simulated target publication failure")

    monkeypatch.setattr(engine, "update_target", failed_commit)
    with pytest.raises(RuntimeError, match="publication"):
        method.update([_batch()])
    with pytest.raises(RuntimeError, match="incomplete"):
        method.state_dict()
    with pytest.raises(ValueError):
        engine.save_checkpoint(tmp_path / "invalid")
    with pytest.raises(RuntimeError):
        engine.export_state_dict()
    monkeypatch.setattr(engine, "update_target", original)
    engine.load_checkpoint(checkpoint)
    assert method.update([_batch()]).updated
    method.config = replace(method.config, log_std=1.0)
    with pytest.raises(ValueError, match="configuration"):
        method.state_dict()


def test_teacher_target_and_schedule_identity_cannot_silently_change(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(541)
    teacher = _model(teacher=True)
    initial_teacher = deepcopy(teacher)
    first = Trainer(_model())
    method = ConsistencyMethod(first, config=_config("cd"), target_factory=_model, teacher=teacher)
    path = first.save_checkpoint(tmp_path / "checkpoint")
    with torch.no_grad():
        next(teacher.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="teacher"):
        method.update([_batch()])
    first.load_checkpoint(path)
    with torch.no_grad():
        next(method.target.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="target weights"):
        method.state_dict()
    other = Trainer(_model())
    second = ConsistencyMethod(
        other,
        config=replace(_config("cd"), time_scale=25.0),
        target_factory=_model,
        teacher=initial_teacher,
    )
    with pytest.raises(ValueError, match="settings"):
        other.load_checkpoint(path)
    with pytest.raises(ValueError, match="iCT"):
        ConsistencyConfig(target_ema=0.9)
    with pytest.raises(ValueError, match="metric"):
        ConsistencyConfig(metric="lpips")
    with pytest.raises(ValueError, match="configuration"):
        ConsistencyMethod(Trainer(_model()), target_factory=lambda: _model(dropout=0.1))


def test_shared_dropout_rng_in_current_target_ict():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = _model(dropout=0.4)

    with torch.no_grad():
        model.output[-1].weight.normal_(std=0.1)
    target = deepcopy(model)
    config = _config("ict")
    objective = _ConsistencyObjective(config, target.requires_grad_(False))
    batch = _batch()
    high = torch.full((2,), 0.5)

    prepared = {**batch, "sigma_high": high, "sigma_low": high}
    objective.config = replace(config, weighting="uniform")
    result = objective(model, prepared)
    assert result.numerator.item() == 0 and result.denominator.dtype == torch.int64


@pytest.mark.parametrize("stage", [0, 3])
def test_consistency_bf16_actual_unet_and_exact_next_step(stage, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(135)
    engine = Trainer(
        _model(),
        zero_stage=stage,
        precision="bf16",
        optimizer_factory=lambda p: torch.optim.RAdam(p, lr=0.004),
    )
    method = ConsistencyMethod(
        engine, config=_config("cd"), target_factory=_model, teacher=_model(teacher=True)
    )
    assert method.update([_batch()]).updated
    path = engine.save_checkpoint(tmp_path / "bf16")
    expected_result = method.update([{"sample": _batch()["sample"]}])
    expected = engine.export_state_dict()
    engine.load_checkpoint(path)
    result = method.update([{"sample": _batch()["sample"]}])
    assert result == expected_result
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ct", "cd", "ict"])
@pytest.mark.parametrize("stage", [0, 3])
def test_unequal_microbatch_accumulation_matches_full_sample_normalization(mode, stage):
    torch.set_num_threads(1)
    torch.manual_seed(731)
    source = _model()
    teacher = _model(teacher=True) if mode == "cd" else None
    factory = lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.8)
    dense = Trainer(deepcopy(source), optimizer_factory=factory, max_grad_norm=None)
    accumulated = Trainer(
        source,
        optimizer_factory=factory,
        zero_stage=stage,
        accumulation_steps=2,
        max_grad_norm=None,
    )
    direct = ConsistencyMethod(
        dense, target_factory=_model, config=_config(mode), teacher=deepcopy(teacher)
    )
    pieces = ConsistencyMethod(
        accumulated, target_factory=_model, config=_config(mode), teacher=teacher
    )
    first, second = _batch(2), _batch(1)
    merged = {key: torch.cat((first[key], second[key])) for key in first}
    expected = direct.update([merged])
    actual = pieces.update([first, second])
    assert actual.loss == pytest.approx(expected.loss, rel=2e-6, abs=3e-7)
    assert actual.terms["consistency"]["denominator"] == 3
    for key, value in accumulated.export_state_dict().items():
        torch.testing.assert_close(value, dense.export_state_dict()[key], rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("mode", ["ct", "ict"])
def test_real_curriculum_growth_and_random_interval_resume(mode, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(379)
    config = ConsistencyConfig(
        mode=mode, total_steps=12, initial_scales=2, final_scales=8, sampling_ema=None
    )
    engine = Trainer(
        _model(), zero_stage=3, optimizer_factory=lambda p: torch.optim.RAdam(p, lr=0.002)
    )
    method = ConsistencyMethod(engine, config=config, target_factory=_model)
    first_scales = config.scales_and_ema(0)[0]
    batch = {"sample": _batch()["sample"]}
    for _ in range(5):
        method.update([batch])
    assert config.scales_and_ema(method.updates)[0] > first_scales
    path = engine.save_checkpoint(tmp_path / "growing")
    expected_result = method.update([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(path)
    assert method.update([batch]) == expected_result
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_student_cannot_bypass_target_update_lifecycle():
    torch.set_num_threads(1)
    engine = Trainer(_model())
    method = ConsistencyMethod(engine, config=_config(), target_factory=_model)

    def foreign(model, batch):
        output = model(batch["sample"], torch.ones(len(batch["sample"])))
        from aster.core import LossTerm

        return LossTerm(
            output.prediction.square().sum() + output.prediction.sum(), torch.tensor(1), "sample"
        )

    assert engine.phase("foreign", objective=foreign, microbatches=[_batch()]).updated
    with pytest.raises(ValueError, match="outside the consistency lifecycle"):
        method.update([_batch()])
    with pytest.raises(ValueError, match="outside the consistency lifecycle"):
        method.state_dict()
