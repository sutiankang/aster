import pytest
import torch
from aster.models.generative import DiT, DiTConfig
from aster.optimization import (
    ResidualCacheCalibration,
    fit_residual_calibration,
    DiTStepCacheSession,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_and_calibration():
    torch.manual_seed(91)
    model = DiT(
        DiTConfig(in_channels=2, hidden_size=16, num_heads=2, num_layers=2, condition_dim=3)
    ).eval()

    with torch.no_grad():
        for block in model.blocks:
            block.ada[-1].weight.normal_(std=0.3)
            block.ada[-1].bias.normal_(std=0.2)
        model.output.weight.normal_(std=0.2)
        model.ada[-1].weight.normal_(std=0.2)
    calibration = ResidualCacheCalibration(
        "dit-fixture", "calibration-fixture-not-benchmark", DiTStepCacheSession.probe_id, (0.0,)
    )
    return model, calibration


def test_disabled_reuse_matches_real_native_dit_for_entire_schedule():
    model, calibration = model_and_calibration()
    condition, sample = torch.randn(1, 3), torch.randn(1, 2, 4, 4)
    schedule = (4.0, 3.0, 2.0, 1.0)
    session = DiTStepCacheSession(
        model,
        policy_artifact_id="dit-fixture",
        condition=condition,
        schedule=schedule,
        calibration=calibration,
        threshold=0,
    )
    for step, time in enumerate(schedule):
        actual = session.predict(sample, step=step)
        with torch.no_grad():
            expected = model(sample, torch.tensor([time]), condition)
        torch.testing.assert_close(actual.prediction, expected.prediction, atol=1e-6, rtol=1e-5)
    assert session.full_backbone_calls == len(schedule) and session.reused_backbone_calls == 0
    assert not session.guard_failed
    assert session.observation()["quality_status"] == "requires_end_to_end_evaluation"
    with pytest.raises(RuntimeError):
        session.predict(sample, step=len(schedule))


def test_actual_residual_reuse_skips_blocks_and_preserves_exact_stationary_output():
    model, calibration = model_and_calibration()
    condition, sample = torch.randn(1, 3), torch.randn(1, 2, 4, 4)
    session = DiTStepCacheSession(
        model,
        policy_artifact_id="dit-fixture",
        condition=condition,
        schedule=(1.0,) * 6,
        calibration=calibration,
        max_skip=2,
        audit_every=10,
    )
    calls = []
    handle = session.model.blocks[0].register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = model(sample, torch.tensor([1.0]), condition).prediction
    condition.zero_()
    for step in range(6):
        torch.testing.assert_close(
            session.predict(sample, step=step).prediction, expected, atol=1e-6, rtol=1e-5
        )
    handle.remove()
    assert len(calls) == session.full_backbone_calls == 3
    assert session.reused_backbone_calls == 3 and not session.guard_failed
    assert session.observation()["evidence_kind"] == "approximate_transform"


def test_failed_output_error_guard_disables_reuse_and_records_failed_quality():
    model, calibration = model_and_calibration()
    condition, sample = torch.randn(1, 3), torch.randn(1, 2, 4, 4)
    session = DiTStepCacheSession(
        model,
        policy_artifact_id="dit-fixture",
        condition=condition,
        schedule=(4.0, 3.0, 2.0, 1.0, 0.0),
        calibration=calibration,
        max_skip=3,
        audit_every=2,
        max_relative_error=0.0,
    )
    for step in range(5):
        value = sample * (step + 1)
        output = session.predict(value, step=step)
        if step >= 2:
            with torch.no_grad():
                expected = model(value, torch.tensor([4.0 - step]), condition).prediction
            torch.testing.assert_close(output.prediction, expected, atol=1e-6, rtol=1e-5)
    observation = session.observation()
    assert observation["guard_failed"] and observation["quality_status"] == "failed_guard"
    assert session.reused_backbone_calls == 1 and session.full_backbone_calls == 4
    assert observation["trace"][2]["checked_relative_output_l1"] > 0


def test_calibration_and_session_identity_shape_order_are_strict():
    model, calibration = model_and_calibration()
    fitted = fit_residual_calibration(
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.2, 0.4, 0.6],
        policy_artifact_id="dit-fixture",
        dataset_fingerprint="paired-true-measurements-fixture",
        degree=1,
    )
    assert fitted.estimate(0.15) == pytest.approx(0.3, abs=1e-4)
    with pytest.raises(ValueError):
        DiTStepCacheSession(
            model,
            policy_artifact_id="other-model",
            condition=None,
            schedule=(1.0, 0.0),
            calibration=calibration,
        )
    session = DiTStepCacheSession(
        model,
        policy_artifact_id="dit-fixture",
        condition=None,
        schedule=(2.0, 1.0, 0.0),
        calibration=calibration,
    )
    sample = torch.randn(1, 2, 4, 4)
    with pytest.raises(RuntimeError):
        session.predict(sample, step=1)
    session.predict(sample, step=0)
    with pytest.raises(ValueError):
        session.predict(torch.randn(2, 2, 4, 4), step=1)
    with torch.no_grad():
        session.model.output.weight.add_(1)
    with pytest.raises(RuntimeError, match="Policy changed"):
        session.predict(sample, step=1)
    session.close()
    with pytest.raises(RuntimeError):
        session.predict(sample, step=1)
