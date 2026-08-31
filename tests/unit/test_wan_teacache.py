from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from aster.core import digest_json
from aster.models.video_world import WanVideoConfig, WanVideoDiT
from aster.methods.video_generation import sample_video_latents
from aster.optimization.wan_teacache import (
    WanCacheSampler,
    WanTeaCacheSettings,
    WanTeaCacheSession,
    WanCacheCalibration,
    calibrate_wan_teacache,
    sample_wan_teacache,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(625)
    model = WanVideoDiT(
        WanVideoConfig(
            latent_channels=2,
            hidden_size=12,
            intermediate_size=24,
            num_heads=2,
            num_layers=2,
            text_dim=4,
            text_length=3,
            frequency_dim=4,
        )
    ).eval()
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.1)
    noise = torch.randn(1, 2, 2, 2, 2)
    positive, negative = {"text": torch.randn(1, 2, 4)}, {"text": torch.randn(1, 2, 4) - 2}
    return model, noise, positive, negative


def calibration(model, noise, positive, negative, sampler, mode="default"):
    return calibrate_wan_teacache(
        model,
        [{"id": "calibration-0", "noise": noise, "positive": positive, "negative": negative}],
        policy_artifact_id="a" * 64,
        dataset_fingerprint="b" * 64,
        sampler=sampler,
        mode=mode,
    )


@pytest.mark.parametrize("solver,guidance", [("euler", 1.0), ("euler", 0.0), ("heun", 2.0)])
def test_full_native_sampler_and_counts_equal_previous_path(solver, guidance):
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=4, solver=solver, guidance_scale=guidance)
    with torch.no_grad():
        expected = sample_video_latents(
            model,
            noise,
            positive,
            steps=4,
            solver=solver,
            guidance_scale=guidance,
            negative_condition=negative,
        )
    actual, report, _ = sample_wan_teacache(
        model,
        noise,
        positive,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        negative_condition=negative,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    count = 4 * (2 if solver == "heun" else 1) * (1 if guidance == 1 else 2)
    assert report["field_calls"] == report["full_backbone_calls"] == report["head_calls"] == count
    assert report["reused_backbone_calls"] == 0

    session = WanTeaCacheSession(
        model,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        negative_condition=negative,
    )
    session.predict(noise, round_index=0, branch="positive")
    assert (
        session._states["positive"]["residual"] is None
        and session._states["positive"]["probe"] is None
    )


def test_prepared_refactor_keeps_parameter_names_forward_and_all_gradients():
    model, noise, positive, _ = fixture()
    reference = copy.deepcopy(model)
    first = noise.clone().requires_grad_()
    second = noise.clone().requires_grad_()
    one = model(first, torch.tensor([0.7]), positive, sequence_length=4).prediction
    prepared = reference.prepare(second, torch.tensor([0.7]), positive, sequence_length=4)
    hidden = prepared.hidden
    for block in reference.blocks:
        hidden = block(hidden, prepared.modulation, prepared.grid, prepared.text, prepared.image)
    two = reference.finish(hidden, prepared).prediction
    torch.testing.assert_close(one, two, atol=0, rtol=0)
    one.square().sum().backward()
    two.square().sum().backward()
    torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["default", "retention"])
def test_real_calibration_branch_residual_and_retention_semantics(mode):
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=8, guidance_scale=2.0)
    fitted = calibration(model, noise, positive, negative, sampler, mode)
    assert WanCacheCalibration.from_dict(fitted.to_dict()).id == fitted.id
    assert len(fitted.measurements) == 14
    settings = WanTeaCacheSettings(threshold=1e8, mode=mode, maximum_relative_output_error=1e8)
    session = WanTeaCacheSession(
        model,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        negative_condition=negative,
        calibration=fitted,
        settings=settings,
    )
    first_outputs = []
    for index in range(8):
        for branch in sampler.branches:
            out = session.predict(noise + index * 0.01, round_index=index, branch=branch)
            if index == 0:
                first_outputs.append(out.prediction.clone())
    expected_full = 4 if mode == "default" else 10
    report = session.observation()
    assert report["full_backbone_calls"] == expected_full
    assert report["field_calls"] == 16 and report["reused_backbone_calls"] == 16 - expected_full
    assert report["trace"][-1]["forced"] == (mode == "default")

    assert not torch.equal(first_outputs[0], first_outputs[1])
    old = report["condition_fingerprint"]
    session.reset(condition={"text": positive["text"] + 3}, negative_condition=negative)
    assert session.observation()["field_calls"] == 0 and session.condition_fingerprint != old
    fresh = session.predict(noise, round_index=0, branch="positive").prediction
    with torch.no_grad():
        expected = model(noise, torch.ones(1), {"text": positive["text"] + 3}).prediction
    torch.testing.assert_close(fresh, expected, atol=0, rtol=0)


def test_polynomial_negative_accumulation_matches_upstream_numpy_not_clamped():
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=4)
    fitted = calibration(model, noise, positive, negative, sampler)

    replaced = replace(fitted, coefficients=(-0.1,))
    assert replaced.estimate(0.5) == float(np.poly1d([-0.1])(0.5)) == -0.1
    session = WanTeaCacheSession(
        model,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        calibration=replaced,
        settings=WanTeaCacheSettings(threshold=0, maximum_relative_output_error=1e8),
    )
    for index in range(4):
        session.predict(noise, round_index=index, branch="positive")
    assert [row["reused"] for row in session.trace] == [False, True, True, False]
    assert session.trace[2]["accumulated"] == -0.2


def test_audit_failure_disables_both_branches_and_actual_full_calls_are_counted():
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=5, guidance_scale=2.0)
    fitted = calibration(model, noise, positive, negative, sampler)
    _, report, _ = sample_wan_teacache(
        model,
        noise,
        positive,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        negative_condition=negative,
        calibration=fitted,
        settings=WanTeaCacheSettings(
            threshold=1e8, audit_every=1, maximum_relative_output_error=0.0
        ),
    )
    assert report["guard_failed"] and report["quality_status"] == "failed_guard"
    assert report["full_backbone_calls"] == report["field_calls"] == 10
    assert report["audit_backbone_calls"] == 1 and report["head_calls"] == 12


def test_fail_closed_order_mutation_profile_solver_geometry_and_recovery():
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=4, guidance_scale=2.0)
    fitted = calibration(model, noise, positive, negative, sampler)
    kwargs = dict(
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        negative_condition=negative,
        calibration=fitted,
        settings=WanTeaCacheSettings(),
    )
    session = WanTeaCacheSession(model, **kwargs)
    with pytest.raises(ValueError, match="order"):
        session.predict(noise, round_index=0, branch="negative")
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(noise, round_index=0, branch="positive")
    session.reset(condition=positive, negative_condition=negative)
    session.predict(noise, round_index=0, branch="positive")
    with torch.no_grad():
        model.head.head.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="changed"):
        session.predict(noise, round_index=0, branch="negative")
    with pytest.raises(ValueError, match="mismatch"):
        WanTeaCacheSession(model, **kwargs)
    with pytest.raises(ValueError):
        WanCacheSampler(solver="unipc")
    with pytest.raises(ValueError, match="profiles"):
        replace(fitted, origin="official_wan_1.3B")
    with pytest.raises(ValueError, match="population"):
        replace(fitted, measurements=fitted.measurements[:-1])
    with pytest.raises(ValueError):
        WanTeaCacheSettings(threshold=float("nan"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="Autocast"):
            WanTeaCacheSession(
                model,
                policy_artifact_id="a" * 64,
                sampler=sampler,
                condition=positive,
                negative_condition=negative,
            )
    assert digest_json(fitted.to_dict()) == fitted.id


def test_condition_change_is_rejected_and_actual_block_hooks_match_report():
    model, noise, positive, negative = fixture()
    sampler = WanCacheSampler(steps=5, guidance_scale=2.0)
    fitted = calibration(model, noise, positive, negative, sampler)
    settings = WanTeaCacheSettings(threshold=1e8, maximum_relative_output_error=1e8)
    calls = {"block": 0, "head": 0}

    def block_hook(*args):
        calls["block"] += 1

    def head_hook(*args):
        calls["head"] += 1

    handles = [b.register_forward_hook(block_hook) for b in model.blocks] + [
        model.head.register_forward_hook(head_hook)
    ]
    try:
        _, observation, _ = sample_wan_teacache(
            model,
            noise,
            positive,
            policy_artifact_id="a" * 64,
            sampler=sampler,
            negative_condition=negative,
            calibration=fitted,
            settings=settings,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert calls["block"] == len(model.blocks) * observation["full_backbone_calls"]
    assert calls["head"] == observation["head_calls"]
    session = WanTeaCacheSession(
        model,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        negative_condition=negative,
        calibration=fitted,
        settings=settings,
    )
    session.predict(noise, round_index=0, branch="positive")
    session.conditions["negative"]["text"].add_(1.0)
    with pytest.raises(RuntimeError, match="changed"):
        session.predict(noise, round_index=0, branch="negative")


def test_image_conditioned_heun_uses_actual_e0_probe_and_keeps_two_image_branches():
    torch.manual_seed(829)
    model = WanVideoDiT(
        WanVideoConfig(
            latent_channels=2,
            condition_channels=4,
            image_conditioned=True,
            hidden_size=12,
            intermediate_size=24,
            num_heads=2,
            num_layers=1,
            text_dim=4,
            text_length=3,
            image_dim=3,
            frequency_dim=4,
        )
    ).eval()
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.1)
    noise = torch.randn(1, 2, 2, 2, 2)
    positive = {
        "text": torch.randn(1, 2, 4),
        "image_features": torch.randn(1, 2, 3),
        "video_condition": torch.randn(1, 4, 2, 2, 2),
    }
    negative = {**positive, "text": torch.zeros(1, 2, 4)}
    sampler = WanCacheSampler(steps=4, solver="heun", guidance_scale=2.0)
    fitted = calibration(model, noise, positive, negative, sampler, mode="retention")
    session = WanTeaCacheSession(
        model,
        policy_artifact_id="a" * 64,
        sampler=sampler,
        condition=positive,
        negative_condition=negative,
        calibration=fitted,
        settings=WanTeaCacheSettings(
            threshold=1e8, mode="retention", maximum_relative_output_error=1e8
        ),
    )
    from aster.optimization.wan_teacache import run_wan_teacache_session

    result = run_wan_teacache_session(session, noise)
    assert result.shape == noise.shape and result.isfinite().all()
    assert session.field_calls == 16 and session.full_backbone_calls == 10
    assert session._states["positive"]["probe"].shape == (1, 6, model.config.hidden_size)
    assert not torch.equal(
        session._states["positive"]["residual"], session._states["negative"]["residual"]
    )
