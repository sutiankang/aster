from copy import deepcopy
import pytest
import torch
from aster.models.lewm import LeWMConfig, LeWorldModel
from aster.methods.lewm import LeWMObjective, LeWMMethod, SIGReg
from aster.training import Trainer


def batch(n=4):
    return dict(pixels=torch.randn(n, 4, 3, 16, 16), actions=torch.randn(n, 3, 2))


@pytest.mark.parametrize("stage", [0, 3])
def test_models_lewm_learns_all_statistics_once_per_update_and_exact_fresh_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(726)
    data = batch()
    c = LeWMConfig()

    def create():
        engine = Trainer(LeWorldModel(c), zero_stage=stage, lr=0.002)
        method = LeWMMethod(engine, objective=LeWMObjective(num_proj=32), seed=727)
        return engine, method

    engine, method = create()
    initial = method.update([data]).loss
    for _ in range(11):
        final = method.update([data]).loss
    assert final < initial * 0.8
    state = engine.export_state_dict()
    assert int(state["projector.net.1.num_batches_tracked"]) == 12
    assert int(state["pred_proj.net.1.num_batches_tracked"]) == 12
    checkpoint = engine.save_checkpoint(tmp_path / "checkpoint")
    expected = method.update([data])
    weights = deepcopy(engine.export_state_dict())
    fresh, fresh_method = create()
    fresh.load_checkpoint(checkpoint, trusted=True)
    actual = fresh_method.update([data])
    assert actual.loss == expected.loss
    assert fresh_method.updates == method.updates
    for name, value in fresh.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0, msg=name)


def test_models_lewm_chunking_preserves_true_global_sigreg_and_bn():
    torch.set_num_threads(1)
    torch.manual_seed(728)
    data = batch(5)
    initial = LeWorldModel().state_dict()
    methods = []
    for _ in range(2):
        model = LeWorldModel()
        model.load_state_dict(initial)
        methods.append(
            LeWMMethod(Trainer(model, lr=0.0003), objective=LeWMObjective(num_proj=16), seed=731)
        )
    left = methods[0].update([data])
    right = methods[1].update(
        [
            {key: value[:2] for key, value in data.items()},
            {key: value[2:] for key, value in data.items()},
        ]
    )
    assert left.loss == right.loss
    for name, value in methods[0].engine.export_state_dict().items():
        torch.testing.assert_close(
            value, methods[1].engine.export_state_dict()[name], atol=0, rtol=0
        )
    projection = torch.randn(8, 32)
    projection /= projection.norm(dim=0)
    emb = torch.randn(4, 10, 8)
    stat = SIGReg(num_proj=32)

    assert not torch.isclose(
        stat(emb, projection), (stat(emb[:, :5], projection) + stat(emb[:, 5:], projection)) / 2
    )


def test_models_lewm_rejects_invalid_raw_boundary_and_implicit_microbatch_statistics():
    torch.set_num_threads(1)
    engine = Trainer(LeWorldModel())
    method = LeWMMethod(engine)
    data = batch()
    data["actions"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        method.update([data])
    assert engine.steps == 0 and not method.incomplete
    with pytest.raises(ValueError, match="one complete"):
        LeWMObjective().preflight_microbatches(engine.model, [batch(), batch()])
    with pytest.raises(ValueError, match="fixed one-logical"):
        LeWMMethod(Trainer(LeWorldModel(), accumulation_steps=2))


def test_models_lewm_factory_save_load_bn_statistics_and_explicit_terminal_windows(tmp_path):
    from aster.models import build_model, load_model
    from aster.models.config import config_from_dict
    from aster.data.lewm import lewm_windows, fit_lewm_actions
    from aster.data.actions import ActionSpec

    torch.set_num_threads(1)
    torch.manual_seed(745)
    model = build_model(config_from_dict(LeWMConfig().to_dict()))
    data = batch()
    for _ in range(3):
        model(**data)
    model.eval()
    expected = model(**data)
    model.save_pretrained(tmp_path / "model")
    restored = load_model(tmp_path / "model").eval()
    torch.testing.assert_close(restored(**data).predictions, expected.predictions, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    pixels, actions = torch.randn(2, 7, 3, 16, 16), torch.randn(2, 6, 2)
    end = torch.zeros(2, 6, dtype=torch.bool)
    end[0, 2] = True
    windows = lewm_windows(pixels, actions, end, history_size=3)

    assert len(windows["pixels"]) == 5
    torch.testing.assert_close(windows["pixels"][0], pixels[0, :4], atol=0, rtol=0)
    torch.testing.assert_close(windows["pixels"][1], pixels[1, :4], atol=0, rtol=0)
    spec = ActionSpec(("x", "y"), ("m", "m"), "test", "delta", 10, 1)
    normalizer = fit_lewm_actions(actions, spec=spec)
    torch.testing.assert_close(
        normalizer.scale, actions.flatten(0, 1).std(0, correction=1), atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="statistics"):
        fit_lewm_actions(actions[:1, :1], spec=spec)


def test_models_lewm_cem_mpc_rng_warmstart_queue_resume_and_bn_mode_preservation():
    from aster.planning.lewm import LeWMCEM, LeWMCEMConfig, LeWMMPC
    from aster.data.actions import ActionSpec, ActionNormalizer

    torch.set_num_threads(1)
    torch.manual_seed(746)
    model = LeWorldModel()
    model.predictor.eval()
    modes = [module.training for module in model.modules()]
    initial = deepcopy(model.state_dict())
    config = LeWMCEMConfig(horizon=3, num_samples=12, topk=4, n_steps=2)
    planner = LeWMCEM(model, config, seed=747)
    pixels, goal = torch.randn(2, 1, 3, 16, 16), torch.randn(2, 1, 3, 16, 16)
    first = planner.solve(pixels, goal)
    saved = deepcopy(planner.state_dict())
    second = planner.solve(pixels, goal, init_action=first.actions[:, 1:])
    planner.load_state_dict(saved)
    repeated = planner.solve(pixels, goal, init_action=first.actions[:, 1:])
    torch.testing.assert_close(second.actions, repeated.actions, atol=0, rtol=0)
    assert modes == [module.training for module in model.modules()]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    normalizer = ActionNormalizer(
        [0.2], [0.5], spec=ActionSpec(("x",), ("m",), "test", "delta", 10, 1)
    )
    mpc = LeWMMPC(planner, normalizer=normalizer, action_block=2, receding_horizon=2)
    mpc.act(pixels, goal)
    state = deepcopy(mpc.state_dict())
    expected = [mpc.act(pixels, goal) for _ in range(7)]
    mpc.load_state_dict(state)
    actual = [mpc.act(pixels, goal) for _ in range(7)]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    mpc.reset()
    assert not mpc.pending and mpc.next_init is None
    with pytest.raises(ValueError, match="topk"):
        LeWMCEMConfig(topk=1)


def test_models_lewm_author_legacy_weights_mapping_rejects_partial_and_mixed_without_mutation():
    from aster.models.vit import convert_vit_state_dict

    torch.set_num_threads(1)
    torch.manual_seed(748)
    model = LeWorldModel()
    state = deepcopy(model.state_dict())
    old = {}
    substitutions = {
        "attention.q_proj": "attention.attention.query",
        "attention.k_proj": "attention.attention.key",
        "attention.v_proj": "attention.attention.value",
        "attention.o_proj": "attention.output.dense",
        "mlp.fc1": "intermediate.dense",
        "mlp.fc2": "output.dense",
    }
    for name, value in state.items():
        if name.startswith("encoder.layers."):
            name = name.replace("encoder.layers.", "encoder.encoder.layer.")
            for new, previous in substitutions.items():
                if new in name:
                    name = name.replace(new, previous)
                    break
        old[name] = value
    restored = LeWorldModel()
    restored.load_author_state_dict(old, vit_layout="transformers_4.57")
    for name, value in state.items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    invalid = dict(old)
    invalid.pop(next(iter(old)))
    with pytest.raises(ValueError, match="not changed"):
        restored.load_author_state_dict(invalid, vit_layout="transformers_4.57")
    for name, value in state.items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Mixed"):
        convert_vit_state_dict({"layers.0.fake": torch.ones(1)}, layout="transformers_4.57")


def test_models_lewm_real_pixel_training_artifact_reload_and_heldout_control(tmp_path):
    from pathlib import Path
    import runpy
    from aster.core import ArtifactStore, read_json

    example = Path(__file__).resolve().parents[2] / "examples" / "lewm_pipeline.py"
    # The default training budget leaves room for CPU-kernel roundoff to alter
    # optimization and CEM trajectories without relaxing the control target.
    report = runpy.run_path(str(example))["run_demo"](tmp_path / "run", seed=743)
    assert report["train_loss_last"] < report["train_loss_first"] * 0.4
    assert report["heldout_latent_std"] > 0.2
    assert report["final_goal_error"] < 0.15 and report["success_rate"] == 1.0, report
    errors = torch.tensor(report["goal_errors"])
    assert len(errors) == report["evaluation_episodes"] == 4
    assert torch.isfinite(errors).all() and (errors < report["success_threshold"]).all(), report
    assert errors.mean().item() == pytest.approx(report["final_goal_error"], abs=1e-7)
    assert report["mean_return"] > report["zero_action_mean_return"]
    assert report["benchmark"] == "local_1d_pixel_control_not_public_pusht"
    artifact = ArtifactStore(tmp_path / "run" / "artifacts").get(report["artifact_id"])
    assert len(artifact.metadata["training_data_fingerprint"]) == 64
    assert read_json(artifact.path / "pixel_contract.json")["type"] == "visual_point_rgb_torch_v1"
