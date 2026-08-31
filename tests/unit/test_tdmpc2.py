import copy
import pytest
import torch
import torch.nn.functional as F

from aster.models import build_model, TDMPC2Config, TDMPC2PolicyConfig
from aster.models.tdmpc2 import SimNorm, random_shift
from aster.methods.tdmpc2 import TDMPC2Method, TDMPC2Planner, MPPIConfig, RunningValueScale
from aster.training import Trainer


def make_world(**extra):
    return build_model(
        TDMPC2Config(
            observation_dim=5,
            action_dim=2,
            latent_dim=16,
            simnorm_dim=4,
            hidden_size=16,
            encoder_size=16,
            num_bins=21,
            num_q=3,
            **extra,
        )
    )


def make_policy(features=16):
    return build_model(TDMPC2PolicyConfig(feature_dim=features, action_dim=2, hidden_size=16))


def test_tdmpc2_simplex_twobin_policy_and_scale_formulas():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    value = torch.randn(2, 16, requires_grad=True)
    torch.testing.assert_close(SimNorm(4)(value), value.reshape(2, 4, 4).softmax(-1).reshape(2, 16))
    world, policy = make_world(), make_policy()
    values = torch.tensor([-1.0, 0.4, 2.0])
    targets = world.value_loss(torch.zeros(3, 21), values)
    torch.testing.assert_close(targets, torch.full((3,), torch.log(torch.tensor(21.0))))
    features, noise = torch.randn(3, 16), torch.randn(3, 2)
    action, info = policy(features, noise=noise)
    mean, raw_std = policy.network(features).chunk(2, -1)
    log_std = -10.0 + 6.0 * (raw_std.tanh() + 1)
    expected = (mean + noise * log_std.exp()).tanh()
    logp = (-0.5 * noise.square() - log_std - 0.9189385175704956).sum(-1)
    corrected = logp - (F.relu(1 - expected.square()) + 1e-6).log().sum(-1)
    torch.testing.assert_close(action, expected)
    torch.testing.assert_close(info["entropy"], -corrected)
    torch.testing.assert_close(info["scaled_entropy"], -corrected * (logp * 2) / (corrected + 1e-8))
    scale = RunningValueScale(0.5)
    scale.update(torch.tensor([0.0, 10.0, 20.0]))
    assert abs(scale.value - 9.5) < 1e-6


def test_tdmpc2_two_phase_update_target_and_exact_next_step_resume(tmp_path):
    torch.manual_seed(15)
    torch.set_num_threads(1)
    world, policy = make_world(episodic=True), make_policy()

    engine = Trainer(world, optimizer=torch.optim.Adam(world.parameters(), lr=0.001), lr=0.001)
    method = TDMPC2Method(
        engine, policy, policy_optimizer=torch.optim.Adam(policy.parameters(), lr=0.001, eps=1e-5)
    )
    data = {
        "observations": torch.randn(3, 4, 5),
        "actions": torch.rand(3, 3, 2) * 2 - 1,
        "rewards": torch.randn(3, 3),
        "terminated": torch.tensor([[False, False, True], [False] * 3, [False] * 3]),
    }
    original = copy.deepcopy(world.state_dict())
    result = method.update([data])
    assert result["world"].updated and result["policy"].updated
    assert any(not torch.equal(value, world.state_dict()[key]) for key, value in original.items())
    for target, source in zip(method.target.parameters(), world.q_heads.parameters()):
        assert not target.requires_grad
    engine.save_checkpoint(tmp_path / "checkpoint")
    method.update([data])
    expected_world, expected_policy = (
        copy.deepcopy(world.state_dict()),
        copy.deepcopy(policy.state_dict()),
    )
    expected_scale = method.scale.value
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    method.update([data])
    assert method.scale.value == expected_scale
    for model, expected in ((world, expected_world), (policy, expected_policy)):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    world.save_pretrained(tmp_path / "world")
    policy.save_pretrained(tmp_path / "policy")
    world_restored = type(world).from_pretrained(tmp_path / "world")
    policy_restored = type(policy).from_pretrained(tmp_path / "policy")
    planner = TDMPC2Planner(
        world_restored,
        policy_restored,
        MPPIConfig(horizon=3, population=12, elites=3, policy_trajectories=2, iterations=2),
    )
    generator = torch.Generator().manual_seed(8)
    action, stats = planner.plan(
        data["observations"][:1, 0], first=True, eval_mode=True, generator=generator
    )
    assert action.shape == (2,) and action.abs().max() <= 1 and stats["mean"].shape == (3, 2)
    previous = copy.deepcopy(planner.state_dict())
    rng = generator.get_state()
    expected, _ = planner.plan(data["observations"][:1, 1], eval_mode=True, generator=generator)
    planner.load_state_dict(previous)
    generator.set_state(rng)
    actual, _ = planner.plan(data["observations"][:1, 1], eval_mode=True, generator=generator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tdmpc2_multitask_mask_and_pixel_encoder_have_real_paths():
    torch.set_num_threads(1)
    world = make_world(task_dim=3, action_dimensions=(1, 2))
    policy = make_policy(features=19)
    task = torch.tensor([0, 1])
    latent = world.encode(torch.randn(2, 5), task)
    action, _ = policy(world.condition(latent, task), action_mask=world.action_mask(latent, task))
    assert action[0, 1] == 0 and latent.shape == (2, 16)
    pixels = torch.randint(256, (2, 3, 64, 64), dtype=torch.uint8)
    pixel_world = build_model(
        TDMPC2Config(
            observation_kind="rgb",
            latent_dim=64,
            conv_channels=4,
            hidden_size=16,
            action_dim=2,
            num_bins=21,
        )
    )
    z = pixel_world.encode(pixels, generator=torch.Generator().manual_seed(9))
    assert z.shape == (2, 64)
    torch.testing.assert_close(z.reshape(2, 8, 8).sum(-1), torch.ones(2, 8))
    shifted = random_shift(pixels, generator=torch.Generator().manual_seed(2))
    assert shifted.shape == pixels.shape and shifted.min() >= 0 and shifted.max() <= 255


@pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
def test_tdmpc2_multitask_persistent_projection_update_resume_and_planning(tmp_path, zero_stage):
    torch.manual_seed(51)
    torch.set_num_threads(1)
    world, prior = make_world(task_dim=3, action_dimensions=(1, 2, 2)), make_policy(features=19)
    with torch.no_grad():
        world.task_embedding.weight.fill_(3.0)
    initial_unvisited = world.task_embedding.weight[2].detach().clone()
    engine = Trainer(
        world,
        optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=0.001),
        zero_stage=zero_stage,
    )
    method = TDMPC2Method(
        engine,
        prior,
        policy_optimizer_factory=lambda parameters: torch.optim.Adam(parameters, lr=0.001),
    )
    batch = {
        "observations": torch.randn(2, 3, 5),
        "actions": torch.rand(2, 2, 2) * 2 - 1,
        "rewards": torch.randn(2, 2),
        "terminated": torch.zeros(2, 2, dtype=torch.bool),
        "task_ids": torch.tensor([[0, 0, 0], [1, 1, 1]]),
    }
    batch["actions"][0, :, 1] = 0
    assert method.update([batch])["policy"].updated
    state = engine.export_state_dict()
    assert state["task_embedding.weight"][:2].norm(dim=-1).max() <= 1 + 1e-6
    torch.testing.assert_close(state["task_embedding.weight"][2], initial_unvisited, atol=0, rtol=0)
    engine.save_checkpoint(tmp_path / "multitask")
    method.update([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "multitask", trusted=True)
    method.update([batch])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    deploy_world = make_world(task_dim=3, action_dimensions=(1, 2, 2))
    deploy_prior = make_policy(features=19)
    deploy_world.load_state_dict(engine.export_state_dict())
    deploy_prior.load_state_dict(engine.export_state_dict(role="policy_prior"))
    planner = TDMPC2Planner(
        deploy_world,
        deploy_prior,
        MPPIConfig(horizon=2, population=6, elites=2, policy_trajectories=1, iterations=1),
    )
    action, _ = planner.plan(batch["observations"][:1, 0], task_id=2, eval_mode=True)
    assert action.shape == (2,) and deploy_world.task_embedding.weight[2].norm() <= 1 + 1e-6


def test_tdmpc2_rejects_reset_crossing_and_nonfinite_before_mutation():
    torch.set_num_threads(1)
    engine = Trainer(make_world(), lr=0.001)
    prior = make_policy()
    with pytest.raises(ValueError, match="finite"):
        TDMPC2Method(engine, prior, entropy_weight=float("nan"))
    assert len(engine.roles) == 1
    method = TDMPC2Method(engine, prior)
    batch = {
        "observations": torch.randn(1, 3, 5),
        "actions": torch.zeros(1, 2, 2),
        "rewards": torch.zeros(1, 2),
        "terminated": torch.zeros(1, 2, dtype=torch.bool),
        "truncated": torch.tensor([[True, False]]),
    }
    before = copy.deepcopy(engine.model.state_dict())
    with pytest.raises(ValueError, match="time-limit reset"):
        method.update([batch])
    for key, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    batch["truncated"] = torch.tensor([[False, True]])
    assert method.update([batch])["world"].updated


@pytest.mark.parametrize("zero_stage", [0, 3])
def test_tdmpc2_bf16_preserves_multi_phase_precision_contract(tmp_path, zero_stage):
    torch.manual_seed(78)
    torch.set_num_threads(1)
    world, prior = make_world(), make_policy()
    factory = lambda parameters: torch.optim.Adam(parameters, lr=0.001)
    engine = Trainer(world, optimizer_factory=factory, precision="bf16", zero_stage=zero_stage)
    method = TDMPC2Method(engine, prior, policy_optimizer_factory=factory)
    batch = {
        "observations": torch.randn(2, 3, 5),
        "actions": torch.zeros(2, 2, 2),
        "rewards": torch.zeros(2, 2),
        "terminated": torch.zeros(2, 2, dtype=torch.bool),
    }
    assert method.update([batch])["policy"].updated
    engine.save_checkpoint(tmp_path / "bf16")
    method.update([batch])
    expected = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "bf16", trusted=True)
    method.update([batch])
    for key, value in engine.export_state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    for bad in (
        {**batch, "observations": torch.randn(2, 3, 6)},
        {**batch, "rewards": torch.zeros(2, 2, dtype=torch.int64)},
    ):
        with pytest.raises(ValueError, match="preflight"):
            method.update([bad])
