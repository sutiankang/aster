from copy import deepcopy
from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
import torch

from aster.core import ArtifactStore, atomic_json, read_json
from aster.evaluation.generative import (
    GenerationCase,
    ImageSamplingPlan,
    generate_image_shard,
    merge_image_shards,
    publish_dmd_generator,
    quantize_image,
)
from aster.evaluation.generation_artifacts import load_native_artifact_model, resolve_image_sampling
from aster.methods.generation import DiffusionObjective, DiffusionSchedule, sample_flow
from aster.methods.generative_distillation import DMDMethod
from aster.models import load_model
from aster.models.generative import UNet2D, UNetConfig
from aster.tensor_recipes import fit_tensors
from aster.training import Trainer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(prediction="epsilon"):
    return UNetConfig(
        in_channels=3,
        model_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        num_heads=2,
        prediction_type=prediction,
    )


def read_pixels(root, manifest, index=0):
    with Image.open(root / manifest.samples[index].files[0].path) as image:
        return np.asarray(image)


def independent_ddim(model, noise, original, indices):

    x = noise.clone()
    with torch.no_grad():
        for cursor in reversed(range(len(indices))):
            index = indices[cursor]
            time = original.timestep_map[index].reshape(1)
            epsilon = model(x, time).prediction
            alpha = original.alpha_bar[index].to(x.dtype)
            previous = (
                original.alpha_bar[indices[cursor - 1]].to(x.dtype) if cursor else x.new_tensor(1.0)
            )
            clean = (x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
            x = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
    return x


@pytest.mark.parametrize("sampler", ["flow_heun", "ddim", "ddpm"])
def test_real_tensor_fit_published_model_to_png_with_training_schedule(tmp_path, sampler):
    torch.manual_seed(709)
    is_flow = sampler.startswith("flow")
    model_config = config("velocity" if is_flow else "epsilon")
    data = {"sample": torch.randn(4, 3, 4, 4).clamp(-1, 1)}
    torch.save(data, tmp_path / "images.pt")
    objective = (
        {"name": "flow"}
        if is_flow
        else {"name": "diffusion", "schedule": {"steps": 19, "name": "cosine"}}
    )
    arguments = {
        "model": model_config.to_dict(),
        "objective": objective,
        "data": str(tmp_path / "images.pt"),
        "preprocessing": {"type": "local_rgb_fixture", "version": "1"},
        "training": {"steps": 2, "batch_size": 2, "seed": 791},
    }
    store = ArtifactStore(tmp_path / "store")
    result = fit_tensors(arguments, {}, tmp_path / "train", store)
    artifact = store.get(result.artifacts["model"])
    assert (artifact.path / "model" / "config.json").is_file() and not (
        artifact.path / "config.json"
    ).exists()
    actual_objective = read_json(artifact.path / "objective.json")
    assert actual_objective["type"] == ("flow_matching" if is_flow else "diffusion")
    plan = ImageSamplingPlan(
        (GenerationCase("case", 888), GenerationCase("case2", 889)),
        (3, 4, 4),
        sampler=sampler,
        steps=4,
    )
    root = tmp_path / "generated"
    manifest = generate_image_shard(store, artifact.id, plan, root)
    manifest.verify(root)
    binding = read_json(root / "shard.json")["sampling_binding"]
    assert binding["model_relative_path"] == "model" and binding["training_semantics_bound"]
    assert binding["actual_successful_objective_bound"] is True
    assert binding["successful_update"] == read_json(artifact.path / "successful_update.json")
    assert binding["successful_update"]["role_updates"] == 2
    assert manifest.producer_artifacts == (artifact.id,)
    model = load_model(artifact.path / "model").eval()
    assert model.output[-1].weight.abs().sum() > 0
    noise = torch.randn((1, 3, 4, 4), generator=torch.Generator().manual_seed(888))
    if is_flow:
        expected = sample_flow(model, noise, steps=4)
        np.testing.assert_array_equal(
            read_pixels(root, manifest), quantize_image(expected[0], plan.quantization)
        )
    else:
        original = DiffusionSchedule.create(19)
        assert actual_objective["betas"] == original.betas.tolist()
        assert binding["diffusion"]["selected_training_indices"] == [0, 6, 12, 18]
        assert binding["diffusion"]["effective_model_times"] == [0, 6, 12, 18]
        if sampler == "ddim":
            expected = independent_ddim(model, noise, original, (0, 6, 12, 18))
            np.testing.assert_array_equal(
                read_pixels(root, manifest), quantize_image(expected[0], plan.quantization)
            )

            parts = [tmp_path / "rank0", tmp_path / "rank1"]
            for rank, part in enumerate(parts):
                generate_image_shard(store, artifact.id, plan, part, rank=rank, world_size=2)
            merged = merge_image_shards(parts, plan, tmp_path / "merged")
            assert merged.samples == manifest.samples
            assert read_json(tmp_path / "merged" / "generation.json")["sampling_binding"] == binding


def test_respacing_preserves_nonidentity_training_model_times_and_exact_full_chain(tmp_path):
    torch.manual_seed(122)
    schedule = DiffusionSchedule(
        [0.002, 0.003, 0.006, 0.02, 0.03, 0.06], timestep_map=[2, 5, 9, 14, 19, 22]
    )
    model, objective = UNet2D(config()), DiffusionObjective(schedule)
    engine = Trainer(model, objective, lr=0.001)
    engine.step([{"sample": torch.randn(2, 3, 4, 4)}])
    model.save_pretrained(tmp_path / "model")
    atomic_json(tmp_path / "model" / "objective.json", objective.config_dict())
    store = ArtifactStore(tmp_path / "store")
    artifact = store.publish(
        tmp_path / "model", kind="native_field", metadata={"evidence": "actual_fixture_update"}
    )
    plan = ImageSamplingPlan(
        (GenerationCase("x", 111),), (3, 4, 4), sampler="ddim", steps=3, respacing_indices=(1, 3, 5)
    )
    output = tmp_path / "out"
    manifest = generate_image_shard(store, artifact.id, plan, output)
    info = read_json(output / "shard.json")["sampling_binding"]["diffusion"]
    assert info["effective_model_times"] == [5, 14, 22]
    torch.testing.assert_close(
        torch.tensor(info["effective_alpha_bar"], dtype=torch.float64),
        schedule.alpha_bar[[1, 3, 5]],
    )
    noise = torch.randn((1, 3, 4, 4), generator=torch.Generator().manual_seed(111))
    expected = independent_ddim(model.eval(), noise, schedule, (1, 3, 5))
    np.testing.assert_array_equal(
        read_pixels(output, manifest), quantize_image(expected[0], plan.quantization)
    )
    loaded, layout = load_native_artifact_model(artifact)
    full, binding = resolve_image_sampling(
        artifact, loaded, layout, replace(plan, steps=6, respacing_indices=None)
    )
    assert torch.equal(full.betas, schedule.betas) and binding["diffusion"][
        "effective_model_times"
    ] == [2, 5, 9, 14, 19, 22]


def test_sampling_rejects_missing_schedule_ambiguous_layout_and_parameter_mismatch(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    model = UNet2D(config())
    model.save_pretrained(tmp_path / "bare")
    bare = store.publish(tmp_path / "bare", kind="native_field", metadata={})
    plan = ImageSamplingPlan((GenerationCase("x", 2),), (3, 4, 4), sampler="ddim", steps=3)
    with pytest.raises(ValueError, match="original betas"):
        generate_image_shard(store, bare.id, plan, tmp_path / "missing-schedule")
    assert not (tmp_path / "missing-schedule").exists()
    objective = DiffusionObjective(DiffusionSchedule.create(11))
    atomic_json(tmp_path / "bare" / "objective.json", objective.config_dict())
    valid = store.publish(tmp_path / "bare", kind="native_field", metadata={})
    for changed in (
        replace(plan, learned_variance=True),
        replace(plan, steps=12),
        replace(plan, respacing_indices=(0, 1, 12)),
    ):
        with pytest.raises(ValueError):
            generate_image_shard(store, valid.id, changed, tmp_path / "invalid")
    for indices in ((0.0, 1, 2), (0, 1, 1), (2, 1, 0)):
        with pytest.raises(ValueError):
            replace(plan, respacing_indices=indices)
    model.save_pretrained(tmp_path / "bare" / "model")
    ambiguous = store.publish(tmp_path / "bare", kind="native_field", metadata={})
    with pytest.raises(ValueError, match="exactly one"):
        generate_image_shard(store, ambiguous.id, plan, tmp_path / "ambiguous")


def make_dmd(sigma_data=0.7):
    generator = UNet2D(config("x0"))
    real, fake = UNet2D(config("edm_residual")), UNet2D(config("edm_residual"))
    engine = Trainer(generator, lr=0.001)
    return engine, DMDMethod(engine, real, fake, generator_time=0.625, sigma_data=sigma_data)


def test_real_dmd_update_checkpoint_publish_and_one_forward_per_png(tmp_path, monkeypatch):
    torch.manual_seed(20)
    store = ArtifactStore(tmp_path / "store")
    engine, method = make_dmd()
    with pytest.raises(ValueError, match="untrained"):
        publish_dmd_generator(method, store, tmp_path / "untrained")
    teacher = deepcopy(method.real_score.state_dict())
    result = method.update([{"noise": torch.randn(2, 3, 4, 4), "sigma": torch.tensor([0.4, 0.9])}])
    assert result["generator"].updated and result["fake_score"][0].updated
    for name, value in teacher.items():
        torch.testing.assert_close(value, method.real_score.state_dict()[name], atol=0, rtol=0)
    engine.save_checkpoint(tmp_path / "checkpoint")
    restored_engine, restored_method = make_dmd()
    restored_engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    assert restored_method.updates == method.updates == 1
    rng = torch.get_rng_state().clone()
    artifact = publish_dmd_generator(restored_method, store, tmp_path / "export")
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    contract = read_json(artifact.path / "generation_contract.json")
    assert contract["generator_time"] == 0.625 and contract["training"]["sigma_data"] == 0.7
    assert set(contract["training"]["role_weight_fingerprints"]) == {
        "model",
        "real_score",
        "fake_score",
    }
    reloaded = load_model(artifact.path / "model")
    for key, tensor in restored_engine.export_state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], tensor, atol=0, rtol=0)
    calls = []
    real_loader = load_native_artifact_model

    def counted_loader(item):
        model, path = real_loader(item)
        model.register_forward_pre_hook(
            lambda module, args: calls.append(args[1].detach().tolist())
        )
        return model, path

    monkeypatch.setattr("aster.evaluation.generative.load_native_artifact_model", counted_loader)
    plan = ImageSamplingPlan(
        (GenerationCase("a", 99), GenerationCase("b", 100)), (3, 4, 4), sampler="direct_x0", steps=1
    )
    manifest = generate_image_shard(store, artifact.id, plan, tmp_path / "images")
    manifest.verify(tmp_path / "images")
    assert calls == [[0.625], [0.625]]
    noise = torch.randn((1, 3, 4, 4), generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        expected = reloaded.eval()(noise, torch.tensor([0.625])).prediction
    np.testing.assert_array_equal(
        read_pixels(tmp_path / "images", manifest), quantize_image(expected[0], plan.quantization)
    )
    assert replace(plan, sampler="flow_euler", steps=5).cohort_id == plan.cohort_id
    with pytest.raises(ValueError, match="reinterpret"):
        generate_image_shard(
            store, artifact.id, replace(plan, sampler="ddim", steps=2), tmp_path / "wrong-solver"
        )
    for change in ({"steps": 2}, {"guidance_scale": 3.0}, {"eta": 1.0}, {"clip_clean": True}):
        with pytest.raises(ValueError):
            replace(plan, **change)

    with torch.no_grad():
        reloaded.output[-1].weight.add_(0.01)
    reloaded.save_pretrained(tmp_path / "rebound" / "model")
    atomic_json(tmp_path / "rebound" / "generation_contract.json", contract)
    rebound = store.publish(tmp_path / "rebound", kind="native_direct_generator", metadata={})
    with pytest.raises(ValueError, match="weights differ"):
        generate_image_shard(store, rebound.id, plan, tmp_path / "rebound-images")


def test_direct_generator_cannot_guess_time_or_accept_drifting_semantics(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    model = UNet2D(config("x0"))
    model.save_pretrained(tmp_path / "model")
    artifact = store.publish(tmp_path / "model", kind="native_field", metadata={})
    plan = ImageSamplingPlan((GenerationCase("a", 2),), (3, 4, 4), sampler="direct_x0", steps=1)
    with pytest.raises(ValueError, match="generation contract"):
        generate_image_shard(store, artifact.id, plan, tmp_path / "no-contract")
    atomic_json(
        tmp_path / "model" / "generation_contract.json",
        {
            "schema_version": 1,
            "method": "drifting",
            "prediction_type": "x0",
            "generator_time": 1.0,
            "training": {"generator_updates": 1},
        },
    )
    wrong = store.publish(tmp_path / "model", kind="native_field", metadata={})
    with pytest.raises(ValueError, match="Unsupported direct"):
        generate_image_shard(store, wrong.id, plan, tmp_path / "wrong-contract")


def test_bound_flow_direction_cannot_be_changed_at_sampling(tmp_path):
    from aster.methods.generation import FlowObjective, FlowPath

    store = ArtifactStore(tmp_path / "store")
    UNet2D(config("velocity")).save_pretrained(tmp_path / "model")
    objective = FlowObjective(FlowPath(direction="data_to_noise"))
    atomic_json(tmp_path / "model" / "objective.json", objective.config_dict())
    artifact = store.publish(tmp_path / "model", kind="native_field", metadata={})
    plan = ImageSamplingPlan((GenerationCase("a", 2),), (3, 4, 4), steps=2)
    with pytest.raises(ValueError, match="direction"):
        generate_image_shard(store, artifact.id, plan, tmp_path / "incorrect")
    generate_image_shard(
        store, artifact.id, replace(plan, flow_direction="data_to_noise"), tmp_path / "correct"
    ).verify(tmp_path / "correct")


@pytest.mark.parametrize("failure", ["invalid_sigma", "nonfinite_gradient"])
def test_dmd_partial_round_cannot_checkpoint_or_publish_and_recovers_exactly(tmp_path, failure):
    torch.manual_seed(24)
    engine, method = make_dmd()
    batch = {"noise": torch.randn(2, 3, 4, 4), "sigma": torch.tensor([0.4, 0.8])}
    method.update([batch])
    checkpoint = engine.save_checkpoint(tmp_path / "complete")
    before = {name: deepcopy(role.model.state_dict()) for name, role in engine.roles.items()}
    bad_batch = dict(batch)
    hook = None
    if failure == "invalid_sigma":
        bad_batch["sigma"] = torch.zeros(2)
    else:
        hook = engine.model.output[-1].weight.register_hook(lambda grad: grad * float("nan"))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            method.update([bad_batch])
    finally:
        if hook is not None:
            hook.remove()
    assert (
        method.updates == 1
        and engine.roles["model"].updates == 1
        and engine.roles["fake_score"].updates == 2
    )
    assert any(
        not torch.equal(value, engine.roles["fake_score"].model.state_dict()[name])
        for name, value in before["fake_score"].items()
    )
    if failure == "nonfinite_gradient":
        assert not engine._failed
    with pytest.raises(ValueError):
        engine.save_checkpoint(tmp_path / "half-round")
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_dmd_generator(method, ArtifactStore(tmp_path / "store"), tmp_path / "half-export")
    with pytest.raises(RuntimeError, match="incomplete"):
        method.update([batch])
    assert not (tmp_path / "half-round").exists() and not (tmp_path / "half-export").exists()

    reference_engine, reference_method = make_dmd()
    reference_engine.load_checkpoint(checkpoint, trusted=True)
    expected_result = reference_method.update([batch])
    expected = {
        name: deepcopy(role.model.state_dict()) for name, role in reference_engine.roles.items()
    }
    engine.load_checkpoint(checkpoint, trusted=True)
    assert not method._round_in_progress and method.updates == 1
    for role, values in before.items():
        for name, tensor in values.items():
            torch.testing.assert_close(
                engine.roles[role].model.state_dict()[name], tensor, atol=0, rtol=0
            )
    result = method.update([batch])
    assert result["generator"].loss == expected_result["generator"].loss and method.updates == 2
    for role, values in expected.items():
        for name, tensor in values.items():
            torch.testing.assert_close(
                engine.roles[role].model.state_dict()[name], tensor, atol=0, rtol=0
            )


def test_dmd_resume_rejects_score_scale_and_fake_distribution_change(tmp_path):
    engine, method = make_dmd()
    engine.save_checkpoint(tmp_path / "checkpoint")
    changed_engine, changed_method = make_dmd(sigma_data=0.8)
    with pytest.raises(ValueError, match="settings changed"):
        changed_engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    changed_engine, changed_method = make_dmd()
    changed_method.fake_objective.log_mean = -0.5
    with pytest.raises(ValueError, match="objective settings changed"):
        changed_engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
