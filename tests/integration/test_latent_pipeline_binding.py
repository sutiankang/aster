from dataclasses import replace
import copy

import pytest
import torch

from aster.core import ArtifactStore, atomic_json, digest_json, read_json
from aster.models.generative import AutoencoderKL, AutoencoderConfig, UNet2D, UNetConfig
from aster.methods.generation import (
    DiffusionObjective,
    DiffusionSchedule,
    EDMObjective,
    sample_diffusion,
)
from aster.pipelines import LatentFieldObjective, LatentGenerationPipeline, LatentPipelineConfig
from aster.training import Trainer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def components(kind="epsilon", learned=False):
    vae = AutoencoderKL(
        AutoencoderConfig(
            in_channels=3,
            latent_channels=2,
            base_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            scaling_factor=0.5,
            shift_factor=0.25,
        )
    )
    field = UNet2D(
        UNetConfig(
            in_channels=2,
            out_channels=4 if learned else 2,
            model_channels=4,
            channel_mult=(1,),
            num_res_blocks=1,
            attention_levels=(),
            num_heads=1,
            prediction_type=kind,
        )
    )
    return vae, field


def chain():
    return DiffusionSchedule(
        [0.01, 0.04, 0.09, 0.15, 0.22, 0.3], timestep_map=[2, 5, 9, 18, 27, 60]
    )


@pytest.mark.parametrize("solver,learned", [("ddim", False), ("ddpm", True)])
def test_caller_explicit_original_chain_clone_respacing_and_restore(
    tmp_path, monkeypatch, solver, learned
):
    torch.manual_seed(285)
    vae, field = components(learned=learned)
    original = chain()
    config = LatentPipelineConfig(
        method="diffusion",
        solver=solver,
        steps=3,
        diffusion_steps=6,
        learned_variance=learned,
        eta=0.4 if solver == "ddim" else 0.0,
        clip_clean=True,
    )
    pipeline = LatentGenerationPipeline(vae, field, config, diffusion_schedule=original)
    assert pipeline.sampling_binding["source"] == {
        "kind": "caller_explicit_schedule",
        "training_semantics_bound": False,
    }
    assert pipeline.sampling_binding["selected_training_indices"] == [0, 2, 5]
    noise = torch.randn(1, 2, 4, 4)
    times = []
    hook = field.register_forward_pre_hook(lambda module, args: times.append(int(args[1][0])))
    output = pipeline.sample(noise, generator=torch.Generator().manual_seed(3))
    hook.remove()
    assert times == [60, 9, 2]
    with torch.no_grad():
        expected = vae.decode(
            sample_diffusion(
                field,
                noise,
                original.respaced([0, 2, 5]),
                method=solver,
                learned_variance=learned,
                eta=config.eta,
                clip_clean=True,
                generator=torch.Generator().manual_seed(3),
            ),
            scaled=True,
        )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    original.betas.fill_(0.5)
    original.timestep_map.add_(100)
    record = pipeline.sampling_binding
    record["original_schedule"]["betas"][0] = 0.9
    torch.testing.assert_close(
        pipeline.sample(noise, generator=torch.Generator().manual_seed(3)), output, atol=0, rtol=0
    )
    pipeline.save_pretrained(tmp_path / "pipeline")
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise AssertionError("Must not reconstruct original chain from a name")

        patch.setattr(DiffusionSchedule, "create", fail)
        restored = LatentGenerationPipeline.from_pretrained(tmp_path / "pipeline")
        torch.testing.assert_close(
            restored.sample(noise, generator=torch.Generator().manual_seed(3)),
            output,
            atol=0,
            rtol=0,
        )
    assert restored.sampling_binding == pipeline.sampling_binding


def trained_artifacts(tmp_path, *, kind="diffusion", learned=False):
    torch.manual_seed(929)
    store = ArtifactStore(tmp_path / "store")
    vae, field = components("edm_residual" if kind == "edm" else "epsilon", learned=learned)
    vae.save_pretrained(tmp_path / "vae")
    va = store.publish(tmp_path / "vae", kind="native_vae_fixture", metadata={})
    inner = (
        EDMObjective(sigma_data=0.7)
        if kind == "edm"
        else DiffusionObjective(chain(), learned_variance=learned)
    )
    objective = LatentFieldObjective(vae, inner, encoder_identity=va.id, sample_posterior=False)
    engine = Trainer(field, objective, lr=0.002)
    engine.add_role("autoencoder", vae, trainable=False)
    batch = {"pixels": torch.randn(2, 3, 4, 4)}
    assert engine.step([batch]).updated
    saved = engine.save_checkpoint(tmp_path / "checkpoint")
    engine.step([batch])
    engine.load_checkpoint(saved, trusted=True)
    field.save_pretrained(tmp_path / "field" / "model")
    atomic_json(tmp_path / "field" / "objective.json", objective.config_dict())
    atomic_json(tmp_path / "field" / "successful_update.json", engine.last_successful_update())
    fa = store.publish(
        tmp_path / "field", kind="actual_latent_training_fixture", metadata={}, parents=(va.id,)
    )
    return store, va, fa, vae, field, objective


@pytest.mark.parametrize(
    "kind,learned", [("diffusion", False), ("diffusion", True), ("edm", False)]
)
def test_actual_training_artifact_schedule_and_encoder_identity_survive_restore(
    tmp_path, kind, learned
):
    store, va, fa, _, _, objective = trained_artifacts(tmp_path, kind=kind, learned=learned)
    config = LatentPipelineConfig(
        method=kind,
        solver="ddpm" if kind == "diffusion" else "heun",
        steps=3,
        diffusion_steps=6,
        sigma_data=0.7,
        sigma_max=2.0,
        learned_variance=learned,
    )
    rng = torch.get_rng_state().clone()
    pipeline = LatentGenerationPipeline.from_artifacts(store, va.id, fa.id, config)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert pipeline.sampling_binding["source"]["training_semantics_bound"] is True
    assert pipeline.sampling_binding["source"]["training_objective"] == objective.config_dict()
    noise = torch.randn(1, 2, 4, 4)
    output = pipeline.sample(noise, generator=torch.Generator().manual_seed(19))
    pipeline.save_pretrained(tmp_path / "pipeline")
    restored = LatentGenerationPipeline.from_pretrained(tmp_path / "pipeline")
    torch.testing.assert_close(
        restored.sample(noise, generator=torch.Generator().manual_seed(19)), output, atol=0, rtol=0
    )
    with torch.no_grad():
        next(restored.field.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="weights changed"):
        restored.sample(noise)
    with pytest.raises(ValueError, match="weights changed"):
        restored.save_pretrained(tmp_path / "bad")
    if kind == "edm":
        with pytest.raises(ValueError, match="EDM preconditioning"):
            LatentGenerationPipeline.from_artifacts(
                store, va.id, fa.id, replace(config, sigma_data=0.5)
            )
    else:
        with pytest.raises(ValueError, match="length"):
            LatentGenerationPipeline.from_artifacts(
                store, va.id, fa.id, replace(config, diffusion_steps=1000)
            )
        with pytest.raises(ValueError, match="variance"):
            LatentGenerationPipeline.from_artifacts(
                store, va.id, fa.id, replace(config, learned_variance=not learned)
            )


def test_reject_wrong_encoder_pixel_goal_tampered_schedule_and_legacy_claims(tmp_path):
    store, va, fa, vae, field, objective = trained_artifacts(tmp_path)
    config = LatentPipelineConfig(method="diffusion", solver="ddim", steps=3, diffusion_steps=6)
    vae.save_pretrained(tmp_path / "other-vae")
    other = store.publish(tmp_path / "other-vae", kind="different_provenance", metadata={})
    with pytest.raises(ValueError, match="encoder identity"):
        LatentGenerationPipeline.from_artifacts(store, other.id, fa.id, config)
    field.save_pretrained(tmp_path / "pixel-field" / "model")
    atomic_json(tmp_path / "pixel-field" / "objective.json", objective.objective.config_dict())
    pixel = store.publish(tmp_path / "pixel-field", kind="pixel_goal_fixture", metadata={})
    with pytest.raises(ValueError, match="latent encoder"):
        LatentGenerationPipeline.from_artifacts(store, va.id, pixel.id, config)

    field.save_pretrained(tmp_path / "legacy-field" / "model")
    atomic_json(tmp_path / "legacy-field" / "objective.json", objective.config_dict())
    legacy_field = store.publish(
        tmp_path / "legacy-field", kind="legacy_goal_declaration", metadata={}
    )
    with pytest.raises(ValueError, match="actual successful objective provenance"):
        LatentGenerationPipeline.from_artifacts(store, va.id, legacy_field.id, config)
    pipeline = LatentGenerationPipeline.from_artifacts(store, va.id, fa.id, config)
    pipeline.save_pretrained(tmp_path / "pipeline")
    metadata_path = tmp_path / "pipeline" / "pipeline.json"
    saved = read_json(metadata_path)
    for change in ("beta", "map", "indices", "source"):
        metadata = copy.deepcopy(saved)
        binding = metadata["sampling_binding"]
        if change == "beta":
            binding["original_schedule"]["betas"][0] *= 1.5
        elif change == "map":
            binding["original_schedule"]["timestep_map"][0] = 1
        elif change == "indices":
            binding["selected_training_indices"] = [0, 3, 5]
        else:
            binding["source"]["kind"] = "caller_config"
        binding["original_schedule_id"] = digest_json(binding["original_schedule"])
        metadata["sampling_binding_id"] = digest_json(binding)
        atomic_json(metadata_path, metadata)
        with pytest.raises(ValueError):
            LatentGenerationPipeline.from_pretrained(tmp_path / "pipeline")
    legacy = copy.deepcopy(saved)
    legacy["schema_version"] = 1
    atomic_json(metadata_path, legacy)
    with pytest.raises(ValueError, match="Legacy"):
        LatentGenerationPipeline.from_pretrained(tmp_path / "pipeline")


def test_strict_configuration_and_parameterization():
    for kwargs in (
        {"steps": True},
        {"shift": float("nan")},
        {"eta": float("nan")},
        {"sigma_data": False},
        {"sigma_min": 80.0},
    ):
        with pytest.raises(ValueError):
            LatentPipelineConfig(**kwargs)
    vae, field = components()
    with pytest.raises(ValueError, match="parameterization"):
        LatentGenerationPipeline(vae, field)
    with pytest.raises(ValueError, match="learned variance"):
        LatentGenerationPipeline(
            vae,
            field,
            LatentPipelineConfig(method="diffusion", solver="ddim", steps=3, learned_variance=True),
        )
