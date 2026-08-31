import pytest
import torch

from aster.models import build_model, MistralConfig, Qwen3NextConfig, Qwen3VLConfig
from aster.models.generative import UNetConfig, UNet2D, AutoencoderConfig, AutoencoderKL
from aster.models.actions import ACTPolicy, ACTConfig
from aster.models.world import RSSMWorldModel, RSSMConfig
from aster.models.vision import CLIPVisionModel, CLIPVisionConfig
from aster.models.qwen_vl import pack_qwen_pixels
from aster.nn import KVState
from aster.data.actions import ActionSpec, ActionNormalizer
from aster.inference import (
    FieldRunner,
    LatentRunner,
    ActionRunner,
    DynamicsRunner,
    EncoderRunner,
    StatefulTokenRunner,
    StateArchive,
    StateError,
    CacheCapacityError,
    SamplingConfig,
)


def test_native_field_latent_action_and_dynamics_contracts():
    torch.set_num_threads(1)
    torch.manual_seed(810)
    field = UNet2D(
        UNetConfig(
            model_channels=8, channel_mult=(1,), num_res_blocks=1, attention_levels=(), num_heads=1
        )
    )
    sample = torch.randn(2, 3, 4, 4)
    runner = FieldRunner(field, policy_artifact_id="unet:1", prediction_type="epsilon")
    torch.testing.assert_close(
        runner.predict(sample, torch.ones(2)).prediction,
        field.eval()(sample, torch.ones(2)).prediction,
    )
    with pytest.raises(ValueError, match="type differs"):
        FieldRunner(field, policy_artifact_id="wrong", prediction_type="velocity").predict(
            sample, torch.ones(2)
        )
    vae = AutoencoderKL(
        AutoencoderConfig(
            base_channels=8,
            latent_channels=2,
            channel_mult=(1,),
            num_res_blocks=1,
            scaling_factor=0.4,
            shift_factor=0.2,
        )
    )
    latent_runner = LatentRunner(vae, policy_artifact_id="vae:1")
    z = latent_runner.encode(sample)
    torch.testing.assert_close(z, vae.eval().latent(sample, sample=False))
    torch.testing.assert_close(latent_runner.decode(z), vae.decode(z, scaled=True))
    torch.testing.assert_close(
        latent_runner.encode(sample, sample=True, seed=7),
        latent_runner.encode(sample, sample=True, seed=7),
    )
    c = ACTConfig(
        proprio_dim=2,
        action_dim=2,
        vision_dim=4,
        hidden_size=8,
        latent_dim=2,
        horizon=3,
        num_heads=2,
        posterior_layers=1,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_size=16,
    )
    policy = ACTPolicy(c).eval()
    with torch.no_grad():
        policy.pad_head.weight.zero_()
        policy.pad_head.bias.fill_(-10)
    spec = ActionSpec(("x", "y"), ("m", "m"), "robot_base", "delta", 20.0, 2)
    normalizer = ActionNormalizer([1.0, 2.0], [0.1, 0.2], spec=spec)
    action_runner = ActionRunner(
        policy, policy_artifact_id="act:1", spec=spec, normalizer=normalizer
    )
    observation = {"proprio": torch.randn(2, 2), "vision_tokens": torch.randn(2, 4, 4)}
    result = action_runner.predict_chunk(observation)
    torch.testing.assert_close(
        result.actions, normalizer.denormalize(policy.predict_chunk(observation).actions)
    )
    assert result.valid.all() and result.spec == spec
    with torch.no_grad():
        action_runner.model.pad_head.bias.fill_(10)
    with pytest.raises(ValueError, match="padding"):
        action_runner.predict_chunk(observation)
    world = RSSMWorldModel(
        RSSMConfig(
            observation_dim=3,
            action_dim=2,
            deter_dim=8,
            stochastic_variables=2,
            classes=3,
            hidden_size=8,
            blocks=2,
            reward_bins=7,
        )
    ).eval()
    dynamics = DynamicsRunner(world, policy_artifact_id="rssm:1")
    obs, act, reset = (
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    reset[:, 0] = True
    actual = dynamics.observe(obs, act, reset)
    expected, _, final = world.observe(obs, act, reset, sample=False)
    torch.testing.assert_close(actual["state"].native_state.features, expected.features)
    imagined = dynamics.imagine(actual["final_state"], act)
    torch.testing.assert_close(
        imagined["state"].native_state.features, world.imagine(final, act, sample=False).features
    )
    with pytest.raises(StateError, match="another policy"):
        DynamicsRunner(world, policy_artifact_id="rssm:2").imagine(actual["final_state"], act)


def test_encoder_cache_real_features_key_content_tenant_and_bound():
    torch.set_num_threads(1)
    vision = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
        )
    ).eval()
    runner = EncoderRunner(
        vision, policy_artifact_id="vision:1", processor_id="crop:4x4", max_cache_bytes=1024
    )
    pixels = torch.randn(1, 3, 4, 4)
    one = runner.encode({"pixel_values": pixels}, tenant="a")
    torch.testing.assert_close(one.last_hidden_state, vision(pixels).last_hidden_state)
    one.last_hidden_state.zero_()
    two = runner.encode({"pixel_values": pixels}, tenant="a")
    assert two.last_hidden_state.abs().sum() > 0 and runner.cache_hits == 1 and runner.calls == 1
    runner.encode({"pixel_values": pixels}, tenant="b")
    runner.encode({"pixel_values": pixels + 1}, tenant="a")
    assert runner.calls == 3 and runner.cache_bytes <= runner.max_cache_bytes


@pytest.mark.parametrize(
    "config",
    [
        MistralConfig(sliding_window=2),
        Qwen3NextConfig(
            num_hidden_layers=2, layer_types=("linear_attention", "full_attention"), num_experts=0
        ),
    ],
)
def test_typed_snapshot_window_and_recurrent_decode_replay(config):
    torch.set_num_threads(1)
    model = build_model(config).eval()
    runner = StatefulTokenRunner(model, policy_artifact_id="native:1")
    ids = torch.tensor([[1, 3, 4, 5, 7]])
    prefix = runner.forward(ids[:, :-1])
    tail = runner.forward(ids[:, -1:], state=prefix.state)
    torch.testing.assert_close(tail.logits, model(ids).logits[:, -1:], atol=4e-6, rtol=4e-5)
    branch = runner.fork(prefix.state)
    torch.testing.assert_close(runner.forward(ids[:, -1:], state=branch).logits, tail.logits)
    torch.testing.assert_close(runner.replay(ids).logits, model(ids).logits)
    archive = StateArchive(max_bytes=100000)
    handle = archive.put(prefix.state.native_state, identity="native:1/tenant:a")
    native = archive.get(handle, identity="native:1/tenant:a")
    assert native.kind == prefix.state.native_state.kind
    torch.testing.assert_close(native.layers[0][0], prefix.state.native_state.layers[0][0])
    with pytest.raises(StateError):
        archive.get(handle, identity="native:2/tenant:a")
    if native.kind == "hybrid_delta":
        with pytest.raises(StateError, match="not implemented"):
            archive.put(native, identity="native:1", quantize=True)
    result = runner.generate(ids[0].tolist(), SamplingConfig(max_new_tokens=3, temperature=0))
    assert len(result.token_ids) == 3 and len(result.raw_model_logprobs) == 3


def test_qwen_vl_snapshot_keeps_per_request_rope_delta():
    torch.set_num_threads(1)
    c = Qwen3VLConfig()
    model = build_model(c).eval()
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 12), c.vision_config)
    tokens = torch.tensor([[1, 26] + [28] * 6 + [27, 3, 5]])
    types = torch.where(tokens == 28, 1, 0)
    runner = StatefulTokenRunner(model, policy_artifact_id="qwen-vl:1", processor_id="grid:2")
    prefix = runner.forward(
        tokens[:, :-1],
        modality_inputs={
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "mm_token_type_ids": types[:, :-1],
        },
    )
    archive = StateArchive(max_bytes=1000000)
    key = archive.put(prefix.state.native_state, identity="qwen-vl:1/grid:2/a")
    restored = archive.get(key, identity="qwen-vl:1/grid:2/a")
    torch.testing.assert_close(
        restored.rope_delta, prefix.state.native_state.rope_delta, atol=0, rtol=0
    )
    expected = model(
        tokens, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types
    ).logits[:, -1:]
    torch.testing.assert_close(
        runner.forward(tokens[:, -1:], state=prefix.state).logits, expected, atol=4e-6, rtol=4e-5
    )


def test_real_int8_state_storage_error_and_lru_eviction():
    tensors = (torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 8))
    state = KVState((tensors,), 8, "mla:1", "mla_latent")
    archive = StateArchive(max_bytes=100000)
    key = archive.put(state, identity="policy1", quantize=True)
    assert archive.stored_bytes < sum(t.numel() * t.element_size() for t in tensors)
    restored = archive.get(key, identity="policy1")
    for original, actual in zip(tensors, restored.layers[0]):
        bound = original.abs().amax(-1, keepdim=True) / 254 + 1e-6
        assert ((original - actual).abs() <= bound).all()
    bounded = StateArchive(max_bytes=archive.stored_bytes)
    old = bounded.put(state, identity="one", quantize=True)
    bounded.put(state, identity="two", quantize=True)
    with pytest.raises(StateError, match="evicted"):
        bounded.get(old, identity="one")
    with pytest.raises(CacheCapacityError):
        StateArchive(max_bytes=1).put(state, identity="one")


@pytest.mark.parametrize("family", ["mamba", "deepseek_v32", "deepseek_v4"])
def test_new_state_families_full_snapshot_archive_and_decode(family):
    from aster.models import MambaConfig, DeepSeekV32Config, DeepSeekV4Config
    from aster.inference import KVStateCodec

    torch.set_num_threads(1)
    config = {
        "mamba": MambaConfig(),
        "deepseek_v32": DeepSeekV32Config(),
        "deepseek_v4": DeepSeekV4Config(),
    }[family]
    model = build_model(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    runner = StatefulTokenRunner(model, policy_artifact_id=family + ":fixture")
    prefix = runner.forward(ids[:, :-1])
    tail = runner.forward(ids[:, -1:], state=prefix.state)
    with torch.no_grad():
        torch.testing.assert_close(tail.logits, model(ids).logits[:, -1:], atol=5e-6, rtol=5e-5)
    archive = StateArchive(max_bytes=10000000)
    handle = archive.put(prefix.state.native_state, identity=family + ":scope")
    restored = archive.get(handle, identity=family + ":scope")
    with torch.no_grad():
        torch.testing.assert_close(
            model(ids[:, -1:], state=restored).logits, tail.logits, atol=5e-6, rtol=5e-5
        )

    with pytest.raises(StateError):
        archive.put(restored, identity=family + ":scope", quantize=True)
    if family != "deepseek_v32":
        with pytest.raises(ValueError):
            KVStateCodec(kind=restored.kind)
        with pytest.raises(ValueError):
            restored.truncate(2)


def test_indexed_mla_pages_preserve_all_three_different_width_leaves_and_cow():
    from aster.models import DeepSeekV32Config
    from aster.inference import KVStateCodec, ModelRunner

    torch.set_num_threads(1)
    model = build_model(DeepSeekV32Config()).eval()
    codec = KVStateCodec(kind="indexed_mla")
    runner = ModelRunner(
        model, policy_artifact_id="dsa-fixture", codec=codec, block_size=2, max_blocks=16
    )
    source = runner.pool.create("dsa-fixture")
    runner.forward_batch([source], [[1, 2, 3]])
    branch = runner.pool.fork(source)
    logits = runner.forward_batch([branch], [[4]])[0]
    with torch.no_grad():
        torch.testing.assert_close(
            logits, model(torch.tensor([[1, 2, 3, 4]])).logits[0, -1], atol=5e-6, rtol=5e-5
        )
    original = runner.pool.materialize(source)
    grown = runner.pool.materialize(branch)
    assert len(original.layers[0]) == 3
    assert all(x.shape[-2] == 3 for x in original.layers[0])
    assert all(x.shape[-2] == 4 for x in grown.layers[0])
    runner.pool.truncate(branch, 2)
    assert all(x.shape[-2] == 2 for x in runner.pool.materialize(branch).layers[0])
    runner.pool.release(source)
    runner.pool.release(branch)
    assert runner.pool.used_blocks == 0
