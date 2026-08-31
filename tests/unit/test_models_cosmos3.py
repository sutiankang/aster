from dataclasses import replace
import torch
import pytest

from aster.models.cosmos3 import Cosmos3Config, Cosmos3MoT, Cosmos3Vision, Cosmos3Sequence


def inputs(c, batch=2):
    ids = torch.tensor([[1, 3, 5, 2]]).expand(batch, -1).clone()

    def positions(length, offset):
        x = torch.arange(length).float()[None].expand(batch, -1) + offset
        return torch.stack((x, x * 0.3, x * 0.1))

    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        vision=Cosmos3Vision(
            torch.randn(batch, c.latent_channel, 2, 3, 4),
            positions(8, 15004),
            torch.tensor([[0.0, 630.0]]).expand(batch, -1),
            torch.tensor([[False, True]]).expand(batch, -1),
        ),
        sound=Cosmos3Sequence(
            torch.randn(batch, 3, c.sound_dim),
            positions(3, 15004),
            torch.full((batch, 3), 630.0),
            torch.ones(batch, 3, dtype=torch.bool),
        ),
        action=Cosmos3Sequence(
            torch.randn(batch, 2, c.action_dim),
            positions(2, 15004),
            torch.full((batch, 2), 630.0),
            torch.ones(batch, 2, dtype=torch.bool),
            domain_ids=torch.arange(batch),
        ),
    )


@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_models_cosmos3_joint_modal_attention_and_understanding_cache(activation):
    torch.set_num_threads(1)
    torch.manual_seed(541)
    c = Cosmos3Config(
        hidden_act=activation,
        qk_norm_for_text=activation == "silu",
        use_und_k_norm_for_gen=activation == "relu2",
    )
    model = Cosmos3MoT(c).eval()
    data = inputs(c)
    output = model(**data)
    assert output.vision.prediction.shape == data["vision"].sample.shape
    assert torch.count_nonzero(output.vision.prediction[:, :, 0]) == 0
    assert output.sound.prediction.shape == data["sound"].sample.shape
    assert output.action.prediction.shape == data["action"].sample.shape
    changed = {**data, "action": replace(data["action"], sample=data["action"].sample + 2)}
    other = model(**changed)
    torch.testing.assert_close(output.text.logits, other.text.logits, atol=0, rtol=0)
    assert not torch.allclose(output.vision.prediction, other.vision.prediction)
    prefix = model.forward_text(data["input_ids"], use_cache=True)
    cached = model(
        vision=data["vision"], sound=data["sound"], action=data["action"], state=prefix.state
    )
    for name in ("vision", "sound", "action"):
        torch.testing.assert_close(
            getattr(cached, name).prediction, getattr(output, name).prediction, atol=2e-6, rtol=3e-5
        )
    next_ids = torch.tensor([[7, 9]]).expand(2, -1)
    expected = model.forward_text(torch.cat((data["input_ids"], next_ids), 1)).logits[:, -2:]
    actual = model.forward_text(next_ids, state=prefix.state).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    with pytest.raises(ValueError, match="snapshot"):
        prefix.state.truncate(1)
    with pytest.raises(ValueError, match="layer count"):
        model.forward_text(next_ids, state=replace(prefix.state, layers=prefix.state.layers[:-1]))
    with pytest.raises(ValueError, match="aligned"):
        broken = tuple(tuple(value[:, :, :-1] for value in layer) for layer in prefix.state.layers)
        model.forward_text(next_ids, state=replace(prefix.state, layers=broken))


def test_models_cosmos3_masks_positions_domain_and_causal_guards():
    torch.set_num_threads(1)
    torch.manual_seed(542)
    c = Cosmos3Config()
    model = Cosmos3MoT(c)
    data = inputs(c)
    first = model(**data)

    times = data["vision"].timesteps.clone()
    times[:, 0] = 999
    other = model(**{**data, "vision": replace(data["vision"], timesteps=times)})
    torch.testing.assert_close(first.vision.prediction, other.vision.prediction, atol=0, rtol=0)
    with pytest.raises(ValueError, match="domain"):
        model(**{**data, "action": replace(data["action"], domain_ids=None)})
    with pytest.raises(ValueError, match="positions"):
        model(**{**data, "sound": replace(data["sound"], positions=None)})
    ids = data["input_ids"].clone()
    ids[:, -1] = 11
    torch.testing.assert_close(
        model.forward_text(ids).logits[:, :-1], first.text.logits[:, :-1], atol=0, rtol=0
    )


@pytest.mark.parametrize("stage", [0, 3])
def test_models_cosmos3_joint_train_export_random_resume_and_joint_sampler(tmp_path, stage):
    from aster.models import build_model, load_model
    from aster.methods.cosmos3 import Cosmos3FlowObjective, sample_cosmos3
    from aster.training import Trainer

    torch.set_num_threads(1)
    torch.manual_seed(543)
    c = Cosmos3Config()
    model = build_model(c)
    data = inputs(c)
    batch = dict(
        model_inputs=data,
        labels=data["input_ids"],
        noise={name: torch.randn_like(data[name].sample) for name in ("vision", "sound", "action")},
    )
    trainer = Trainer(
        model,
        Cosmos3FlowObjective(text_weight=0.2, time_distribution="provided"),
        lr=0.003,
        zero_stage=stage,
    )
    first = trainer.step([batch]).loss
    for _ in range(19):
        last = trainer.step([batch]).loss
    assert last < first * 0.65
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert actual.loss == expected.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    export = build_model(c)
    export.load_state_dict(weights, strict=True)
    export.eval()
    export.save_pretrained(tmp_path / "model")
    restored = load_model(tmp_path / "model").eval()
    torch.testing.assert_close(
        export(**data).action.prediction, restored(**data).action.prediction, atol=0, rtol=0
    )
    sampled = sample_cosmos3(restored, data, steps=2, shift=2)
    repeated = sample_cosmos3(restored, data, steps=2, shift=2, reuse_understanding=False)
    for name in sampled:
        torch.testing.assert_close(sampled[name], repeated[name], atol=3e-6, rtol=4e-5)
    torch.testing.assert_close(
        sampled["vision"][:, :, 0], data["vision"].sample[:, :, 0], atol=0, rtol=0
    )

    stochastic = Trainer(
        build_model(c),
        Cosmos3FlowObjective(time_distribution="logit_normal"),
        lr=0.001,
        zero_stage=stage,
    )
    random_batch = {"model_inputs": data}
    stochastic.step([random_batch])
    stochastic.save_checkpoint(tmp_path / "random")
    expected = stochastic.step([random_batch])
    values = stochastic.export_state_dict()
    stochastic.load_checkpoint(tmp_path / "random", trusted=True)
    actual = stochastic.step([random_batch])
    assert expected.loss == actual.loss
    for name, value in stochastic.export_state_dict().items():
        torch.testing.assert_close(value, values[name], atol=0, rtol=0)


def test_models_cosmos3_shared_time_coordinates_and_conditioned_loss_counts():
    from aster.models.cosmos3 import cosmos3_positions
    from aster.methods.cosmos3 import Cosmos3FlowObjective

    video = cosmos3_positions((7, 1, 1), fps=24, temporal_compression=4, temporal_offset=15004)
    sound = cosmos3_positions(
        (26, 1, 1),
        fps=25,
        temporal_compression=1,
        base_temporal_compression=4,
        temporal_offset=15004,
    )
    assert video[0, 0, -1] == sound[0, 0, -1] == 15010
    torch.manual_seed(544)
    c = Cosmos3Config()
    data = inputs(c)
    bundle = Cosmos3FlowObjective(time_distribution="provided")(
        Cosmos3MoT(c), {"model_inputs": data}
    )
    counts = {term.name: int(term.denominator) for term in bundle.terms}
    assert counts == dict(
        vision_flow=2 * 2 * 1 * 3 * 4, sound_flow=2 * 3 * 4, action_flow=2 * 2 * 3
    )
