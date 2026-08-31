from dataclasses import replace
import pytest
import torch
from aster.models import Qwen4ExpTextConfig, build_model, load_model
from aster.models.qwen4_exp import NGramEmbedding, PLELayer
from aster.methods import CrossEntropyObjective
from aster.training import Trainer


@pytest.mark.parametrize("stage", [0, 3])
def test_models_qwen4exp_training_export_exact_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(401)
    config = Qwen4ExpTextConfig()
    model = build_model(config)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 2], [1, 4, 6, 8, 10, 12, 2]])
    batch = dict(input_ids=ids, labels=ids)
    trainer = Trainer(model, CrossEntropyObjective(), lr=0.006, zero_stage=stage)
    initial = trainer.step([batch]).loss
    for _ in range(18):
        final = trainer.step([batch]).loss
    assert final < 0.5 * initial
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    deployed = build_model(config)
    deployed.load_state_dict(weights, strict=True)
    deployed.eval()
    deployed.save_pretrained(tmp_path / "export")
    torch.testing.assert_close(
        deployed(ids).logits, load_model(tmp_path / "export").eval()(ids).logits, atol=0, rtol=0
    )


def test_models_qwen4exp_real_qsa_ple_cache_and_3axis_positions():
    torch.set_num_threads(1)
    torch.manual_seed(402)
    model = build_model(Qwen4ExpTextConfig()).eval()
    ids = torch.tensor([[0, 1, 3, 2, 5, 7, 9, 11, 2], [1, 4, 6, 8, 10, 12, 13, 2, 0]])
    mask = ids.ne(0)
    positions = torch.arange(9)[None, None].expand(3, 2, -1).clone()
    positions[1, :, 2:] += 3
    positions[2, :, 3:] += 5
    full = model(ids, position_ids=positions, attention_mask=mask, output_hidden_states=True)
    assert full.hidden_states[0].shape[-1] == 4 * model.config.hidden_size
    assert full.hidden_states[-1].shape[-1] == model.config.hidden_size
    records = full.auxiliary["qsa_indexer"][-1]
    assert any(
        len(blocks) > model.config.indexer_budget // model.config.indexer_compress_ratio
        for _, _, blocks, _ in records
    )
    state = None
    parts = []
    for start, end in ((0, 1), (1, 4), (4, 5), (5, 9)):
        value = model(
            ids[:, start:end],
            position_ids=positions[:, :, start:end],
            attention_mask=mask[:, :end],
            state=state,
            use_cache=True,
        )
        parts.append(value.logits)
        state = value.state
    torch.testing.assert_close(full.logits, torch.cat(parts, 1), atol=4e-6, rtol=5e-5)
    assert state.position_ids.shape == (3, 2, 9)
    assert len(state.layers[-1].attention) == 3
    assert state.layers[1].ple_tokens.shape == (2, 2)
    with pytest.raises(ValueError, match="snapshot"):
        state.truncate(4)
    clone = state.fork()
    clone.layers[0].attention[1].zero_()
    assert state.layers[0].attention[1].abs().sum() > 0
    suffix = torch.tensor([[3], [4]])
    left = model(
        suffix, state=state, attention_mask=torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), 1)
    ).logits
    right = model(
        suffix.flip(0),
        state=state.reorder(torch.tensor([1, 0])),
        attention_mask=torch.cat((mask.flip(0), torch.ones(2, 1, dtype=torch.bool)), 1),
    ).logits
    torch.testing.assert_close(left.flip(0), right, atol=4e-6, rtol=5e-5)


def test_models_qwen4exp_ngram_eos_isolation_and_ple_history():
    torch.manual_seed(403)
    c = Qwen4ExpTextConfig()
    table = NGramEmbedding(c, 0)
    tokens = torch.tensor([[4, 6, 2, 9, 11, 2]])
    a, _ = table.lookup_ids(tokens)
    b, _ = table.lookup_ids(tokens[:, 3:])
    torch.testing.assert_close(a[:, 3:], b, atol=0, rtol=0)
    first, history = table.lookup_ids(tokens[:, :4])
    second, _ = table.lookup_ids(tokens[:, 4:], history)
    torch.testing.assert_close(a, torch.cat((first, second), 1), atol=0, rtol=0)
    layer = PLELayer(c, 0)

    nn = torch.nn
    nn.init.normal_(layer.conv1d.weight, std=0.03)
    hidden = torch.randn(1, 6, c.hidden_size * c.hc_count)
    whole, _, _ = layer(hidden, tokens)
    left, conv, context = layer(hidden[:, :2], tokens[:, :2])
    right, _, _ = layer(hidden[:, 2:], tokens[:, 2:], conv, context)
    torch.testing.assert_close(whole, torch.cat((left, right), 1), atol=3e-6, rtol=3e-5)


def test_models_qwen4exp_explicit_ple_ids_and_invalid_configs():
    c = Qwen4ExpTextConfig()
    model = build_model(c)
    ids = torch.tensor([[1, 3, 5, 2]])
    embedded = model.get_input_embeddings()(ids)
    with pytest.raises(ValueError, match="PLE needs explicit"):
        model(inputs_embeds=embedded)
    torch.testing.assert_close(
        model(ids).logits, model(inputs_embeds=embedded, ple_input_ids=ids).logits, atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="one-indexed"):
        replace(c, ple_layer_ids=(4,))
    with pytest.raises(ValueError, match="whole-microblock"):
        replace(c, indexer_budget=3)
    with pytest.raises(ValueError, match="fit both"):
        replace(c, indexer_head_dim=4)
    with pytest.raises(ValueError, match="GDN/QSA"):
        replace(c, layer_types=("full_attention",) * 4)
    from aster.methods.sparse_indexer import QSAIndexerObjective

    with pytest.raises(ValueError, match="layer records"):
        QSAIndexerObjective((0,))(
            model, dict(input_ids=ids, qsa_teacher_attention={0: torch.ones(1, 4, 4)})
        )


@pytest.mark.parametrize("stage", [0, 3])
def test_models_qwen4exp_combined_indexer_kd_trainer_resume(tmp_path, stage):
    from aster.methods.sparse_indexer import QSAIndexerObjective

    torch.set_num_threads(1)
    torch.manual_seed(404)
    c = Qwen4ExpTextConfig()
    model = build_model(c)
    initial = model.state_dict()["model.layers.3.self_attn.indexer.index_qk_proj.weight"].clone()
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 2], [1, 4, 6, 8, 10, 12, 14, 2]])
    teacher = torch.rand(2, 3, 8, 8)
    batch = dict(input_ids=ids, labels=ids, qsa_teacher_attention={3: teacher})
    trainer = Trainer(
        model, QSAIndexerObjective((3,), indexer_weight=0.4), lr=0.004, zero_stage=stage
    )
    trainer.step([batch])
    current = trainer.export_state_dict()["model.layers.3.self_attn.indexer.index_qk_proj.weight"]
    assert not torch.equal(current, initial)
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, x in trainer.export_state_dict().items():
        torch.testing.assert_close(x, weights[name], atol=0, rtol=0)
