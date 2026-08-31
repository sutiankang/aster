from dataclasses import replace
import pytest
import torch
from aster.models import KimiK3TextConfig, Qwen35TextConfig, build_model, load_model
from aster.methods import CrossEntropyObjective
from aster.training import Trainer


@pytest.mark.parametrize("stage", [0, 3])
def test_models_k3_train_export_and_exact_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(312)
    config = KimiK3TextConfig()
    model = build_model(config)
    objective = CrossEntropyObjective()
    batch = {"input_ids": torch.tensor([[1, 3, 5, 7, 9, 2], [1, 4, 6, 8, 10, 2]])}
    batch["labels"] = batch["input_ids"]
    trainer = Trainer(model, objective, lr=0.005, zero_stage=stage)
    initial = trainer.step([batch]).loss
    for _ in range(18):
        final = trainer.step([batch]).loss
    assert final < 0.5 * initial
    trainer.save_checkpoint(tmp_path / "resume")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "resume", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, value in trainer.export_state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    native = build_model(config)
    native.load_state_dict(weights, strict=True)
    native.eval()
    native.save_pretrained(tmp_path / "model")
    torch.testing.assert_close(
        load_model(tmp_path / "model").eval()(batch["input_ids"]).logits,
        native(batch["input_ids"]).logits,
        atol=0,
        rtol=0,
    )


def test_models_k3_padding_cache_fork_and_nope():
    torch.set_num_threads(1)
    torch.manual_seed(313)
    model = build_model(KimiK3TextConfig()).eval()
    tokens = torch.tensor([[0, 1, 3, 6, 8, 2], [1, 4, 7, 2, 0, 0]])
    mask = tokens.ne(0)
    full = model(tokens, attention_mask=mask).logits
    first = model(tokens[:, :3], attention_mask=mask[:, :3], use_cache=True)
    final = model(tokens[:, 3:], attention_mask=mask, state=first.state, use_cache=True)
    torch.testing.assert_close(
        full[:, 3:][mask[:, 3:]], final.logits[mask[:, 3:]], atol=3e-6, rtol=3e-5
    )
    for row in range(2):
        compact = model(tokens[row, mask[row]][None]).logits[0]
        torch.testing.assert_close(full[row, mask[row]], compact, atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match="snapshot"):
        first.state.truncate(1)
    saved = first.state.fork()
    saved.layers[0][1].zero_()
    assert first.state.layers[0][1].abs().sum() > 0
    reordered = first.state.reorder(torch.tensor([1, 0]))
    back = model(tokens.flip(0)[:, 3:], attention_mask=mask.flip(0), state=reordered)

    torch.testing.assert_close(back.logits, final.logits.flip(0), atol=3e-6, rtol=3e-5)
    arbitrary_positions = torch.arange(20, 26)[None].expand(2, -1)
    torch.testing.assert_close(
        full,
        model(tokens, attention_mask=mask, position_ids=arbitrary_positions).logits,
        atol=0,
        rtol=0,
    )
    wrong = build_model(Qwen35TextConfig())(tokens[:, :3], use_cache=True).state
    with pytest.raises(ValueError, match="state/config"):
        model(tokens[:, 3:], state=wrong)


def test_models_k3_rejects_unimplemented_aliases():
    with pytest.raises(ValueError, match="KDA/MLA"):
        KimiK3TextConfig(layer_types=("linear_attention",) * 4)
    with pytest.raises(ValueError, match="NoPE"):
        KimiK3TextConfig(mla_use_nope=False)
    with pytest.raises(ValueError, match="normalization"):
        KimiK3TextConfig(latent_moe_use_norm=False)
    with pytest.raises(ValueError, match="constants"):
        KimiK3TextConfig(gate_lower_bound=0)
