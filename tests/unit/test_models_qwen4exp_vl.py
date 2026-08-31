from dataclasses import replace
import pytest
import torch
from aster.models import Qwen4ExpConfig, build_model, load_model
from aster.models.qwen_vl import pack_qwen_pixels
from aster.methods import CrossEntropyObjective
from aster.methods.sparse_indexer import qsa_indexer_distillation
from aster.training import Trainer


@pytest.mark.parametrize("stage", [0, 3])
def test_models_qwen4exp_visual_train_save_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(421)
    c = Qwen4ExpConfig()
    model = build_model(c)
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.vision_config)
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 3, 2]])
    batch = dict(
        labels=ids,
        model_inputs=dict(
            input_ids=ids,
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=(ids == 28).long(),
        ),
    )
    trainer = Trainer(model, CrossEntropyObjective(), lr=0.006, zero_stage=stage)
    initial = trainer.step([batch]).loss
    for _ in range(18):
        final = trainer.step([batch]).loss
    assert final < 0.55 * initial
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, x in trainer.export_state_dict().items():
        torch.testing.assert_close(x, weights[name], atol=0, rtol=0)
    native = build_model(c)
    native.load_state_dict(weights, strict=True)
    native.eval()
    native.save_pretrained(tmp_path / "model")
    data = batch["model_inputs"]
    torch.testing.assert_close(
        native(**data).logits, load_model(tmp_path / "model").eval()(**data).logits, atol=0, rtol=0
    )


def test_models_qwen4exp_video_span_cache_and_frozen_language_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(422)
    c = Qwen4ExpConfig()
    model = build_model(c).eval()
    pixels, grid = pack_qwen_pixels(torch.randn(4, 3, 4, 4), c.vision_config)

    ids = torch.tensor([[1, 26, 29, 27, 4, 26, 29, 27, 2]])
    kwargs = dict(
        pixel_values_videos=pixels, video_grid_thw=grid, mm_token_type_ids=(ids == 29).long() * 2
    )
    prefix = model(ids, use_cache=True, **kwargs)
    suffix = torch.tensor([[5, 7]])
    actual = model(suffix, state=prefix.state, use_cache=True).logits
    extended = torch.cat((ids, suffix), 1)
    full = model(extended, **{**kwargs, "mm_token_type_ids": (extended == 29).long() * 2}).logits
    torch.testing.assert_close(actual, full[:, -2:], atol=5e-6, rtol=6e-5)
    assert prefix.state.token_state.layers[1].ple_tokens.shape == (1, 2)
    assert prefix.state.token_state.position_ids.shape == (3, 1, len(ids[0]))
    for p in model.language_model.parameters():
        p.requires_grad_(False)
    output = model(ids, **kwargs)
    output.logits.square().sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision_tower.parameters()
    )
    assert all(p.grad is None for p in model.language_model.parameters())
    with pytest.raises(ValueError, match="fresh prefill"):
        model(ids, state=prefix.state, **kwargs)
    with pytest.raises(ValueError, match="snapshot"):
        prefix.state.truncate(2)
    with pytest.raises(ValueError, match="placeholders"):
        model(ids, **{**kwargs, "mm_token_type_ids": torch.zeros_like(ids)})


def test_models_qwen4exp_qsa_kd_teacher_detach_and_ragged_block_mass():
    torch.set_num_threads(1)
    torch.manual_seed(423)
    c = Qwen4ExpConfig().text_config
    model = build_model(c)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 2], [0, 1, 4, 6, 8, 10, 12, 2]])
    out = model(ids, attention_mask=ids.ne(0))
    records = out.auxiliary["qsa_indexer"][-1]
    teacher = torch.rand(2, 3, 8, 8, requires_grad=True)
    term = qsa_indexer_distillation(records, teacher, query_mask=ids.ne(0))
    explicit = []
    for row, query, indices, scores in records:
        if not bool(ids[row, query]):
            continue
        probability = teacher.detach()[row, :, query].sum(0)[indices].sum(-1)
        probability = probability / probability.sum()
        explicit.append((probability * (probability.log() - scores.log_softmax(-1))).sum())
    torch.testing.assert_close(term.mean, torch.stack(explicit).mean())
    assert term.denominator.dtype == torch.int64
    term.mean.backward()
    indexer = model.model.layers[-1].self_attn.indexer
    assert indexer.index_qk_proj.weight.grad.abs().sum() > 0
    assert teacher.grad is None

    model.zero_grad(set_to_none=True)
    model(ids).logits.square().mean().backward()
    assert indexer.index_qk_proj.weight.grad is None
