from dataclasses import replace
import pytest
import torch
from aster.models import OCR2Config, OCR2TextConfig, build_model, load_model
from aster.methods import CrossEntropyObjective
from aster.training import Trainer


def observation(c, *, local=False):
    count = c.vision_config.global_queries + 1 + (2 * c.vision_config.local_queries if local else 0)
    ids = torch.tensor([[1] + [c.image_token_id] * count + [3, 5, 7, 2]])
    pixels = torch.randn(
        1, 3, c.vision_config.sam_config.image_size, c.vision_config.sam_config.image_size
    )
    crops = (
        torch.randn(2, 3, c.vision_config.local_image_size, c.vision_config.local_image_size)
        if local
        else None
    )
    return dict(
        input_ids=ids,
        pixel_values=pixels,
        pixel_values_local=(crops,),
        images_seq_mask=ids.eq(c.image_token_id),
        images_spatial_crop=torch.tensor([[2, 1] if local else [1, 1]]),
    )


@pytest.mark.parametrize("stage", [0, 3])
def test_models_ocr2_full_train_export_exact_resume(tmp_path, stage):
    torch.set_num_threads(1)
    torch.manual_seed(521)
    c = OCR2Config()
    model = build_model(c)
    data = observation(c)
    labels = data["input_ids"].masked_fill(data["images_seq_mask"], -100)
    batch = dict(labels=labels, model_inputs=data)
    trainer = Trainer(
        model,
        CrossEntropyObjective(auxiliary_weights={"router_aux": 0.001}),
        lr=0.006,
        zero_stage=stage,
    )
    initial = trainer.step([batch]).loss
    for _ in range(11):
        final = trainer.step([batch]).loss
    assert final < initial * 0.5
    trainer.save_checkpoint(tmp_path / "checkpoint")
    expected = trainer.step([batch])
    weights = trainer.export_state_dict()
    trainer.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = trainer.step([batch])
    assert expected.loss == actual.loss
    for name, x in trainer.export_state_dict().items():
        torch.testing.assert_close(x, weights[name], atol=0, rtol=0)
    export = build_model(c)
    export.load_state_dict(weights, strict=True)
    export.eval()
    export.save_pretrained(tmp_path / "model")
    torch.testing.assert_close(
        load_model(tmp_path / "model").eval()(**data).logits, export(**data).logits, atol=0, rtol=0
    )


def test_models_ocr2_local_global_separator_order_cache_and_freeze():
    torch.set_num_threads(1)
    torch.manual_seed(522)
    c = OCR2Config()
    model = build_model(c).eval()
    data = observation(c, local=True)
    features = model.encode_document_views(
        data["pixel_values"], data["pixel_values_local"], data["images_spatial_crop"]
    )[0]
    local = model.projector.layers(model.vision_encoder(data["pixel_values_local"][0])).flatten(
        0, 1
    )
    global_ = model.projector.layers(model.vision_encoder(data["pixel_values"])).flatten(0, 1)
    torch.testing.assert_close(
        features, torch.cat((local, global_, model.separator.weight[None]), 0), atol=0, rtol=0
    )
    prefix = model(**data, use_cache=True)
    suffix = torch.tensor([[9, 11]])
    out = model(suffix, state=prefix.state, use_cache=True)
    ids = torch.cat((data["input_ids"], suffix), 1)
    full = model(**{**data, "input_ids": ids, "images_seq_mask": ids.eq(c.image_token_id)})
    torch.testing.assert_close(out.logits, full.logits[:, -2:], atol=3e-6, rtol=4e-5)
    with pytest.raises(ValueError, match="snapshot"):
        prefix.state.truncate(3)
    with pytest.raises(ValueError, match="new OCR prefill"):
        model(**data, state=prefix.state)
    frozen = build_model(replace(c, freeze_vision=True))
    output = frozen(**data)
    output.logits.square().mean().backward()
    assert all(p.grad is None for p in frozen.vision_encoder.parameters())
    assert frozen.projector.layers.weight.grad.abs().sum() > 0
    names = [model.official_weight_name(name) for name in model.state_dict()]
    assert len(set(names)) == len(names)
    assert "model.sam_model.pos_embed" in names and "model.qwen2_model.query_768.weight" in names


def test_models_ocr2_text_cache_and_unrenormalized_softmax_moe():
    torch.set_num_threads(1)
    torch.manual_seed(523)
    c = OCR2TextConfig()
    model = build_model(c)
    tokens = torch.tensor([[1, 3, 5, 7, 2], [1, 4, 6, 8, 2]])
    output = model(tokens)
    weights = output.auxiliary["router"][0]["weights"]
    assert (weights.sum(-1) < 1).all()
    prefix = model(tokens[:, :3], use_cache=True)
    actual = model(tokens[:, 3:], state=prefix.state)
    torch.testing.assert_close(actual.logits, output.logits[:, 3:], atol=2e-6, rtol=3e-5)
