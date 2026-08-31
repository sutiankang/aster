from dataclasses import replace
import pytest
import torch
from aster.models import (
    Gemma4VisionConfig,
    Gemma4TextConfig,
    Gemma4Config,
    build_model,
    load_model,
    pack_gemma4_images,
)


def test_models_gemma4_vision_train_patch_order_and_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    c = Gemma4VisionConfig(standardize=True)
    model = build_model(c)
    images = [torch.rand(3, 8, 8), torch.rand(3, 4, 8)]
    batch = pack_gemma4_images(images, c)
    torch.testing.assert_close(
        batch["pixel_values"][0, 0], images[0][:, :2, :2].permute(1, 2, 0).reshape(-1)
    )
    target = torch.randn(6, c.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    first = None
    for _ in range(12):
        optimizer.zero_grad()
        loss = (model(**batch).last_hidden_state - target).square().mean()
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.8
    model.eval()
    expected = model(**batch)
    model.save_pretrained(tmp_path / "vision")
    restored = load_model(tmp_path / "vision").eval()
    torch.testing.assert_close(
        restored(**batch).last_hidden_state, expected.last_hidden_state, atol=0, rtol=0
    )

    permutation = torch.randperm(batch["pixel_values"].shape[1])
    shuffled = model(
        batch["pixel_values"][:, permutation], batch["pixel_position_ids"][:, permutation]
    )
    torch.testing.assert_close(
        expected.last_hidden_state, shuffled.last_hidden_state, atol=2e-5, rtol=2e-5
    )


def test_models_gemma4_vision_rejects_ambiguous_grids():
    c = Gemma4VisionConfig()
    model = build_model(c)
    batch = pack_gemma4_images([torch.rand(3, 8, 8)], c)
    with pytest.raises(ValueError, match="Pool=1"):
        replace(c, pooling_kernel_size=1)
    with pytest.raises(ValueError, match="divisible"):
        pack_gemma4_images([torch.rand(3, 6, 8)], c)
    batch["pixel_position_ids"][0, 1] = batch["pixel_position_ids"][0, 0]
    with pytest.raises(ValueError, match="rectangle"):
        model(**batch)


def test_models_gemma4_full_train_reload_cache_and_ownership(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(33)
    c = Gemma4Config(text_config=replace(Gemma4TextConfig(), use_bidirectional_attention="vision"))
    model = build_model(c)
    tokens = torch.tensor([[1, 60, 60, 60, 60, 4, 5, 6], [1, 60, 60, 60, 60, 7, 8, 9]])
    batch = pack_gemma4_images(torch.rand(2, 3, 8, 8), c.vision_config)
    kwargs = dict(
        pixel_values=batch["pixel_values"],
        image_position_ids=batch["pixel_position_ids"],
        mm_token_type_ids=(tokens == 60).long(),
    )
    labels = tokens[:, 6:].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    first = None
    for _ in range(12):
        optimizer.zero_grad()
        logits = model(tokens, **kwargs).logits[:, 5:7]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 48), labels.reshape(-1))
        if first is None:
            first = loss.item()
        loss.backward()
        assert model.model.vision_tower.patch_embedder.input_proj.weight.grad.abs().sum() > 0
        optimizer.step()
    assert loss.item() < first * 0.7
    model.eval()
    full = model(tokens, **kwargs).logits
    model.save_pretrained(tmp_path / "vlm")
    restored = load_model(tmp_path / "vlm").eval()
    assert restored.lm_head.weight is restored.get_input_embeddings().weight
    torch.testing.assert_close(restored(tokens, **kwargs).logits, full, atol=0, rtol=0)
    prefix = model(
        tokens[:, :6],
        **{**kwargs, "mm_token_type_ids": kwargs["mm_token_type_ids"][:, :6]},
        use_cache=True,
    )
    continuation = model(tokens[:, 6:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(continuation.logits, full[:, 6:], atol=2e-6, rtol=2e-5)

    reversed_inputs = {
        **kwargs,
        "pixel_values": kwargs["pixel_values"].flip(0),
        "image_position_ids": kwargs["image_position_ids"].flip(0),
        "image_batch_indices": torch.tensor([1, 0]),
    }
    torch.testing.assert_close(model(tokens, **reversed_inputs).logits, full, atol=0, rtol=0)
    with pytest.raises(ValueError, match="sample"):
        model(tokens, **kwargs, image_batch_indices=torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="complete image"):
        model(tokens)
    with pytest.raises(ValueError, match="disagree"):
        model(tokens, **{**kwargs, "mm_token_type_ids": torch.zeros_like(tokens)})
