import pytest
import torch
import torch.nn.functional as F
from aster.models import (
    Qwen35TextConfig,
    Qwen35MoETextConfig,
    Qwen35Config,
    build_model,
    load_model,
)
from aster.models.qwen_vl import pack_qwen_pixels


@pytest.mark.parametrize("moe", [False, True])
def test_models_qwen35_train_and_storage(tmp_path, moe):
    torch.set_num_threads(1)
    torch.manual_seed(55)
    c = Qwen35MoETextConfig() if moe else Qwen35TextConfig()
    model = build_model(c)
    ids = torch.tensor([[1, 3, 5, 7, 2]])
    names = model.state_dict()
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in names
    assert "model.layers.0.linear_attn.in_proj_qkvz.weight" not in names
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        result = model(ids)
        loss = F.cross_entropy(
            result.logits[:, :-1].reshape(-1, c.vocab_size), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    prefix = model(ids[:, :2], use_cache=True)
    torch.testing.assert_close(
        model(ids[:, 2:], state=prefix.state, use_cache=True).logits,
        model(ids).logits[:, 2:],
        atol=3e-6,
        rtol=3e-5,
    )
    model.save_pretrained(tmp_path / "qwen35")
    torch.testing.assert_close(load_model(tmp_path / "qwen35")(ids).logits, model(ids).logits)


def test_models_qwen35_multimodal_state_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(56)
    c = Qwen35Config()
    model = build_model(c)
    pixels, grid = pack_qwen_pixels(torch.randn(1, 3, 8, 8), c.vision_config)
    ids = torch.tensor([[1, 26, 28, 28, 28, 28, 27, 2]])
    args = dict(
        pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=torch.where(ids == 28, 1, 0)
    )
    output = model(ids, use_cache=True, **args)
    assert output.state.kind == "qwen3_5_vl_hybrid"
    assert output.state.token_state.kind == "hybrid_delta"
    output.logits.square().mean().backward()
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    suffix = torch.tensor([[3, 4]])
    torch.testing.assert_close(
        model(suffix, state=output.state, use_cache=True).logits,
        model(
            torch.cat((ids, suffix), -1),
            **{
                **args,
                "mm_token_type_ids": torch.cat(
                    (args["mm_token_type_ids"], torch.zeros_like(suffix)), -1
                ),
            },
        ).logits[:, -2:],
        atol=3e-6,
        rtol=3e-5,
    )
    model.save_pretrained(tmp_path / "vl")
    torch.testing.assert_close(load_model(tmp_path / "vl")(ids, **args).logits, output.logits)
