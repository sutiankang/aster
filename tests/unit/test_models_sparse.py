from dataclasses import replace
import pytest
import torch
from aster.models import DeepSeekV32Config, DeepSeekV3Config, build_model, load_model
from aster.methods.sparse_indexer import indexer_distillation


def test_models_dsa_dense_limit_cache_indexer_training_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(46)
    dense_config = DeepSeekV3Config()
    config = DeepSeekV32Config(index_topk=128)
    dense, model = build_model(dense_config), build_model(config)
    dense.load_state_dict(
        {k: v for k, v in model.state_dict().items() if ".indexer." not in k}, strict=True
    )
    ids = torch.tensor([[1, 3, 5, 7, 2]])
    torch.testing.assert_close(dense(ids).logits, model(ids).logits, atol=2e-6, rtol=2e-5)
    prefix = model(ids[:, :3], use_cache=True)
    suffix = model(ids[:, 3:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(suffix.logits, model(ids).logits[:, 3:], atol=2e-6, rtol=2e-5)
    assert suffix.state.layers[0][2].shape == (1, 1, 5, config.index_head_dim)
    assert suffix.state.truncate(3).seen_tokens == 3
    output = model(ids)
    teacher = torch.rand(1, 2, 5, 5, requires_grad=True)
    terms = [
        indexer_distillation(info["scores"], teacher, info["visible"]).mean
        for info in output.auxiliary["indexer"]
    ]
    sum(terms).backward()
    assert teacher.grad is None
    assert model.model.embed_tokens.weight.grad is None
    assert model.model.layers[0].self_attn.indexer.wq_b.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path / "dsa")
    torch.testing.assert_close(load_model(tmp_path / "dsa")(ids).logits, output.logits)


def test_models_dsa_sparse_selection_and_empty_query():
    torch.set_num_threads(1)
    model = build_model(DeepSeekV32Config(index_topk=2))
    info = model(torch.tensor([[1, 3, 5, 7]])).auxiliary["indexer"][0]
    assert info["indices"].shape == (1, 4, 2)
    scores = torch.randn(1, 2, 3, requires_grad=True)
    valid = torch.tensor([[[False, False, False], [True, True, False]]])
    result = indexer_distillation(scores, torch.ones_like(scores), valid)
    result.mean.backward()
    assert result.denominator.item() == 1 and torch.isfinite(scores.grad).all()
    with pytest.raises(ValueError):
        DeepSeekV32Config(q_lora_rank=None)
