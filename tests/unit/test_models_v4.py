from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from aster.models import DeepSeekV4Config, build_model, load_model


def test_models_v4_train_cache_fork_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(51)
    c = DeepSeekV4Config()
    model = build_model(c)
    ids = torch.tensor([[1, 4, 7, 5, 8, 9, 11, 2, 6], [1, 5, 3, 7, 2, 6, 4, 8, 9]])
    model.set_hash_routes(
        0, torch.stack((torch.arange(c.vocab_size) % 4, (torch.arange(c.vocab_size) + 1) % 4), -1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        result = model(ids)
        loss = F.cross_entropy(
            result.logits[:, :-1].reshape(-1, c.vocab_size), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    model.eval()
    entire = model(ids).logits
    for prefix in (1, 2, 3, 4, 7):
        state = model(ids[:, :prefix], use_cache=True).state
        old = state.fork()
        torch.testing.assert_close(
            model(ids[:, prefix:], state=state).logits, entire[:, prefix:], atol=3e-6, rtol=3e-5
        )
        torch.testing.assert_close(
            state.layers[1].compressor.entries, old.layers[1].compressor.entries
        )
        order = torch.tensor([1, 0])
        torch.testing.assert_close(
            model(ids[order, prefix:], state=state.reorder(order)).logits,
            entire[order, prefix:],
            atol=3e-6,
            rtol=3e-5,
        )
        with pytest.raises(ValueError):
            state.truncate(0)
    model.save_pretrained(tmp_path / "v4")
    torch.testing.assert_close(load_model(tmp_path / "v4")(ids).logits, entire)
    with pytest.raises(ValueError):
        model(ids, attention_mask=torch.zeros_like(ids))
    with pytest.raises(ValueError):
        model(inputs_embeds=model.get_input_embeddings()(ids))
    torch.testing.assert_close(
        model(inputs_embeds=model.get_input_embeddings()(ids), routing_input_ids=ids).logits, entire
    )


def test_models_v4_causal_closed_window_and_swiglu_clipping():
    torch.set_num_threads(1)
    torch.manual_seed(52)
    c = replace(DeepSeekV4Config(), swiglu_limit=0.01)
    model = build_model(c)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13]])
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[8, 2, 4]])
    torch.testing.assert_close(model(ids).logits[:, :4], model(changed).logits[:, :4])
    prefix = model(ids[:, :3], use_cache=True).state
    assert prefix.layers[1].compressor.entries.shape[1] == 0
    assert prefix.layers[1].compressor.pending_kv.shape[1] == 3
    assert prefix.layers[2].compressor.entries.shape[1] == 1
    full = model(ids, use_cache=True)
    assert full.state.layers[2].compressor.entries.shape[1] == 3
    for info in full.auxiliary["indexer"]:
        assert not info["visible"][:, :1].any()
