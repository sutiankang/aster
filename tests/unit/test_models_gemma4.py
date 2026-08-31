from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from aster.models import Gemma4TextConfig, build_model, load_model


@pytest.mark.parametrize("moe", [False, True])
def test_models_gemma4_shared_cache_ple_gradient_and_reload(tmp_path, moe):
    torch.set_num_threads(1)
    torch.manual_seed(121)
    c = Gemma4TextConfig(enable_moe_block=moe)
    model = build_model(c)
    tokens = torch.randint(1, c.vocab_size, (2, 9))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = None
    for _ in range(8):
        loss = F.cross_entropy(model(tokens).logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial
    model.eval()
    whole = model(tokens).logits
    prefix = model(tokens[:, :3], use_cache=True)
    tail = model(tokens[:, 3:], state=prefix.state, use_cache=True)
    torch.testing.assert_close(tail.logits, whole[:, 3:], atol=1e-5, rtol=1e-4)
    assert len(tail.state.layers) == c.independent_layers == 2
    assert tail.state.layers[0][0].shape[-2] == c.sliding_window - 1
    assert tail.state.layers[1][0].shape[-2] == 9
    assert not hasattr(model.model.layers[2].self_attn, "k_proj")
    assert model.model.layers[1].self_attn.v_proj is None
    assert model.model.layers[2].mlp.gate_proj.out_features == 2 * c.intermediate_size
    with pytest.raises(ValueError, match="replay"):
        tail.state.truncate(4)
    original = prefix.state.layers[0][0].clone()
    fork = prefix.state.fork()
    fork.layers[0][0].add_(1)
    torch.testing.assert_close(original, prefix.state.layers[0][0], atol=0, rtol=0)
    embeddings, ple = model.get_input_embeddings()(tokens), model.get_per_layer_inputs(tokens)
    torch.testing.assert_close(model(inputs_embeds=embeddings, per_layer_inputs=ple).logits, whole)
    with pytest.raises(ValueError, match="PLE"):
        model(inputs_embeds=embeddings)
    model.save_pretrained(tmp_path / "gemma")
    restored = load_model(tmp_path / "gemma").eval()
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    torch.testing.assert_close(restored(tokens).logits, whole, atol=0, rtol=0)


def test_models_gemma4_config_and_causality():
    torch.set_num_threads(1)
    torch.manual_seed(122)
    c = Gemma4TextConfig()
    model = build_model(c)
    tokens = torch.randint(1, c.vocab_size, (1, 8))
    changed = tokens.clone()
    changed[:, 5:] = 2
    torch.testing.assert_close(model(tokens).logits[:, :5], model(changed).logits[:, :5])
    with pytest.raises(ValueError, match="earlier"):
        replace(c, num_kv_shared_layers=3)
    with pytest.raises(ValueError, match="ending"):
        replace(c, layer_types=("full_attention",) * 3 + ("sliding_attention",))
    with pytest.raises(ValueError, match="configuration"):
        wrong = build_model(replace(c, global_rotary_fraction=0.5))
        wrong(tokens[:, -1:], state=model(tokens[:, :-1], use_cache=True).state)
