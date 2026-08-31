import torch
from aster.models import GPT2Config, build_model, load_model


def test_models_gpt_training_state_tie_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(40)
    model = build_model(GPT2Config())
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    loss = torch.nn.functional.cross_entropy(
        model(ids[:, :-1]).logits.flatten(0, 1), ids[:, 1:].flatten()
    )
    loss.backward()
    assert model.transformer.h[0].attn.c_attn.weight.grad.abs().sum() > 0
    prefix = model(ids[:, :3], use_cache=True)
    torch.testing.assert_close(
        model(ids[:, 3:], state=prefix.state).logits, model(ids).logits[:, 3:]
    )
    model.save_pretrained(tmp_path / "gpt")
    restored = load_model(tmp_path / "gpt")
    assert restored.lm_head.weight is restored.transformer.wte.weight
    torch.testing.assert_close(restored(ids).logits, model(ids).logits)
