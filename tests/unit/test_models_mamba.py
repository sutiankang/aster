import pytest
import torch
from aster.models import MambaConfig, build_model, load_model


@pytest.mark.parametrize("kernel", [1, 4])
def test_models_mamba_real_scan_chunks_train_store(tmp_path, kernel):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model(MambaConfig(conv_kernel=kernel))
    ids = torch.tensor([[1, 4, 7, 3, 8, 2], [1, 3, 6, 5, 9, 2]])
    full = model(ids).logits
    first = model(ids[:, :2], use_cache=True)
    snapshot = first.state.fork()
    second = model(ids[:, 2:5], state=first.state)
    third = model(ids[:, 5:], state=second.state)
    torch.testing.assert_close(
        torch.cat((first.logits, second.logits, third.logits), 1), full, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(first.state.layers[0][1], snapshot.layers[0][1])
    assert third.state.layers[0][0].shape == (2, 64, kernel - 1)
    with pytest.raises(ValueError):
        third.state.truncate(2)
    full.square().mean().backward()
    assert model.backbone.layers[0].mixer.A_log.grad.abs().sum() > 0
    model.save_pretrained(tmp_path / "mamba")
    torch.testing.assert_close(load_model(tmp_path / "mamba")(ids).logits, full)
