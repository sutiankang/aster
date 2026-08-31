import pytest
import torch
import torch.nn.functional as F
from aster.models import LLaDAConfig, build_model, load_model
from aster.methods.masked_diffusion import MaskedDiffusionObjective, sample_masked_diffusion


def test_models_llada_denoising_training_bidirectional_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    model = build_model(LLaDAConfig())
    clean = torch.tensor([[1, 4, 7, 2], [1, 5, 8, 2]])
    masked = clean.clone()
    masked[:, 1:3] = 31
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

    def objective():
        return F.cross_entropy(
            model(masked).logits[:, 1:3].reshape(-1, 32), clean[:, 1:3].reshape(-1)
        )

    before = objective().item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    assert objective().item() < before
    changed = masked.clone()
    changed[:, -1] = 9
    assert not torch.allclose(model(masked).logits[:, 0], model(changed).logits[:, 0])
    with pytest.raises(ValueError):
        model(masked, use_cache=True)
    model.save_pretrained(tmp_path / "llada")
    torch.testing.assert_close(load_model(tmp_path / "llada")(masked).logits, model(masked).logits)


def test_models_llada_shared_method_and_mask_sampler():
    torch.set_num_threads(1)
    torch.manual_seed(48)
    model = build_model(LLaDAConfig())
    clean = torch.tensor([[1, 4, 7, 2], [1, 5, 8, 2]])
    objective = MaskedDiffusionObjective(31, sft=True)
    term = objective(
        model,
        {
            "input_ids": clean,
            "prompt_lengths": torch.tensor([1, 1]),
            "time": torch.ones(2),
            "masked_indices": torch.tensor([[False, True, True, True]]).expand(2, -1),
        },
    )
    term.mean.backward()
    assert term.denominator.item() == 2 and torch.isfinite(term.mean)
    sampled = sample_masked_diffusion(
        model,
        clean[:, :1],
        mask_token_id=31,
        generation_length=4,
        steps=4,
        block_length=2,
        forbidden_token_ids=(31,),
    )
    assert sampled.shape == (2, 5) and not (sampled == 31).any()
    torch.testing.assert_close(sampled[:, :1], clean[:, :1])
