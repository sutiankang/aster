from types import SimpleNamespace
import torch
from aster.methods.masked_diffusion import MaskedDiffusionObjective, sample_masked_diffusion


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 1.0, 3.0, -5.0]))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def test_masked_loss_inverse_noise_probability_and_sft_denominator():
    model = Fixture()
    clean = torch.tensor([[0, 1, 2], [1, 1, 2]])
    masked = torch.tensor([[False, True, False], [False, False, True]])
    batch = {"input_ids": clean, "time": torch.tensor([0.0, 1.0]), "masked_indices": masked}
    objective = MaskedDiffusionObjective(3, epsilon=0.5, random_length_probability=0.0)
    loss = objective(model, batch)
    logp = model.logits.log_softmax(-1)
    torch.testing.assert_close(loss.mean, (-logp[1] * 2 - logp[2]) / 6)
    sft = MaskedDiffusionObjective(3, epsilon=0.5, sft=True)(
        model, {**batch, "prompt_lengths": torch.tensor([1, 2])}
    )
    torch.testing.assert_close(sft.mean, (-logp[1] - logp[2]) / 2)
    sft.mean.backward()
    assert model.logits.grad.isfinite().all()


def test_masked_sampling_blocks_cfg_and_prompt_preservation():
    model = Fixture()
    prompt = torch.tensor([[0, 1], [1, 0]])
    output = sample_masked_diffusion(
        model, prompt, mask_token_id=3, generation_length=6, block_length=3, steps=4, cfg_scale=0.5
    )
    torch.testing.assert_close(output[:, :2], prompt)
    assert (output[:, 2:] == 2).all() and model.training
    left = sample_masked_diffusion(
        model,
        prompt,
        mask_token_id=3,
        generation_length=6,
        steps=3,
        temperature=0.5,
        forbidden_token_ids=(3,),
        generator=torch.Generator().manual_seed(7),
    )
    right = sample_masked_diffusion(
        model,
        prompt,
        mask_token_id=3,
        generation_length=6,
        steps=3,
        temperature=0.5,
        forbidden_token_ids=(3,),
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(left, right)
