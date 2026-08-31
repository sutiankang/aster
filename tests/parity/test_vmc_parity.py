import math

import pytest
import torch
import torch.nn.functional as F

from aster.models.vmc import MDNRNN, MDNRNNConfig, VMCVAE, VMCVAEConfig
from aster.methods.vmc import VMCVAEObjective, MDNRNNObjective


@pytest.mark.parametrize("layer_norm", [False, True])
def test_vmc_lstm_mixture_loss_complete_source_formula_gradients(layer_norm):
    torch.manual_seed(449)
    model = MDNRNN(
        MDNRNNConfig(latent_size=3, action_dim=1, hidden_size=7, mixtures=2, layer_norm=layer_norm)
    )
    weights = {k: v.detach().clone().requires_grad_() for k, v in model.named_parameters()}
    z = torch.randn(2, 5, 3)
    action = torch.randn(2, 5, 1)
    restart = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    h, c = torch.zeros(2, 7), torch.zeros(2, 7)
    outputs = []

    def norm(value, index):
        mean = value.mean(-1, keepdim=True)
        variance = (value - mean).square().mean(-1, keepdim=True)
        return (value - mean) * (variance + 1e-12).rsqrt() * weights[
            f"lstm.norms.{index}.weight"
        ] + weights[f"lstm.norms.{index}.bias"]

    for time in range(4):
        keep = (~restart[:, time, None]).float()
        c, h = c * keep, h * keep
        incoming = torch.cat((z[:, time], action[:, time], restart[:, time, None].float(), h), -1)
        gate = incoming @ weights["lstm.projection.weight"].T
        if not layer_norm:
            gate = gate + weights["lstm.projection.bias"]
        gates = gate.chunk(4, -1)
        if layer_norm:
            gates = [norm(value, index) for index, value in enumerate(gates)]
        i, j, f, o = gates
        c = c * torch.sigmoid(f + 1) + torch.sigmoid(i) * torch.tanh(j)
        if layer_norm:
            c = norm(c, 4)
        h = torch.tanh(c) * torch.sigmoid(o)
        outputs.append(h)
    projected = torch.stack(outputs, 1) @ weights["output.weight"].T + weights["output.bias"]
    logmix, mean, logstd = projected[..., 1:].reshape(2, 4, 3, 6).chunk(3, -1)
    logmix = logmix - torch.logsumexp(logmix, -1, keepdim=True)
    logdensity = (
        -0.5 * ((z[:, 1:, :, None] - mean) / logstd.exp()).square()
        - logstd
        - 0.5 * math.log(2 * math.pi)
    )
    reference_nll = -torch.logsumexp(logmix + logdensity, -1).mean()
    labels = restart[:, 1:].float()

    logits = projected[..., 0]
    bce = logits.clamp_min(0) - logits * labels + torch.log1p(torch.exp(-logits.abs()))
    reference_restart = (bce * (1 + labels * 9)).mean()
    actual_output = model(z[:, :-1], action[:, :-1], restart[:, :-1])
    torch.testing.assert_close(actual_output.mean, mean, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_output.state.hidden, h, atol=2e-6, rtol=2e-5)
    terms = MDNRNNObjective(sequence_length=5)(
        model, dict(latents=z, actions=action, restart=restart)
    ).terms
    torch.testing.assert_close(terms[0].mean, reference_nll, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(terms[1].mean, reference_restart, atol=2e-6, rtol=2e-5)
    (terms[0].mean + terms[1].mean).backward()
    (reference_nll + reference_restart).backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, weights[name].grad, atol=5e-6, rtol=5e-5, msg=name
        )


def test_vmc_vae_reparameterization_kl_and_full_image_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(631)
    model = VMCVAE(VMCVAEConfig(latent_size=3, conv_channels=2))
    w = {k: v.detach().clone().requires_grad_() for k, v in model.named_parameters()}
    pixels = torch.rand(2, 3, 64, 64)
    noise = torch.randn(2, 3)
    value = pixels
    for index in range(4):
        key = f"encoder.convolutions.{2 * index}"
        value = F.conv2d(value, w[key + ".weight"], w[key + ".bias"], stride=2).relu()
    value = value.permute(0, 2, 3, 1).flatten(1)
    mean = value @ w["mean.weight"].T + w["mean.bias"]
    logvar = value @ w["logvar.weight"].T + w["logvar.bias"]
    latent = mean + (0.5 * logvar).exp() * noise
    value = (latent @ w["decoder.projection.weight"].T + w["decoder.projection.bias"])[
        ..., None, None
    ]
    for index in range(4):
        key = f"decoder.deconvolutions.{2 * index}"
        value = F.conv_transpose2d(value, w[key + ".weight"], w[key + ".bias"], stride=2)
        if index < 3:
            value = value.relu()
    reconstructed = value.sigmoid()
    reference = (pixels - reconstructed).square().flatten(1).sum(1).mean()
    reference += (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).clamp_min(0).mean()
    terms = VMCVAEObjective(kl_tolerance=0)(model, dict(images=pixels, noise=noise)).terms
    actual = sum(term.mean for term in terms)
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-6)
    actual.backward()
    reference.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, w[name].grad, atol=2e-5, rtol=3e-5, msg=name)
