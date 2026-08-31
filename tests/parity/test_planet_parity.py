from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from aster.models.planet import PlaNetConfig, PlaNetWorldModel


def _linear(weights, prefix, value):
    return value @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]


def _hidden(weights, prefix, value, layers):
    for index in range(layers):
        value = _linear(weights, f"{prefix}.{2 * index}", value).relu()
    return value


@pytest.mark.parametrize("future_rnn", [True, False])
def test_planet_full_continuous_rssm_source_formula_all_gradients(future_rnn):
    torch.manual_seed(222)
    c = PlaNetConfig(
        observation_dim=4,
        action_dim=2,
        state_size=3,
        belief_size=7,
        hidden_size=8,
        model_layers=2,
        reward_hidden_size=9,
        reward_layers=2,
        future_rnn=future_rnn,
    )
    model = PlaNetWorldModel(c)
    weights = {k: v.detach().clone().requires_grad_() for k, v in model.named_parameters()}
    observations = torch.randn(2, 4, 4, requires_grad=True)
    actions = torch.randn(2, 4, 2, requires_grad=True)
    x, a = observations.detach().clone().requires_grad_(), actions.detach().clone().requires_grad_()
    resets = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    prior_noise, post_noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    actual = model(
        observations, actions, resets, prior_noise=prior_noise, posterior_noise=post_noise
    )
    encoded = _hidden(weights, "encoder", x, 2)
    z, h = torch.zeros(2, 3), torch.zeros(2, 7)
    predictions, priors, posterior, reward = [], [], [], []
    for step in range(4):
        keep = (~resets[:, step, None]).float()
        h, z = h * keep, z * keep
        hidden = _hidden(weights, "transition_input", torch.cat((z, a[:, step] * keep), -1), 2)
        gates = torch.cat((hidden, h), -1) @ weights["gru.gate_kernel"] + weights["gru.gate_bias"]
        reset, update = gates.sigmoid().chunk(2, -1)
        candidate = (
            torch.cat((hidden, h * reset), -1) @ weights["gru.candidate_kernel"]
            + weights["gru.candidate_bias"]
        ).tanh()
        h = (1 - update) * candidate + update * h
        prior_hidden = _hidden(weights, "prior_hidden", h if future_rnn else hidden, 2)
        pm = _linear(weights, "prior_mean", prior_hidden)
        ps = F.softplus(_linear(weights, "prior_stddev", prior_hidden)) + 0.1
        priors.append(torch.cat((pm, ps, pm + ps * prior_noise[:, step], h), -1))
        hidden = _hidden(weights, "posterior_hidden", torch.cat((h, encoded[:, step]), -1), 2)
        qm = _linear(weights, "posterior_mean", hidden)
        qs = F.softplus(_linear(weights, "posterior_stddev", hidden)) + 0.1
        z = qm + qs * post_noise[:, step]
        posterior.append(torch.cat((qm, qs, z, h), -1))
        features = torch.cat((z, h), -1)
        predictions.append(
            _linear(weights, "decoder.1", _hidden(weights, "decoder.0", features, 2))
        )
        reward.append(
            _linear(weights, "reward_head.1", _hidden(weights, "reward_head.0", features, 2))
        )
    expected = [torch.stack(v, 1) for v in (predictions, reward, priors, posterior)]
    actuals = [
        actual["reconstruction"],
        actual["reward"][..., None],
        torch.cat(
            [getattr(actual["prior"], k) for k in ("mean", "stddev", "sample", "belief")], -1
        ),
        torch.cat(
            [getattr(actual["state"], k) for k in ("mean", "stddev", "sample", "belief")], -1
        ),
    ]
    for first, second in zip(actuals, expected):
        torch.testing.assert_close(first, second, atol=4e-7, rtol=2e-6)
    sum(v.square().sum() for v in actuals).backward()
    sum(v.square().sum() for v in expected).backward()
    torch.testing.assert_close(observations.grad, x.grad, atol=8e-6, rtol=2e-5)
    torch.testing.assert_close(actions.grad, a.grad, atol=8e-6, rtol=2e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, weights[name].grad, atol=2e-5, rtol=3e-5, msg=name
        )


def test_planet_image_encoder_nhwc_flatten_and_decoder_operator_formula():
    torch.set_num_threads(1)
    torch.manual_seed(221)
    c = PlaNetConfig(
        conv_channels=2,
        state_size=3,
        belief_size=5,
        hidden_size=8,
        action_dim=2,
        reward_hidden_size=8,
        reward_layers=1,
    )
    model = PlaNetWorldModel(c)
    values = torch.randn(2, 3, 64, 64, requires_grad=True)
    source = values.detach().clone().requires_grad_()
    features = torch.randn(2, 8, requires_grad=True)
    source_features = features.detach().clone().requires_grad_()
    w = {k: v.detach().clone().requires_grad_() for k, v in model.named_parameters()}
    h = source
    for index in range(4):
        key = f"encoder.convolutions.{2 * index}"
        h = F.conv2d(h, w[key + ".weight"], w[key + ".bias"], stride=2).relu()
    reference_encoded = h.permute(0, 2, 3, 1).reshape(2, -1)
    h = _linear(w, "decoder.projection", source_features)[..., None, None]
    for index in range(4):
        key = f"decoder.deconvolutions.{2 * index}"
        h = F.conv_transpose2d(h, w[key + ".weight"], w[key + ".bias"], stride=2)
        if index < 3:
            h = h.relu()
    encoded, decoded = model.encoder(values), model.decoder(features)
    torch.testing.assert_close(encoded, reference_encoded, atol=0, rtol=0)
    torch.testing.assert_close(decoded, h, atol=0, rtol=0)
    (encoded.square().sum() + decoded.square().sum()).backward()
    (reference_encoded.square().sum() + h.square().sum()).backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder.", "decoder.")):
            torch.testing.assert_close(parameter.grad, w[name].grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(values.grad, source.grad, atol=0, rtol=0)
    torch.testing.assert_close(features.grad, source_features.grad, atol=2e-5, rtol=2e-5)
