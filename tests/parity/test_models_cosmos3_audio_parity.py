import math
import pytest
import torch
import torch.nn.functional as F
from aster.models.cosmos3_audio import Cosmos3AudioConfig, Cosmos3AudioCodec, AudioWeightNormConv


def formula(c, state, waveform):
    used = set()

    def parameter(name):
        used.add(name)
        return state[name]

    def conv(x, key, stride=1, padding=0, dilation=1, transpose=False, groups=1):
        if key + ".weight_g" in state:
            w = torch._weight_norm(parameter(key + ".weight_v"), parameter(key + ".weight_g"), 0)
        else:
            w = parameter(key + ".weight")
        bias = parameter(key + ".bias") if key + ".bias" in state else None
        if transpose:
            return F.conv_transpose1d(x, w, bias, stride, padding, stride % 2)
        return F.conv1d(x, w, bias, stride, padding, dilation, groups)

    def snake(x, key):
        return (
            x
            + (parameter(key + ".beta").exp() + 1e-9).reciprocal()
            * torch.sin(parameter(key + ".alpha").exp() * x).square()
        )

    def residual(x, key, d):
        return x + conv(
            snake(
                conv(snake(x, key + ".snake1"), key + ".conv1", padding=3 * d, dilation=d),
                key + ".snake2",
            ),
            key + ".conv2",
        )

    if c.normalize_volume:
        waveform = waveform / (waveform.abs().max() + 1e-5) * 0.95
    waveform = F.pad(waveform, (0, (-waveform.shape[-1]) % c.hop_size))
    b, channels, n = waveform.shape
    pad = c.enc_n_fft - c.enc_hop_length
    spec = torch.stft(
        F.pad(waveform.reshape(b * channels, n), (pad // 2, pad - pad // 2)).float(),
        n_fft=c.enc_n_fft,
        hop_length=c.enc_hop_length,
        win_length=c.enc_n_fft,
        window=torch.hann_window(c.enc_n_fft),
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    x = (
        torch.cat((spec.real, spec.imag), 1)
        .to(waveform.dtype)
        .reshape(b, channels * (c.enc_n_fft + 2), -1)
    )
    x = conv(x, "encoder.layers.0")
    index = 1
    for width, stride in zip(c.enc_c_mults, c.enc_strides):
        for _ in range(c.enc_num_blocks):
            key = f"encoder.layers.{index}"
            value = conv(F.pad(x, (3, 3)), key + ".dwconv.1", groups=width * c.enc_dim)
            value = (
                F.layer_norm(
                    value.transpose(1, 2).float(),
                    (width * c.enc_dim,),
                    parameter(key + ".norm.weight").float(),
                    None,
                    1e-5,
                )
                .to(x.dtype)
                .transpose(1, 2)
            )
            value = conv(value, key + ".pwconv1")
            value = snake(value, key + ".act") if c.enc_use_snake else F.gelu(value)
            x = x + conv(value, key + ".pwconv2")
            index += 1
        x = conv(x, f"encoder.layers.{index}", stride=stride, padding=math.ceil(stride / 2))
        index += 1
    moments = conv(x, f"encoder.layers.{index}")
    mean, scale = moments.chunk(2, 1)
    x = conv(mean, "decoder.conv1", padding=3)
    for i, stride in enumerate(c.dec_strides[::-1]):
        key = f"decoder.block.{i}"
        x = conv(
            snake(x, key + ".snake1"),
            key + ".conv_t1",
            stride,
            math.ceil(stride / 2),
            transpose=True,
        )
        for unit, dilation in enumerate((1, 3, 9), 1):
            x = residual(x, key + f".res_unit{unit}", dilation)
    x = conv(snake(x, "decoder.snake1"), "decoder.conv2", padding=3).clamp(-1, 1)
    assert used == set(state), (
        "Every declared source parameter must participate",
        set(state) - used,
    )
    return x, mean, scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_avae2_full_source_formula_all_weights_waveform_gradients(dtype):
    torch.set_num_threads(1)
    torch.manual_seed(682)
    c = Cosmos3AudioConfig()
    model = Cosmos3AudioCodec(c).to(dtype).eval()
    state = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    waveform = (torch.randn(2, 2, 47) * 0.2).to(dtype).requires_grad_()
    other = waveform.detach().clone().requires_grad_()
    output, posterior = model(waveform)
    expected, mean, scale = formula(c, state, other)
    tolerance = (
        dict(atol=2e-6, rtol=4e-5) if dtype == torch.float32 else dict(atol=0.012, rtol=0.05)
    )
    torch.testing.assert_close(output, expected, **tolerance)
    torch.testing.assert_close(posterior.mean, mean, **tolerance)
    torch.testing.assert_close(posterior.scale, scale, **tolerance)
    factor = torch.randn_like(output).float() / output.numel()
    factor_mean = torch.randn_like(mean).float() / mean.numel()
    factor_scale = torch.randn_like(scale).float() / scale.numel()
    std = F.softplus(scale) + 1e-4
    oracle_kl = (mean.square() + std.square() - std.square().log() - 1).sum(1).mean()
    torch.testing.assert_close(posterior.kl(), oracle_kl, **tolerance)

    (
        (output.float() * factor).sum()
        + (posterior.mean.float() * factor_mean).sum() * 0.1
        + (posterior.scale.float() * factor_scale).sum() * 0.1
        + posterior.kl() * 0.01
    ).backward()
    (
        (expected.float() * factor).sum()
        + (mean.float() * factor_mean).sum() * 0.1
        + (scale.float() * factor_scale).sum() * 0.1
        + oracle_kl * 0.01
    ).backward()
    gradient_tolerance = (
        dict(atol=3e-6, rtol=4e-4) if dtype == torch.float32 else dict(atol=0.02, rtol=0.12)
    )
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and state[name].grad is not None
        torch.testing.assert_close(parameter.grad, state[name].grad, **gradient_tolerance, msg=name)
    torch.testing.assert_close(waveform.grad, other.grad, **gradient_tolerance)


@pytest.mark.parametrize("transpose", [False, True])
def test_models_avae2_weightnorm_leaf_actual_torch_forward_and_grad(transpose):
    torch.set_num_threads(1)
    torch.manual_seed(683)
    c = AudioWeightNormConv(3, 5, 6, stride=3, padding=2, output_padding=1, transpose=transpose)
    options = dict(stride=3, padding=2)
    if transpose:
        options["output_padding"] = 1
    oracle = torch.nn.utils.weight_norm(
        (torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d)(3, 5, 6, **options)
    )
    oracle.load_state_dict(c.state_dict(), strict=True)
    x = torch.randn(2, 3, 9)
    left, right = c(x), oracle(x)
    torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
    left.square().mean().backward()
    right.square().mean().backward()
    for name, parameter in c.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(oracle.named_parameters())[name].grad, atol=2e-7, rtol=3e-6
        )
