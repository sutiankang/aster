import math
import pytest
import torch
import torch.nn.functional as F
from aster.models.wan22_vae import Wan22VAEConfig, Wan22VideoVAE


class Formula:
    def __init__(self, config, weights):
        self.c, self.w, self.cache = config, weights, {}

    def norm(self, x, name):
        y = F.normalize(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x, dim=1).to(
            x.dtype
        )
        return y * math.sqrt(x.shape[1]) * self.w[name + ".gamma"]

    def conv(self, x, name, *, padding=(0, 0, 0, 0, 0, 0), stride=1, previous=None):
        padding = list(padding)
        if previous is not None:
            x = torch.cat((previous, x), 2)
            padding[4] -= previous.shape[2]
        return F.conv3d(
            F.pad(x, padding), self.w[name + ".weight"], self.w[name + ".bias"], stride=stride
        )

    def causal(self, x, name):
        old = self.cache.get(name)
        kept = x[:, :, -2:].clone()
        if kept.shape[2] < 2 and old is not None:
            kept = torch.cat((old[:, :, -1:], kept), 2)
        result = self.conv(x, name, padding=(1, 1, 1, 1, 2, 0), previous=old)
        self.cache[name] = kept
        return result

    def resnet(self, x, name):
        shortcut = (
            self.conv(x, name + ".conv_shortcut") if name + ".conv_shortcut.weight" in self.w else x
        )
        x = self.causal(F.silu(self.norm(x, name + ".norm1")), name + ".conv1")
        return shortcut + self.causal(F.silu(self.norm(x, name + ".norm2")), name + ".conv2")

    def attention(self, x, name):
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        qkv = F.conv2d(
            self.norm(frames, name + ".norm"),
            self.w[name + ".to_qkv.weight"],
            self.w[name + ".to_qkv.bias"],
        )
        q, k, v = qkv.reshape(b * t, 1, c * 3, h * w).permute(0, 1, 3, 2).contiguous().chunk(3, -1)
        out = (
            F.scaled_dot_product_attention(q, k, v)
            .squeeze(1)
            .permute(0, 2, 1)
            .reshape(b * t, c, h, w)
        )
        out = F.conv2d(out, self.w[name + ".proj.weight"], self.w[name + ".proj.bias"])
        return x + out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def mid(self, x, name):
        return self.resnet(
            self.attention(self.resnet(x, name + ".resnets.0"), name + ".attentions.0"),
            name + ".resnets.1",
        )

    def resample(self, x, name, up, temporal):
        b, c, t, h, w = x.shape
        if up and temporal:
            if name not in self.cache:
                self.cache[name] = "Rep"
            else:
                old = self.cache[name]
                kept = x[:, :, -2:].clone()
                if kept.shape[2] < 2:
                    kept = torch.cat(
                        (torch.zeros_like(kept) if isinstance(old, str) else old[:, :, -1:], kept),
                        2,
                    )
                x = self.conv(
                    x,
                    name + ".time_conv",
                    padding=(0, 0, 0, 0, 2, 0),
                    previous=None if isinstance(old, str) else old,
                )
                self.cache[name] = kept
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = (
            F.interpolate(frames.float(), scale_factor=2.0, mode="nearest-exact").to(frames.dtype)
            if up
            else F.pad(frames, (0, 1, 0, 1))
        )
        frames = F.conv2d(
            frames,
            self.w[name + ".resample.1.weight"],
            self.w[name + ".resample.1.bias"],
            stride=1 if up else 2,
            padding=1 if up else 0,
        )
        x = frames.reshape(b, t, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        if not up and temporal:
            if name not in self.cache:
                self.cache[name] = x.clone()
            else:
                kept = x[:, :, -1:].clone()
                x = self.conv(
                    torch.cat((self.cache[name][:, :, -1:], x), 2),
                    name + ".time_conv",
                    stride=(2, 1, 1),
                )
                self.cache[name] = kept
        return x

    @staticmethod
    def avg_down(x, outgoing, r, p):
        x = F.pad(x, (0, 0, 0, 0, (-x.shape[2]) % r, 0))
        b, c, t, h, w = x.shape
        x = (
            x.view(b, c, t // r, r, h // p, p, w // p, p)
            .permute(0, 1, 3, 5, 7, 2, 4, 6)
            .contiguous()
        )
        return x.view(b, outgoing, c * r * p * p // outgoing, t // r, h // p, w // p).mean(2)

    @staticmethod
    def dup_up(x, outgoing, r, first):
        b, c, t, h, w = x.shape
        x = x.repeat_interleave(outgoing * r * 4 // c, 1).view(b, outgoing, r, 2, 2, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous().view(b, outgoing, t * r, h * 2, w * 2)
        return x[:, :, r - 1 :] if first else x

    def encoder(self, x):
        c = self.c
        x = self.causal(x, "encoder.conv_in")
        for i, mult in enumerate(c.dim_mult):
            down = i < len(c.dim_mult) - 1
            temporal = c.temperal_downsample[i] if down else False
            residual = self.avg_down(x, c.base_dim * mult, 2 if temporal else 1, 2 if down else 1)
            name = f"encoder.down_blocks.{i}"
            for j in range(c.num_res_blocks):
                x = self.resnet(x, f"{name}.resnets.{j}")
            if down:
                x = self.resample(x, name + ".downsampler", False, temporal)
            x = x + residual
        return self.causal(
            F.silu(self.norm(self.mid(x, "encoder.mid_block"), "encoder.norm_out")),
            "encoder.conv_out",
        )

    def decoder(self, x, first):
        c = self.c
        x = self.mid(self.causal(x, "decoder.conv_in"), "decoder.mid_block")
        for i, mult in enumerate(c.dim_mult[::-1]):
            up = i < len(c.dim_mult) - 1
            temporal = c.temperal_downsample[::-1][i] if up else False
            residual = (
                self.dup_up(x, c.decoder_base_dim * mult, 2 if temporal else 1, first)
                if up
                else None
            )
            name = f"decoder.up_blocks.{i}"
            for j in range(c.num_res_blocks + 1):
                x = self.resnet(x, f"{name}.resnets.{j}")
            if up:
                x = self.resample(x, name + ".upsampler", True, temporal) + residual
        return self.causal(F.silu(self.norm(x, "decoder.norm_out")), "decoder.conv_out")

    def run(self, video):
        b, c, t, h, w = video.shape
        p = self.c.patch_size
        x = (
            video.view(b, c, t, h // p, p, w // p, p)
            .permute(0, 1, 6, 4, 2, 3, 5)
            .contiguous()
            .view(b, c * p * p, t, h // p, w // p)
        )
        pieces = [self.encoder(x[:, :, :1])]
        for start in range(1, t, 4):
            pieces.append(self.encoder(x[:, :, start : start + 4]))
        mean, logvar = self.conv(torch.cat(pieces, 2), "quant_conv").chunk(2, 1)
        latent = self.conv(mean, "post_quant_conv")
        self.cache = {}
        output = torch.cat(
            [self.decoder(latent[:, :, i : i + 1], i == 0) for i in range(latent.shape[2])], 2
        )
        b, c, t, h, w = output.shape
        output = (
            output.view(b, c // (p * p), p, p, t, h, w)
            .permute(0, 1, 4, 5, 3, 6, 2)
            .contiguous()
            .view(b, c // (p * p), t, h * p, w * p)
        )
        return output.clamp(-1, 1), mean, logvar.clamp(-30, 20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_models_wan22_vae_full_official_weight_layout_formula_and_gradients(dtype):
    torch.set_num_threads(1)
    torch.manual_seed(556)
    config = Wan22VAEConfig()
    model = Wan22VideoVAE(config).to(dtype)

    video = torch.randn(1, 3, 9, 16, 32, dtype=dtype).requires_grad_()
    original = video.detach().clone().requires_grad_()
    weights = {
        name: value.detach().clone().requires_grad_() for name, value in model.state_dict().items()
    }
    oracle = Formula(config, weights)
    expected = oracle.run(original)
    output, posterior = model(video, sample_posterior=False)
    actual = (output, posterior.mean, posterior.logvar)
    tolerance = (
        dict(atol=2e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=0.005, rtol=0.02)
    )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, **tolerance)
    probes = [torch.randn_like(value) for value in actual]
    sum((value.float() * probe.float()).mean() for value, probe in zip(actual, probes)).backward()
    sum((value.float() * probe.float()).mean() for value, probe in zip(expected, probes)).backward()
    tolerance = dict(atol=3e-6, rtol=6e-4) if dtype == torch.float32 else dict(atol=0.01, rtol=0.15)
    torch.testing.assert_close(video.grad, original.grad, **tolerance)
    assert video.grad[:, :, :1].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and weights[name].grad is not None, name
        torch.testing.assert_close(parameter.grad, weights[name].grad, msg=name, **tolerance)
