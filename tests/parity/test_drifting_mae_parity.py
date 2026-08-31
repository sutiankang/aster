import torch
import torch.nn.functional as F

from aster.models.drifting_features import MAEResNet, MAEResNetConfig
from aster.methods.masked_autoencoding import MaskedAutoencodingObjective


def _source_function(params, x, mask, config):
    def convolution(x, key, stride=1, kernel=3):
        return F.conv2d(x, params[key + ".weight"], params.get(key + ".bias"), stride, kernel // 2)

    def normalize(x, key, group_count=None):
        channels = x.shape[1]
        groups = min(32, channels) if group_count is None else group_count
        while channels % groups:
            groups -= 1

        reshaped = x.reshape(len(x), groups, -1)
        mean = reshaped.mean(-1, keepdim=True)
        variance = ((reshaped - mean) ** 2).mean(-1, keepdim=True)
        y = ((reshaped - mean) / (variance + 1e-6).sqrt()).reshape_as(x)
        return (
            y * params[key + ".weight"][None, :, None, None]
            + params[key + ".bias"][None, :, None, None]
        )

    def cgr(x, key):
        return F.relu(normalize(convolution(x, key + ".0"), key + ".1"))

    x = F.relu(normalize(convolution(x * (~mask), "encoder.conv1"), "encoder.gn1"))
    features = dict(conv1=x)
    for level, count in enumerate(config.layers):
        for index in range(count):
            key = f"encoder.stages.{level}.{index}"
            stride = 2 if level > 0 and index == 0 else 1
            skip = x
            y = F.relu(normalize(convolution(x, key + ".conv1", stride=stride), key + ".gn1"))
            y = normalize(convolution(y, key + ".conv2"), key + ".gn2")
            if skip.shape != y.shape:
                skip = normalize(
                    convolution(skip, key + ".projection.0", stride, 1), key + ".projection.1"
                )
            x = F.relu(skip + y)
        x = normalize(x, f"encoder.stage_norms.{level}")
        features[f"layer{level + 1}"] = x
    logits = F.linear(x.mean((2, 3)), params["classifier.weight"], params["classifier.bias"])
    x = cgr(x, "decoder.bridge")
    for name, skip in [
        ("up43", "layer3"),
        ("up32", "layer2"),
        ("up21", "layer1"),
        ("up10", "conv1"),
    ]:
        key = "decoder." + name
        x = F.interpolate(x, size=features[skip].shape[-2:], mode="bilinear", align_corners=False)
        x = normalize(torch.cat([x, features[skip]], 1), key + ".concat_norm", 32)
        x = cgr(cgr(x, key + ".proj"), key + ".refine")
    return convolution(x, "decoder.head", kernel=1), logits


def test_author_mae_graph_full_forward_and_all_parameter_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(680)
    config = MAEResNetConfig(
        in_channels=2, num_classes=3, base_channels=32, layers=(1, 1, 1, 1), patch_size=2
    )
    model = MAEResNet(config)
    parameters = {
        name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()
    }
    samples = torch.randn(2, 2, 8, 8, requires_grad=True)
    independent = samples.detach().clone().requires_grad_()
    mask = torch.rand(2, 1, 4, 4).gt(0.4).repeat_interleave(2, -2).repeat_interleave(2, -1)
    output = model(samples, mask)
    reconstruction, logits = _source_function(parameters, independent, mask, config)
    torch.testing.assert_close(output.reconstruction, reconstruction, atol=1e-5, rtol=5e-5)
    torch.testing.assert_close(output.logits, logits, atol=1e-5, rtol=5e-5)
    labels = torch.tensor([0, 2])
    actual = MaskedAutoencodingObjective(classification_weight=0.2)(
        model, dict(samples=samples, mask=mask, labels=labels)
    )
    sum(term.mean * term.weight for term in actual.terms).backward()
    expected = (
        ((reconstruction - independent).square() * mask).sum((1, 2, 3))
        / (mask.sum((1, 2, 3)) + 1e-8)
    ).mean() * 0.8
    expected = expected + 0.2 * F.cross_entropy(logits, labels)
    expected.backward()
    torch.testing.assert_close(samples.grad, independent.grad, atol=3e-5, rtol=2e-4)
    for name, value in model.named_parameters():
        assert value.grad is not None and parameters[name].grad is not None
        torch.testing.assert_close(
            value.grad, parameters[name].grad, atol=5e-5, rtol=3e-4, msg=name
        )
