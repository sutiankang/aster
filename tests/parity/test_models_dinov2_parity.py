from dataclasses import asdict
import pytest
import torch
from aster.models.dinov2 import DinoVisionConfig, DinoVisionModel


def dino_oracle_state(native):
    c = native.config
    source = native.state_dict()

    result = {
        "embeddings.cls_token": source["cls_token"],
        "embeddings.register_tokens": source["reg_token"],
        "embeddings.mask_token": torch.zeros(1, c.hidden_size),
        "embeddings.position_embeddings": torch.cat(
            (torch.zeros(1, 1, c.hidden_size), source["pos_embed"]), 1
        ),
    }
    for suffix in ("weight", "bias"):
        result["embeddings.patch_embeddings.projection." + suffix] = source[
            "patch_embed.proj." + suffix
        ]
        result["layernorm." + suffix] = source["norm." + suffix]
    for index in range(c.num_hidden_layers):
        a, b = f"blocks.{index}.", f"encoder.layer.{index}."
        for part in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
            for suffix in ("weight", "bias"):
                result[b + part + "." + suffix] = source[a + part + "." + suffix]
        for source_name, dest in (
            ("ls1.scale_factor", "layer_scale1.lambda1"),
            ("ls2.scale_factor", "layer_scale2.lambda1"),
        ):
            result[b + dest] = source[a + source_name]
        for suffix in ("weight", "bias"):
            if a + "attn.qkv." + suffix in source:
                values = source[a + "attn.qkv." + suffix]
                for label, tensor in zip(
                    ("query", "key", "value"),
                    (None, None, None) if values is None else values.chunk(3, 0),
                ):
                    result[b + "attention.attention." + label + "." + suffix] = tensor
            result[b + "attention.output.dense." + suffix] = source[a + "attn.proj." + suffix]
    return result


def dino_oracle(tf, native):
    c = native.config
    values = asdict(c)
    values.pop("intermediate_size")
    values["mlp_ratio"] = c.intermediate_size // c.hidden_size
    oracle = tf.Dinov2WithRegistersModel(
        tf.Dinov2WithRegistersConfig(**values, _attn_implementation="eager")
    )
    oracle.load_state_dict(dino_oracle_state(native), strict=True)
    return oracle


@pytest.mark.oracle
def test_models_dinov2_registers_official_weights_intermediate_and_gradients():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(57)
    model = DinoVisionModel(DinoVisionConfig(layerscale_value=0.1))
    oracle = dino_oracle(tf, model)
    pixels = torch.randn(2, 3, 8, 8, requires_grad=True)
    other = pixels.detach().clone().requires_grad_()
    left, right = model(pixels, output_hidden_states=True), oracle(other, output_hidden_states=True)
    torch.testing.assert_close(
        left.last_hidden_state, right.last_hidden_state, atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        model.patch_features(pixels), right.hidden_states[-2][:, 5:], atol=2e-6, rtol=2e-5
    )
    factor = torch.randn_like(left.last_hidden_state)
    (left.last_hidden_state * factor).sum().backward()
    (right.last_hidden_state * factor).sum().backward()
    torch.testing.assert_close(pixels.grad, other.grad, atol=4e-5, rtol=4e-4)

    grads = {name: p.grad for name, p in model.named_parameters()}

    class GradView:
        config = model.config

        def state_dict(self):
            return grads

    mapped = dino_oracle_state(GradView())
    for name, p in oracle.named_parameters():
        if name == "embeddings.mask_token":
            continue
        if name == "embeddings.position_embeddings":
            torch.testing.assert_close(mapped[name][:, 1:], p.grad[:, 1:], atol=6e-5, rtol=5e-4)
        else:
            torch.testing.assert_close(mapped[name], p.grad, atol=6e-5, rtol=5e-4, msg=name)
