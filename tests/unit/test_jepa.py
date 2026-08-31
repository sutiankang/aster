import copy
import torch
from aster.models.jepa import (
    JEPAEncoderConfig,
    JEPAConfig,
    JEPAModel,
    JEPAEncoder,
    JEPABlock,
    tube_masks,
    jepa_positions,
)
from aster.methods.jepa import JEPAMethod
from aster.training import Trainer


def tiny():
    return JEPAConfig(
        encoder=JEPAEncoderConfig(
            image_size=8,
            num_frames=4,
            patch_size=4,
            tubelet_size=2,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
        ),
        predictor_hidden_size=16,
        predictor_intermediate_size=24,
        predictor_layers=1,
        predictor_heads=2,
    )


def test_jepa_transformer_reference_forward_and_all_grads():
    torch.manual_seed(16)
    block = JEPABlock(16, 24, 2)
    reference = torch.nn.TransformerEncoderLayer(
        16,
        2,
        dim_feedforward=24,
        dropout=0.0,
        activation="gelu",
        layer_norm_eps=1e-6,
        batch_first=True,
        norm_first=True,
    )
    pairs = [
        (block.qkv, reference.self_attn),
        (block.proj, reference.self_attn.out_proj),
        (block.fc1, reference.linear1),
        (block.fc2, reference.linear2),
        (block.norm1, reference.norm1),
        (block.norm2, reference.norm2),
    ]
    with torch.no_grad():
        for source, target in pairs:
            if target is reference.self_attn:
                target.in_proj_weight.copy_(source.weight)
                target.in_proj_bias.copy_(source.bias)
            else:
                target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = block(x), reference(y)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-7, rtol=2e-5)
    for source, target in pairs:
        target_weight = target.in_proj_weight if target is reference.self_attn else target.weight
        target_bias = target.in_proj_bias if target is reference.self_attn else target.bias
        torch.testing.assert_close(source.weight.grad, target_weight.grad, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(source.bias.grad, target_bias.grad, atol=2e-7, rtol=2e-5)


def test_context_selected_before_attention_and_tube_semantics():
    torch.set_num_threads(1)
    encoder = JEPAEncoder(tiny().encoder)
    context = torch.tensor([[0, 4]])  # same top-left spatial tube in both temporal positions
    pixels = torch.randn(1, 3, 4, 8, 8)
    changed = pixels.clone()
    changed[..., 4:, 4:] += 100
    torch.testing.assert_close(encoder(pixels, context), encoder(changed, context))
    assert not torch.allclose(encoder(pixels), encoder(changed))
    context, target = tube_masks(2, tiny().encoder.grid, keep_fraction=0.5)
    assert context.shape == target.shape == (2, 4)
    for first, second in zip(context, target):
        assert sorted(torch.cat((first, second)).tolist()) == list(range(8))
    # Position dimensions use temporal half first; times differ while spatial axes repeat.
    position = jepa_positions((2, 2, 2), 16)[0]
    torch.testing.assert_close(position[0, 8:], position[4, 8:])
    assert not torch.equal(position[0, :8], position[4, :8])


def test_jepa_training_target_encoder_only_and_resume(tmp_path):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    model = JEPAModel(tiny())
    engine = Trainer(model, lr=0.001, zero_stage=3)
    method = JEPAMethod(engine, total_updates=5, regularization_weight=0.1)
    context, target = tube_masks(2, model.config.encoder.grid, keep_fraction=0.5)
    batch = {
        "pixel_values": torch.randn(2, 3, 4, 8, 8),
        "context_indices": context,
        "target_indices": target,
    }
    assert method.update([batch]).updated
    assert not hasattr(engine.roles["target_encoder"].model, "predictor")
    engine.save_checkpoint(tmp_path / "checkpoint")
    expected = method.update([batch])
    weights = engine.export_state_dict()
    engine.load_checkpoint(tmp_path / "checkpoint", trusted=True)
    actual = method.update([batch])
    assert abs(actual.loss - expected.loss) < 1e-7
    for name, tensor in engine.export_state_dict().items():
        torch.testing.assert_close(tensor, weights[name])
