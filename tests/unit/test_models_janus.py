import pytest
import torch
import torch.nn.functional as F
from aster.models import JanusConfig, JanusVQConfig, JanusVisionConfig, build_model, load_model


def test_models_janus_dual_path_training_cache_store(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(35)
    model = build_model(JanusConfig())
    pixels = torch.randn(1, 3, 8, 8)
    ids = torch.tensor([[1] + [31] * 16 + [4, 7]])
    result = model(ids, pixel_values=pixels)
    F.cross_entropy(result.logits[:, -1], torch.tensor([2])).backward()
    assert model.model.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    model.zero_grad()
    encoded = model.model.vqmodel.encode(pixels).image_tokens
    prefix = model(torch.tensor([[1, 3]]), use_cache=True, output_kind="image_codes")
    image_embeddings = model.prepare_embeddings_for_image_generation(encoded[:, :-1])
    generated = model(inputs_embeds=image_embeddings, state=prefix.state, output_kind="image_codes")
    F.cross_entropy(generated.logits.flatten(0, 1), encoded[:, 1:].flatten()).backward()
    assert model.model.generation_embeddings.weight.grad.abs().sum() > 0
    assert model.model.generation_head.vision_head.weight.grad.abs().sum() > 0
    assert model.decode_image_tokens(encoded).shape == pixels.shape
    model.save_pretrained(tmp_path / "janus")
    torch.testing.assert_close(
        result.logits, load_model(tmp_path / "janus")(ids, pixel_values=pixels).logits
    )


def test_models_janus_vq_train_and_detach_directions(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(36)
    model = build_model(JanusVQConfig())
    pixels = torch.randn(2, 3, 8, 8)
    reconstruction, result = model.reconstruct(pixels)

    loss = (
        F.mse_loss(reconstruction, pixels)
        + result.commitment_errors.mean()
        + model.config.beta * result.codebook_errors.mean()
    )
    loss.backward()
    assert model.encoder.conv_in.weight.grad.abs().sum() > 0
    assert model.quantize.embedding.weight.grad.abs().sum() > 0
    assert model.decoder.conv_out.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path / "vq")
    torch.testing.assert_close(
        model.decode(result.image_tokens), load_model(tmp_path / "vq").decode(result.image_tokens)
    )
    with pytest.raises(ValueError):
        model.decode(result.image_tokens[:, :-1])
    with pytest.raises(ValueError):
        JanusVisionConfig(use_qk_norm=True)
