from dataclasses import asdict, replace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from aster.models import OpenVLAConfig, build_model
from test_models_dinov2_parity import dino_oracle, dino_oracle_state


@pytest.mark.oracle
def test_models_openvla_composed_official_forward_gradients_and_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(58)
    c = OpenVLAConfig()
    c = replace(c, dino_config=replace(c.dino_config, layerscale_value=0.1))
    native = build_model(c)
    dino = dino_oracle(tf, native.vision_backbone.featurizer)
    siglip = tf.SiglipVisionModel(
        tf.SiglipVisionConfig(**asdict(c.siglip_config), _attn_implementation="eager")
    )
    siglip.load_state_dict(native.vision_backbone.fused_featurizer.state_dict(), strict=True)
    tc = c.text_config
    language = tf.LlamaForCausalLM(
        tf.LlamaConfig(
            vocab_size=tc.vocab_size,
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            num_hidden_layers=tc.num_hidden_layers,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=tc.num_key_value_heads,
            rms_norm_eps=tc.rms_norm_eps,
            max_position_embeddings=tc.max_position_embeddings,
            tie_word_embeddings=tc.tie_word_embeddings,
            rope_parameters={"rope_type": "default", "rope_theta": tc.rope.theta},
            _attn_implementation="eager",
        )
    )
    language.load_state_dict(native.language_model.state_dict(), strict=True)
    projectors = nn.ModuleList(
        nn.Linear(layer.in_features, layer.out_features)
        for layer in (native.projector.fc1, native.projector.fc2, native.projector.fc3)
    )
    for a, b in zip((native.projector.fc1, native.projector.fc2, native.projector.fc3), projectors):
        b.load_state_dict(a.state_dict(), strict=True)
    left_pixels = torch.randn(2, 6, 8, 8, requires_grad=True)
    right_pixels = left_pixels.detach().clone().requires_grad_()
    ids = torch.tensor([[1, 3, 5, 7, 2], [1, 4, 6, 8, 2]])
    features = torch.cat(
        (
            dino(right_pixels[:, :3], output_hidden_states=True).hidden_states[-2][:, 5:],
            siglip(right_pixels[:, 3:], output_hidden_states=True).hidden_states[-2],
        ),
        -1,
    )
    projected = projectors[2](F.gelu(projectors[1](F.gelu(projectors[0](features)))))
    embedded = language.get_input_embeddings()(ids)
    embedded = torch.cat((embedded[:, :1], projected, embedded[:, 1:]), 1)
    left = native(ids, pixel_values=left_pixels, use_cache=True)
    right = language(inputs_embeds=embedded, use_cache=True)
    torch.testing.assert_close(left.logits, right.logits, atol=4e-6, rtol=4e-5)
    factor = torch.randn_like(left.logits)
    (left.logits * factor).sum().backward()
    (right.logits * factor).sum().backward()
    torch.testing.assert_close(left_pixels.grad, right_pixels.grad, atol=5e-5, rtol=5e-4)
    for prefix, other in (
        ("language_model.", language),
        ("vision_backbone.fused_featurizer.", siglip),
    ):
        for name, p in other.named_parameters():
            torch.testing.assert_close(
                dict(native.named_parameters())[prefix + name].grad,
                p.grad,
                atol=8e-5,
                rtol=8e-4,
                msg=prefix + name,
            )
    for original, reference in zip(
        (native.projector.fc1, native.projector.fc2, native.projector.fc3), projectors
    ):
        for name, p in original.named_parameters():
            torch.testing.assert_close(
                p.grad, dict(reference.named_parameters())[name].grad, atol=8e-5, rtol=8e-4
            )
    grads = {name: p.grad for name, p in native.vision_backbone.featurizer.named_parameters()}

    class GradientView:
        config = c.dino_config

        def state_dict(self):
            return grads

    mapped = dino_oracle_state(GradientView())
    for name, p in dino.named_parameters():
        if name == "embeddings.mask_token":
            continue
        if name == "embeddings.position_embeddings":
            torch.testing.assert_close(mapped[name][:, 1:], p.grad[:, 1:], atol=8e-5, rtol=8e-4)
        else:
            torch.testing.assert_close(mapped[name], p.grad, atol=8e-5, rtol=8e-4, msg=name)
    suffix = torch.tensor([[9, 2], [7, 2]])
    torch.testing.assert_close(
        native(suffix, state=left.state, use_cache=True).logits,
        language(suffix, past_key_values=right.past_key_values, use_cache=True).logits,
        atol=4e-6,
        rtol=4e-5,
    )
