from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from aster.models import OpenVLAConfig, build_model, load_model
from aster.models.openvla import normalize_openvla_pixels, convert_prismatic_state_dict
from aster.data.actions import ActionSpec


def config():
    spec = ActionSpec(("dx", "dy", "gripper"), ("m", "m", "unitless"), "test-base", "delta", 10, 1)
    return OpenVLAConfig(
        action_spec=spec,
        norm_stats={
            "tiny": {
                "action": {
                    "q01": [-2.0, -4.0, 0.0],
                    "q99": [2.0, 4.0, 0.0],
                    "mask": [True, True, False],
                }
            }
        },
    )


def test_models_openvla_teacher_forcing_to_cached_actions_and_storage(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(59)
    c = config()
    model = build_model(c)
    pixels = normalize_openvla_pixels(torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8))
    prompt = model.prepare_action_prompt(torch.tensor([[1, 3, 5]]))
    physical = torch.tensor([[0.4, -1.0, 0.5]])
    targets = model.action_tokens(physical)
    ids = torch.cat((prompt, targets), 1)
    labels = torch.cat((torch.full_like(prompt, -100), targets), 1)
    labels = model.align_labels(labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        out = model(ids, pixel_values=pixels)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, c.text_config.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.15 * losses[0]
    model.eval()
    out = model(prompt, pixel_values=pixels, use_cache=True)
    generated = []
    for _ in range(3):
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token)
        out = model(token, state=out.state, use_cache=True)
    generated = torch.cat(generated, 1)
    assert torch.equal(generated, targets)
    decoded = model.decode_actions(generated)
    assert decoded.spec == c.action_spec and decoded.statistics_key == "tiny"
    assert (decoded.actions - physical).abs().max() < 0.3
    model.save_pretrained(tmp_path / "vla")
    restored = load_model(tmp_path / "vla")
    torch.testing.assert_close(
        restored(ids, pixel_values=pixels).logits, model(ids, pixel_values=pixels).logits
    )
    torch.testing.assert_close(restored.decode_actions(generated).actions, decoded.actions)


def test_models_openvla_shapes_codec_and_native_timm_weight_conversion():
    torch.set_num_threads(1)
    model = build_model(config())
    raw = model.state_dict()
    official = {}
    prefix = "vision_backbone.fused_featurizer."

    for key, tensor in raw.items():
        if not key.startswith(prefix):
            official[key] = tensor
            continue
        tail = key[len(prefix) :]
        if tail == "embeddings.position_embedding.weight":
            official[prefix + "pos_embed"] = tensor[None]
        elif tail.startswith("embeddings.patch_embedding."):
            official[prefix + "patch_embed.proj." + tail.split(".")[-1]] = tensor
        elif tail.startswith("post_layernorm."):
            official[prefix + "norm." + tail.split(".")[-1]] = tensor
        elif tail.startswith("encoder.layers."):
            parts = tail.split(".")
            layer = parts[2]
            name = ".".join(parts[3:])
            base = prefix + f"blocks.{layer}."
            if name.startswith("self_attn.q_proj."):
                suffix = parts[-1]
                official[base + "attn.qkv." + suffix] = torch.cat(
                    [
                        raw[prefix + f"encoder.layers.{layer}.self_attn.{proj}_proj.{suffix}"]
                        for proj in ("q", "k", "v")
                    ]
                )
            elif name.startswith(("self_attn.k_proj.", "self_attn.v_proj.")):
                continue
            else:
                name = (
                    name.replace("layer_norm1.", "norm1.")
                    .replace("layer_norm2.", "norm2.")
                    .replace("self_attn.out_proj.", "attn.proj.")
                )
                official[base + name] = tensor
        else:
            raise AssertionError(tail)
    mapped, ignored = convert_prismatic_state_dict(official, model)
    assert not ignored
    for key in raw:
        torch.testing.assert_close(mapped[key], raw[key])
    with pytest.raises(ValueError):
        convert_prismatic_state_dict({**official, "unexpected.weight": torch.zeros(1)}, model)
    with pytest.raises(ValueError):
        model.decode_actions(torch.tensor([[1, 2, 3]]))
    compatibility = model.decode_actions(torch.tensor([[1, 2, 3]]), strict=False)
    assert torch.isfinite(compatibility.actions).all()
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2]]), pixel_values=torch.randn(1, 3, 8, 8))
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2]]), position_ids=torch.tensor([[0, 1]]))
