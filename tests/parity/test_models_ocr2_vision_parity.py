from dataclasses import asdict
import pytest
import torch
import torch.nn.functional as F
from aster.models import OCR2SAMConfig, OCR2VisualConfig, build_model
from aster.nn.parameter_codec import public_parameter_names


def sam_name(name):
    if name == "pos_embed":
        return name
    if name.startswith("patch_embed.proj."):
        return name.replace("patch_embed.proj.", "patch_embed.projection.", 1)
    if name.startswith("blocks."):
        return (
            name.replace("blocks.", "layers.", 1)
            .replace(".norm1.", ".layer_norm1.")
            .replace(".norm2.", ".layer_norm2.")
        )
    for old, new in (
        ("neck.0.", "neck.conv1."),
        ("neck.1.", "neck.layer_norm1."),
        ("neck.2.", "neck.conv2."),
        ("neck.3.", "neck.layer_norm2."),
    ):
        if name.startswith(old):
            return new + name[len(old) :]
    return name


@pytest.mark.oracle
@pytest.mark.parametrize("side", [24, 32])
def test_models_ocr2_sam_official_blocks_and_full_gradient(side):
    tf = pytest.importorskip("transformers")
    from transformers.models.sam.modeling_sam import SamVisionEncoder

    torch.set_num_threads(1)
    torch.manual_seed(511)
    c = OCR2SAMConfig()
    native = build_model(c)
    with torch.no_grad():
        native.position.pos_embed.normal_(0, 0.04)
        for block in native.blocks:
            block.attn.relative.rel_pos_h.normal_(0, 0.03)
            block.attn.relative.rel_pos_w.normal_(0, 0.03)
    config = tf.SamVisionConfig(
        hidden_size=c.hidden_size,
        output_channels=c.neck_channels,
        num_hidden_layers=c.depth,
        num_attention_heads=c.num_heads,
        num_channels=c.in_channels,
        image_size=c.image_size,
        patch_size=c.patch_size,
        mlp_dim=c.intermediate_size,
        layer_norm_eps=c.norm_eps,
        window_size=c.window_size,
        global_attn_indexes=list(c.global_attn_indexes),
    )
    config._attn_implementation = "sdpa"
    oracle = SamVisionEncoder(config)
    source = native.state_dict()
    oracle.load_state_dict(
        {sam_name(k): v for k, v in source.items() if not k.startswith(("net_2.", "net_3."))},
        strict=True,
    )
    net2, net3 = (source[f"net_{i}.weight"].detach().clone().requires_grad_() for i in (2, 3))
    left = torch.randn(1, 3, side, side, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    actual = native(left)
    hidden = oracle.patch_embed.projection(right).permute(0, 2, 3, 1)
    position = oracle.pos_embed
    if position.shape[1] != hidden.shape[1]:
        position = F.interpolate(
            position.permute(0, 3, 1, 2).float(),
            size=hidden.shape[1:3],
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).permute(0, 2, 3, 1)
    hidden = hidden + position
    for layer in oracle.layers:
        hidden = layer(hidden)
    expected = F.conv2d(
        F.conv2d(oracle.neck(hidden), net2, stride=2, padding=1), net3, stride=2, padding=1
    )
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=5e-5)
    factor = torch.randn_like(actual)
    (actual * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=3e-4)
    mapped = public_parameter_names(native)
    reference = dict(oracle.named_parameters())
    reference.update({"net_2.weight": net2, "net_3.weight": net3})
    for name, p in native.named_parameters():
        torch.testing.assert_close(
            p.grad, reference[sam_name(mapped[name])].grad, atol=6e-5, rtol=8e-4, msg=name
        )


@pytest.mark.oracle
@pytest.mark.parametrize("local", [False, True])
def test_models_ocr2_causal_query_real_official_qwen2(local):
    tf = pytest.importorskip("transformers")
    from aster.models.ocr2_vision import OCR2QueryEncoder

    torch.set_num_threads(1)
    torch.manual_seed(512)
    c = OCR2VisualConfig()
    text = c.decoder_config
    config = tf.Qwen2Config(
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        intermediate_size=text.intermediate_size,
        num_hidden_layers=text.num_hidden_layers,
        num_attention_heads=text.num_attention_heads,
        num_key_value_heads=text.num_key_value_heads,
        max_position_embeddings=text.max_position_embeddings,
        rms_norm_eps=text.rms_norm_eps,
        rope_theta=text.rope.theta,
        attention_dropout=text.attention_dropout,
    )
    config._attn_implementation = "sdpa"
    oracle = tf.Qwen2Model(config)
    del oracle.embed_tokens
    native = OCR2QueryEncoder(c)
    oracle.load_state_dict(native.model.model.state_dict(), strict=True)
    side = 3 if local else 4
    left = torch.randn(2, text.hidden_size, side, side, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    query = (
        (native.query_768 if local else native.query_1024).weight.detach().clone().requires_grad_()
    )
    x = right.flatten(2).transpose(1, 2)
    n = x.shape[1]
    combined = torch.cat((x, query[None].expand(len(x), -1, -1)), 1)

    mask = torch.full((2 * n, 2 * n), -torch.inf)
    mask[:n, :n] = 0
    for index in range(n):
        mask[n + index, : n + index + 1] = 0
    expected = oracle(
        inputs_embeds=combined,
        attention_mask=mask[None, None].expand(2, -1, -1, -1),
        use_cache=False,
    ).last_hidden_state[:, n:]
    actual = native(left)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=4e-5)
    factor = torch.randn_like(actual)
    (actual * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=4e-4)
    for name, p in native.model.model.named_parameters():
        torch.testing.assert_close(
            p.grad, dict(oracle.named_parameters())[name].grad, atol=4e-5, rtol=8e-4, msg=name
        )
    torch.testing.assert_close(
        (native.query_768 if local else native.query_1024).weight.grad,
        query.grad,
        atol=1e-5,
        rtol=3e-4,
    )
