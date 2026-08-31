from dataclasses import asdict
import pytest
import torch
from aster.models import BertConfig, T5Config, build_model


@pytest.mark.oracle
@pytest.mark.parametrize(
    "family,config",
    [
        ("Bert", BertConfig()),
        ("T5", T5Config()),
        ("T5", T5Config(feed_forward_proj="gated-gelu")),
        ("T5", T5Config(feed_forward_proj="gated-silu")),
    ],
)
def test_models_encoder_official_forward_gradient(family, config):
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    torch.manual_seed(13)
    official_config = getattr(tf, family + "Config")(**asdict(config))
    official_config._attn_implementation = "eager"
    official = getattr(
        tf, family + ("ForMaskedLM" if family == "Bert" else "ForConditionalGeneration")
    )(official_config)
    model = build_model(config)
    official.load_state_dict(model.state_dict(), strict=True)
    source = torch.tensor([[1, 3, 5, 9], [5, 7, 2, 0]])
    padding = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    kwargs = {"input_ids": source, "attention_mask": padding}
    if family == "T5":
        kwargs["decoder_input_ids"] = torch.tensor([[0, 3, 4], [0, 5, 6]])
    else:
        kwargs["token_type_ids"] = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    left, right = model(**kwargs).logits, official(**kwargs).logits
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    coefficients = torch.randn_like(left)
    (left * coefficients).sum().backward()
    (right * coefficients).sum().backward()
    oracle_parameters = dict(official.named_parameters())
    for name, parameter in model.named_parameters():
        alias = "cls.predictions.bias" if name == "cls.predictions.decoder.bias" else name
        torch.testing.assert_close(
            parameter.grad,
            oracle_parameters.get(name, oracle_parameters.get(alias)).grad,
            atol=3e-5,
            rtol=3e-4,
            msg=name,
        )


@pytest.mark.oracle
def test_models_t5_official_incremental_cache():
    tf = pytest.importorskip("transformers")
    torch.set_num_threads(1)
    config = T5Config()
    model = build_model(config).eval()
    oracle = tf.T5ForConditionalGeneration(tf.T5Config(**asdict(config))).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    source, target = torch.tensor([[1, 5, 9]]), torch.tensor([[0, 3, 8, 6]])
    first = model(source, decoder_input_ids=target[:, :2], use_cache=True)
    reference = oracle(source, decoder_input_ids=target[:, :2], use_cache=True)
    expected = oracle(
        encoder_outputs=(reference.encoder_last_hidden_state,),
        decoder_input_ids=target[:, 2:],
        past_key_values=reference.past_key_values,
    ).logits
    actual = model(decoder_input_ids=target[:, 2:], state=first.state).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
