import pytest
import torch
from aster.models import BertConfig, T5Config, build_model, load_model
from aster.models.t5 import relative_position_bucket


def test_models_bert_mlm_bidirectional_and_storage(tmp_path):
    torch.set_num_threads(1)
    model = build_model(BertConfig()).eval()
    tokens = torch.tensor([[1, 3, 5, 7]])
    output = model(tokens, output_hidden_states=True)
    altered = tokens.clone()
    altered[:, -1] = 9
    assert not torch.equal(output.logits[:, 0], model(altered).logits[:, 0])
    assert len(output.hidden_states) == model.config.num_hidden_layers + 1
    output.logits.square().mean().backward()
    assert model.bert.encoder.layer[0].attention.self.query.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(
        load_model(tmp_path).eval()(tokens).logits, output.logits, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="Bidirectional"):
        model(tokens, use_cache=True)


@pytest.mark.parametrize("projection", ["relu", "gated-gelu", "gated-silu"])
def test_models_t5_cache_condition_and_storage(tmp_path, projection):
    torch.set_num_threads(1)
    model = build_model(T5Config(feed_forward_proj=projection)).eval()
    source, target = torch.tensor([[4, 2, 9, 1]]), torch.tensor([[0, 5, 7, 3]])
    full = model(source, decoder_input_ids=target, output_hidden_states=True)
    first = model(source, decoder_input_ids=target[:, :2], use_cache=True)
    tail = model(decoder_input_ids=target[:, 2:], state=first.state, use_cache=True)
    torch.testing.assert_close(full.logits[:, 2:], tail.logits, atol=2e-6, rtol=2e-5)
    assert first.state.seen_tokens == 2 and tail.state.seen_tokens == 4
    assert tail.state.truncate(2).layers[0][0].shape[-2] == 2
    assert tail.state.truncate(2).layers[0][2].shape[-2] == source.shape[1]
    assert len(full.hidden_states) == model.config.num_decoder_layers + 1
    with pytest.raises(ValueError, match="condition is fixed"):
        model(source, decoder_input_ids=target[:, 2:], state=first.state)
    torch.testing.assert_close(
        model.shift_right(torch.tensor([[4, -100, 2]])), torch.tensor([[0, 4, 0]])
    )
    full.logits.square().mean().backward()
    assert model.encoder.block[0].layer[0].SelfAttention.q.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    torch.testing.assert_close(
        load_model(tmp_path).eval()(source, decoder_input_ids=target).logits,
        full.logits,
        atol=0,
        rtol=0,
    )


def test_models_t5_buckets_edges():
    distances = torch.tensor([-10000, -20, -1, 0, 1, 20, 10000])
    result = relative_position_bucket(
        distances, bidirectional=True, num_buckets=16, max_distance=64
    )
    assert result.tolist() == [7, 6, 1, 0, 9, 14, 15]
