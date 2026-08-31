import json
import pytest
import torch
from aster.data import (
    ByteBPETokenizer,
    WordPieceTokenizer,
    UnigramTokenizer,
    load_tokenizer,
    JsonlDataset,
    StatefulSampler,
    causal_collate,
    pack_documents,
)


def test_native_byte_bpe_training_roundtrip(tmp_path):
    texts = ["hello hello 中文🙂", "hello world 中文🙂"] * 4
    tokenizer = ByteBPETokenizer.train(texts, vocab_size=290)
    assert tokenizer.vocab_size > 259
    for text in [*texts, "unseen 🐈 words"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(texts[0])) < len(texts[0].encode("utf-8"))
    tokenizer.save_pretrained(tmp_path)
    restored = load_tokenizer(tmp_path)
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode(texts[0]) == tokenizer.encode(texts[0])


def test_wordpiece_and_global_unigram_optimum():
    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "##s", "中", ","]
        )
    }
    tokenizer = WordPieceTokenizer(vocabulary)
    assert tokenizer.encode("Héllos, 中", False) == [5, 6, 8, 7]
    assert tokenizer.encode("helloworld", False) == [1]
    unigram = UnigramTokenizer([("<unk>", -10), ("a", -0.1), ("b", -0.1), ("ab", -2)])
    assert unigram.encode("ab") == [1, 2]


def test_sharding_resume_and_data_mutation(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps({"id": i}) for i in range(9)), encoding="utf-8")
    data = JsonlDataset(path)
    first = StatefulSampler(data, rank=0, world_size=2, seed=7)
    second = StatefulSampler(data, rank=1, world_size=2, seed=7)
    prefix, state = first.take(2), first.state_dict()
    suffix = first.take(99)
    restored = StatefulSampler(data, rank=0, world_size=2, seed=7)
    restored.load_state_dict(state)
    assert restored.take(99) == suffix
    assert sorted(x["id"] for x in prefix + suffix + second.take(99)) == list(range(9))
    path.write_text('{"id":99}', encoding="utf-8")
    with pytest.raises(ValueError):
        restored.load_state_dict(state)


def test_packing_and_padding_do_not_lose_boundary_targets():
    records = list(pack_documents([[3, 4, 5], [6, 7, 8]], length=3))
    targets = [t for record in records for t in record["input_ids"][1:]]
    assert targets == [4, 5, 2, 6, 7, 8, 2]
    batch = causal_collate(
        [{"input_ids": [1, 3, 4], "labels": [-100, -100, 4]}, {"input_ids": [1, 9]}], multiple_of=4
    )
    assert torch.equal(
        batch["labels"], torch.tensor([[-100, -100, 4, -100], [-100, 9, -100, -100]])
    )
