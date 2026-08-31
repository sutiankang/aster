import random
import torch
from aster.data.pretraining import (
    masked_language_modeling,
    span_corruption,
    audit_split_overlap,
    WeightedMixture,
)


def test_mlm_specials_are_never_prediction_targets():
    ids = torch.tensor([[1, 5, 6, 7, 0]] * 100)
    result = masked_language_modeling(
        ids,
        ids != 0,
        mask_token_id=3,
        vocab_size=20,
        special_token_ids=(0, 1),
        generator=torch.Generator().manual_seed(7),
    )
    assert (result["labels"][:, 0] == -100).all() and (result["labels"][:, -1] == -100).all()
    selected = result["labels"] != -100
    assert selected.any()
    torch.testing.assert_close(result["labels"][selected], ids[selected])


def test_t5_spans_can_reconstruct_original_tokens():
    source = list(range(10, 30))
    result = span_corruption(
        source,
        sentinel_ids=list(range(99, 89, -1)),
        eos_token_id=1,
        decoder_start_token_id=0,
        noise_density=0.3,
        rng=random.Random(2),
    )
    replacement = {}
    current = None
    for token in result["labels"][:-1]:
        if token >= 90:
            current = token
            replacement[current] = []
        else:
            replacement[current].append(token)
    reconstructed = []
    for token in result["input_ids"][:-1]:
        reconstructed.extend(replacement[token] if token in replacement else [token])
    assert reconstructed == source
    assert result["decoder_input_ids"][1:] == result["labels"][:-1]


def test_split_audit_does_not_silently_drop_leaks():
    result = audit_split_overlap(["Hello   WORLD", "training"], ["hello world", "heldout"])
    assert not result["passed"] and result["collisions"][0]["evaluation_index"] == 0
