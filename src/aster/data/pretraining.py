"""Self-supervised corruption and deterministic mixtures without fitting on test splits."""

from dataclasses import dataclass
import hashlib
import math
import random
import re
import unicodedata
import torch
from ..core import digest_json


def masked_language_modeling(
    input_ids,
    attention_mask,
    *,
    mask_token_id,
    vocab_size,
    special_token_ids=(),
    probability=0.15,
    generator=None,
):
    if (
        input_ids.shape != attention_mask.shape
        or input_ids.ndim != 2
        or not 0 < probability < 1
        or not 0 <= mask_token_id < vocab_size
    ):
        raise ValueError("Invalid MLM configuration")
    available = attention_mask.bool()
    for token in special_token_ids:
        available &= input_ids != token
    selected = (
        torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < probability
    ) & available
    labels = input_ids.clone().masked_fill(~selected, -100)
    choices = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    corrupted = input_ids.clone()
    corrupted[selected & (choices < 0.8)] = mask_token_id
    random_words = selected & (choices >= 0.8) & (choices < 0.9)
    replacements = torch.randint(
        vocab_size, input_ids.shape, device=input_ids.device, generator=generator
    )
    corrupted[random_words] = replacements[random_words]

    return {"input_ids": corrupted, "attention_mask": attention_mask.clone(), "labels": labels}


def _positive_partition(total, count, rng):
    if not 1 <= count <= total:
        raise ValueError("Cannot partition into positive segments")
    boundaries = sorted(rng.sample(range(1, total), count - 1))
    points = [0, *boundaries, total]
    return [right - left for left, right in zip(points[:-1], points[1:])]


def span_corruption(
    tokens,
    *,
    sentinel_ids,
    eos_token_id,
    decoder_start_token_id,
    noise_density=0.15,
    mean_span_length=3.0,
    rng=None,
):

    rng = random.Random() if rng is None else rng
    tokens = list(tokens)
    n = len(tokens)
    if n < 2 or not 0 < noise_density < 1 or mean_span_length <= 0:
        raise ValueError("Invalid T5 corruption inputs")
    noisy = min(n - 1, max(1, round(n * noise_density)))
    spans = min(noisy, n - noisy, max(1, round(noisy / mean_span_length)))
    if (
        len(sentinel_ids) < spans + 1
        or len(set(sentinel_ids)) != len(sentinel_ids)
        or set(tokens) & set(sentinel_ids)
    ):
        raise ValueError("Insufficient or colliding sentinel vocabulary")
    noise_lengths = _positive_partition(noisy, spans, rng)
    clean_lengths = _positive_partition(n - noisy, spans, rng)
    inputs, targets, position = [], [], 0
    for index, (clean, masked) in enumerate(zip(clean_lengths, noise_lengths)):
        inputs.extend(tokens[position : position + clean])
        position += clean
        inputs.append(sentinel_ids[index])
        targets.extend([sentinel_ids[index], *tokens[position : position + masked]])
        position += masked
    inputs.append(eos_token_id)
    targets.extend((sentinel_ids[spans], eos_token_id))
    return {
        "input_ids": inputs,
        "labels": targets,
        "decoder_input_ids": [decoder_start_token_id, *targets[:-1]],
    }


def normalize_document(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def document_fingerprint(text):
    return hashlib.sha256(normalize_document(text).encode("utf-8")).hexdigest()


def audit_split_overlap(train_texts, evaluation_texts):
    index = {}
    for position, text in enumerate(train_texts):
        index.setdefault(document_fingerprint(text), []).append(position)
    collisions = []
    for position, text in enumerate(evaluation_texts):
        digest = document_fingerprint(text)
        if digest in index:
            collisions.append(
                {
                    "evaluation_index": position,
                    "training_indices": index[digest],
                    "fingerprint": digest,
                }
            )
    return {
        "method": "exact_nfkc_casefold_whitespace",
        "collisions": collisions,
        "passed": not collisions,
        "limitation": "Does not certify absence of paraphrases, near duplicates or pretraining contamination.",
    }


class WeightedMixture:
    """Choose each global item deterministically from its seed, independent of worker order."""

    def __init__(self, datasets, weights, *, size, seed=0, split="train"):
        self.datasets, self.weights, self.size, self.seed, self.split = (
            tuple(datasets),
            tuple(weights),
            size,
            seed,
            split,
        )
        if (
            not self.datasets
            or len(self.datasets) != len(self.weights)
            or size < 1
            or any(len(dataset) < 1 for dataset in datasets)
            or any(not math.isfinite(w) or w <= 0 for w in weights)
        ):
            raise ValueError("Invalid mixture datasets or weights")
        self.fingerprint = digest_json(
            {
                "datasets": [dataset.fingerprint for dataset in datasets],
                "weights": weights,
                "size": size,
                "seed": seed,
                "split": split,
            }
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        rng = random.Random(
            int.from_bytes(hashlib.sha256(f"{self.seed}:{index}".encode()).digest()[:16], "big")
        )
        choice = rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        return self.datasets[choice][rng.randrange(len(self.datasets[choice]))]

    def verify(self):
        for dataset in self.datasets:
            if hasattr(dataset, "verify"):
                dataset.verify()
