"""Native byte BPE, WordPiece, and Unigram tokenizers with explicit normalization."""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import unicodedata
from ..core import atomic_json, read_json, digest_json


class ByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id, vocab_size = 0, 1, 2, 259

    def encode(self, text, add_special_tokens=True):
        values = [byte + 3 for byte in text.encode("utf-8")]
        return [self.bos_token_id] + values if add_special_tokens else values

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(int(x) - 3 for x in tokens if 3 <= int(x) < 259).decode(
            "utf-8", errors="replace"
        )

    def to_dict(self):
        return {"type": "byte", "schema_version": 1}

    def save_pretrained(self, directory):
        atomic_json(Path(directory) / "tokenizer.json", self.to_dict())

    @property
    def fingerprint(self):
        return digest_json(self.to_dict())


class ByteBPETokenizer(ByteTokenizer):
    """Merge byte pairs by learned rank rather than greedy longest-string matching."""

    def __init__(self, merges=()):
        self.merges = tuple(tuple(pair) for pair in merges)
        self.rank, self.pieces = {}, {index + 3: bytes([index]) for index in range(256)}
        for rank, pair in enumerate(self.merges):
            if len(pair) != 2 or pair in self.rank or any(p not in self.pieces for p in pair):
                raise ValueError(
                    "Merge pairs must reference earlier byte/merge tokens exactly once"
                )
            token = 259 + rank
            self.rank[pair] = (rank, token)
            self.pieces[token] = self.pieces[pair[0]] + self.pieces[pair[1]]
        self.vocab_size = 259 + len(self.merges)

    @lru_cache(maxsize=8192)
    def _encode(self, text):
        values = [byte + 3 for byte in text.encode("utf-8")]
        while len(values) > 1:
            candidates = [
                (self.rank[pair][0], pair) for pair in zip(values, values[1:]) if pair in self.rank
            ]
            if not candidates:
                break
            _, selected = min(candidates)
            merged, index = [], 0
            while index < len(values):
                if index + 1 < len(values) and (values[index], values[index + 1]) == selected:
                    merged.append(self.rank[selected][1])
                    index += 2
                else:
                    merged.append(values[index])
                    index += 1
            values = merged
        return tuple(values)

    def encode(self, text, add_special_tokens=True):
        values = list(self._encode(text))
        return [self.bos_token_id] + values if add_special_tokens else values

    def decode(self, tokens, skip_special_tokens=True):
        pieces = []
        for token in tokens:
            token = int(token)
            if token in (0, 1, 2):
                continue
            if token not in self.pieces:
                raise ValueError("Unknown BPE token")
            pieces.append(self.pieces[token])
        return b"".join(pieces).decode("utf-8", errors="replace")

    @classmethod
    def train(cls, texts, vocab_size=512, min_frequency=2):
        if vocab_size < 259 or min_frequency < 1:
            raise ValueError("BPE vocabulary must retain the 256 bytes and 3 special tokens")
        documents = [[value + 3 for value in text.encode("utf-8")] for text in texts]
        merges = []
        for token in range(259, vocab_size):
            counts = Counter(pair for doc in documents for pair in zip(doc, doc[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < min_frequency:
                break
            merges.append(pair)
            for index, doc in enumerate(documents):
                output, cursor = [], 0
                while cursor < len(doc):
                    if cursor + 1 < len(doc) and tuple(doc[cursor : cursor + 2]) == pair:
                        output.append(token)
                        cursor += 2
                    else:
                        output.append(doc[cursor])
                        cursor += 1
                documents[index] = output
        return cls(merges)

    def to_dict(self):
        return {
            "type": "byte_bpe",
            "schema_version": 1,
            "merges": [list(pair) for pair in self.merges],
        }


class WordPieceTokenizer:
    """Use longest-match subwords with explicit normalization and whole-word UNK fallback."""

    def __init__(
        self,
        vocabulary,
        *,
        lowercase=True,
        strip_accents=True,
        continuation="##",
        max_word_chars=100,
    ):
        self.vocabulary = dict(vocabulary)
        if len(set(self.vocabulary.values())) != len(self.vocabulary) or any(
            x < 0 for x in self.vocabulary.values()
        ):
            raise ValueError("Vocabulary IDs must be distinct nonnegative integers")
        for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
            if token not in vocabulary:
                raise ValueError(f"Missing special token: {token}")
        self.lowercase, self.strip_accents, self.continuation = (
            lowercase,
            strip_accents,
            continuation,
        )
        self.max_word_chars = max_word_chars
        self.pad_token_id, self.bos_token_id, self.eos_token_id = [
            vocabulary[x] for x in ("[PAD]", "[CLS]", "[SEP]")
        ]
        self.vocab_size = max(vocabulary.values()) + 1

    def _words(self, text):
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = "".join(
                c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
            )
        words, current = [], ""
        for char in text:
            code = ord(char)
            boundary = (
                char.isspace()
                or unicodedata.category(char).startswith("P")
                or 0x4E00 <= code <= 0x9FFF
            )
            if boundary:
                if current:
                    words.append(current)
                    current = ""
                if not char.isspace():
                    words.append(char)
            elif not unicodedata.category(char).startswith("C"):
                current += char
        if current:
            words.append(current)
        return words

    def encode(self, text, add_special_tokens=True):
        output = []
        for word in self._words(text):
            pieces, start = [], 0
            if len(word) > self.max_word_chars:
                pieces = ["[UNK]"]
            else:
                while start < len(word):
                    chosen = None
                    for end in range(len(word), start, -1):
                        candidate = (self.continuation if start else "") + word[start:end]
                        if candidate in self.vocabulary:
                            chosen = candidate
                            break
                    if chosen is None:
                        pieces = ["[UNK]"]
                        break
                    pieces.append(chosen)
                    start = end
            output.extend(self.vocabulary[piece] for piece in pieces)
        return [self.bos_token_id] + output + [self.eos_token_id] if add_special_tokens else output

    def decode(self, tokens, skip_special_tokens=True):
        inverse = {v: k for k, v in self.vocabulary.items()}
        words = [
            inverse[int(token)]
            for token in tokens
            if not skip_special_tokens
            or int(token) not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]
        return " ".join(words).replace(" " + self.continuation, "")

    def to_dict(self):
        return {
            "type": "wordpiece",
            "schema_version": 1,
            "vocabulary": self.vocabulary,
            "lowercase": self.lowercase,
            "strip_accents": self.strip_accents,
            "continuation": self.continuation,
            "max_word_chars": self.max_word_chars,
        }

    def save_pretrained(self, directory):
        atomic_json(Path(directory) / "tokenizer.json", self.to_dict())


class UnigramTokenizer:
    """Viterbi-segment explicit piece log probabilities; normalization is caller-declared."""

    def __init__(self, pieces, unk_id=0):
        self.pieces = tuple((str(text), float(score)) for text, score in pieces)
        if (
            not self.pieces
            or not 0 <= unk_id < len(self.pieces)
            or any(not text or not math.isfinite(score) for text, score in self.pieces)
        ):
            raise ValueError("Unigram pieces need finite log probabilities and a valid UNK")
        if len({text for text, _ in self.pieces}) != len(self.pieces):
            raise ValueError("Duplicate unigram piece")
        self.unk_id, self.vocab_size = unk_id, len(self.pieces)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("No implicit special-token template for this unigram vocabulary")
        costs = [float("inf") for _ in range(len(text) + 1)]
        back = [None] * (len(text) + 1)
        costs[0] = 0.0
        fallback = min(score for _, score in self.pieces) - 10
        for start in range(len(text)):
            candidates = [
                (index, piece, score)
                for index, (piece, score) in enumerate(self.pieces)
                if index != self.unk_id and text.startswith(piece, start)
            ]
            if not any(len(piece) == 1 for _, piece, _ in candidates):
                candidates.append((self.unk_id, text[start : start + 1], fallback))
            for index, piece, score in candidates:
                end, cost = start + len(piece), costs[start] - score
                if cost < costs[end]:
                    costs[end], back[end] = cost, (start, index)
        output, end = [], len(text)
        while end:
            start, token = back[end]
            output.append(token)
            end = start
        return list(reversed(output))

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(self.pieces[int(token)][0] for token in tokens)

    def to_dict(self):
        return {
            "type": "unigram",
            "schema_version": 1,
            "pieces": list(self.pieces),
            "unk_id": self.unk_id,
        }

    def save_pretrained(self, directory):
        atomic_json(Path(directory) / "tokenizer.json", self.to_dict())


def load_tokenizer(directory):
    config = read_json(Path(directory) / "tokenizer.json")
    kind, version = config.pop("type"), config.pop("schema_version")
    if version != 1:
        raise ValueError("Unsupported tokenizer schema")
    factory = {
        "byte": ByteTokenizer,
        "byte_bpe": ByteBPETokenizer,
        "wordpiece": WordPieceTokenizer,
        "unigram": UnigramTokenizer,
    }
    if kind not in factory:
        raise ValueError("Unknown tokenizer; do not silently substitute a byte tokenizer")
    return factory[kind](**config)
