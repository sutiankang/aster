"""Local content fingerprints, duplicate candidates, and bounded sensitive-data checks."""

from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import hmac
import itertools
import math
from pathlib import Path
import platform
import random
import re
import secrets
import unicodedata
from urllib.parse import urlsplit

from ..core import atomic_json, digest_json, read_json


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TextSample:
    dataset_id: str
    revision: str
    split: str
    sample_id: str
    text: str

    def __post_init__(self):
        if not all(
            isinstance(x, str) and x
            for x in (self.dataset_id, self.revision, self.split, self.sample_id)
        ) or not isinstance(self.text, str):
            raise ValueError("Text sample needs explicit dataset/revision/split/sample identity")

    @property
    def key(self):
        return self.dataset_id, self.revision, self.split, self.sample_id


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    revision: str
    split: str
    source_uri: str
    license_id: str | None
    license_reference: str | None
    expected_samples: tuple[tuple[str, str], ...]

    def __post_init__(self):
        object.__setattr__(
            self, "expected_samples", tuple(tuple(row) for row in self.expected_samples)
        )
        if not all(
            isinstance(x, str) and x
            for x in (self.dataset_id, self.revision, self.split, self.source_uri)
        ):
            raise ValueError("Manifest requires fixed dataset revision, split and source")
        if not self.expected_samples or any(
            len(row) != 2 or not row[0] or not re.fullmatch(r"[0-9a-f]{64}", row[1])
            for row in self.expected_samples
        ):
            raise ValueError("Manifest must enumerate every sample and exact raw-text hash")
        if len({row[0] for row in self.expected_samples}) != len(self.expected_samples):
            raise ValueError("Duplicate manifest sample identity")
        for value in (self.license_id, self.license_reference):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Empty license metadata")
        for reference in (self.source_uri, self.license_reference):
            if reference is not None:
                parsed = urlsplit(reference)
                if parsed.username or parsed.password or parsed.query:
                    raise ValueError(
                        "Provenance must use credential-free stable references, not signed/query URLs"
                    )

    @classmethod
    def from_samples(cls, samples, *, source_uri, license_id=None, license_reference=None):
        samples = tuple(samples)
        if not samples or len({s.key[:3] for s in samples}) != 1:
            raise ValueError("Each manifest describes one nonempty exact dataset/revision/split")
        first = samples[0]
        return cls(
            first.dataset_id,
            first.revision,
            first.split,
            source_uri,
            license_id,
            license_reference,
            tuple((s.sample_id, _text_hash(s.text)) for s in samples),
        )

    @property
    def fingerprint(self):
        return digest_json(asdict(self))


@dataclass(frozen=True)
class ContaminationConfig:
    training_splits: tuple[str, ...] = ("train", "calibration")
    protected_splits: tuple[str, ...] = ("validation", "test")
    search: str = "exhaustive"
    shingle_size: int = 5
    jaccard_threshold: float = 0.8
    containment_min_chars: int = 64
    permutations: int = 64
    bands: int = 16
    seed: int = 42
    max_samples: int = 10000
    max_text_chars: int = 1000000
    max_total_chars: int = 10000000
    max_candidate_pairs: int = 250000
    max_pii_findings: int = 10000

    def __post_init__(self):
        for key in ("training_splits", "protected_splits"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if (
            not self.training_splits
            or not self.protected_splits
            or set(self.training_splits) & set(self.protected_splits)
            or any(
                not isinstance(x, str) or not x
                for x in (*self.training_splits, *self.protected_splits)
            )
        ):
            raise ValueError(
                "Train/calibration and protected split identities must be explicit and disjoint"
            )
        integers = (
            self.shingle_size,
            self.containment_min_chars,
            self.permutations,
            self.bands,
            self.max_samples,
            self.max_text_chars,
            self.max_total_chars,
            self.max_candidate_pairs,
            self.max_pii_findings,
        )
        if any(type(x) is not int or x < 1 for x in integers) or type(self.seed) is not int:
            raise ValueError("Audit budgets and MinHash layout require positive integer limits")
        if (
            self.permutations % self.bands
            or not math.isfinite(self.jaccard_threshold)
            or not 0 < self.jaccard_threshold <= 1
        ):
            raise ValueError(
                "MinHash bands must divide permutations; Jaccard threshold lies in (0,1]"
            )
        if self.search not in {"exhaustive", "minhash_lsh"}:
            raise ValueError("Unknown candidate search")


def normalize_text(text):

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def character_shingles(text, size):

    return (
        frozenset(text[i : i + size] for i in range(max(1, len(text) - size + 1)))
        if text
        else frozenset()
    )


def jaccard_similarity(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def minhash_signature(shingles, *, permutations=64, seed=42):

    if not shingles or type(permutations) is not int or permutations < 1:
        raise ValueError("Nonempty shingles and positive permutations required")
    prime = (1 << 61) - 1
    values = [
        int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % prime
        for s in sorted(shingles)
    ]
    rng = random.Random(seed)
    pairs = [(rng.randrange(1, prime), rng.randrange(prime)) for _ in range(permutations)]
    return tuple(min((a * value + b) % prime for value in values) for a, b in pairs)


_PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w.-])", re.I),
    "phone_candidate": re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,20}\d)(?!\w)"),
    "secret_token_candidate": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
    "private_key_marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def scan_pii(text, *, hmac_key, max_findings=10000):

    if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
        raise ValueError("PII correlation requires a private >=32-byte HMAC key")
    if not isinstance(text, str) or type(max_findings) is not int or max_findings < 0:
        raise ValueError("PII input and finding limit are invalid")
    result = []
    for kind, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "phone_candidate":
                value = re.sub(r"\D", "", value)
                if not 9 <= len(value) <= 15 or len(set(value)) < 3:
                    continue
            else:
                value = value.casefold() if kind == "email" else value
            if len(result) >= max_findings:
                raise ValueError("PII finding budget exceeded; refusing a truncated clean report")
            fingerprint = hmac.new(
                hmac_key, (kind + "\0" + value).encode(), hashlib.sha256
            ).hexdigest()
            result.append(
                {
                    "kind": kind,
                    "start": match.start(),
                    "end": match.end(),
                    "fingerprint": fingerprint,
                }
            )
    return sorted(result, key=lambda item: (item["start"], item["kind"]))


@dataclass(frozen=True)
class DataAuditReport:
    payload: dict
    sha256: str

    def verify(self):
        if digest_json(self.payload) != self.sha256 or self.payload.get("schema_version") != 1:
            raise ValueError("Data audit report content/schema changed")
        return self

    def save(self, path):
        self.verify()
        path = Path(path)
        if path.exists():
            raise FileExistsError("Data audit reports are append-free; choose a new evidence path")
        atomic_json(path, {"payload": self.payload, "sha256": self.sha256})
        return path

    @classmethod
    def load(cls, path):
        envelope = read_json(path)
        if set(envelope) != {"payload", "sha256"}:
            raise ValueError("Unexpected data audit envelope")
        return cls(**envelope).verify()


def audit_text_splits(samples, manifests, *, config=None, allowed_licenses=(), pii_hmac_key=None):
    config = config or ContaminationConfig()
    if isinstance(allowed_licenses, str):
        raise TypeError("License allowlist must be a sequence, not a single string")
    allowed_licenses = tuple(allowed_licenses)
    if any(not isinstance(value, str) or not value for value in allowed_licenses):
        raise ValueError("License declarations must be explicit identifiers/expressions")
    records, total_chars = [], 0
    for sample in samples:
        if not isinstance(sample, TextSample):
            raise TypeError("Audit input must use TextSample")
        total_chars += len(sample.text)
        if (
            len(records) >= config.max_samples
            or len(sample.text) > config.max_text_chars
            or total_chars > config.max_total_chars
        ):
            raise ValueError(
                "Text audit budget exceeded; no partial report may be labeled complete"
            )
        records.append(sample)
    if not records or len({s.key for s in records}) != len(records):
        raise ValueError("Empty audit or duplicate sample identity")
    records.sort(key=lambda s: s.key)
    manifests = tuple(sorted(manifests, key=lambda m: (m.dataset_id, m.revision, m.split)))
    by_manifest = {(m.dataset_id, m.revision, m.split): m for m in manifests}
    if len(by_manifest) != len(manifests) or set(by_manifest) != {s.key[:3] for s in records}:
        raise ValueError("Manifest coverage differs from exact supplied split set")
    allowed_splits = set(config.training_splits) | set(config.protected_splits)
    if {s.split for s in records} - allowed_splits:
        raise ValueError("Unclassified split would evade contamination policy")
    if not any(s.split in config.training_splits for s in records) or not any(
        s.split in config.protected_splits for s in records
    ):
        raise ValueError("Audit must include both training/calibration and protected samples")
    for identity, manifest in by_manifest.items():
        observed = {s.sample_id: _text_hash(s.text) for s in records if s.key[:3] == identity}
        if observed != dict(manifest.expected_samples):
            raise ValueError("Missing/extra/modified samples violate the fixed split manifest")
    corpus_fingerprint = digest_json(
        [{"identity": s.key, "raw_text_sha256": _text_hash(s.text)} for s in records]
    )
    normalized = [normalize_text(s.text) for s in records]
    shingles = [character_shingles(text, config.shingle_size) for text in normalized]
    candidates = set()

    def add_pair(left, right):
        pair = tuple(sorted((left, right)))
        if pair in candidates:
            return
        if len(candidates) >= config.max_candidate_pairs:
            raise ValueError("Candidate budget exceeded; refusing silently incomplete audit")
        candidates.add(pair)

    exact = defaultdict(list)
    for index, text in enumerate(normalized):
        if text:
            exact[_text_hash(text)].append(index)
    for indices in exact.values():
        for left, right in itertools.combinations(indices, 2):
            add_pair(left, right)
    if config.search == "exhaustive":
        for left, right in itertools.combinations(range(len(records)), 2):
            add_pair(left, right)
    else:
        buckets = defaultdict(list)
        rows = config.permutations // config.bands
        for index, grams in enumerate(shingles):
            if not grams:
                continue
            signature = minhash_signature(grams, permutations=config.permutations, seed=config.seed)
            for band in range(config.bands):
                key = band, signature[band * rows : (band + 1) * rows]
                for previous in buckets[key]:
                    add_pair(previous, index)
                buckets[key].append(index)
    matches, exclusions = [], set()
    for left, right in sorted(candidates):
        a, b = records[left], records[right]
        score = jaccard_similarity(shingles[left], shingles[right])
        equal = bool(normalized[left]) and normalized[left] == normalized[right]
        contained = None

        if a.split in config.protected_splits and b.split in config.training_splits:
            protected, training = left, right
        elif b.split in config.protected_splits and a.split in config.training_splits:
            protected, training = right, left
        else:
            protected = training = None
        if (
            protected is not None
            and len(normalized[protected]) >= config.containment_min_chars
            and normalized[protected] in normalized[training]
        ):
            contained = "protected_text_contained_in_training"
        if not equal and not contained and score < config.jaccard_threshold:
            continue
        kind = "exact_normalized" if equal else (contained or "near_jaccard")
        cross = protected is not None
        matches.append(
            {
                "left": a.key,
                "right": b.key,
                "kind": kind,
                "jaccard": score,
                "cross_protected_boundary": cross,
            }
        )
        if cross:
            exclusions.add(records[training].key)
    hmac_key = secrets.token_bytes(32) if pii_hmac_key is None else pii_hmac_key
    pii, value_owners = [], defaultdict(set)
    for sample in records:
        findings = scan_pii(
            sample.text, hmac_key=hmac_key, max_findings=config.max_pii_findings - len(pii)
        )
        for finding in findings:
            pii.append({"sample": sample.key, **finding})
            value_owners[(finding["kind"], finding["fingerprint"])].add(sample.key)
    pii_leaks = []
    for (kind, fingerprint), owners in sorted(value_owners.items()):
        train = sorted(key for key in owners if key[2] in config.training_splits)
        protected = sorted(key for key in owners if key[2] in config.protected_splits)
        if train and protected:
            pii_leaks.append(
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "training": train,
                    "protected": protected,
                }
            )
    license_inventory = []
    for manifest in manifests:
        status = (
            "unknown_license"
            if manifest.license_id is None
            else "missing_reference"
            if manifest.license_reference is None
            else "declared_allowlisted"
            if manifest.license_id in allowed_licenses
            else "declared_not_allowlisted"
        )
        license_inventory.append(
            {
                "dataset_id": manifest.dataset_id,
                "revision": manifest.revision,
                "split": manifest.split,
                "source_uri": manifest.source_uri,
                "license_id": manifest.license_id,
                "license_reference": manifest.license_reference,
                "status": status,
                "manifest_fingerprint": manifest.fingerprint,
            }
        )
    empty = [records[i].key for i, text in enumerate(normalized) if not text]
    payload = {
        "schema_version": 1,
        "implementation": "aster_text_contamination_v1",
        "config": asdict(config),
        "runtime": {
            "python_version": platform.python_version(),
            "unicode_version": unicodedata.unidata_version,
        },
        "normalization": "unicode_nfkc_casefold_whitespace_v1",
        "corpus_fingerprint": corpus_fingerprint,
        "split_manifest_fingerprints": [m.fingerprint for m in manifests],
        "sample_count": len(records),
        "checked_candidate_pairs": len(candidates),
        "near_duplicate_coverage": "exhaustive_pairs"
        if config.search == "exhaustive"
        else "probabilistic_lsh_candidates",
        "matches": matches,
        "training_exclusion_candidates": sorted(exclusions),
        "empty_samples": empty,
        "pii_findings": pii,
        "pii_cross_split_repetitions": pii_leaks,
        "pii_hmac_key_id": hashlib.sha256(hmac_key).hexdigest(),
        "license_inventory": license_inventory,
        "allowed_license_declarations": sorted(set(allowed_licenses)),
        "status": "review_required"
        if matches
        or pii
        or empty
        or any(x["status"] != "declared_allowlisted" for x in license_inventory)
        else "no_findings_in_checked_scope",
        "limitations": [
            "MinHash candidate retrieval may miss near duplicates and embedded test text.",
            "Unicode/case normalization can produce semantic false positives, especially in code.",
            "PII pattern rules have false positives and false negatives; hashes are not anonymization.",
            "License metadata/allowlists are declarations, not legal permission or legal compliance.",
            "No raw text is included; supplied identifiers/source metadata may themselves be sensitive.",
            "No input row was deleted or rewritten; proposed exclusions require a new training split manifest.",
        ],
    }
    return DataAuditReport(payload, digest_json(payload))


def data_audit_gate(
    report,
    *,
    corpus_fingerprint,
    require_exhaustive=True,
    allow_pii_candidates=False,
    allow_unreviewed_license=False,
):

    if any(
        type(flag) is not bool
        for flag in (require_exhaustive, allow_pii_candidates, allow_unreviewed_license)
    ):
        raise ValueError("Data gate relaxations require explicit booleans, not truthy strings")
    report.verify()
    value, reasons = report.payload, []
    if value["corpus_fingerprint"] != corpus_fingerprint:
        reasons.append("corpus_identity_mismatch")
    if require_exhaustive and value["near_duplicate_coverage"] != "exhaustive_pairs":
        reasons.append("probabilistic_coverage_not_authorized")
    if any(match["cross_protected_boundary"] for match in value["matches"]):
        reasons.append("train_eval_contamination")
    if value["empty_samples"]:
        reasons.append("empty_samples")
    if value["pii_findings"] and not allow_pii_candidates:
        reasons.append("pii_candidates_require_review")
    if not allow_unreviewed_license and any(
        row["status"] != "declared_allowlisted" for row in value["license_inventory"]
    ):
        reasons.append("license_metadata_requires_review")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "report_sha256": report.sha256,
        "scope": "configured_data_pipeline_gate_not_legal_compliance",
    }
