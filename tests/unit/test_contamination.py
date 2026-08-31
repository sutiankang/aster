from dataclasses import replace
import json
import pytest

from aster.data.contamination import (
    TextSample,
    SplitManifest,
    ContaminationConfig,
    DataAuditReport,
    normalize_text,
    character_shingles,
    jaccard_similarity,
    minhash_signature,
    scan_pii,
    audit_text_splits,
    data_audit_gate,
)


KEY = b"fixture-only-not-a-production-key!"


def sample(identifier, text, split="train", *, revision="commit-fixed-v1"):
    return TextSample("local-data", revision, split, identifier, text)


def manifests(records, *, license_id="CC0-1.0", license_reference="https://example.test/LICENSE"):
    groups = {}
    for record in records:
        groups.setdefault(record.key[:3], []).append(record)
    return [
        SplitManifest.from_samples(
            rows,
            source_uri="local://fixture/text",
            license_id=license_id,
            license_reference=license_reference,
        )
        for rows in groups.values()
    ]


def audit(records, **kwargs):
    return audit_text_splits(
        records, manifests(records), allowed_licenses=("CC0-1.0",), pii_hmac_key=KEY, **kwargs
    )


def test_normalized_exact_overlap_preserves_eval_and_proposes_only_train_exclusion():
    rows = [
        sample("a", "Ａ  cat\n sits here"),
        sample("b", "a cat sits here", "test"),
        sample("c", "completely unrelated train content"),
    ]
    report = audit(rows)
    matches = report.payload["matches"]
    assert (
        len(matches) == 1
        and matches[0]["kind"] == "exact_normalized"
        and matches[0]["jaccard"] == 1.0
    )
    assert report.payload["training_exclusion_candidates"] == [rows[0].key]
    assert data_audit_gate(report, corpus_fingerprint=report.payload["corpus_fingerprint"])[
        "reasons"
    ] == ["train_eval_contamination"]
    assert rows[0].text == "Ａ  cat\n sits here" and rows[1].text == "a cat sits here"


@pytest.mark.parametrize("search", ["exhaustive", "minhash_lsh"])
def test_near_duplicate_real_candidate_search_and_independent_jaccard(search):
    text = "A reproducible machine learning experiment must preserve its exact data split and preprocessing configuration."
    modified = text.replace("exact", "fixed")
    rows = [sample("train", text), sample("test", modified, "test")]
    config = ContaminationConfig(search=search, jaccard_threshold=0.75)
    report = audit(rows, config=config)
    match = report.payload["matches"][0]
    a = character_shingles(normalize_text(text), 5)
    b = character_shingles(normalize_text(modified), 5)
    assert match["kind"] == "near_jaccard" and match["jaccard"] == len(a & b) / len(a | b)
    assert match["cross_protected_boundary"]
    expected_coverage = (
        "exhaustive_pairs" if search == "exhaustive" else "probabilistic_lsh_candidates"
    )
    assert report.payload["near_duplicate_coverage"] == expected_coverage


def test_embedded_protected_question_is_detected_even_when_whole_doc_jaccard_is_low():
    heldout = "This held out evaluation question must never be present inside a longer training document or calibration source."
    training = (
        "Here is unrelated context about optimization methods and unrelated content. " * 30
        + heldout
    )
    rows = [sample("train", training), sample("test", heldout, "test")]
    report = audit(rows, config=ContaminationConfig(jaccard_threshold=0.99))
    match = report.payload["matches"][0]
    assert match["kind"] == "protected_text_contained_in_training" and match["jaccard"] < 0.99


def test_manifest_enforces_revision_split_full_membership_and_raw_text_identity():
    rows = [
        sample("train", "raw training text"),
        sample("test", "protected evaluation sample", "test"),
    ]
    pinned = manifests(rows)
    for invalid in (
        [rows[0]],
        [replace(rows[0], text="changed"), rows[1]],
        [replace(rows[0], revision="other"), rows[1]],
        [rows[0], replace(rows[1], split="train")],
    ):
        with pytest.raises(ValueError):
            audit_text_splits(invalid, pinned, pii_hmac_key=KEY)
    with pytest.raises(ValueError):
        audit_text_splits(rows + [rows[0]], pinned, pii_hmac_key=KEY)
    with pytest.raises(ValueError):
        SplitManifest.from_samples(
            rows[:1], source_uri="https://token@example.test/data", license_id="CC0-1.0"
        )
    with pytest.raises(ValueError):
        SplitManifest.from_samples(
            rows[:1], source_uri="https://example.test/data?signature=secret", license_id="CC0-1.0"
        )


def test_pii_report_has_only_hmac_fingerprints_not_raw_matches_and_tracks_cross_split_repetition():
    address = "fixture.person@example.test"
    token = "sk-" + "AbCdEf0123456789" * 2
    rows = [
        sample("train", f"Contact {address} using {token}"),
        sample("test", f"The audit fixture includes {address} for a private contact test.", "test"),
    ]
    report = audit(rows)
    payload = json.dumps(report.payload)
    assert address not in payload and token not in payload
    assert {x["kind"] for x in report.payload["pii_findings"]} == {
        "email",
        "secret_token_candidate",
    }
    repeat = report.payload["pii_cross_split_repetitions"][0]
    assert (
        repeat["kind"] == "email"
        and repeat["training"] == [rows[0].key]
        and repeat["protected"] == [rows[1].key]
    )
    one = scan_pii(address, hmac_key=KEY)[0]
    other = scan_pii(address, hmac_key=b"different-private-fixture-key-000")[0]
    assert (
        one["fingerprint"] != other["fingerprint"]
        and one["start"] == 0
        and one["end"] == len(address)
    )
    assert (
        "pii_candidates_require_review"
        in data_audit_gate(report, corpus_fingerprint=report.payload["corpus_fingerprint"])[
            "reasons"
        ]
    )


def test_license_inventory_is_explicit_declaration_not_legal_approval():
    rows = [
        sample("train", "zero repeated train string"),
        sample("test", "a different protected item", "test"),
    ]
    report = audit_text_splits(
        rows, manifests(rows, license_id=None, license_reference=None), pii_hmac_key=KEY
    )
    assert {item["status"] for item in report.payload["license_inventory"]} == {"unknown_license"}
    gate = data_audit_gate(report, corpus_fingerprint=report.payload["corpus_fingerprint"])
    assert not gate["passed"] and gate["scope"].endswith("not_legal_compliance")
    approved_declarations = audit(rows)
    assert all(
        item["status"] == "declared_allowlisted"
        for item in approved_declarations.payload["license_inventory"]
    )
    assert data_audit_gate(
        approved_declarations,
        corpus_fingerprint=approved_declarations.payload["corpus_fingerprint"],
    )["passed"]
    assert not data_audit_gate(approved_declarations, corpus_fingerprint="unrelated-corpus")[
        "passed"
    ]
    with pytest.raises(ValueError):
        data_audit_gate(
            approved_declarations,
            corpus_fingerprint=approved_declarations.payload["corpus_fingerprint"],
            allow_pii_candidates="false",
        )


def test_probabilistic_clean_result_requires_explicit_pipeline_authorization():
    rows = [
        sample("train", "alpha training input"),
        sample("test", "entirely different heldout objective", "test"),
    ]
    report = audit(rows, config=ContaminationConfig(search="minhash_lsh"))
    assert report.payload["status"] == "no_findings_in_checked_scope"
    assert not data_audit_gate(report, corpus_fingerprint=report.payload["corpus_fingerprint"])[
        "passed"
    ]
    assert data_audit_gate(
        report, corpus_fingerprint=report.payload["corpus_fingerprint"], require_exhaustive=False
    )["passed"]


def test_budget_overrun_and_empty_text_are_not_silently_labeled_clean():
    rows = [
        sample("a", "one text"),
        sample("b", "second text"),
        sample("test", "heldout text", "test"),
    ]
    with pytest.raises(ValueError, match="Candidate budget"):
        audit(rows, config=ContaminationConfig(max_candidate_pairs=1))
    with pytest.raises(ValueError, match="budget"):
        audit(rows, config=ContaminationConfig(max_total_chars=5))
    with pytest.raises(ValueError, match="PII finding budget"):
        scan_pii("a@example.test b@example.test", hmac_key=KEY, max_findings=1)
    empty = [sample("train", " \t "), sample("test", "some heldout content", "test")]
    report = audit(empty)
    assert report.payload["empty_samples"] == [empty[0].key]
    assert not data_audit_gate(report, corpus_fingerprint=report.payload["corpus_fingerprint"])[
        "passed"
    ]


def test_deterministic_minhash_report_roundtrip_tamper_and_no_overwrite(tmp_path):
    grams = character_shingles("中文字符也必须能处理而不依赖按空格分词", 3)
    assert minhash_signature(grams, permutations=32, seed=4) == minhash_signature(
        grams, permutations=32, seed=4
    )
    assert minhash_signature(grams, permutations=32, seed=4) != minhash_signature(
        grams, permutations=32, seed=5
    )
    assert jaccard_similarity(frozenset(), frozenset()) == 0.0
    rows = [sample("train", "red green blue"), sample("test", "orange violet cyan", "test")]
    report = audit(rows)
    assert report.sha256 == audit(list(reversed(rows))).sha256
    path = report.save(tmp_path / "report.json")
    loaded = DataAuditReport.load(path)
    assert (
        loaded.sha256 == report.sha256
        and loaded.payload["corpus_fingerprint"] == report.payload["corpus_fingerprint"]
    )
    with pytest.raises(FileExistsError):
        loaded.save(path)
    loaded.payload["sample_count"] = 999
    with pytest.raises(ValueError):
        loaded.verify()
