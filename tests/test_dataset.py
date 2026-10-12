"""Unit tests for evaluation/dataset.py (documents, base cases, and variants validation)."""

from copy import deepcopy
import pytest

from evaluation.dataset import (
    DatasetValidationError,
    compute_dataset_hash,
    load_documents,
    load_jsonl_base_cases,
    validate_base_cases,
    validate_variants,
)
from src.schemas import BaseCaseRecord, VariantRecord


def test_load_documents_word_counts() -> None:
    """All 8 counterfactual documents must load and fall within 300-1500 words."""
    docs = load_documents(validate_word_count=True)
    assert len(docs) == 8
    for doc_id in [f"DOC-00{i}" for i in range(1, 9)]:
        assert doc_id in docs
        word_count = len(docs[doc_id].split())
        assert 300 <= word_count <= 1500


def test_base_cases_structure_and_verbatim_quotes_verified() -> None:
    """All 50 human-verified base cases must pass strict runner validation (require_verified=True)."""
    docs = load_documents()
    records = load_jsonl_base_cases()
    assert len(records) == 50

    summary = validate_base_cases(records, docs, require_verified=True)
    assert summary["total_cases"] == 50
    assert summary["answerable_cases"] == 38
    assert summary["unanswerable_cases"] == 12
    assert summary["verified_cases"] == 50
    assert summary["draft_cases"] == 0

    for rec in records:
        assert rec.status == "verified"
        assert rec.verified_by == "sayanpal1499"
        assert rec.verified_on == "2026-10-12"


def test_strict_verification_gate_rejects_draft_cases() -> None:
    """Runner mode (require_verified=True) must reject any record with status == 'draft'."""
    docs = load_documents()
    records = load_jsonl_base_cases()
    draft_records = [deepcopy(r) for r in records]
    draft_records[0].status = "draft"
    draft_records[0].verified_by = None
    draft_records[0].verified_on = None

    with pytest.raises(DatasetValidationError, match="runner requires status == 'verified'"):
        validate_base_cases(draft_records, docs, require_verified=True)


def test_rejects_non_verbatim_evidence_and_unsupported_answer() -> None:
    """Validator must catch fabricated evidence quotes and unsupported numbers in expected_answer."""
    docs = load_documents()
    records = load_jsonl_base_cases()

    bad_quote_records = [deepcopy(r) for r in records]
    bad_quote_records[0].expected_evidence = ["This sentence does not exist in DOC-001."]
    with pytest.raises(DatasetValidationError, match="not found verbatim"):
        validate_base_cases(bad_quote_records, docs, require_verified=False)

    bad_number_records = [deepcopy(r) for r in records]
    bad_number_records[0].expected_answer = "999 mA"
    with pytest.raises(DatasetValidationError, match="numeric value '999'"):
        validate_base_cases(bad_number_records, docs, require_verified=False)


def test_rejects_duplicate_ids_and_missing_documents() -> None:
    """Validator must reject duplicate base case IDs or unknown document IDs."""
    docs = load_documents()
    records = load_jsonl_base_cases()

    dup_records = [deepcopy(r) for r in records]
    dup_records[1].id = dup_records[0].id
    with pytest.raises(DatasetValidationError, match="Duplicate base case ID"):
        validate_base_cases(dup_records, docs, require_verified=False)

    missing_doc_records = [deepcopy(r) for r in records]
    missing_doc_records[0].document_id = "DOC-999"
    with pytest.raises(DatasetValidationError, match="does not exist in documents"):
        validate_base_cases(missing_doc_records, docs, require_verified=False)


def test_validate_variants_checks_links_and_verification_gate() -> None:
    """Variant validator must check base_id, document_id, and status == 'verified'."""
    docs = load_documents()
    records = load_jsonl_base_cases()
    base_map: dict[str, BaseCaseRecord] = {r.id: r for r in records}

    draft_variant = VariantRecord(
        variant_id="VAR-001",
        base_id="BASE-001",
        document_id="DOC-001",
        transformation="paraphrase_hard",
        original_input=records[0].question,
        modified_input="At 20 °C in active telemetry acquisition mode, how much continuous current does the Helios-X9 draw?",
        expected_behavior="Return the same expected answer (42 mA) and verbatim quote as BASE-001.",
        expected_answer="42 mA",
        should_abstain=False,
        expected_evidence=["In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."],
        notes="Pure lexical rephrasing of BASE-001.",
        status="draft",
    )

    summary = validate_variants([draft_variant], base_map, docs, require_verified=False)
    assert summary["total_variants"] == 1
    assert summary["draft_variants"] == 1

    with pytest.raises(DatasetValidationError, match="runner requires status == 'verified'"):
        validate_variants([draft_variant], base_map, docs, require_verified=True)

    assert len(compute_dataset_hash()) == 16
