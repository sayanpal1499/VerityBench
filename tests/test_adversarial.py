"""Unit tests for Phase 3 adversarial variant generator (evaluation/adversarial.py)."""

import pytest

from evaluation.adversarial import (
    IMPLEMENTED_CATEGORIES,
    RESERVED_INTERVIEW_CATEGORY,
    generate_all_variants,
)
from evaluation.dataset import (
    DatasetValidationError,
    load_documents,
    load_jsonl_base_cases,
    load_jsonl_variants,
    validate_variants,
)


def test_variants_file_counts_and_categories() -> None:
    """All 100 variants must cover the 10 implemented categories and exclude the reserved category."""
    docs = load_documents()
    base_records = load_jsonl_base_cases()
    base_map = {b.id: b for b in base_records}
    variants = load_jsonl_variants()

    assert len(variants) == 100
    summary = validate_variants(variants, base_map, docs, require_verified=False)
    assert summary["total_variants"] == 100

    by_cat = summary["by_category"]
    for cat in IMPLEMENTED_CATEGORIES:
        assert cat in by_cat and by_cat[cat] >= 6
    assert RESERVED_INTERVIEW_CATEGORY not in by_cat


def test_deterministic_variant_generation() -> None:
    """Calling generate_all_variants twice must yield bit-for-bit identical VariantRecord lists."""
    docs = load_documents()
    base_records = load_jsonl_base_cases()
    run1 = generate_all_variants(base_records, docs)
    run2 = generate_all_variants(base_records, docs)
    assert [v.model_dump() for v in run1] == [v.model_dump() for v in run2]
    assert all(v.status == "draft" and v.verified_by is None for v in run1)


def test_long_context_positions_and_padding() -> None:
    """long_context variants must test start, middle, and end positions at ~5x length."""
    docs = load_documents()
    variants = [v for v in load_jsonl_variants() if v.transformation == "long_context"]
    assert len(variants) == 9
    positions = [v.context_position for v in variants]
    assert positions.count("start") == 3
    assert positions.count("middle") == 3
    assert positions.count("end") == 3

    for v in variants:
        assert v.modified_document is not None
        orig_words = len(docs[v.document_id].split())
        padded_words = len(v.modified_document.split())
        assert padded_words >= orig_words * 4.5


def test_empty_malformed_variants_rejected_without_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """All 6 empty_malformed variants must raise InputValidationError before any LLM call."""
    import src.qa_system as qa_mod
    from src.qa_system import InputValidationError, answer_question

    docs = load_documents()
    called = False

    def _forbid_llm(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("LLM should never be called for empty_malformed variants")

    monkeypatch.setattr(qa_mod, "call_llm", _forbid_llm)

    malformed = [v for v in load_jsonl_variants() if v.transformation == "empty_malformed"]
    assert len(malformed) == 6
    for v in malformed:
        with pytest.raises(InputValidationError):
            answer_question(docs[v.document_id], v.modified_input, prompt_version="grounded_v1")
    assert not called
