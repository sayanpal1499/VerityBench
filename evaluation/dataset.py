"""Dataset loader and validator for source documents, base cases, and adversarial variants."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Optional

from pydantic import ValidationError

from src.qa_system import normalize_text_for_quote_match
from src.schemas import BaseCaseRecord, VariantRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "documents"
DEFAULT_BASE_CASES_PATH = PROJECT_ROOT / "data" / "base_cases.jsonl"
DEFAULT_VARIANTS_PATH = PROJECT_ROOT / "data" / "variants.jsonl"

MIN_DOC_WORDS = 300
MAX_DOC_WORDS = 1500

STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "with",
    "by", "under", "after", "before", "between", "into", "during", "until",
    "and", "or", "but", "if", "than", "that", "this", "these", "those",
    "its", "their", "his", "her", "our", "your", "many", "much", "long",
    "can", "may", "must", "shall", "will", "would", "should", "could",
    "according", "document", "stated", "specified", "maximum", "minimum",
    "nominal", "standard", "total", "rate", "value", "amount",
}


class DatasetValidationError(Exception):
    """Raised when dataset records or source documents fail validation checks."""


def load_documents(
    docs_dir: Path = DEFAULT_DOCS_DIR, validate_word_count: bool = True
) -> dict[str, str]:
    """Load all .txt documents from docs_dir and optionally validate 300-1500 word count."""
    if not docs_dir.exists():
        raise DatasetValidationError(f"Documents directory not found: {docs_dir}")

    docs: dict[str, str] = {}
    for file_path in sorted(docs_dir.glob("*.txt")):
        doc_id = file_path.stem
        text = file_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        if validate_word_count and not (MIN_DOC_WORDS <= word_count <= MAX_DOC_WORDS):
            raise DatasetValidationError(
                f"Document '{doc_id}' has {word_count} words; expected between "
                f"{MIN_DOC_WORDS} and {MAX_DOC_WORDS} words."
            )
        docs[doc_id] = text

    if not docs:
        raise DatasetValidationError(f"No .txt documents found in {docs_dir}")
    return docs


def load_jsonl_base_cases(path: Path = DEFAULT_BASE_CASES_PATH) -> list[BaseCaseRecord]:
    """Parse base_cases.jsonl into a list of BaseCaseRecord objects."""
    if not path.exists():
        raise DatasetValidationError(f"Base cases file not found: {path}")

    records: list[BaseCaseRecord] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(BaseCaseRecord.model_validate(data))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetValidationError(
                f"Invalid base case record at {path}:{line_no}: {exc}"
            ) from exc
    return records


def load_jsonl_variants(path: Path = DEFAULT_VARIANTS_PATH) -> list[VariantRecord]:
    """Parse variants.jsonl into a list of VariantRecord objects."""
    if not path.exists():
        raise DatasetValidationError(f"Variants file not found: {path}")

    records: list[VariantRecord] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(VariantRecord.model_validate(data))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetValidationError(
                f"Invalid variant record at {path}:{line_no}: {exc}"
            ) from exc
    return records


def _extract_content_tokens(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens excluding common stopwords."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {t for t in tokens if len(t) >= 2 and t not in STOPWORDS}


def _extract_numeric_strings(text: str) -> list[str]:
    """Extract numeric literals (removing thousand-separator commas)."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.findall(r"\d+(?:\.\d+)?", cleaned)


def _check_answer_supported_by_evidence(
    record: BaseCaseRecord, combined_evidence: str
) -> Optional[str]:
    """Verify that expected_answer is genuinely supported by expected_evidence, not just a coincidence."""
    norm_ev = normalize_text_for_quote_match(combined_evidence).lower()
    norm_ans = normalize_text_for_quote_match(record.expected_answer).lower()

    # 1. Check question subject/entity overlap with the cited evidence so a quote for
    #    Entity B cannot be used as evidence for a question about Entity A.
    q_tokens = _extract_content_tokens(record.question)
    ev_tokens = _extract_content_tokens(combined_evidence)
    if q_tokens and not (q_tokens & ev_tokens):
        return (
            f"Case {record.id}: question content tokens {sorted(q_tokens)} have zero "
            f"overlap with expected_evidence."
        )

    # 2. For structured factual types, verify answer numbers/tokens appear in evidence
    if record.answer_type in ("number_with_unit", "date", "named_entity", "list"):
        ans_nums = _extract_numeric_strings(norm_ans)
        ev_nums = set(_extract_numeric_strings(norm_ev))
        for num in ans_nums:
            if num not in ev_nums:
                return (
                    f"Case {record.id}: numeric value '{num}' in expected_answer "
                    f"('{record.expected_answer}') is not present in expected_evidence."
                )
        ans_tokens = _extract_content_tokens(norm_ans)
        if ans_tokens and not (ans_tokens & ev_tokens):
            return (
                f"Case {record.id}: expected_answer '{record.expected_answer}' shares no "
                f"content tokens with expected_evidence."
            )

    # 3. For two_sentence_fact and yes_no, require explicit entailment notes
    if record.answer_type == "two_sentence_fact":
        if len(record.expected_evidence) < 2:
            return (
                f"Case {record.id}: answer_type is 'two_sentence_fact' but only "
                f"{len(record.expected_evidence)} evidence quote(s) provided."
            )
        if len(record.notes.strip()) < 15:
            return (
                f"Case {record.id}: 'two_sentence_fact' requires a clear entailment "
                f"explanation in 'notes' (>= 15 chars)."
            )

    if record.answer_type == "yes_no":
        if len(record.notes.strip()) < 15:
            return (
                f"Case {record.id}: 'yes_no' answerable case requires an entailment "
                f"explanation in 'notes' (>= 15 chars) showing why the evidence entails Yes/No."
            )

    return None


def validate_base_cases(
    records: list[BaseCaseRecord],
    documents: dict[str, str],
    require_verified: bool = True,
    min_cases: int = 50,
) -> dict[str, Any]:
    """Validate base cases against all Phase 2 rules.

    Never mutates record status. Raises DatasetValidationError if any check fails.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(records) < min_cases:
        errors.append(f"Expected at least {min_cases} base cases, found {len(records)}.")

    seen_ids: set[str] = set()
    answerable_count = 0
    unanswerable_count = 0
    verified_count = 0
    draft_count = 0

    for rec in records:
        if rec.id in seen_ids:
            errors.append(f"Duplicate base case ID: '{rec.id}'.")
        seen_ids.add(rec.id)

        if rec.document_id not in documents:
            errors.append(
                f"Case {rec.id}: document_id '{rec.document_id}' does not exist in documents."
            )
            continue

        doc_text = documents[rec.document_id]
        norm_doc = normalize_text_for_quote_match(doc_text)

        if rec.status == "verified":
            verified_count += 1
            if not rec.verified_by or not rec.verified_on:
                errors.append(
                    f"Case {rec.id}: status is 'verified' but 'verified_by' or 'verified_on' is null."
                )
        else:
            draft_count += 1
            if require_verified:
                errors.append(
                    f"Case {rec.id}: status is '{rec.status}'; runner requires status == 'verified'."
                )

        if rec.should_abstain:
            unanswerable_count += 1
            if rec.answer_type != "unanswerable":
                errors.append(
                    f"Case {rec.id}: should_abstain is True, so answer_type must be 'unanswerable' "
                    f"(got '{rec.answer_type}')."
                )
            if rec.expected_evidence:
                errors.append(
                    f"Case {rec.id}: should_abstain is True, so expected_evidence must be empty []."
                )
            if len(rec.notes.strip()) < 20:
                errors.append(
                    f"Case {rec.id}: unanswerable case must record in 'notes' (>= 20 chars) "
                    f"why the document does not entail the requested answer."
                )
        else:
            answerable_count += 1
            if rec.answer_type == "unanswerable":
                errors.append(
                    f"Case {rec.id}: should_abstain is False, but answer_type is 'unanswerable'."
                )
            if not rec.expected_answer.strip():
                errors.append(f"Case {rec.id}: answerable case has empty expected_answer.")
            if not rec.expected_evidence:
                errors.append(f"Case {rec.id}: answerable case must have non-empty expected_evidence.")
            else:
                for ev in rec.expected_evidence:
                    norm_ev = normalize_text_for_quote_match(ev)
                    if ev not in doc_text and norm_ev not in norm_doc:
                        errors.append(
                            f"Case {rec.id}: expected_evidence quote not found verbatim in "
                            f"{rec.document_id}: '{ev[:80]}...'"
                        )
                support_err = _check_answer_supported_by_evidence(
                    rec, " ".join(rec.expected_evidence)
                )
                if support_err:
                    errors.append(support_err)

    if errors:
        bullet_list = "\n  - ".join(errors[:25])
        more = f"\n  ... and {len(errors) - 25} more errors." if len(errors) > 25 else ""
        raise DatasetValidationError(
            f"Base dataset validation failed with {len(errors)} error(s):\n  - {bullet_list}{more}"
        )

    return {
        "total_cases": len(records),
        "answerable_cases": answerable_count,
        "unanswerable_cases": unanswerable_count,
        "verified_cases": verified_count,
        "draft_cases": draft_count,
        "warnings": warnings,
    }


def validate_variants(
    variants: list[VariantRecord],
    base_cases_by_id: dict[str, BaseCaseRecord],
    documents: dict[str, str],
    require_verified: bool = True,
) -> dict[str, Any]:
    """Validate adversarial variants in data/variants.jsonl."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    verified_count = 0
    draft_count = 0
    by_category: dict[str, int] = {}

    for var in variants:
        if var.variant_id in seen_ids:
            errors.append(f"Duplicate variant_id: '{var.variant_id}'.")
        seen_ids.add(var.variant_id)

        if var.base_id not in base_cases_by_id:
            errors.append(
                f"Variant {var.variant_id}: base_id '{var.base_id}' not found in base_cases."
            )
        if var.document_id not in documents:
            errors.append(
                f"Variant {var.variant_id}: document_id '{var.document_id}' not found in documents."
            )
        elif var.transformation != "empty_malformed" and not var.should_abstain:
            eff_doc = (
                var.modified_document
                if var.modified_document is not None
                else documents[var.document_id]
            )
            norm_doc = normalize_text_for_quote_match(eff_doc)
            if not var.expected_evidence:
                errors.append(
                    f"Variant {var.variant_id}: answerable variant must have non-empty expected_evidence."
                )
            for ev in var.expected_evidence:
                norm_ev = normalize_text_for_quote_match(ev)
                if ev not in eff_doc and norm_ev not in norm_doc:
                    errors.append(
                        f"Variant {var.variant_id}: expected_evidence not found verbatim in effective document: '{ev[:80]}...'"
                    )

        by_category[var.transformation] = by_category.get(var.transformation, 0) + 1

        if var.status == "verified":
            verified_count += 1
            if not var.verified_by or not var.verified_on:
                errors.append(
                    f"Variant {var.variant_id}: status is 'verified' but 'verified_by' or 'verified_on' is null."
                )
        else:
            draft_count += 1
            if require_verified:
                errors.append(
                    f"Variant {var.variant_id}: status is '{var.status}'; runner requires status == 'verified'."
                )

    if errors:
        bullet_list = "\n  - ".join(errors[:25])
        raise DatasetValidationError(
            f"Variant validation failed with {len(errors)} error(s):\n  - {bullet_list}"
        )

    return {
        "total_variants": len(variants),
        "verified_variants": verified_count,
        "draft_variants": draft_count,
        "by_category": by_category,
    }


def compute_dataset_hash(
    base_path: Path = DEFAULT_BASE_CASES_PATH,
    variants_path: Optional[Path] = None,
) -> str:
    """Compute a deterministic SHA-256 hash of the dataset files for run metadata."""
    hasher = hashlib.sha256()
    if base_path.exists():
        hasher.update(base_path.read_bytes())
    if variants_path and variants_path.exists():
        hasher.update(variants_path.read_bytes())
    return hasher.hexdigest()[:16]


def main() -> None:
    """CLI validator for inspecting dataset integrity and verification status."""
    parser = argparse.ArgumentParser(description="Validate documents, base cases, and variants.")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Validate structure and verbatim quotes even when records have status == 'draft'.",
    )
    parser.add_argument(
        "--check-variants",
        action="store_true",
        help="Also validate data/variants.jsonl if present.",
    )
    args = parser.parse_args()

    docs = load_documents()
    base_records = load_jsonl_base_cases()
    summary = validate_base_cases(
        records=base_records,
        documents=docs,
        require_verified=not args.allow_draft,
    )
    output: dict[str, Any] = {
        "documents_loaded": len(docs),
        "document_word_counts": {k: len(v.split()) for k, v in docs.items()},
        "base_cases_summary": summary,
    }

    if args.check_variants and DEFAULT_VARIANTS_PATH.exists():
        variants = load_jsonl_variants()
        base_map = {b.id: b for b in base_records}
        output["variants_summary"] = validate_variants(
            variants=variants,
            base_cases_by_id=base_map,
            documents=docs,
            require_verified=not args.allow_draft,
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
