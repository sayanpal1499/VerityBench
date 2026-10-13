"""Document QA system implementing the Evidence-Quote Gate with Forced Abstention."""

import argparse
import difflib
import json
from pathlib import Path
import re
from typing import Any, Optional

from pydantic import ValidationError

from src.llm_client import DEFAULT_SUT_MODEL, LLMClientError, call_llm
from src.prompts import get_prompt_config
from src.schemas import QAOutput, QATrace, QuoteCheckResult

MAX_QUESTION_CHARS = 2000
MIN_QUOTE_CHARS = 5
GATE_FALLBACK_CONFIDENCE = 0.50


def normalize_text_for_quote_match(text: str) -> str:
    """Normalize whitespace, smart quotes, and dashes for deterministic quote verification."""
    if not text:
        return ""
    # Replace smart double and single quotes with ASCII equivalents
    cleaned = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u201f", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201a", "'")
        .replace("\u201b", "'")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2015", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )
    # Collapse all whitespace sequences (spaces, tabs, newlines) into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_outer_quotes_and_Ellipsis(text: str) -> str:
    """Strip outer wrapping quote marks or leading/trailing ellipses added by models."""
    stripped = text.strip()
    stripped = re.sub(r"^[\.\u2026\"'\s]+", "", stripped)
    stripped = re.sub(r"[\.\u2026\"'\s]+$", "", stripped)
    return stripped.strip()


def _best_fuzzy_ratio(norm_quote: str, norm_doc: str) -> float:
    """Compute the highest difflib SequenceMatcher ratio for norm_quote within norm_doc."""
    if not norm_quote or not norm_doc:
        return 0.0
    q_len = len(norm_quote)
    d_len = len(norm_doc)
    if q_len >= d_len:
        return round(difflib.SequenceMatcher(None, norm_quote.lower(), norm_doc.lower()).ratio(), 4)

    # Split document into sentence-like segments and sliding word windows around quote length
    q_lower = norm_quote.lower()
    doc_words = norm_doc.split()
    q_word_count = max(1, len(norm_quote.split()))

    best = 0.0
    step = max(1, q_word_count // 4)
    for window_words in (max(1, q_word_count - 2), q_word_count, q_word_count + 2):
        for i in range(0, max(1, len(doc_words) - window_words + 1), step):
            candidate = " ".join(doc_words[i : i + window_words]).lower()
            ratio = difflib.SequenceMatcher(None, q_lower, candidate).ratio()
            if ratio > best:
                best = ratio
                if best >= 0.99:
                    return round(best, 4)

    # Refine around the best window with stride 1 if close to threshold
    if best >= 0.75:
        for i in range(0, max(1, len(doc_words) - q_word_count + 1)):
            candidate = " ".join(doc_words[i : i + q_word_count]).lower()
            ratio = difflib.SequenceMatcher(None, q_lower, candidate).ratio()
            if ratio > best:
                best = ratio
    return round(best, 4)


def verify_evidence_quote(quote: str, document: str) -> QuoteCheckResult:
    """Verify whether a single evidence quote exists in the source document.

    Primary pass:
      - 'exact': raw quote substring is directly in raw document.
      - 'normalized': whitespace/quote-normalized quote is in normalized document.
    Diagnostic candidate (does NOT pass the verbatim gate):
      - 'fuzzy_candidate': difflib similarity ratio >= 0.90.
      - 'missing': similarity ratio < 0.90.
    """
    raw_q = (quote or "").strip()
    norm_q = _strip_outer_quotes_and_Ellipsis(normalize_text_for_quote_match(raw_q))
    norm_doc = normalize_text_for_quote_match(document)

    if len(norm_q) < MIN_QUOTE_CHARS:
        return QuoteCheckResult(
            original_quote=quote,
            normalized_quote=norm_q,
            match_type="missing",
            similarity_ratio=0.0,
            passed_gate=False,
        )

    if raw_q in document:
        return QuoteCheckResult(
            original_quote=quote,
            normalized_quote=norm_q,
            match_type="exact",
            similarity_ratio=1.0,
            passed_gate=True,
        )

    if norm_q in norm_doc or norm_q.lower() in norm_doc.lower():
        return QuoteCheckResult(
            original_quote=quote,
            normalized_quote=norm_q,
            match_type="normalized",
            similarity_ratio=1.0,
            passed_gate=True,
        )

    ratio = _best_fuzzy_ratio(norm_q, norm_doc)
    if ratio >= 0.90:
        return QuoteCheckResult(
            original_quote=quote,
            normalized_quote=norm_q,
            match_type="fuzzy_candidate",
            similarity_ratio=ratio,
            passed_gate=False,
        )

    return QuoteCheckResult(
        original_quote=quote,
        normalized_quote=norm_q,
        match_type="missing",
        similarity_ratio=ratio,
        passed_gate=False,
    )


def apply_evidence_quote_gate(
    raw_output: QAOutput, document: str, gate_enabled: bool = True
) -> tuple[QAOutput, list[QuoteCheckResult], bool, str]:
    """Apply the Evidence-Quote Gate to a raw QAOutput.

    Returns (final_output, quote_checks, gate_triggered, gate_reason).
    """
    quote_checks = [verify_evidence_quote(q, document) for q in raw_output.evidence]

    if not gate_enabled:
        return (
            raw_output.model_copy(deep=True),
            quote_checks,
            False,
            "gate_disabled_for_prompt_version",
        )

    if raw_output.abstained:
        # Model voluntarily abstained; ensure evidence list is clean
        clean_abstain = QAOutput(
            answer=raw_output.answer,
            evidence=[],
            confidence=raw_output.confidence,
            abstained=True,
        )
        return clean_abstain, quote_checks, False, "model_voluntarily_abstained"

    verified_quotes = [
         qc.original_quote for qc in quote_checks if qc.passed_gate
    ]
    fuzzy_count = sum(1 for qc in quote_checks if qc.match_type == "fuzzy_candidate")
    missing_count = sum(1 for qc in quote_checks if qc.match_type == "missing")

    if verified_quotes:
        final_out = QAOutput(
            answer=raw_output.answer,
            evidence=verified_quotes,
            confidence=raw_output.confidence,
            abstained=False,
        )
        return (
            final_out,
            quote_checks,
            False,
            f"passed_verbatim_quote_check (verified={len(verified_quotes)}, fuzzy={fuzzy_count})",
        )

    # Gate forces abstention because model gave a substantive answer without any verified quote
    if not raw_output.evidence:
        reason = "forced_abstention_no_evidence_quotes_returned"
    elif fuzzy_count > 0:
        reason = (
            f"forced_abstention_only_fuzzy_or_missing_quotes "
            f"(fuzzy_candidates={fuzzy_count}, missing={missing_count})"
        )
    else:
        reason = f"forced_abstention_all_quotes_missing (missing={missing_count})"

    forced_output = QAOutput(
        answer=(
            "Abstained (Evidence-Quote Gate): The model's answer could not be verified "
            "by any verbatim quote from the document."
        ),
        evidence=[],
        confidence=GATE_FALLBACK_CONFIDENCE,
        abstained=True,
    )
    return forced_output, quote_checks, True, reason


def _extract_json_object(raw_text: str) -> str:
    """Extract a JSON object from raw model text, stripping markdown code fences if present."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_qa_output(raw_text: str) -> QAOutput:
    """Parse raw JSON text into a validated QAOutput instance."""
    json_str = _extract_json_object(raw_text)
    data = json.loads(json_str)
    if isinstance(data, dict):
        # Coerce evidence if model accidentally returned a single string instead of list[str]
        if isinstance(data.get("evidence"), str):
            ev_str = data["evidence"].strip()
            data["evidence"] = [ev_str] if ev_str else []
        elif data.get("evidence") is None:
            data["evidence"] = []
    return QAOutput.model_validate(data)


class InputValidationError(ValueError):
    """Raised when document or question fails pre-API input validation."""


def validate_inputs(document: str, question: str) -> None:
    """Validate document and question before making any LLM API call."""
    if not isinstance(document, str) or not document.strip():
        raise InputValidationError("Validation error: 'document' must be a non-empty string.")
    if not isinstance(question, str) or not question.strip():
        raise InputValidationError(
            "Validation error: 'question' must be a non-empty, non-whitespace string."
        )
    if len(question) > MAX_QUESTION_CHARS:
        raise InputValidationError(
            f"Validation error: 'question' length ({len(question)} chars) exceeds "
            f"maximum allowed length ({MAX_QUESTION_CHARS} chars)."
        )
    if len(re.findall(r"[a-zA-Z0-9]", question)) < 2:
        raise InputValidationError(
            "Validation error: 'question' must contain at least 2 alphanumeric characters."
        )


def answer_question(
    document: str,
    question: str,
    prompt_version: str = "grounded_v1",
    model: str = DEFAULT_SUT_MODEL,
    seed: int = 42,
    bypass_cache: bool = False,
    cache_dir: Optional[Path] = None,
) -> QAOutput:
    """Answer a question over a document using the specified prompt version and evidence gate.

    Returns the final QAOutput (post-gate) with `_trace` attached for inspection of
    `raw_output` (pre-gate), `quote_checks`, `gate_triggered`, and `llm_metadata`.
    """
    validate_inputs(document=document, question=question)
    prompt_cfg = get_prompt_config(prompt_version)
    gate_enabled = prompt_cfg.get("gate_enabled", "true").lower() == "true"

    user_prompt = prompt_cfg["user_template"].format(document=document, question=question)
    system_prompt = prompt_cfg["system"]

    llm_metadata: dict[str, Any] = {}
    error_msg: Optional[str] = None
    error_category: Optional[str] = None
    raw_output: Optional[QAOutput] = None

    try:
        response = call_llm(
            prompt=user_prompt,
            model=model,
            seed=seed,
            temperature=0.0,
            system_prompt=system_prompt,
            prompt_version=prompt_version,
            json_mode=True,
            response_schema=QAOutput.model_json_schema(),
            max_output_tokens=768,
            bypass_cache=bypass_cache,
            cache_dir=cache_dir,
        )
        llm_metadata = response.get("metadata", {})
        try:
            raw_output = _parse_qa_output(response["text"])
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as parse_err:
            # Malformed JSON or schema error: retry once with explicit repair prompt
            repair_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous output was not valid JSON matching the schema. "
                "Return ONLY a valid JSON object starting with {{ and ending with }}. Keys required: "
                "\"answer\" (str), \"evidence\" (list of str), \"confidence\" (float 0..1), "
                "\"abstained\" (bool)."
            )
            retry_resp = call_llm(
                prompt=repair_prompt,
                model=model,
                seed=seed + 1,
                temperature=0.0,
                system_prompt=system_prompt,
                prompt_version=f"{prompt_version}_json_retry",
                json_mode=True,
                response_schema=QAOutput.model_json_schema(),
                max_output_tokens=768,
                bypass_cache=bypass_cache,
                cache_dir=cache_dir,
            )
            llm_metadata = retry_resp.get("metadata", {})
            llm_metadata["json_repair_retried"] = True
            try:
                raw_output = _parse_qa_output(retry_resp["text"])
            except Exception as second_err:
                error_msg = f"Malformed JSON after 1 retry: { parse_err } | { second_err }"
                error_category = "schema_validation_error"
    except LLMClientError as api_err:
        error_msg = f"API failure: {api_err}"
        error_category = getattr(api_err, "error_category", "infrastructure_error")

    if raw_output is None:
        # Clear structured error fallback (never crash on malformed JSON or API error)
        fallback = QAOutput(
            answer=f"System Error Abstention: {error_msg}",
            evidence=[],
            confidence=0.0,
            abstained=True,
        )
        is_infra = bool(error_category in ("rate_limit", "provider_server_error", "connection_timeout", "infrastructure_error"))
        trace = QATrace(
            prompt_version=prompt_version,
            raw_output=fallback,
            final_output=fallback,
            gate_enabled=gate_enabled,
            gate_triggered=False,
            gate_reason="error_fallback",
            quote_checks=[],
            fuzzy_candidate_count=0,
            error=error_msg,
            error_category=error_category,
            is_infrastructure_error=is_infra,
            llm_metadata=llm_metadata,
        )
        fallback._trace = trace
        return fallback

    final_output, quote_checks, gate_triggered, gate_reason = apply_evidence_quote_gate(
        raw_output=raw_output,
        document=document,
        gate_enabled=gate_enabled,
    )
    fuzzy_count = sum(1 for qc in quote_checks if qc.match_type == "fuzzy_candidate")

    trace = QATrace(
        prompt_version=prompt_version,
        raw_output=raw_output,
        final_output=final_output,
        gate_enabled=gate_enabled,
        gate_triggered=gate_triggered,
        gate_reason=gate_reason,
        quote_checks=quote_checks,
        fuzzy_candidate_count=fuzzy_count,
        error=None,
        llm_metadata=llm_metadata,
    )
    final_output._trace = trace
    return final_output


def main() -> None:
    """Simple CLI entry point for running a single question against a .txt document."""
    parser = argparse.ArgumentParser(description="Run Document QA system on a .txt file.")
    parser.add_argument("--document", required=True, help="Path to the .txt document.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--prompt", default="grounded_v1", help="Prompt version.")
    parser.add_argument("--model", default=DEFAULT_SUT_MODEL, help="Model identifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bypass-cache", action="store_true", help="Bypass local disk cache.")
    args = parser.parse_args()

    doc_path = Path(args.document)
    doc_text = doc_path.read_text(encoding="utf-8")
    output = answer_question(
        document=doc_text,
        question=args.question,
        prompt_version=args.prompt,
        model=args.model,
        seed=args.seed,
        bypass_cache=args.bypass_cache,
    )
    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
