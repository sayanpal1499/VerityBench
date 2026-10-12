"""Unit tests for src/qa_system.py covering all 5 quote-gate cases, validation, and malformed JSON."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm_client import LLMClientError
from src.qa_system import answer_question, apply_evidence_quote_gate, verify_evidence_quote
from src.schemas import QAOutput

SAMPLE_DOCUMENT = (
    "The Helios-X9 Microcontroller operates at a nominal supply voltage of 3.3 V "
    "and supports a maximum clock frequency of 180 MHz.\n"
    "By contrast, the legacy Helios-X7 Microcontroller operates at 5.0 V and a "
    "maximum clock frequency of 96 MHz.\n"
    "Thermal shutdown engages automatically when die temperature exceeds 115 °C."
)


def test_real_verbatim_quote_passes_gate(tmp_path: Path) -> None:
    """Case 1: Real verbatim quote in document passes the gate."""
    mock_resp = {
        "text": (
            '{"answer": "3.3 V", '
            '"evidence": ["The Helios-X9 Microcontroller operates at a nominal supply voltage of 3.3 V"], '
            '"confidence": 0.96, "abstained": false}'
        ),
        "metadata": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    }
    with patch("src.qa_system.call_llm", return_value=mock_resp):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the nominal supply voltage of the Helios-X9?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert out.abstained is False
        assert out.answer == "3.3 V"
        assert out.confidence == pytest.approx(0.96)
        assert out._trace is not None
        assert out._trace.gate_triggered is False
        assert out._trace.quote_checks[0].match_type == "exact"
        assert out._trace.quote_checks[0].passed_gate is True


def test_fake_quote_forces_abstention(tmp_path: Path) -> None:
    """Case 2: Fabricated quote not in document forces abstention while preserving raw_output."""
    mock_resp = {
        "text": (
            '{"answer": "1.8 V", '
            '"evidence": ["The Helios-X9 requires a regulated 1.8 V core supply."], '
            '"confidence": 0.92, "abstained": false}'
        ),
        "metadata": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    }
    with patch("src.qa_system.call_llm", return_value=mock_resp):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the nominal supply voltage of the Helios-X9?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert out.abstained is True
        assert out.evidence == []
        assert out.confidence == pytest.approx(0.50)
        assert out._trace is not None
        assert out._trace.gate_triggered is True
        assert out._trace.raw_output.abstained is False
        assert out._trace.raw_output.answer == "1.8 V"
        assert out._trace.raw_output.confidence == pytest.approx(0.92)
        assert out._trace.quote_checks[0].match_type == "missing"


def test_fuzzy_candidate_quote_recorded_and_does_not_pass_verbatim_gate() -> None:
    """Near-miss quote with difflib ratio >= 0.90 is recorded as fuzzy_candidate, not verbatim pass."""
    # Alter one short word ("nominal" -> "typical") so similarity ratio is >= 0.90 but not exact
    near_quote = "The Helios-X9 Microcontroller operates at a typical supply voltage of 3.3 V"
    check = verify_evidence_quote(near_quote, SAMPLE_DOCUMENT)
    assert check.match_type == "fuzzy_candidate"
    assert check.similarity_ratio >= 0.90
    assert check.passed_gate is False

    raw = QAOutput(answer="3.3 V", evidence=[near_quote], confidence=0.90, abstained=False)
    final_out, checks, triggered, reason = apply_evidence_quote_gate(raw, SAMPLE_DOCUMENT, True)
    assert triggered is True
    assert final_out.abstained is True
    assert "fuzzy_candidates=1" in reason


def test_irrelevant_but_real_quote_passes_quote_gate(tmp_path: Path) -> None:
    """Case 3: Real quote that does not support the answer passes Layer-1 quote presence gate."""
    mock_resp = {
        "text": (
            '{"answer": "3.3 V", '
            '"evidence": ["Thermal shutdown engages automatically when die temperature exceeds 115 °C."], '
            '"confidence": 0.85, "abstained": false}'
        ),
        "metadata": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    }
    with patch("src.qa_system.call_llm", return_value=mock_resp):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the nominal supply voltage of the Helios-X9?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        # Gate checks verbatim quote existence only; claim support is checked in scoring/hallucination
        assert out.abstained is False
        assert out._trace is not None
        assert out._trace.gate_triggered is False
        assert out._trace.quote_checks[0].passed_gate is True


def test_wrong_entity_quote_passes_quote_gate(tmp_path: Path) -> None:
    """Case 4: Quote associated with the wrong entity (Helios-X7 instead of X9) passes quote gate."""
    mock_resp = {
        "text": (
            '{"answer": "5.0 V", '
            '"evidence": ["By contrast, the legacy Helios-X7 Microcontroller operates at 5.0 V"], '
            '"confidence": 0.88, "abstained": false}'
        ),
        "metadata": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    }
    with patch("src.qa_system.call_llm", return_value=mock_resp):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the nominal supply voltage of the Helios-X9?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert out.abstained is False
        assert out.answer == "5.0 V"
        assert out._trace is not None
        assert out._trace.gate_triggered is False


def test_unanswerable_question_voluntary_abstention(tmp_path: Path) -> None:
    """Case 5: Question that should abstain passes cleanly when model abstains."""
    mock_resp = {
        "text": (
            '{"answer": "The document does not specify the package pin count.", '
            '"evidence": [], "confidence": 0.95, "abstained": true}'
        ),
        "metadata": {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    }
    with patch("src.qa_system.call_llm", return_value=mock_resp):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="How many pins does the Helios-X9 package have?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert out.abstained is True
        assert out.evidence == []
        assert out.confidence == pytest.approx(0.95)
        assert out._trace is not None
        assert out._trace.gate_triggered is False


def test_empty_or_malformed_question_raises_validation_error() -> None:
    """Empty, whitespace-only, or absurdly long questions fail validation before any API call."""
    with patch("src.qa_system.call_llm") as mock_llm:
        with pytest.raises(ValueError, match="Validation error"):
            answer_question(document=SAMPLE_DOCUMENT, question="   ")
        with pytest.raises(ValueError, match="Validation error"):
            answer_question(document=SAMPLE_DOCUMENT, question="x" * 3000)
        assert mock_llm.call_count == 0


def test_malformed_json_retries_once_then_returns_safe_fallback(tmp_path: Path) -> None:
    """Malformed JSON triggers 1 retry and returns a structured fallback error without crashing."""
    bad_resp = {"text": "not valid json at all", "metadata": {}}
    with patch("src.qa_system.call_llm", side_effect=[bad_resp, bad_resp]) as mock_llm:
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the voltage?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert mock_llm.call_count == 2
        assert out.abstained is True
        assert out.confidence == 0.0
        assert out._trace is not None
        assert out._trace.error is not None
        assert "Malformed JSON after 1 retry" in out._trace.error


def test_api_failure_returns_safe_fallback(tmp_path: Path) -> None:
    """API failure returns a structured error QAOutput without crashing."""
    with patch("src.qa_system.call_llm", side_effect=LLMClientError("Simulated 503 outage")):
        out = answer_question(
            document=SAMPLE_DOCUMENT,
            question="What is the voltage?",
            prompt_version="grounded_v1",
            cache_dir=tmp_path,
        )
        assert out.abstained is True
        assert out.confidence == 0.0
        assert out._trace is not None
        assert out._trace.error is not None
        assert "API failure" in out._trace.error
