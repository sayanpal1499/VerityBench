"""Unit tests for Phase 7 failure clustering, CSV export/import, and causal probes (evaluation/clustering.py)."""

from pathlib import Path

from evaluation.clustering import (
    assign_initial_failure_cluster,
    build_failure_records_for_review,
    export_failure_review_csv,
    load_failure_review_csv,
    plot_failure_distribution,
    run_causal_probes_for_clusters,
    summarize_failure_clusters,
)
from evaluation.dataset import (
    load_documents,
    load_jsonl_base_cases,
    load_jsonl_variants,
)
from src.schemas import QAOutput


def test_assign_initial_failure_cluster_rules() -> None:
    """Verify rule-based failure labels across all failure categories."""
    assert (
        assign_initial_failure_cluster({"is_correct": True}, {"has_hallucination": False})
        is None
    )
    assert (
        assign_initial_failure_cluster({"is_correct": False, "injection_followed": True})
        == "INJECTION_FOLLOWED"
    )
    assert (
        assign_initial_failure_cluster({"is_correct": False, "distractor_captured": True})
        == "DISTRACTOR_CAPTURE"
    )
    assert (
        assign_initial_failure_cluster({"is_correct": False, "false_premise_accepted": True})
        == "FALSE_PREMISE_ACCEPTED"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "abstained": True, "should_abstain": False}
        )
        == "OVER_ABSTENTION"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "abstained": False, "should_abstain": True}
        )
        == "UNDER_ABSTENTION"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "category": "source_conflict"}
        )
        == "SOURCE_CONFLICT_MISSED"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "category": "long_context"}
        )
        == "POSITION_SENSITIVITY"
    )
    # API error disaggregation (rate limit vs token exhaustion)
    err_trunc = {"answer": "System Error Abstention: max completion tokens reached before generating a valid document"}
    err_rate = {"answer": "System Error Abstention: API failure: Groq HTTP 429: Rate limit reached on tokens per day"}
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "category": "source_conflict"}, output_record=err_trunc
        )
        == "OUTPUT_SCHEMA_TRUNCATION"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "category": "prompt_injection"}, output_record=err_rate
        )
        == "API_RATE_LIMIT_EXHAUSTION"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": False, "category": "format_edge"}, output_record=err_trunc
        )
        == "OUTPUT_SCHEMA_TRUNCATION"
    )
    assert (
        assign_initial_failure_cluster(
            {"is_correct": True}, {"has_hallucination": True}
        )
        == "FABRICATED_CONTENT"
    )


def test_csv_roundtrip_and_three_cluster_requirement_check(tmp_path: Path) -> None:
    """Verify failure_review.csv round-trip, >=3 cluster check, and <3 cluster warning flag."""
    scored = [
        {"case_id": "VAR-016", "base_id": "BASE-001", "document_id": "DOC-001", "category": "distractor_near_miss", "is_correct": False, "distractor_captured": True, "abstained": False, "should_abstain": False, "reason": "Distractor"},
        {"case_id": "VAR-038", "base_id": "BASE-001", "document_id": "DOC-001", "category": "false_premise_numeric", "is_correct": False, "false_premise_accepted": True, "abstained": False, "should_abstain": False, "reason": "Premise"},
        {"case_id": "BASE-006", "base_id": "BASE-006", "document_id": "DOC-001", "category": "base", "is_correct": False, "abstained": False, "should_abstain": True, "reason": "Under-abstained"},
    ]
    outputs = {
        "VAR-016": {"question": "Q1", "expected_answer": "42 mA", "answer": "28 mA", "gate_triggered": False},
        "VAR-038": {"question": "Q2", "expected_answer": "42 mA", "answer": "3.8 V", "gate_triggered": False},
        "BASE-006": {"question": "Q3", "expected_answer": "N/A", "answer": "$12", "gate_triggered": False},
    }

    failures = build_failure_records_for_review(scored, outputs)
    assert len(failures) == 3

    csv_path = tmp_path / "failure_review.csv"
    export_failure_review_csv(failures, csv_path)
    loaded = load_failure_review_csv(csv_path)
    assert len(loaded) == 3

    summary_ok = summarize_failure_clusters(loaded)
    assert summary_ok["observed_cluster_count"] == 3
    assert summary_ok["meets_three_cluster_requirement"] is True
    assert summary_ok["unmet_three_cluster_requirement"] is False

    # If only 2 clusters occur, unmet_three_cluster_requirement must be flagged honestly
    summary_low = summarize_failure_clusters(loaded[:2])
    assert summary_low["observed_cluster_count"] == 2
    assert summary_low["unmet_three_cluster_requirement"] is True
    assert summary_low["escalation_notice"] is not None

    fig_path = tmp_path / "failure_distribution.png"
    plot_failure_distribution(summary_ok, fig_path)
    assert fig_path.exists()


def test_causal_probes_with_mock_qa_runner() -> None:
    """Causal probe runner must execute counterfactual inputs for observed clusters."""
    docs = load_documents()
    base_map = {b.id: b for b in load_jsonl_base_cases()}
    var_map = {v.variant_id: v for v in load_jsonl_variants()}

    failures = [
        {
            "case_id": "VAR-016",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "category": "distractor_near_miss",
            "actual_answer": "28 mA",
            "human_verified_cluster": "DISTRACTOR_CAPTURE",
        },
        {
            "case_id": "VAR-038",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "category": "false_premise_numeric",
            "actual_answer": "3.8 V and 42 mA",
            "human_verified_cluster": "FALSE_PREMISE_ACCEPTED",
        },
        {
            "case_id": "BASE-006",
            "base_id": "BASE-006",
            "document_id": "DOC-001",
            "category": "base",
            "question": base_map["BASE-006"].question,
            "actual_answer": "$50 per unit",
            "human_verified_cluster": "UNDER_ABSTENTION",
        },
    ]

    def _mock_qa(document: str, question: str, prompt_version: str = "grounded_v1") -> QAOutput:
        if "PROBE-EMPTY" in document:
            return QAOutput(answer="Abstained.", evidence=[], confidence=0.95, abstained=True)
        return QAOutput(
            answer="The Helios-X9 draws 42 mA in active mode at 20 °C.",
            evidence=["In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."],
            confidence=0.95,
            abstained=False,
        )

    probes = run_causal_probes_for_clusters(
        failures, base_map, var_map, docs, qa_runner=_mock_qa
    )
    assert "DISTRACTOR_CAPTURE" in probes
    assert "FALSE_PREMISE_ACCEPTED" in probes
    assert "UNDER_ABSTENTION" in probes
    assert probes["DISTRACTOR_CAPTURE"]["probes_executed"] == 1
    assert "attention capture" in probes["DISTRACTOR_CAPTURE"]["probes"][0]["finding"]
