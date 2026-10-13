"""Unit tests for Phase 8 (evaluation/regression.py) and Phase 9 (evaluation/runner.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.dataset import DatasetValidationError, load_jsonl_variants
from evaluation.regression import (
    freeze_regression_baseline,
    load_regression_cases,
    run_noise_floor_experiment,
    run_prompt_regression_comparison,
)
from evaluation.runner import (
    execute_verified_evaluation,
    populate_evaluation_report_from_saved_results,
    verify_all_records_before_run,
)
from src.schemas import BaseCaseRecord, QAOutput, VariantRecord


def _sample_cases_and_docs() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    docs = {
        "DOC-001": (
            "The Helios-X9 Optical Transceiver operates at a continuous supply current "
            "of 42 mA under nominal load. Its thermal shutdown threshold is 118 °C."
        )
    }
    cases = {
        "BASE-001": {
            "case_id": "BASE-001",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "category": "base",
            "answer_type": "number_with_unit",
            "question": "What is the continuous supply current of the Helios-X9?",
            "expected_answer": "42 mA",
            "expected_evidence": [
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            "should_abstain": False,
        },
        "BASE-002": {
            "case_id": "BASE-002",
            "base_id": "BASE-002",
            "document_id": "DOC-001",
            "category": "base",
            "answer_type": "number_with_unit",
            "question": "What is the thermal shutdown threshold of the Helios-X9?",
            "expected_answer": "118 °C",
            "expected_evidence": ["Its thermal shutdown threshold is 118 °C."],
            "should_abstain": False,
        },
        "BASE-003": {
            "case_id": "BASE-003",
            "base_id": "BASE-003",
            "document_id": "DOC-001",
            "category": "base",
            "answer_type": "unanswerable",
            "question": "What is the acoustic noise rating of the Helios-X9?",
            "expected_answer": "ABSTAIN",
            "expected_evidence": [],
            "should_abstain": True,
        },
        "VAR-001": {
            "case_id": "VAR-001",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "category": "paraphrase_hard",
            "answer_type": "number_with_unit",
            "question": "State the nominal steady-state current draw for the Helios-X9.",
            "expected_answer": "42 mA",
            "expected_evidence": [
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            "should_abstain": False,
        },
    }
    return cases, docs


def test_freeze_and_load_regression_baseline(tmp_path: Path) -> None:
    cases_by_id, docs = _sample_cases_and_docs()
    scored_records = [
        {"case_id": "BASE-001", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.95},
        {"case_id": "BASE-002", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.92},
        {"case_id": "BASE-003", "category": "base", "is_correct": True, "abstained": True, "confidence": 0.90},
        {"case_id": "VAR-001", "category": "paraphrase_hard", "is_correct": False, "abstained": True, "confidence": 0.50},
    ]

    reg_path = tmp_path / "regression_cases.jsonl"
    frozen = freeze_regression_baseline(
        scored_records=scored_records,
        cases_by_id=cases_by_id,
        documents=docs,
        output_path=reg_path,
        model_id="groq/openai/gpt-oss-20b",
        prompt_version="grounded_v1",
        dataset_version="v1.0-test",
    )

    assert len(frozen) == 3
    loaded = load_regression_cases(reg_path)
    assert [c["case_id"] for c in loaded] == ["BASE-001", "BASE-002", "BASE-003"]
    assert loaded[0]["expected_answer"] == "42 mA"
    assert loaded[0]["baseline_model_id"] == "groq/openai/gpt-oss-20b"
    assert loaded[0]["baseline_prompt_version"] == "grounded_v1"


def test_noise_floor_experiment_flags_flaky_cases(tmp_path: Path) -> None:
    cases_by_id, docs = _sample_cases_and_docs()
    scored_records = [
        {"case_id": "BASE-001", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.95},
        {"case_id": "BASE-002", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.92},
    ]
    frozen = freeze_regression_baseline(
        scored_records=scored_records,
        cases_by_id=cases_by_id,
        documents=docs,
        output_path=tmp_path / "reg.jsonl",
    )

    call_counts: dict[str, int] = {"BASE-001": 0, "BASE-002": 0}
    bypass_flags_seen: list[bool] = []

    def mock_qa_runner(
        document: str,
        question: str,
        prompt_version: str = "grounded_v1",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        bypass_flags_seen.append(bypass_cache)
        if "continuous supply current" in question:
            call_counts["BASE-001"] += 1
            return QAOutput(
                answer="42 mA",
                evidence=[
                    "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
                ],
                confidence=0.95,
                abstained=False,
            )
        call_counts["BASE-002"] += 1
        # Make BASE-002 fail on run #2 out of 3 -> flaky!
        if call_counts["BASE-002"] == 2:
            return QAOutput(
                answer="Cannot determine from document.",
                evidence=[],
                confidence=0.50,
                abstained=True,
            )
        return QAOutput(
            answer="118 °C",
            evidence=["Its thermal shutdown threshold is 118 °C."],
            confidence=0.92,
            abstained=False,
        )

    noise_report = run_noise_floor_experiment(
        regression_cases=frozen,
        num_runs=3,
        qa_runner_fn=mock_qa_runner,
    )

    assert all(bypass_flags_seen), "Noise floor runs must force bypass_cache=True"
    assert noise_report["total_frozen_cases"] == 2
    assert noise_report["stable_count"] == 1
    assert noise_report["flaky_count"] == 1
    assert noise_report["flaky_rate"] == 0.5
    assert noise_report["flaky_ids"] == ["BASE-002"]
    assert noise_report["stable_ids"] == ["BASE-001"]


def test_prompt_regression_comparison_and_planted_bad_detector() -> None:
    cases_by_id, docs = _sample_cases_and_docs()
    test_cases = []
    for cid, c in cases_by_id.items():
        item = dict(c)
        item["document"] = docs[c["document_id"]]
        item["baseline_passed"] = cid != "VAR-001"
        test_cases.append(item)

    baseline_scored = {
        "BASE-001": {"case_id": "BASE-001", "is_correct": True},
        "BASE-002": {"case_id": "BASE-002", "is_correct": True},
        "BASE-003": {"case_id": "BASE-003", "is_correct": True},
        "VAR-001": {"case_id": "VAR-001", "is_correct": False},
    }
    # Mark BASE-002 as flaky from noise floor so it must be excluded from regression counting
    flaky_ids = {"BASE-002"}

    # 1. Test grounded_v2 where BASE-001 stays PASS, BASE-002 (flaky) fails, BASE-003 stays PASS, and VAR-001 is NEWLY_FIXED
    def mock_v2_runner(
        document: str,
        question: str,
        prompt_version: str = "grounded_v2",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        if "acoustic noise" in question:
            return QAOutput(answer="Not stated.", evidence=[], confidence=0.95, abstained=True)
        if "thermal shutdown" in question:
            # Flaky case fails, but should be classified as FLAKY_EXCLUDED, not REGRESSION
            return QAOutput(answer="Abstaining.", evidence=[], confidence=0.50, abstained=True)
        return QAOutput(
            answer="The current draw is 42 mA.",
            evidence=[
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            confidence=0.95,
            abstained=False,
        )

    v2_report = run_prompt_regression_comparison(
        test_cases=test_cases,
        baseline_scored_by_id=baseline_scored,
        flaky_ids=flaky_ids,
        candidate_prompt_version="grounded_v2",
        qa_runner_fn=mock_v2_runner,
    )
    assert v2_report["regression_detected"] is False
    assert v2_report["regression_count"] == 0
    assert v2_report["flaky_excluded_count"] == 1
    assert v2_report["newly_fixed_count"] == 1
    assert v2_report["newly_fixed"][0]["case_id"] == "VAR-001"

    # 2. Test planted bad prompt `grounded_v2_bad` (never abstains, guesses wrong values)
    def mock_v2_bad_runner(
        document: str,
        question: str,
        prompt_version: str = "grounded_v2_bad",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        # Never abstains even on unanswerable BASE-003, and guesses wrong on BASE-001
        return QAOutput(
            answer="Approximately 999 mA / 35 dB based on typical transceivers.",
            evidence=["Paraphrased guess not in document"],
            confidence=0.99,
            abstained=False,
        )

    bad_report = run_prompt_regression_comparison(
        test_cases=test_cases,
        baseline_scored_by_id=baseline_scored,
        flaky_ids=flaky_ids,
        candidate_prompt_version="grounded_v2_bad",
        qa_runner_fn=mock_v2_bad_runner,
    )
    assert bad_report["regression_detected"] is True
    assert bad_report["regression_count"] == 2  # BASE-001 and BASE-003 regressed; BASE-002 excluded as flaky
    reg_ids = {r["case_id"] for r in bad_report["regressions"]}
    assert reg_ids == {"BASE-001", "BASE-003"}


def test_runner_strictly_refuses_draft_variants(tmp_path: Path) -> None:
    """Verify the strict evaluation runner refuses to run while variants are draft or audit claims have needs_human_review=True."""
    variants = load_jsonl_variants(Path("data/variants.jsonl"))
    assert len(variants) == 100
    assert all(v.status == "draft" for v in variants), (
        "All 100 variants in data/variants.jsonl must remain status='draft' until human approval."
    )

    with pytest.raises(DatasetValidationError, match="runner requires status == 'verified'"):
        verify_all_records_before_run(
            base_cases_path=Path("data/base_cases.jsonl"),
            variants_path=Path("data/variants.jsonl"),
            docs_dir=Path("data/documents"),
        )

    with pytest.raises(DatasetValidationError, match="runner requires status == 'verified'"):
        execute_verified_evaluation(
            base_cases_path=Path("data/base_cases.jsonl"),
            variants_path=Path("data/variants.jsonl"),
            docs_dir=Path("data/documents"),
        )

    # When variants are verified in a temporary copy, canonical audit_claims.jsonl (45 verified, 0 flagged) passes
    temp_variants_path = tmp_path / "verified_variants.jsonl"
    temp_variants_path.write_text(
        "\n".join(
            json.dumps(
                {
                    **v.model_dump(),
                    "status": "verified",
                    "verified_by": "unit_test_verifier",
                    "verified_on": "2026-10-13",
                }
            )
            for v in variants
        )
        + "\n",
        encoding="utf-8",
    )
    b_ok, v_ok, d_ok = verify_all_records_before_run(
        base_cases_path=Path("data/base_cases.jsonl"),
        variants_path=temp_variants_path,
        docs_dir=Path("data/documents"),
        audit_claims_path=Path("data/audit_claims.jsonl"),
    )
    assert len(b_ok) == 50 and len(v_ok) == 100 and len(d_ok) == 8

    # And if any audit claim still has needs_human_review=True, verify_all_records_before_run must refuse
    audit_lines = [
        json.loads(line)
        for line in Path("data/audit_claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_lines[0]["needs_human_review"] = True
    temp_flagged_audit_path = tmp_path / "flagged_audit_claims.jsonl"
    temp_flagged_audit_path.write_text(
        "\n".join(json.dumps(r) for r in audit_lines) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="needs_human_review is True"):
        verify_all_records_before_run(
            base_cases_path=Path("data/base_cases.jsonl"),
            variants_path=temp_variants_path,
            docs_dir=Path("data/documents"),
            audit_claims_path=temp_flagged_audit_path,
        )



def test_runner_executes_and_populates_report_when_verified_mocked(tmp_path: Path) -> None:
    """Verify end-to-end runner and report builder on synthetic verified records using a mocked QA runner."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    doc_text = " ".join(
        [
            "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load.",
            "Its thermal shutdown threshold is 118 °C.",
        ]
        * 25
    )  # >300 words to satisfy document validator
    (docs_dir / "DOC-001.txt").write_text(doc_text, encoding="utf-8")

    base_path = tmp_path / "base_cases.jsonl"
    variants_path = tmp_path / "variants.jsonl"

    base_rows = []
    for i in range(1, 51):
        cid = f"BASE-{i:03d}"
        base_rows.append(
            BaseCaseRecord(
                id=cid,
                document_id="DOC-001",
                question=f"What is the continuous supply current of the Helios-X9 (case {i})?",
                expected_answer="42 mA",
                expected_evidence=[
                    "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
                ],
                should_abstain=False,
                answer_type="number_with_unit",
                status="verified",
                verified_by="unit_test_verifier",
                verified_on="2026-10-12",
                notes="Synthetic verified base case for unit testing runner gate.",
            ).model_dump()
        )
    base_path.write_text(
        "\n".join(json.dumps(r) for r in base_rows) + "\n", encoding="utf-8"
    )

    var_rows = [
        VariantRecord(
            variant_id="VAR-001",
            base_id="BASE-001",
            document_id="DOC-001",
            transformation="paraphrase_hard",
            original_input="What is the continuous supply current of the Helios-X9 (case 1)?",
            modified_input="State the nominal continuous supply current of the Helios-X9.",
            modified_document=None,
            expected_behavior="Return 42 mA.",
            expected_answer="42 mA",
            expected_evidence=[
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            should_abstain=False,
            rationale="Synthetic verified variant for runner test.",
            status="verified",
            verified_by="unit_test_verifier",
            verified_on="2026-10-12",
        ).model_dump()
    ]
    variants_path.write_text(
        "\n".join(json.dumps(r) for r in var_rows) + "\n", encoding="utf-8"
    )

    def mock_qa(
        document: str,
        question: str,
        prompt_version: str = "grounded_v1",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        return QAOutput(
            answer="The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load.",
            evidence=[
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            confidence=0.95,
            abstained=False,
        )

    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    rep_path = tmp_path / "evaluation_report.md"

    metrics_payload = execute_verified_evaluation(
        base_cases_path=base_path,
        variants_path=variants_path,
        docs_dir=docs_dir,
        results_dir=res_dir,
        figures_dir=fig_dir,
        qa_runner_fn=mock_qa,
    )

    assert (res_dir / "metrics.json").exists()
    assert (fig_dir / "accuracy_by_category.png").exists()
    assert (fig_dir / "hallucination_rate_by_category.png").exists()
    assert (fig_dir / "calibration_curve.png").exists()
    assert metrics_payload["grounded_v1"]["overall_all_150"]["numerator"] == 51
    assert metrics_payload["grounded_v1"]["overall_all_150"]["denominator"] == 51

    populated = populate_evaluation_report_from_saved_results(
        results_dir=res_dir, report_path=rep_path
    )
    assert populated.exists()
    report_text = populated.read_text(encoding="utf-8")
    assert "51/51 (100.0%)" in report_text


def test_runner_extracts_qatrace_and_scores_edge_cases() -> None:
    """Verify QATrace extraction in run_qa_suite_for_prompt and deterministic scoring edge cases."""
    from evaluation.runner import run_qa_suite_for_prompt
    from evaluation.scoring import score_single_output
    from src.qa_system import QATrace

    doc_text = (
        "Under nominal operation, the Helios-X9 requires a regulated core supply voltage of 3.3 V "
        "and an auxiliary I/O rail of 1.8 V. In active telemetry acquisition mode at 20 °C, "
        "the controller draws a continuous supply current of 42 mA."
    )
    base_case = BaseCaseRecord(
        id="BASE-001",
        document_id="DOC-001",
        question="What is the continuous supply current drawn by the Helios-X9 in active telemetry acquisition mode at 20 °C?",
        expected_answer="42 mA",
        expected_evidence=[
            "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."
        ],
        should_abstain=False,
        answer_type="number_with_unit",
        status="verified",
        verified_by="sayanpal1499",
        verified_on="2026-10-12",
        notes="Direct lookup.",
    )

    def mock_gated_runner(**kwargs: Any) -> QAOutput:
        raw_out = QAOutput(
            answer="42 mA",
            evidence=["Non-verbatim quote that triggers post-generation gate"],
            confidence=0.91,
            abstained=False,
        )
        final_out = QAOutput(
            answer="Insufficient verbatim evidence in document.",
            evidence=[],
            confidence=0.0,
            abstained=True,
        )
        object.__setattr__(
            final_out,
            "_trace",
            QATrace(
                raw_output=raw_out,
                final_output=final_out,
                evidence_checks=[],
                gate_enabled=True,
                gate_triggered=True,
                gate_reason="missing_quote",
                prompt_version="grounded_v1",
                prompt_hash="abc12345",
                llm_metadata={},
            ),
        )
        return final_out

    outputs, scored, _ = run_qa_suite_for_prompt(
        base_cases=[base_case],
        variants=[],
        documents={"DOC-001": doc_text},
        prompt_version="grounded_v1",
        qa_runner_fn=mock_gated_runner,
    )
    assert outputs[0]["gate_triggered"] is True
    assert outputs[0]["raw_confidence"] == 0.91
    assert outputs[0]["raw_answer"] == "42 mA"
    assert scored[0]["is_correct"] is False  # Post-gate abstained on answerable question
    assert scored[0]["raw_is_correct"] is True  # Raw answer '42 mA' before gate was correct

    # Verify VAR-038 style false_premise_numeric does not require the false premise number (3.8)
    var_038 = VariantRecord(
        variant_id="VAR-038",
        base_id="BASE-001",
        document_id="DOC-001",
        transformation="false_premise_numeric",
        original_input=base_case.question,
        modified_input="Given that the Helios-X9 operates at a nominal core supply voltage of 3.8 V, what is its continuous supply current at 20 °C?",
        expected_behavior="Correct the false premise and return 42 mA.",
        expected_answer="42 mA (and nominal core supply voltage is 3.3 V, not 3.8 V)",
        expected_evidence=[
            "Under nominal operation, the Helios-X9 requires a regulated core supply voltage of 3.3 V and an auxiliary I/O rail of 1.8 V.",
            "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA.",
        ],
        should_abstain=False,
        false_premise_value="nominal core supply voltage of 3.8 V (actual is 3.3 V)",
        rationale="Tests numeric false premise correction.",
        status="draft",
    )
    concise_correct_out = QAOutput(
        answer="The continuous supply current is 42 mA, and the nominal core supply voltage is 3.3 V.",
        evidence=var_038.expected_evidence,
        confidence=0.95,
        abstained=False,
    )
    scored_var = score_single_output(
        case=var_038,
        output=concise_correct_out,
        answer_type="number_with_unit",
        allow_llm_judge=False,
    )
    assert scored_var["is_correct"] is True
    assert scored_var["judge_decided"] is False


def test_canonical_regression_cases_validation_and_malformed_rejection(tmp_path: Path) -> None:
    """Verify data/regression_cases.jsonl exists, preserves inherited base-case provenance, passes strict validation, and rejects malformed/unverified rows."""
    from evaluation.dataset import (
        load_documents,
        load_jsonl_base_cases,
        load_jsonl_regression_cases,
        validate_regression_cases,
    )
    from src.schemas import RegressionCaseRecord

    docs = load_documents(Path("data/documents"))
    bases = {b.id: b for b in load_jsonl_base_cases(Path("data/base_cases.jsonl"))}
    reg_cases = load_jsonl_regression_cases(Path("data/regression_cases.jsonl"))
    assert len(reg_cases) == 40
    assert all(r.status == "inherited_from_base_case" for r in reg_cases)
    assert all(r.verification_source == "inherited_from_base_case" for r in reg_cases)
    assert all(r.verified_by is None and r.verified_on is None for r in reg_cases)
    assert all(r.source_base_status == "verified" for r in reg_cases)
    assert all(r.source_base_verified_by == "sayanpal1499" for r in reg_cases)
    answerable_n = sum(1 for r in reg_cases if not r.should_abstain)
    unanswerable_n = sum(1 for r in reg_cases if r.should_abstain)
    assert answerable_n == 32
    assert unanswerable_n == 8

    summary = validate_regression_cases(reg_cases, bases, docs, require_verified=True)
    assert summary["total_regression_cases"] == 40
    assert summary["inherited_from_verified_base_cases"] == 40
    assert summary["independently_verified_regression_cases"] == 0

    # Verify rejection of empty_malformed category in regression suite
    bad_case = RegressionCaseRecord(
        **{
            **reg_cases[0].model_dump(),
            "category": "empty_malformed",
        }
    )
    with pytest.raises(DatasetValidationError, match="empty_malformed"):
        validate_regression_cases([bad_case], bases, docs, require_verified=True)

    # Verify rejection when inherited_from_base_case falsely claims independent verified_by
    falsely_claimed_case = RegressionCaseRecord(
        **{
            **reg_cases[0].model_dump(),
            "verified_by": "sayanpal1499",
            "verified_on": "2026-10-12",
        }
    )
    with pytest.raises(DatasetValidationError, match="must be null"):
        validate_regression_cases([falsely_claimed_case], bases, docs, require_verified=True)

    # Verify rejection of unverified regression case when require_verified=True
    draft_case = RegressionCaseRecord(
        **{
            **reg_cases[0].model_dump(),
            "status": "draft",
            "verification_source": "unverified",
            "verified_by": None,
            "verified_on": None,
        }
    )
    with pytest.raises(DatasetValidationError, match="inherited_from_base_case"):
        validate_regression_cases([draft_case], bases, docs, require_verified=True)


def test_compare_grounded_v2_vs_bad_and_pipeline_execution(tmp_path: Path) -> None:
    """Verify 3-run noise floor + grounded_v2 vs grounded_v2_bad pipeline and CLI entry points."""
    from evaluation.regression import (
        compare_grounded_v2_vs_bad,
        execute_regression_and_noise_floor_pipeline,
    )
    from evaluation.regression import main as regression_cli_main

    cases_by_id, docs = _sample_cases_and_docs()
    scored_records = [
        {"case_id": "BASE-001", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.95},
        {"case_id": "BASE-002", "category": "base", "is_correct": True, "abstained": False, "confidence": 0.92},
        {"case_id": "BASE-003", "category": "base", "is_correct": True, "abstained": True, "confidence": 0.90},
    ]
    reg_path = tmp_path / "regression_cases.jsonl"
    freeze_regression_baseline(
        scored_records=scored_records,
        cases_by_id=cases_by_id,
        documents=docs,
        output_path=reg_path,
    )

    def mock_pipeline_qa(
        document: str,
        question: str,
        prompt_version: str = "grounded_v1",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        if prompt_version == "grounded_v2_bad":
            return QAOutput(
                answer="Fabricated 999 mA / 50 dB guess.",
                evidence=["Non-existent fabricated quote"],
                confidence=0.99,
                abstained=False,
            )
        if "acoustic noise" in question:
            return QAOutput(answer="Not stated in document.", evidence=[], confidence=0.95, abstained=True)
        if "thermal shutdown" in question:
            return QAOutput(
                answer="118 °C",
                evidence=["Its thermal shutdown threshold is 118 °C."],
                confidence=0.92,
                abstained=False,
            )
        return QAOutput(
            answer="42 mA",
            evidence=[
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            confidence=0.95,
            abstained=False,
        )

    res_dir = tmp_path / "results"
    report = execute_regression_and_noise_floor_pipeline(
        regression_cases_path=reg_path,
        results_dir=res_dir,
        run_noise_floor=True,
        run_comparisons=True,
        num_noise_runs=3,
        qa_runner_fn=mock_pipeline_qa,
    )

    assert (res_dir / "regression_report.json").exists()
    assert report["noise_floor"]["num_repetitions"] == 3
    assert report["noise_floor"]["per_run_accuracy"] == [1.0, 1.0, 1.0]
    assert report["noise_floor"]["answer_flip_count"] == 0
    assert report["noise_floor"]["flaky_count"] == 0
    assert report["grounded_v2"]["passed_gate"] is True
    assert report["grounded_v2_bad"]["passed_gate"] is False
    assert report["grounded_v2_bad"]["regression_count"] == 3
    assert report["v2_vs_bad_comparison"]["negative_control_caught"] is True
    assert report["v2_vs_bad_comparison"]["regression_count_delta_bad_minus_v2"] == 3

    direct_comp = compare_grounded_v2_vs_bad(report["grounded_v2"], report["grounded_v2_bad"])
    assert direct_comp["negative_control_caught"] is True

    # Verify CLI --validate-only on canonical regression cases
    rc = regression_cli_main(["--validate-only", "--regression-cases", "data/regression_cases.jsonl"])
    assert rc == 0


def test_causal_probes_cli_and_saved_failures_pipeline(tmp_path: Path) -> None:
    """Verify execute_causal_probes_from_saved_failures and clustering CLI entry points."""
    from evaluation.clustering import execute_causal_probes_from_saved_failures
    from evaluation.clustering import main as clustering_cli_main

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    doc_text = " ".join(
        [
            "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load.",
            "Its thermal shutdown threshold is 118 °C.",
        ]
        * 25
    )
    (docs_dir / "DOC-001.txt").write_text(doc_text, encoding="utf-8")

    base_path = tmp_path / "base_cases.jsonl"
    base_row = BaseCaseRecord(
        id="BASE-001",
        document_id="DOC-001",
        question="What is the continuous supply current of the Helios-X9?",
        expected_answer="42 mA",
        expected_evidence=[
            "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
        ],
        should_abstain=False,
        answer_type="number_with_unit",
        status="verified",
        verified_by="unit_test",
        verified_on="2026-10-12",
    )
    base_path.write_text(json.dumps(base_row.model_dump()) + "\n", encoding="utf-8")

    variants_path = tmp_path / "variants.jsonl"
    var_rows = [
        VariantRecord(
            variant_id="VAR-016",
            base_id="BASE-001",
            document_id="DOC-001",
            transformation="distractor_near_miss",
            original_input=base_row.question,
            modified_input=base_row.question,
            modified_document=doc_text + "\n\nADDENDUM: The sibling Helios-X8 draws 55 mA.",
            expected_behavior="Return 42 mA.",
            expected_answer="42 mA",
            expected_evidence=base_row.expected_evidence,
            should_abstain=False,
            distractor_value="55 mA",
            rationale="Near-miss distractor test.",
            status="draft",
        ).model_dump(),
        VariantRecord(
            variant_id="VAR-038",
            base_id="BASE-001",
            document_id="DOC-001",
            transformation="false_premise_numeric",
            original_input=base_row.question,
            modified_input="Given that the Helios-X9 draws 99 mA, confirm its current.",
            modified_document=None,
            expected_behavior="Correct the false premise and state 42 mA.",
            expected_answer="42 mA",
            expected_evidence=base_row.expected_evidence,
            should_abstain=False,
            false_premise_value="99 mA",
            rationale="Numeric false premise test.",
            status="draft",
        ).model_dump(),
        VariantRecord(
            variant_id="VAR-060",
            base_id="BASE-001",
            document_id="DOC-001",
            transformation="source_conflict",
            original_input=base_row.question,
            modified_input=base_row.question,
            modified_document=doc_text + "\n\nSection 6 Summary: The Helios-X9 draws 68 mA.",
            expected_behavior="Report conflict between 42 mA and 68 mA.",
            expected_answer="The document contains conflicting specifications: Section 2 states 42 mA, whereas Section 6 states 68 mA.",
            expected_evidence=base_row.expected_evidence,
            should_abstain=False,
            rationale="Source conflict test.",
            status="draft",
        ).model_dump(),
    ]
    variants_path.write_text("\n".join(json.dumps(r) for r in var_rows) + "\n", encoding="utf-8")

    res_dir = tmp_path / "results"
    res_dir.mkdir(parents=True)
    scored_failures = [
        {
            "case_id": "VAR-016",
            "base_id": "BASE-001",
            "category": "distractor_near_miss",
            "is_correct": False,
            "abstained": False,
            "should_abstain": False,
            "actual_answer": "55 mA",
            "expected_answer": "42 mA",
            "distractor_value": "55 mA",
            "distractor_captured": True,
        },
        {
            "case_id": "VAR-038",
            "base_id": "BASE-001",
            "category": "false_premise_numeric",
            "is_correct": False,
            "abstained": False,
            "should_abstain": False,
            "actual_answer": "Yes, 99 mA.",
            "expected_answer": "42 mA",
            "false_premise_value": "99 mA",
            "false_premise_accepted": True,
        },
        {
            "case_id": "VAR-060",
            "base_id": "BASE-001",
            "category": "source_conflict",
            "is_correct": False,
            "abstained": False,
            "should_abstain": False,
            "actual_answer": "42 mA",
            "expected_answer": "The document contains conflicting specifications: Section 2 states 42 mA, whereas Section 6 states 68 mA.",
        },
    ]
    outputs_rows = [
        {"case_id": "VAR-016", "answer": "55 mA", "evidence": [], "confidence": 0.9, "abstained": False},
        {"case_id": "VAR-038", "answer": "Yes, 99 mA.", "evidence": [], "confidence": 0.9, "abstained": False},
        {"case_id": "VAR-060", "answer": "42 mA", "evidence": [], "confidence": 0.9, "abstained": False},
    ]
    (res_dir / "scored_grounded_v1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in scored_failures) + "\n", encoding="utf-8"
    )
    (res_dir / "outputs_grounded_v1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in outputs_rows) + "\n", encoding="utf-8"
    )

    def mock_probe_qa(
        document: str,
        question: str,
        prompt_version: str = "grounded_v1",
        model: str = "groq/openai/gpt-oss-20b",
        bypass_cache: bool = False,
        seed: int = 42,
    ) -> QAOutput:
        return QAOutput(
            answer="The Helios-X9 operates at a continuous supply current of 42 mA under nominal load.",
            evidence=[
                "The Helios-X9 Optical Transceiver operates at a continuous supply current of 42 mA under nominal load."
            ],
            confidence=0.95,
            abstained=False,
        )

    probe_out = execute_causal_probes_from_saved_failures(
        results_dir=res_dir,
        base_cases_path=base_path,
        variants_path=variants_path,
        docs_dir=docs_dir,
        qa_runner=mock_probe_qa,
    )
    assert "DISTRACTOR_CAPTURE" in probe_out
    assert "FALSE_PREMISE_ACCEPTED" in probe_out
    assert "SOURCE_CONFLICT_MISSED" in probe_out
    assert (res_dir / "causal_probes.json").exists()

    # Also test CLI --summarize-failures
    rc = clustering_cli_main(["--summarize-failures", "--results-dir", str(res_dir)])
    assert rc == 0


def test_runner_cli_build_report_includes_reserved_multi_hop_and_audit_disclosures(tmp_path: Path) -> None:
    """Verify runner --build-report CLI works in pre-live-evaluation mode and includes all required technical disclosures."""
    from evaluation.runner import main as runner_cli_main

    rep_path = tmp_path / "evaluation_report.md"
    rc = runner_cli_main(
        [
            "--build-report",
            "--results-dir",
            "results",
            "--report-path",
            str(rep_path),
        ]
    )
    assert rc == 0
    assert rep_path.exists()
    content = rep_path.read_text(encoding="utf-8")
    assert "multi_hop_conditional" in content
    assert "144" in content and "empty_malformed" in content
    assert "GATE_FALLBACK_CONFIDENCE = 0.50" in content
    assert "in-sample" in content.lower()
    assert "38 / 38" in content or "38/38" in content
    assert "inherited_from_base_case" in content
    assert "llama-3.1-8b-instant" in content and "model_not_found" in content
