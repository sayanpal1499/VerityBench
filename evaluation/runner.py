"""Phase 9: Strict Evaluation Runner, Orchestrator & Report Populator (evaluation/runner.py).

Enforces the Hard Ground-Truth Governance Gate:
- Refuses to execute the full evaluation if ANY record in `data/base_cases.jsonl`
  or `data/variants.jsonl` has `status != "verified"` (or lacks `verified_by` / `verified_on`).
- Never auto-promotes `"draft"` records to `"verified"` or alters `expected_answer`.
- Populates `reports/evaluation_report.md` strictly from saved artifacts in `results/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Optional

from evaluation.calibration import (
    compute_full_calibration_report,
    plot_calibration_curve,
)
from evaluation.clustering import (
    build_failure_records_for_review,
    export_failure_review_csv,
    load_failure_review_csv,
    plot_failure_distribution,
    summarize_failure_clusters,
)
from evaluation.dataset import (
    DatasetValidationError,
    compute_dataset_hash,
    load_documents,
    load_jsonl_audit_claims,
    load_jsonl_base_cases,
    load_jsonl_variants,
    validate_audit_claims,
    validate_base_cases,
    validate_variants,
)
from evaluation.hallucination import (
    evaluate_output_hallucinations,
    run_audit_benchmark,
)
from evaluation.metrics import (
    compare_ablation_metrics,
    compute_all_metrics,
    plot_accuracy_by_category,
    plot_hallucination_rate_by_category,
)
from evaluation.scoring import score_single_output
from src.prompts import get_prompt_hash
from src.qa_system import InputValidationError, answer_question
from src.schemas import BaseCaseRecord, QAOutput, VariantRecord


def verify_all_records_before_run(
    base_cases_path: Path | str = Path("data/base_cases.jsonl"),
    variants_path: Path | str = Path("data/variants.jsonl"),
    docs_dir: Path | str = Path("data/documents"),
    audit_claims_path: Optional[Path | str] = None,
) -> tuple[list[BaseCaseRecord], list[VariantRecord], dict[str, str]]:
    """Hard verification gate: refuse to run unless all base cases, variants, and audit claims are verified.

    Raises `DatasetValidationError` immediately if any record has `status != "verified"`,
    lacks `verified_by` / `verified_on`, or (for audit claims) has `needs_human_review == True`.
    """
    docs = load_documents(Path(docs_dir))
    base_cases = load_jsonl_base_cases(Path(base_cases_path))
    validate_base_cases(
        records=base_cases,
        documents=docs,
        require_verified=True,
    )
    variants = load_jsonl_variants(Path(variants_path))
    base_map = {b.id: b for b in base_cases}
    validate_variants(
        variants=variants,
        base_cases_by_id=base_map,
        documents=docs,
        require_verified=True,
    )
    resolved_audit_path = (
        Path(audit_claims_path)
        if audit_claims_path is not None
        else Path(base_cases_path).parent / "audit_claims.jsonl"
    )
    if resolved_audit_path.exists():
        audit_claims = load_jsonl_audit_claims(resolved_audit_path)
        validate_audit_claims(
            claims=audit_claims,
            base_cases_by_id=base_map,
            documents=docs,
            require_verified=True,
        )
    return base_cases, variants, docs


def run_qa_suite_for_prompt(
    base_cases: list[BaseCaseRecord],
    variants: list[VariantRecord],
    documents: dict[str, str],
    prompt_version: str,
    model_id: str = "groq/openai/gpt-oss-20b",
    judge_model_id: str = "gemini/gemini-3.5-flash-lite",
    bypass_cache: bool = False,
    qa_runner_fn: Optional[Callable[..., QAOutput]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run QA system, scorer, and 3-layer hallucination detector across all verified records."""
    runner = qa_runner_fn or answer_question
    base_by_id = {b.id: b for b in base_cases}

    outputs: list[dict[str, Any]] = []
    scored_records: list[dict[str, Any]] = []
    claim_records: list[dict[str, Any]] = []

    all_items: list[BaseCaseRecord | VariantRecord] = [*base_cases, *variants]
    for item in all_items:
        is_variant = isinstance(item, VariantRecord)
        cid = item.variant_id if is_variant else item.id
        base_id = item.base_id if is_variant else item.id
        category = item.transformation if is_variant else "base"
        answer_type = (
            base_by_id[base_id].answer_type if is_variant else item.answer_type
        )
        doc_text = (
            item.modified_document
            if (is_variant and item.modified_document is not None)
            else documents[item.document_id]
        )
        question_text = item.modified_input if is_variant else item.question

        validation_rejected = False
        try:
            qa_out = runner(
                document=doc_text,
                question=question_text,
                prompt_version=prompt_version,
                model=model_id,
                bypass_cache=bypass_cache,
                seed=42,
            )
        except InputValidationError as exc:
            validation_rejected = True
            qa_out = QAOutput(
                answer=f"InputValidationError: {exc}",
                evidence=[],
                confidence=1.0,
                abstained=True,
            )

        trace = getattr(qa_out, "_trace", None)
        gate_triggered = bool(
            getattr(trace, "gate_triggered", getattr(qa_out, "gate_triggered", False))
        )
        raw_out = getattr(trace, "raw_output", None)
        raw_answer = (
            raw_out.answer
            if raw_out is not None
            else getattr(qa_out, "raw_answer", qa_out.answer)
        )
        raw_confidence = (
            raw_out.confidence
            if raw_out is not None
            else getattr(qa_out, "raw_confidence", qa_out.confidence)
        )
        raw_abstained = (
            raw_out.abstained
            if raw_out is not None
            else getattr(qa_out, "raw_abstained", qa_out.abstained)
        )

        out_row = {
            "case_id": cid,
            "base_id": base_id,
            "document_id": item.document_id,
            "category": category,
            "question": question_text,
            "expected_answer": item.expected_answer,
            "answer": qa_out.answer,
            "evidence": qa_out.evidence,
            "confidence": qa_out.confidence,
            "abstained": qa_out.abstained,
            "validation_rejected": validation_rejected,
            "gate_triggered": gate_triggered,
            "raw_answer": raw_answer,
            "raw_confidence": raw_confidence,
            "raw_abstained": raw_abstained,
        }
        outputs.append(out_row)

        scored = score_single_output(
            case=item,
            output=qa_out,
            answer_type=answer_type,
            validation_rejected=validation_rejected,
            judge_model=judge_model_id,
            bypass_cache=bypass_cache,
        )
        if gate_triggered and raw_out is not None:
            raw_scored = score_single_output(
                case=item,
                output=raw_out,
                answer_type=answer_type,
                validation_rejected=False,
                judge_model=judge_model_id,
                bypass_cache=bypass_cache,
            )
            raw_is_correct = raw_scored["is_correct"]
        else:
            raw_is_correct = scored["is_correct"]

        scored["gate_triggered"] = gate_triggered
        scored["raw_confidence"] = raw_confidence
        scored["raw_is_correct"] = raw_is_correct
        scored_records.append(scored)

        hall = evaluate_output_hallucinations(
            case_id=cid,
            category=category,
            document=doc_text,
            question=question_text,
            output=qa_out,
            judge_model=judge_model_id,
            bypass_cache=bypass_cache,
        )
        claim_records.append(hall)

    return outputs, scored_records, claim_records


def execute_verified_evaluation(
    base_cases_path: Path | str = Path("data/base_cases.jsonl"),
    variants_path: Path | str = Path("data/variants.jsonl"),
    docs_dir: Path | str = Path("data/documents"),
    results_dir: Path | str = Path("results"),
    figures_dir: Path | str = Path("reports/figures"),
    model_id: str = "groq/openai/gpt-oss-20b",
    judge_model_id: str = "gemini/gemini-3.5-flash-lite",
    bypass_cache: bool = False,
    qa_runner_fn: Optional[Callable[..., QAOutput]] = None,
    audit_claims_path: Optional[Path | str] = None,
    allow_llm_judge: bool = True,
) -> dict[str, Any]:
    """Execute the full evaluation pipeline ONLY after verifying all dataset records."""
    # Gate 1: Strict ground-truth verification check
    base_cases, variants, docs = verify_all_records_before_run(
        base_cases_path=base_cases_path,
        variants_path=variants_path,
        docs_dir=docs_dir,
        audit_claims_path=audit_claims_path,
    )

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # RUN-01-MAIN: grounded_v1
    out_v1, scored_v1, claims_v1 = run_qa_suite_for_prompt(
        base_cases=base_cases,
        variants=variants,
        documents=docs,
        prompt_version="grounded_v1",
        model_id=model_id,
        judge_model_id=judge_model_id,
        bypass_cache=bypass_cache,
        qa_runner_fn=qa_runner_fn,
    )
    _write_jsonl(out_dir / "outputs_grounded_v1.jsonl", out_v1)
    _write_jsonl(out_dir / "scored_grounded_v1.jsonl", scored_v1)
    _write_jsonl(out_dir / "claims_grounded_v1.jsonl", claims_v1)

    # RUN-02-ABLATION: naive_v0
    out_v0, scored_v0, claims_v0 = run_qa_suite_for_prompt(
        base_cases=base_cases,
        variants=variants,
        documents=docs,
        prompt_version="naive_v0",
        model_id=model_id,
        judge_model_id=judge_model_id,
        bypass_cache=bypass_cache,
        qa_runner_fn=qa_runner_fn,
    )
    _write_jsonl(out_dir / "outputs_naive_v0.jsonl", out_v0)
    _write_jsonl(out_dir / "scored_naive_v0.jsonl", scored_v0)
    _write_jsonl(out_dir / "claims_naive_v0.jsonl", claims_v0)

    # Phase 6: Metrics + Calibration
    g_metrics = compute_all_metrics(scored_v1, claims_v1)
    n_metrics = compute_all_metrics(scored_v0, claims_v0)
    ablation = compare_ablation_metrics(g_metrics, n_metrics)
    calibration_v1 = compute_full_calibration_report(scored_v1)
    calibration_v0 = compute_full_calibration_report(scored_v0)

    # Phase 5 Human Audit Benchmark (when audit_claims.jsonl exists)
    resolved_audit_path = (
        Path(audit_claims_path)
        if audit_claims_path is not None
        else Path(base_cases_path).parent / "audit_claims.jsonl"
    )
    human_audit_summary: dict[str, Any] = {}
    if resolved_audit_path.exists():
        audit_claims = load_jsonl_audit_claims(resolved_audit_path)
        human_audit_summary = run_audit_benchmark(
            audit_claims=audit_claims,
            documents=docs,
            judge_model=judge_model_id,
            bypass_cache=bypass_cache,
            allow_llm_judge=allow_llm_judge,
        )
        _write_json(out_dir / "audit_benchmark.json", human_audit_summary)

    # Phase 7: Failure clustering export
    outputs_v1_by_id = {str(o["case_id"]): o for o in out_v1}
    claims_v1_by_id = {str(c["case_id"]): c for c in claims_v1}
    review_csv_path = out_dir / "failure_review.csv"
    if not review_csv_path.exists():
        cand_failures = build_failure_records_for_review(
            scored_records=scored_v1,
            outputs_by_id=outputs_v1_by_id,
            hallucinations_by_id=claims_v1_by_id,
        )
        export_failure_review_csv(cand_failures, review_csv_path)
    reviewed_failures = load_failure_review_csv(review_csv_path)
    cluster_summary = summarize_failure_clusters(reviewed_failures)

    # Generate figures strictly from saved metrics
    plot_accuracy_by_category(
        grounded_metrics=g_metrics,
        output_path=fig_dir / "accuracy_by_category.png",
        naive_metrics=n_metrics,
    )
    plot_hallucination_rate_by_category(
        grounded_metrics=g_metrics,
        output_path=fig_dir / "hallucination_rate_by_category.png",
        naive_metrics=n_metrics,
    )
    plot_calibration_curve(calibration_v1, fig_dir / "calibration_curve.png")
    plot_failure_distribution(cluster_summary, fig_dir / "failure_distribution.png")

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sut_model_id": model_id,
        "judge_model_id": judge_model_id,
        "dataset_hash": compute_dataset_hash(Path(base_cases_path), Path(variants_path)),
        "prompt_hashes": {
            "grounded_v1": get_prompt_hash("grounded_v1"),
            "naive_v0": get_prompt_hash("naive_v0"),
            "grounded_v2": get_prompt_hash("grounded_v2"),
        },
        "grounded_v1": g_metrics,
        "naive_v0": n_metrics,
        "ablation": ablation,
        "human_audit": human_audit_summary,
        "calibration": {
            "grounded_v1": calibration_v1,
            "naive_v0": calibration_v0,
        },
        "failure_clusters": cluster_summary,
    }
    _write_json(out_dir / "metrics.json", metrics_payload)
    return metrics_payload


def populate_evaluation_report_from_saved_results(
    results_dir: Path | str = Path("results"),
    report_path: Path | str = Path("reports/evaluation_report.md"),
) -> Path:
    """Populate `reports/evaluation_report.md` strictly from saved JSON artifacts in `results/`."""
    res_dir = Path(results_dir)
    metrics_file = res_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Cannot populate report: required saved results file '{metrics_file}' does not exist."
        )

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    g1 = metrics["grounded_v1"]
    n0 = metrics["naive_v0"]
    ablation = metrics["ablation"]
    cal_v1 = metrics.get("calibration", {}).get("grounded_v1", {})
    cal_final = cal_v1.get("post_gate_system_calibration", {})
    cal_raw = cal_v1.get("intrinsic_raw_model_calibration", {})
    clusters = metrics.get("failure_clusters", {})
    human_audit = metrics.get("human_audit", {})

    unmet_flag_banner = ""
    if clusters.get("unmet_three_cluster_requirement", False):
        unmet_flag_banner = (
            "\n> [!WARNING]\n"
            f"> **UNMET THREE-CLUSTER REQUIREMENT FLAG:** Observed `{clusters.get('observed_cluster_count', 0)}` "
            "distinct failure clusters (`< 3` required by R8). Per Ground-Truth Governance Rule §8.2, "
            "clusters were never fabricated or split artificially.\n"
        )

    ov_acc = g1["overall_all_150"]
    ov_cb = ov_acc["cluster_bootstrap_ci_95"]["ci_95"]
    ov_hall = g1["hallucination"]["output_level_all"]
    cl_hall = g1["hallucination"].get("claim_level_all", {})
    abs_g1 = g1.get("abstention", {})
    prec_g1 = abs_g1.get("abstention_precision", {})
    rec_g1 = abs_g1.get("abstention_recall", {})
    triv_g1 = abs_g1.get("trivial_always_abstain_baseline", {}).get("overall_accuracy", {})

    cat_rows = []
    for cat, stats in g1.get("by_category", {}).items():
        w_ci = stats.get("wilson_ci_95", [0.0, 0.0])
        cat_rows.append(
            f"| `{cat}` | `{stats.get('formatted', '')}` | `[{w_ci[0]*100:.1f}%, {w_ci[1]*100:.1f}%]` |"
        )
    cat_table = "\n".join(cat_rows)

    audit_section = ""
    if human_audit:
        raw_l2 = human_audit.get("raw_layer2_resolution", {})
        mapped_ag = human_audit.get("exact_agreement", {})
        kappa = human_audit.get("cohens_kappa", {})
        audit_section = f"""
## 4. Hallucination Detector Human Audit Benchmark

- **Confirmed Layer-2 Deterministic Agreement (Unmapped):** `{raw_l2.get('deterministic_correct_count', 0)}/{human_audit.get('total_audited_claims', 0)}` (`{raw_l2.get('strict_unmapped_agreement', 0.0)*100:.2f}%`)
- **Deferred `needs_judge` Claims (Not Counted as Confirmed Detector Successes):** `{raw_l2.get('needs_judge_count', 0)}/{human_audit.get('total_audited_claims', 0)}` (`{raw_l2.get('needs_judge_rate', 0.0)*100:.2f}%`, IDs: `{raw_l2.get('needs_judge_claim_ids', [])}`)
- **Mapped 3-Class Agreement (Including `needs_judge -> unsupported` Fallback on `{raw_l2.get('fallback_inflated_claim_ids', [])}`):** `{mapped_ag.get('numerator', 0)}/{mapped_ag.get('denominator', 0)}` (`{mapped_ag.get('rate', 0.0)*100:.2f}%`), Cohen's $\\kappa = {kappa.get('value', 0.0):.4f}$ (`95% CI: {kappa.get('ci_95', [0.0, 0.0])}`)
"""

    cluster_items = clusters.get("clusters", [])
    cluster_rows = []
    for c in cluster_items:
        rep_ids = ", ".join(c.get("representative_case_ids", []))
        cluster_rows.append(
            f"| `{c.get('cluster_label', '')}` | `{c.get('formatted_share', '')}` | `{rep_ids}` | "
            f"{c.get('model_behavior_hypothesis', '')} | {c.get('concrete_improvement', '')} |"
        )
    cluster_table = (
        "\n".join(cluster_rows)
        if cluster_rows
        else "| *(No failures observed)* | `0/0 (0.0%)` | — | — | — |"
    )
    cluster_section = f"""
---

## 5. Failure Mode Distribution & Root-Cause Clusters (`grounded_v1`)

- **Total Observed Failures:** `{clusters.get('total_failures', 0)}` across `{clusters.get('observed_cluster_count', 0)}` distinct failure cluster(s) (minimum required: `{clusters.get('min_required_clusters', 3)}`).

| Cluster Label | Share (`k/M`) | Representative Cases | Root-Cause Hypothesis | Concrete Remediation |
| :--- | :--- | :--- | :--- | :--- |
{cluster_table}
"""

    content = f"""# VerityBench — Adversarial Reliability Evaluation Report

**Generated from Saved Results (`results/metrics.json`):** `{metrics.get('generated_at_utc', '')}`  
**Fixed SUT Model:** `{metrics.get('sut_model_id', '')}` | **Fixed Judge Model:** `{metrics.get('judge_model_id', '')}`  
**Dataset Hash:** `{metrics.get('dataset_hash', '')}`
{unmet_flag_banner}
---

## 1. Executive Summary

- **Overall Accuracy (`grounded_v1`):** `{ov_acc['formatted']}`, Base-Case Cluster Bootstrap 95% CI: `[{ov_cb[0]*100:.1f}%, {ov_cb[1]*100:.1f}%]`.
- **Output-Level Hallucination Rate (`grounded_v1`):** `{ov_hall['numerator']}/{ov_hall['denominator']}` (`{ov_hall['rate']*100:.1f}%`).
- **Claim-Level Hallucination & Deferral Breakdown (`grounded_v1`):** Bad claims `{cl_hall.get('numerator', 0)}/{cl_hall.get('denominator', 0)}` (`{cl_hall.get('rate', 0.0)*100:.1f}%`); Confirmed supported claims: `{cl_hall.get('supported_numerator', 0)}`; Layer-2 deterministically decided: `{cl_hall.get('layer2_decided_claims', 0)}`; Layer-2 deferred (`needs_judge`): `{cl_hall.get('needs_judge_claims', 0)}` (unresolved `needs_judge`: `{cl_hall.get('unresolved_needs_judge_claims', 0)}`).
- **Ablation vs. `naive_v0`:** Accuracy delta `{ablation['overall_accuracy_delta']*100:+.1f} pp`; Output hallucination delta `{ablation['output_hallucination_rate_delta']*100:+.1f} pp`.
- **Confidence Calibration (`grounded_v1`):** Post-Gate ECE = `{cal_final.get('ece', 0.0):.4f}` (Brier = `{cal_final.get('brier_score', 0.0):.4f}`); Intrinsic Raw-Model ECE = `{cal_raw.get('ece', 0.0):.4f}` (Brier = `{cal_raw.get('brier_score', 0.0):.4f}`).

---

## 2. Per-Category Accuracy (`grounded_v1`)

| Category | Accuracy (`k/n`) | 95% Wilson CI |
| :--- | :--- | :--- |
{cat_table}

---

## 3. Selective Abstention Metrics (`grounded_v1`)

- **Abstention Precision:** `{prec_g1.get('numerator', 0)}/{prec_g1.get('denominator', 0)}` (`{prec_g1.get('rate', 0.0)*100:.1f}%`)
- **Abstention Recall:** `{rec_g1.get('numerator', 0)}/{rec_g1.get('denominator', 0)}` (`{rec_g1.get('rate', 0.0)*100:.1f}%`)
- **Over-Abstention (FP) / Under-Abstention (FN):** `{abs_g1.get('fp_abstain_over_abstention', 0)}` / `{abs_g1.get('fn_abstain_under_abstention', 0)}`
- **Trivial Always-Abstain Baseline Accuracy:** `{triv_g1.get('numerator', 0)}/{triv_g1.get('denominator', 0)}` (`{triv_g1.get('rate', 0.0)*100:.1f}%`)
{audit_section}{cluster_section}"""

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VerityBench Evaluation Runner (Strict Verification Gate Enforced)"
    )
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Validate that all base cases and variants have status == 'verified'.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Execute full evaluation suite (requires all records verified).",
    )
    parser.add_argument(
        "--build-report",
        action="store_true",
        help="Populate reports/evaluation_report.md from saved results/metrics.json.",
    )
    parser.add_argument(
        "--base-cases",
        type=Path,
        default=Path("data/base_cases.jsonl"),
        help="Path to base_cases.jsonl",
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path("data/variants.jsonl"),
        help="Path to variants.jsonl",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data/documents"),
        help="Path to source documents directory",
    )
    args = parser.parse_args(argv)

    if args.check_gate:
        base_cases, variants, _ = verify_all_records_before_run(
            base_cases_path=args.base_cases,
            variants_path=args.variants,
            docs_dir=args.docs_dir,
        )
        print(
            f"VERIFICATION GATE PASSED: {len(base_cases)} base cases and "
            f"{len(variants)} variants verified."
        )
        return 0

    if args.run_all:
        execute_verified_evaluation(
            base_cases_path=args.base_cases,
            variants_path=args.variants,
            docs_dir=args.docs_dir,
        )
        return 0

    if args.build_report:
        out = populate_evaluation_report_from_saved_results()
        print(f"Populated evaluation report at {out}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
