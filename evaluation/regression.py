"""Phase 8: Prompt Regression Runner & Non-Determinism Noise-Floor Harness (evaluation/regression.py).

Implements:
1. Freezing baseline passing cases (`data/regression_cases.jsonl`) with full
   provenance (`model_id`, `prompt_version`, `dataset_version`) without ever
   modifying ground-truth `expected_answer`.
2. Uncached 3-run noise-floor measurement (`bypass_cache=True`) to identify
   non-deterministic (`flaky`) cases and establish the exact instability rule.
3. Prompt-change regression comparison (`grounded_v1` vs `grounded_v2` and
   deliberate negative control `grounded_v2_bad`), excluding `flaky` cases from
   strict regression counting while reporting both regressions and newly fixed
   cases (`progress`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

from evaluation.clustering import assign_initial_failure_cluster
from evaluation.dataset import (
    DEFAULT_BASE_CASES_PATH,
    DEFAULT_DOCS_DIR,
    DEFAULT_REGRESSION_CASES_PATH,
    load_documents,
    load_jsonl_base_cases,
    validate_regression_cases,
)
from evaluation.scoring import score_single_output
from src.qa_system import InputValidationError, answer_question
from src.schemas import BaseCaseRecord, QAOutput, VariantRecord

INSTABILITY_RULE = (
    "A baseline-passing test case is marked 'flaky' if it fails (is_correct == False) "
    "in >= 1 out of 3 uncached repetitions (bypass_cache=True, temperature=0.0, seed=42) "
    "under the identical baseline model and prompt (pass_count < 3)."
)


def _extract_case_id(case_or_dict: Any) -> str:
    if isinstance(case_or_dict, dict):
        return str(
            case_or_dict.get("case_id")
            or case_or_dict.get("variant_id")
            or case_or_dict.get("id")
        )
    return str(
        getattr(case_or_dict, "variant_id", None) or getattr(case_or_dict, "id", "")
    )


def freeze_regression_baseline(
    scored_records: list[dict[str, Any]],
    cases_by_id: dict[str, BaseCaseRecord | VariantRecord | dict[str, Any]],
    documents: dict[str, str],
    output_path: Path | str,
    model_id: str = "groq/openai/gpt-oss-20b",
    prompt_version: str = "grounded_v1",
    dataset_version: str = "v1.0",
    max_cases: Optional[int] = 40,
) -> list[dict[str, Any]]:
    """Freeze cases that passed under baseline `grounded_v1` to `regression_cases.jsonl`.

    Never alters `expected_answer` or ground-truth metadata.
    """
    passing_scored = [
        r
        for r in scored_records
        if bool(r.get("is_correct", False)) and r.get("category") != "empty_malformed"
    ]
    passing_scored.sort(key=lambda r: _extract_case_id(r))

    if max_cases is not None and len(passing_scored) > max_cases:
        base_passing = [
            r for r in passing_scored if _extract_case_id(r).startswith("BASE-")
        ]
        var_passing = [
            r for r in passing_scored if _extract_case_id(r).startswith("VAR-")
        ]
        half = max_cases // 2
        selected = base_passing[:half] + var_passing[: max_cases - len(base_passing[:half])]
        if len(selected) < max_cases:
            remaining_ids = {_extract_case_id(r) for r in selected}
            for r in passing_scored:
                if _extract_case_id(r) not in remaining_ids:
                    selected.append(r)
                    if len(selected) >= max_cases:
                        break
        passing_scored = sorted(selected, key=lambda r: _extract_case_id(r))

    frozen_cases: list[dict[str, Any]] = []
    for scored in passing_scored:
        cid = _extract_case_id(scored)
        case_obj = cases_by_id[cid]
        if isinstance(case_obj, dict):
            doc_id = case_obj.get("document_id") or case_obj.get("doc_id", "")
            doc_text = (
                case_obj.get("modified_document")
                or case_obj.get("document")
                or documents.get(doc_id, "")
            )
            question = case_obj.get("modified_input") or case_obj["question"]
            expected_answer = case_obj["expected_answer"]
            expected_evidence = case_obj.get("expected_evidence", [])
            should_abstain = bool(case_obj.get("should_abstain", False))
            base_id = case_obj.get("base_id", cid)
            category = case_obj.get("transformation") or case_obj.get("category", "base")
            answer_type = case_obj.get("answer_type", scored.get("answer_type", "number_with_unit"))
            false_premise_val = case_obj.get("false_premise_value")
            injected_val = case_obj.get("injected_target_value")
            conflict_vals = case_obj.get("conflict_values")
        else:
            doc_id = case_obj.document_id
            mod_doc = getattr(case_obj, "modified_document", None)
            doc_text = mod_doc if mod_doc is not None else documents.get(doc_id, "")
            transformation = getattr(case_obj, "transformation", None)
            question = (
                getattr(case_obj, "modified_input", None)
                if transformation
                else getattr(case_obj, "question", "")
            )
            expected_answer = case_obj.expected_answer
            expected_evidence = list(case_obj.expected_evidence)
            should_abstain = bool(case_obj.should_abstain)
            base_id = getattr(case_obj, "base_id", cid)
            category = transformation or "base"
            answer_type = getattr(case_obj, "answer_type", scored.get("answer_type", "number_with_unit"))
            false_premise_val = getattr(case_obj, "false_premise_value", None)
            injected_val = getattr(case_obj, "injected_target_value", None)
            conflict_vals = getattr(case_obj, "conflict_values", None)

        frozen_cases.append(
            {
                "case_id": cid,
                "base_id": base_id,
                "document_id": doc_id,
                "category": category,
                "answer_type": answer_type,
                "document": doc_text,
                "question": question,
                "expected_answer": expected_answer,
                "expected_evidence": expected_evidence,
                "should_abstain": should_abstain,
                "false_premise_value": false_premise_val,
                "injected_target_value": injected_val,
                "conflict_values": conflict_vals,
                "baseline_passed": True,
                "baseline_abstained": bool(scored.get("abstained", False)),
                "baseline_confidence": float(scored.get("confidence", 0.0)),
                "baseline_model_id": model_id,
                "baseline_prompt_version": prompt_version,
                "dataset_version": dataset_version,
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in frozen_cases:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    return frozen_cases


def load_regression_cases(path: Path | str) -> list[dict[str, Any]]:
    """Load frozen regression cases from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    cases: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


class _RegressionCaseAdapter:
    """Lightweight adapter so dict regression cases work seamlessly with score_single_output."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.id = d.get("case_id") or d.get("id", "")
        self.variant_id = d.get("case_id") if d.get("category", "base") != "base" else None
        self.base_id = d.get("base_id", self.id)
        self.document_id = d.get("document_id", "")
        cat = d.get("category", "base")
        self.transformation = cat if cat != "base" else None
        self.question = d.get("question", "")
        self.modified_input = d.get("question", "") if self.transformation else None
        self.expected_answer = d.get("expected_answer", "")
        self.expected_evidence = d.get("expected_evidence", [])
        self.should_abstain = bool(d.get("should_abstain", False))
        self.answer_type = d.get("answer_type", "number_with_unit")
        self.false_premise_value = d.get("false_premise_value")
        self.injected_target_value = d.get("injected_target_value")
        self.conflict_values = d.get("conflict_values")


def _default_score_case(
    case_dict: dict[str, Any],
    qa_output: QAOutput,
    validation_rejected: bool = False,
) -> dict[str, Any]:
    adapter = _RegressionCaseAdapter(case_dict)
    return score_single_output(
        case=adapter,  # type: ignore[arg-type]
        output=qa_output,
        answer_type=adapter.answer_type,
        validation_rejected=validation_rejected,
    )


def _coerce_qa_output(res: Any) -> QAOutput:
    if isinstance(res, QAOutput):
        return res
    if isinstance(res, dict):
        if "final_output" in res and isinstance(res["final_output"], dict):
            fo = res["final_output"]
            return QAOutput(
                answer=fo.get("answer", ""),
                evidence=fo.get("evidence", []),
                confidence=float(fo.get("confidence", 0.5)),
                abstained=bool(fo.get("abstained", False)),
            )
        return QAOutput(
            answer=res.get("answer", ""),
            evidence=res.get("evidence", []),
            confidence=float(res.get("confidence", 0.5)),
            abstained=bool(res.get("abstained", False)),
        )
    raise TypeError(f"Unsupported QA output type: {type(res)}")


def build_canonical_regression_cases(
    base_cases_path: Path | str = DEFAULT_BASE_CASES_PATH,
    docs_dir: Path | str = DEFAULT_DOCS_DIR,
    output_path: Path | str = DEFAULT_REGRESSION_CASES_PATH,
    max_cases: int = 40,
    model_id: str = "groq/openai/gpt-oss-20b",
    prompt_version: str = "grounded_v1",
    dataset_version: str = "v1.0",
) -> list[dict[str, Any]]:
    """Build `data/regression_cases.jsonl` by inheriting ground truth from verified canonical base cases."""
    docs = load_documents(Path(docs_dir))
    base_cases = load_jsonl_base_cases(Path(base_cases_path))
    verified_bases = [b for b in base_cases if b.status == "verified"]
    selected = verified_bases[:max_cases]

    frozen_cases: list[dict[str, Any]] = []
    for b in selected:
        frozen_cases.append(
            {
                "case_id": b.id,
                "base_id": b.id,
                "document_id": b.document_id,
                "category": "base",
                "answer_type": b.answer_type,
                "document": docs[b.document_id],
                "question": b.question,
                "expected_answer": b.expected_answer,
                "expected_evidence": list(b.expected_evidence),
                "should_abstain": bool(b.should_abstain),
                "false_premise_value": None,
                "injected_target_value": None,
                "conflict_values": None,
                "baseline_passed": True,
                "baseline_abstained": bool(b.should_abstain),
                "baseline_confidence": 0.95,
                "baseline_model_id": model_id,
                "baseline_prompt_version": prompt_version,
                "dataset_version": dataset_version,
                "status": "inherited_from_base_case",
                "verification_source": "inherited_from_base_case",
                "source_base_status": b.status,
                "source_base_verified_by": b.verified_by,
                "source_base_verified_on": b.verified_on,
                "verified_by": None,
                "verified_on": None,
                "baseline_source": "canonical_base_case_seed",
                "notes": (
                    f"Ground-truth question, expected_answer, expected_evidence, and should_abstain "
                    f"inherited directly from human-verified canonical base case {b.id} in "
                    f"data/base_cases.jsonl (source_base_verified_by={b.verified_by}, "
                    f"source_base_verified_on={b.verified_on}); not independently human-reviewed "
                    f"as a separate regression dataset and not yet frozen from a live grounded_v1 run."
                ),
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in frozen_cases:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    base_map = {b.id: b for b in base_cases}
    validate_regression_cases(
        records=frozen_cases,
        base_cases_by_id=base_map,
        documents=docs,
        require_verified=True,
        min_cases=min(20, len(frozen_cases)),
    )
    return frozen_cases


def run_noise_floor_experiment(
    regression_cases: list[dict[str, Any]],
    num_runs: int = 3,
    model_id: str = "groq/openai/gpt-oss-20b",
    prompt_version: str = "grounded_v1",
    qa_runner_fn: Optional[Callable[..., Any]] = None,
    scorer_fn: Optional[Callable[[dict[str, Any], QAOutput], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run frozen baseline cases `num_runs` times with `bypass_cache=True` to measure the noise floor."""
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    runner = qa_runner_fn or answer_question
    scorer = scorer_fn or (lambda c, out: _default_score_case(c, out))

    per_case_results: list[dict[str, Any]] = []
    flaky_ids: list[str] = []
    stable_ids: list[str] = []
    answer_flip_ids: list[str] = []
    run_pass_counts = [0 for _ in range(num_runs)]

    for case in regression_cases:
        cid = _extract_case_id(case)
        run_outcomes: list[dict[str, Any]] = []
        pass_count = 0
        normalized_answers_seen: set[tuple[str, bool]] = set()

        for run_idx in range(1, num_runs + 1):
            try:
                raw_res = runner(
                    document=case["document"],
                    question=case["question"],
                    prompt_version=prompt_version,
                    model=model_id,
                    bypass_cache=True,
                    seed=42,
                )
                qa_out = _coerce_qa_output(raw_res)
            except InputValidationError as exc:
                qa_out = QAOutput(
                    answer=f"InputValidationError: {exc}",
                    evidence=[],
                    confidence=1.0,
                    abstained=True,
                )
            scored = scorer(case, qa_out)
            is_correct = bool(scored.get("is_correct", False))
            if is_correct:
                pass_count += 1
                run_pass_counts[run_idx - 1] += 1

            normalized_answers_seen.add(
                (" ".join(qa_out.answer.strip().lower().split()), bool(qa_out.abstained))
            )

            run_outcomes.append(
                {
                    "run_index": run_idx,
                    "is_correct": is_correct,
                    "answer": qa_out.answer,
                    "abstained": qa_out.abstained,
                    "confidence": qa_out.confidence,
                    "scoring_method": scored.get("scoring_method", "deterministic"),
                    "reason": scored.get("reason", ""),
                }
            )

        is_flaky = pass_count < num_runs
        has_answer_flip = len(normalized_answers_seen) > 1
        if is_flaky:
            flaky_ids.append(cid)
        else:
            stable_ids.append(cid)
        if has_answer_flip:
            answer_flip_ids.append(cid)

        per_case_results.append(
            {
                "case_id": cid,
                "category": case.get("category", "base"),
                "num_runs": num_runs,
                "pass_count": pass_count,
                "fail_count": num_runs - pass_count,
                "is_flaky": is_flaky,
                "has_answer_flip": has_answer_flip,
                "status": "flaky" if is_flaky else "stable_pass",
                "runs": run_outcomes,
            }
        )

    total = len(regression_cases)
    flaky_count = len(flaky_ids)
    stable_count = len(stable_ids)
    per_run_accuracy = [
        round(c / total, 4) if total > 0 else 0.0 for c in run_pass_counts
    ]

    return {
        "model_id": model_id,
        "prompt_version": prompt_version,
        "num_repetitions": num_runs,
        "bypass_cache": True,
        "instability_rule": INSTABILITY_RULE,
        "total_frozen_cases": total,
        "stable_count": stable_count,
        "flaky_count": flaky_count,
        "flaky_rate": round(flaky_count / total, 4) if total > 0 else 0.0,
        "flaky_fraction": f"{flaky_count}/{total}",
        "flaky_ids": flaky_ids,
        "stable_ids": stable_ids,
        "answer_flip_count": len(answer_flip_ids),
        "answer_flip_rate": round(len(answer_flip_ids) / total, 4) if total > 0 else 0.0,
        "answer_flip_ids": answer_flip_ids,
        "per_run_accuracy": per_run_accuracy,
        "cases": per_case_results,
    }


def run_prompt_regression_comparison(
    test_cases: list[dict[str, Any]],
    baseline_scored_by_id: dict[str, dict[str, Any]],
    flaky_ids: set[str] | list[str],
    candidate_prompt_version: str,
    baseline_prompt_version: str = "grounded_v1",
    model_id: str = "groq/openai/gpt-oss-20b",
    qa_runner_fn: Optional[Callable[..., Any]] = None,
    scorer_fn: Optional[Callable[[dict[str, Any], QAOutput], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compare a candidate prompt version against the baseline, excluding flaky cases from regression counting."""
    runner = qa_runner_fn or answer_question
    scorer = scorer_fn or (lambda c, out: _default_score_case(c, out))
    flaky_set = set(flaky_ids)

    regressions: list[dict[str, Any]] = []
    newly_fixed: list[dict[str, Any]] = []
    unchanged_pass: list[dict[str, Any]] = []
    unchanged_fail: list[dict[str, Any]] = []
    flaky_excluded: list[dict[str, Any]] = []
    comparison_table: list[dict[str, Any]] = []

    for case in test_cases:
        cid = _extract_case_id(case)
        base_scored = baseline_scored_by_id.get(cid, {})
        old_correct = bool(
            base_scored.get("is_correct", case.get("baseline_passed", True))
        )
        old_status = "PASS" if old_correct else "FAIL"

        raw_res = runner(
            document=case["document"],
            question=case["question"],
            prompt_version=candidate_prompt_version,
            model=model_id,
            bypass_cache=False,
            seed=42,
        )
        qa_out = _coerce_qa_output(raw_res)
        cand_scored = scorer(case, qa_out)
        new_correct = bool(cand_scored.get("is_correct", False))
        new_status = "PASS" if new_correct else "FAIL"

        is_flaky = cid in flaky_set
        failure_type = "NONE"
        if not new_correct:
            failure_type = assign_initial_failure_cluster(cand_scored, None) or "FORMAT_REASONING_ERROR"

        row = {
            "case_id": cid,
            "base_id": case.get("base_id", cid),
            "category": case.get("category", "base"),
            "old_status": old_status,
            "new_status": new_status,
            "is_flaky": is_flaky,
            "failure_type": failure_type,
            "reason": cand_scored.get("reason", ""),
            "candidate_answer": qa_out.answer,
            "candidate_abstained": qa_out.abstained,
            "candidate_confidence": qa_out.confidence,
        }

        if is_flaky and old_correct:
            row["transition"] = "FLAKY_EXCLUDED"
            flaky_excluded.append(row)
        elif old_correct and not new_correct:
            row["transition"] = "REGRESSION"
            regressions.append(row)
        elif not old_correct and new_correct:
            row["transition"] = "NEWLY_FIXED"
            newly_fixed.append(row)
        elif old_correct and new_correct:
            row["transition"] = "UNCHANGED_PASS"
            unchanged_pass.append(row)
        else:
            row["transition"] = "UNCHANGED_FAIL"
            unchanged_fail.append(row)

        comparison_table.append(row)

    stable_baseline_pass_evaluated = len(regressions) + len(unchanged_pass)
    regression_count = len(regressions)
    regression_rate = (
        round(regression_count / stable_baseline_pass_evaluated, 4)
        if stable_baseline_pass_evaluated > 0
        else 0.0
    )

    return {
        "model_id": model_id,
        "baseline_prompt_version": baseline_prompt_version,
        "candidate_prompt_version": candidate_prompt_version,
        "total_evaluated": len(test_cases),
        "stable_baseline_pass_evaluated": stable_baseline_pass_evaluated,
        "flaky_excluded_count": len(flaky_excluded),
        "regression_count": regression_count,
        "regression_rate": regression_rate,
        "regression_fraction": f"{regression_count}/{stable_baseline_pass_evaluated}",
        "regression_detected": regression_count > 0,
        "passed_gate": regression_count == 0,
        "newly_fixed_count": len(newly_fixed),
        "unchanged_pass_count": len(unchanged_pass),
        "unchanged_fail_count": len(unchanged_fail),
        "regressions": regressions,
        "newly_fixed": newly_fixed,
        "flaky_excluded": flaky_excluded,
        "comparison_table": comparison_table,
    }


def compare_grounded_v2_vs_bad(
    v2_report: dict[str, Any],
    bad_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate prompt `grounded_v2` directly against planted bad prompt `grounded_v2_bad`."""
    v2_rows = {str(r["case_id"]): r for r in v2_report.get("comparison_table", [])}
    bad_rows = {str(r["case_id"]): r for r in bad_report.get("comparison_table", [])}
    shared_ids = sorted(set(v2_rows.keys()) & set(bad_rows.keys()))

    v2_pass_bad_fail_ids: list[str] = []
    head_to_head_rows: list[dict[str, Any]] = []

    for cid in shared_ids:
        r_v2 = v2_rows[cid]
        r_bad = bad_rows[cid]
        if r_v2.get("new_status") == "PASS" and r_bad.get("new_status") == "FAIL":
            v2_pass_bad_fail_ids.append(cid)
        head_to_head_rows.append(
            {
                "case_id": cid,
                "category": r_v2.get("category", "base"),
                "is_flaky": bool(r_v2.get("is_flaky", False)),
                "baseline_status": r_v2.get("old_status", "PASS"),
                "grounded_v2_status": r_v2.get("new_status", ""),
                "grounded_v2_transition": r_v2.get("transition", ""),
                "grounded_v2_bad_status": r_bad.get("new_status", ""),
                "grounded_v2_bad_transition": r_bad.get("transition", ""),
                "grounded_v2_bad_failure_type": r_bad.get("failure_type", "NONE"),
            }
        )

    v2_reg_count = int(v2_report.get("regression_count", 0))
    bad_reg_count = int(bad_report.get("regression_count", 0))
    v2_reg_rate = float(v2_report.get("regression_rate", 0.0))
    bad_reg_rate = float(bad_report.get("regression_rate", 0.0))
    v2_passed_gate = bool(v2_report.get("passed_gate", v2_reg_count == 0))
    bad_passed_gate = bool(bad_report.get("passed_gate", bad_reg_count == 0))

    return {
        "shared_cases_evaluated": len(shared_ids),
        "grounded_v2": {
            "prompt_version": v2_report.get("candidate_prompt_version", "grounded_v2"),
            "regression_count": v2_reg_count,
            "regression_rate": v2_reg_rate,
            "regression_fraction": v2_report.get("regression_fraction", "0/0"),
            "newly_fixed_count": int(v2_report.get("newly_fixed_count", 0)),
            "passed_gate": v2_passed_gate,
        },
        "grounded_v2_bad": {
            "prompt_version": bad_report.get("candidate_prompt_version", "grounded_v2_bad"),
            "regression_count": bad_reg_count,
            "regression_rate": bad_reg_rate,
            "regression_fraction": bad_report.get("regression_fraction", "0/0"),
            "newly_fixed_count": int(bad_report.get("newly_fixed_count", 0)),
            "passed_gate": bad_passed_gate,
        },
        "negative_control_caught": not bad_passed_gate,
        "regression_count_delta_bad_minus_v2": bad_reg_count - v2_reg_count,
        "regression_rate_delta_bad_minus_v2": round(bad_reg_rate - v2_reg_rate, 4),
        "v2_pass_but_bad_failed_count": len(v2_pass_bad_fail_ids),
        "v2_pass_but_bad_failed_ids": v2_pass_bad_fail_ids,
        "head_to_head_table": head_to_head_rows,
    }


def execute_regression_and_noise_floor_pipeline(
    regression_cases_path: Path | str = DEFAULT_REGRESSION_CASES_PATH,
    results_dir: Path | str = Path("results"),
    num_noise_runs: int = 3,
    bad_subset_size: Optional[int] = 20,
    model_id: str = "groq/openai/gpt-oss-20b",
    qa_runner_fn: Optional[Callable[..., Any]] = None,
    scorer_fn: Optional[Callable[[dict[str, Any], QAOutput], dict[str, Any]]] = None,
    run_noise_floor: bool = True,
    run_comparisons: bool = True,
) -> dict[str, Any]:
    """Execute the Phase 8 noise-floor experiment and `grounded_v2` vs `grounded_v2_bad` regression suite."""
    reg_path = Path(regression_cases_path)
    if not reg_path.exists():
        raise FileNotFoundError(
            f"Regression cases file not found at '{reg_path}'. Run --build-baseline first."
        )
    regression_cases = load_regression_cases(reg_path)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_floor_path = out_dir / "regression_noise_floor.json"
    if run_noise_floor:
        noise_report = run_noise_floor_experiment(
            regression_cases=regression_cases,
            num_runs=num_noise_runs,
            model_id=model_id,
            prompt_version="grounded_v1",
            qa_runner_fn=qa_runner_fn,
            scorer_fn=scorer_fn,
        )
        noise_floor_path.write_text(
            json.dumps(noise_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    elif noise_floor_path.exists():
        noise_report = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    else:
        noise_report = {
            "flaky_ids": [],
            "stable_ids": [_extract_case_id(c) for c in regression_cases],
            "flaky_count": 0,
            "stable_count": len(regression_cases),
            "total_frozen_cases": len(regression_cases),
            "flaky_rate": 0.0,
        }

    if not run_comparisons:
        return {"noise_floor": noise_report}

    flaky_ids = set(noise_report.get("flaky_ids", []))
    baseline_scored_by_id = {
        _extract_case_id(c): {
            "case_id": _extract_case_id(c),
            "is_correct": bool(c.get("baseline_passed", True)),
        }
        for c in regression_cases
    }

    v2_report = run_prompt_regression_comparison(
        test_cases=regression_cases,
        baseline_scored_by_id=baseline_scored_by_id,
        flaky_ids=flaky_ids,
        candidate_prompt_version="grounded_v2",
        baseline_prompt_version="grounded_v1",
        model_id=model_id,
        qa_runner_fn=qa_runner_fn,
        scorer_fn=scorer_fn,
    )
    (out_dir / "regression_grounded_v2.json").write_text(
        json.dumps(v2_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bad_cases = (
        regression_cases[:bad_subset_size]
        if (bad_subset_size is not None and bad_subset_size > 0)
        else regression_cases
    )
    bad_report = run_prompt_regression_comparison(
        test_cases=bad_cases,
        baseline_scored_by_id=baseline_scored_by_id,
        flaky_ids=flaky_ids,
        candidate_prompt_version="grounded_v2_bad",
        baseline_prompt_version="grounded_v1",
        model_id=model_id,
        qa_runner_fn=qa_runner_fn,
        scorer_fn=scorer_fn,
    )
    (out_dir / "regression_planted_bad.json").write_text(
        json.dumps(bad_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    head_to_head = compare_grounded_v2_vs_bad(v2_report, bad_report)
    combined_payload = {
        "model_id": model_id,
        "noise_floor": noise_report,
        "grounded_v2": v2_report,
        "grounded_v2_bad": bad_report,
        "v2_vs_bad_comparison": head_to_head,
    }
    (out_dir / "regression_report.json").write_text(
        json.dumps(combined_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return combined_payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 8 regression baseline construction, noise floor, and prompt diffing."""
    parser = argparse.ArgumentParser(
        description="VerityBench Phase 8 Prompt Regression & Noise-Floor Runner"
    )
    parser.add_argument(
        "--build-baseline",
        action="store_true",
        help="Build data/regression_cases.jsonl from canonical verified base cases.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data/regression_cases.jsonl against canonical base cases and documents.",
    )
    parser.add_argument(
        "--run-noise-floor",
        action="store_true",
        help="Run 3-repetition uncached noise-floor experiment on data/regression_cases.jsonl.",
    )
    parser.add_argument(
        "--run-regression",
        action="store_true",
        help="Run prompt regression comparison for grounded_v2 and grounded_v2_bad.",
    )
    parser.add_argument(
        "--regression-cases",
        type=Path,
        default=DEFAULT_REGRESSION_CASES_PATH,
        help="Path to regression_cases.jsonl",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write regression JSON outputs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of uncached repetitions for noise-floor estimation (default: 3)",
    )
    args = parser.parse_args(argv)

    if args.build_baseline:
        cases = build_canonical_regression_cases(output_path=args.regression_cases)
        print(f"Built and validated {len(cases)} regression cases at {args.regression_cases}")
        return 0

    if args.validate_only:
        docs = load_documents()
        bases = {b.id: b for b in load_jsonl_base_cases()}
        reg_cases = load_regression_cases(args.regression_cases)
        summary = validate_regression_cases(
            records=reg_cases,
            base_cases_by_id=bases,
            documents=docs,
            require_verified=True,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.run_noise_floor or args.run_regression:
        res = execute_regression_and_noise_floor_pipeline(
            regression_cases_path=args.regression_cases,
            results_dir=args.results_dir,
            num_noise_runs=args.num_runs,
            run_noise_floor=args.run_noise_floor or not (args.results_dir / "regression_noise_floor.json").exists(),
            run_comparisons=args.run_regression,
        )
        print(
            json.dumps(
                {
                    "noise_floor_flaky_fraction": res["noise_floor"].get("flaky_fraction"),
                    "v2_vs_bad_comparison": res.get("v2_vs_bad_comparison"),
                },
                indent=2,
            )
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
