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

import json
from pathlib import Path
from typing import Any, Callable, Optional

from evaluation.clustering import assign_initial_failure_cluster
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

    for case in regression_cases:
        cid = _extract_case_id(case)
        run_outcomes: list[dict[str, Any]] = []
        pass_count = 0

        for run_idx in range(1, num_runs + 1):
            raw_res = runner(
                document=case["document"],
                question=case["question"],
                prompt_version=prompt_version,
                model=model_id,
                bypass_cache=True,
                seed=42,
            )
            qa_out = _coerce_qa_output(raw_res)
            scored = scorer(case, qa_out)
            is_correct = bool(scored.get("is_correct", False))
            if is_correct:
                pass_count += 1

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
        if is_flaky:
            flaky_ids.append(cid)
        else:
            stable_ids.append(cid)

        per_case_results.append(
            {
                "case_id": cid,
                "category": case.get("category", "base"),
                "num_runs": num_runs,
                "pass_count": pass_count,
                "fail_count": num_runs - pass_count,
                "is_flaky": is_flaky,
                "status": "flaky" if is_flaky else "stable_pass",
                "runs": run_outcomes,
            }
        )

    total = len(regression_cases)
    flaky_count = len(flaky_ids)
    stable_count = len(stable_ids)

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
        "newly_fixed_count": len(newly_fixed),
        "unchanged_pass_count": len(unchanged_pass),
        "unchanged_fail_count": len(unchanged_fail),
        "regressions": regressions,
        "newly_fixed": newly_fixed,
        "flaky_excluded": flaky_excluded,
        "comparison_table": comparison_table,
    }
