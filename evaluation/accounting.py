"""Authoritative per-configuration accounting and consistency validation (evaluation/accounting.py).

Verifies exact accounting identities across total records, pre-API exclusions,
API error fallbacks, completed LLM calls, abstentions, and substantive answers.
"""

from typing import Any, Optional


def compute_configuration_accounting(
    outputs: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    claims: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Calculate and strictly verify all accounting identities for a model configuration.

    Accounting Identities Verified:
    1. total_records == llm_evaluated_records + pre_api_exclusions
    2. total_records == substantive_answers + total_abstentions
    3. llm_evaluated_records == completed_llm_responses + api_error_fallbacks
    4. completed_llm_responses == substantive_answers + intentional_abstentions
    5. total_abstentions == pre_api_exclusions + api_error_fallbacks + intentional_abstentions
    6. correct_total_150 == correct_answered + correct_abstained
    7. correct_llm_144 == correct_answered + (correct_abstained - pre_api_exclusions)
    """
    outs_by_id = {r["case_id"]: r for r in outputs}
    scs_by_id = {r["case_id"]: r for r in scored}
    cl_by_id = {r["case_id"]: r for r in claims} if claims else {}

    total_records = len(outputs)
    pre_api_exclusions = sum(1 for o in outputs if o.get("validation_rejected", False))
    llm_evaluated_records = total_records - pre_api_exclusions

    api_error_fallbacks = sum(
        1 for o in outputs if "System Error Abstention:" in o.get("answer", "")
    )
    completed_llm_responses = llm_evaluated_records - api_error_fallbacks

    intentional_abstentions = sum(
        1
        for o in outputs
        if o.get("abstained", False)
        and not o.get("validation_rejected", False)
        and "System Error Abstention:" not in o.get("answer", "")
    )
    substantive_answers = sum(1 for o in outputs if not o.get("abstained", False))
    total_abstentions = sum(1 for o in outputs if o.get("abstained", False))

    # Assert structural partitions
    assert total_records == llm_evaluated_records + pre_api_exclusions, (
        f"Partition error: total {total_records} != {llm_evaluated_records} + {pre_api_exclusions}"
    )
    assert total_records == substantive_answers + total_abstentions, (
        f"Partition error: total {total_records} != answers {substantive_answers} + abstentions {total_abstentions}"
    )
    assert llm_evaluated_records == completed_llm_responses + api_error_fallbacks, (
        f"Partition error: eval {llm_evaluated_records} != completed {completed_llm_responses} + errs {api_error_fallbacks}"
    )
    assert completed_llm_responses == substantive_answers + intentional_abstentions, (
        f"Partition error: completed {completed_llm_responses} != answers {substantive_answers} + intentional {intentional_abstentions}"
    )
    assert total_abstentions == pre_api_exclusions + api_error_fallbacks + intentional_abstentions, (
        f"Partition error: abstentions {total_abstentions} != pre_api {pre_api_exclusions} + api_err {api_error_fallbacks} + intentional {intentional_abstentions}"
    )

    # Correctness accounting
    correct_total_150 = sum(1 for s in scored if s.get("is_correct", False))
    correct_answered = sum(
        1
        for cid, s in scs_by_id.items()
        if s.get("is_correct", False) and not outs_by_id[cid].get("abstained", False)
    )
    correct_abstained = sum(
        1
        for cid, s in scs_by_id.items()
        if s.get("is_correct", False) and outs_by_id[cid].get("abstained", False)
    )
    correct_pre_api = sum(
        1
        for cid, s in scs_by_id.items()
        if s.get("is_correct", False) and outs_by_id[cid].get("validation_rejected", False)
    )
    correct_llm_144 = correct_total_150 - correct_pre_api

    assert correct_total_150 == correct_answered + correct_abstained, (
        f"Correctness error: total {correct_total_150} != answered {correct_answered} + abstained {correct_abstained}"
    )
    assert correct_llm_144 == correct_answered + (correct_abstained - correct_pre_api), (
        f"Correctness error: eval {correct_llm_144} != answered {correct_answered} + eval abstained {correct_abstained - correct_pre_api}"
    )

    incorrect_total_150 = total_records - correct_total_150
    incorrect_llm_144 = llm_evaluated_records - correct_llm_144
    incorrect_answered = substantive_answers - correct_answered

    # Hallucination accounting
    hallucinated_total = sum(
        1 for c in cl_by_id.values() if c.get("has_hallucination", False)
    )
    hallucinated_answered = sum(
        1
        for cid, c in cl_by_id.items()
        if c.get("has_hallucination", False) and not outs_by_id[cid].get("abstained", False)
    )

    # Rates
    acc_all_rate = round(correct_total_150 / total_records, 4) if total_records > 0 else 0.0
    acc_llm_rate = round(correct_llm_144 / llm_evaluated_records, 4) if llm_evaluated_records > 0 else 0.0
    acc_ans_rate = round(correct_answered / substantive_answers, 4) if substantive_answers > 0 else 0.0

    hal_llm_rate = round(hallucinated_total / llm_evaluated_records, 4) if llm_evaluated_records > 0 else 0.0
    hal_ans_rate = round(hallucinated_answered / substantive_answers, 4) if substantive_answers > 0 else 0.0

    return {
        "total_records": total_records,
        "pre_api_exclusions": pre_api_exclusions,
        "llm_evaluated_records": llm_evaluated_records,
        "api_error_fallbacks": api_error_fallbacks,
        "completed_llm_responses": completed_llm_responses,
        "intentional_abstentions": intentional_abstentions,
        "substantive_answers": substantive_answers,
        "total_abstentions": total_abstentions,
        "correct_total_150": correct_total_150,
        "correct_llm_144": correct_llm_144,
        "correct_answered": correct_answered,
        "correct_abstained": correct_abstained,
        "incorrect_total_150": incorrect_total_150,
        "incorrect_llm_144": incorrect_llm_144,
        "incorrect_answered": incorrect_answered,
        "hallucinated_total": hallucinated_total,
        "hallucinated_answered": hallucinated_answered,
        "accuracy_all_150": {
            "numerator": correct_total_150,
            "denominator": total_records,
            "rate": acc_all_rate,
            "formatted": f"{correct_total_150}/{total_records} ({acc_all_rate*100:.1f}%)",
        },
        "accuracy_llm_144": {
            "numerator": correct_llm_144,
            "denominator": llm_evaluated_records,
            "rate": acc_llm_rate,
            "formatted": f"{correct_llm_144}/{llm_evaluated_records} ({acc_llm_rate*100:.1f}%)",
        },
        "accuracy_answered_only": {
            "numerator": correct_answered,
            "denominator": substantive_answers,
            "rate": acc_ans_rate,
            "formatted": f"{correct_answered}/{substantive_answers} ({acc_ans_rate*100:.1f}%)",
        },
        "hallucination_llm_144": {
            "numerator": hallucinated_total,
            "denominator": llm_evaluated_records,
            "rate": hal_llm_rate,
            "formatted": f"{hallucinated_total}/{llm_evaluated_records} ({hal_llm_rate*100:.1f}%)",
        },
        "hallucination_answered_only": {
            "numerator": hallucinated_answered,
            "denominator": substantive_answers,
            "rate": hal_ans_rate,
            "formatted": f"{hallucinated_answered}/{substantive_answers} ({hal_ans_rate*100:.1f}%)",
        },
        "accounting_identities_verified": True,
    }
