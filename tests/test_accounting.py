"""Unit tests for authoritative per-configuration accounting (evaluation/accounting.py)."""

import json
from pathlib import Path
import pytest

from evaluation.accounting import compute_configuration_accounting


def test_accounting_on_grounded_v1_saved_records() -> None:
    """Verify all accounting identities hold strictly on saved grounded_v1 artifacts."""
    res_dir = Path("results")
    outs = [
        json.loads(line)
        for line in (res_dir / "outputs_grounded_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scs = [
        json.loads(line)
        for line in (res_dir / "scored_grounded_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cls = (
        [
            json.loads(line)
            for line in (res_dir / "claims_grounded_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (res_dir / "claims_grounded_v1.jsonl").exists()
        else None
    )

    acct = compute_configuration_accounting(outs, scs, cls)

    assert acct["accounting_identities_verified"] is True
    assert acct["total_records"] == 150
    assert acct["pre_api_exclusions"] == 6
    assert acct["llm_evaluated_records"] == 144
    assert acct["api_error_fallbacks"] == 32
    assert acct["completed_llm_responses"] == 112
    assert acct["intentional_abstentions"] == 23
    assert acct["substantive_answers"] == 89
    assert acct["total_abstentions"] == 61

    assert acct["correct_total_150"] == 118
    assert acct["correct_llm_144"] == 112
    assert acct["correct_answered"] == 89
    assert acct["correct_abstained"] == 29

    assert acct["hallucinated_total"] == 5
    assert acct["hallucinated_answered"] == 5

    assert acct["accuracy_llm_144"]["formatted"] == "112/144 (77.8%)"
    assert acct["accuracy_answered_only"]["formatted"] == "89/89 (100.0%)"
    assert acct["hallucination_llm_144"]["formatted"] == "5/144 (3.5%)"
    assert acct["hallucination_answered_only"]["formatted"] == "5/89 (5.6%)"


def test_accounting_on_naive_v0_saved_records() -> None:
    """Verify all accounting identities hold strictly on saved naive_v0 artifacts."""
    res_dir = Path("results")
    outs = [
        json.loads(line)
        for line in (res_dir / "outputs_naive_v0.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scs = [
        json.loads(line)
        for line in (res_dir / "scored_naive_v0.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cls = (
        [
            json.loads(line)
            for line in (res_dir / "claims_naive_v0.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (res_dir / "claims_naive_v0.jsonl").exists()
        else None
    )

    acct = compute_configuration_accounting(outs, scs, cls)

    assert acct["accounting_identities_verified"] is True
    assert acct["total_records"] == 150
    assert acct["pre_api_exclusions"] == 6
    assert acct["llm_evaluated_records"] == 144
    assert acct["api_error_fallbacks"] == 129
    assert acct["completed_llm_responses"] == 15
    assert acct["intentional_abstentions"] == 0
    assert acct["substantive_answers"] == 15
    assert acct["total_abstentions"] == 135

    assert acct["correct_total_150"] == 43
    assert acct["correct_llm_144"] == 37
    assert acct["correct_answered"] == 14
    assert acct["correct_abstained"] == 29

    assert acct["hallucinated_total"] == 1
    assert acct["hallucinated_answered"] == 1

    assert acct["accuracy_llm_144"]["formatted"] == "37/144 (25.7%)"
    assert acct["accuracy_answered_only"]["formatted"] == "14/15 (93.3%)"
    assert acct["hallucination_llm_144"]["formatted"] == "1/144 (0.7%)"
    assert acct["hallucination_answered_only"]["formatted"] == "1/15 (6.7%)"


def test_accounting_synthetic_partition_violations() -> None:
    """Verify that compute_configuration_accounting raises AssertionError on broken partitions."""
    # Broken total: output says 2 records, but 1 has conflicting flags
    broken_outs = [
        {"case_id": "C-1", "validation_rejected": False, "abstained": False, "answer": "Ans"},
        {"case_id": "C-2", "validation_rejected": True, "abstained": False, "answer": "Ans"},  # rejected but not abstained
    ]
    broken_scs = [
        {"case_id": "C-1", "is_correct": True},
        {"case_id": "C-2", "is_correct": False},
    ]
    with pytest.raises(AssertionError):
        compute_configuration_accounting(broken_outs, broken_scs)
