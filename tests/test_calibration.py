"""Unit tests for Phase 6 calibration and metrics (evaluation/calibration.py, evaluation/metrics.py)."""

from pathlib import Path

from evaluation.calibration import (
    compute_ece_and_brier,
    compute_full_calibration_report,
    plot_calibration_curve,
)
from evaluation.metrics import (
    cluster_bootstrap_ci_95,
    compute_all_metrics,
    wilson_ci_95,
)


def test_hand_computed_ece_and_brier_toy_example() -> None:
    """Verify exact ECE, Brier score, bin counts, and empty bins on a hand-computed 4-sample toy example."""
    # Bin [0.5, 0.6): conf=[0.5, 0.5], corr=[1, 0] -> acc=0.5, mean_conf=0.5, gap=0.0
    # Bin [0.9, 1.0]: conf=[0.9, 0.9], corr=[1, 0] -> acc=0.5, mean_conf=0.9, gap=0.4
    # ECE = (2/4)*0.0 + (2/4)*0.4 = 0.20
    # Brier = (0.25 + 0.25 + 0.01 + 0.81) / 4 = 1.32 / 4 = 0.33
    confs = [0.50, 0.50, 0.90, 0.90]
    corrs = [1, 0, 1, 0]

    res = compute_ece_and_brier(confs, corrs, n_bins=10, n_bootstrap=200, seed=42)
    assert res["total_samples"] == 4
    assert res["ece"] == 0.20
    assert res["brier_score"] == 0.33
    assert len(res["empty_bins"]) == 8
    assert res["bins"][5]["count"] == 2
    assert res["bins"][9]["count"] == 2

    high_low = res["high_vs_low_confidence_table"]
    assert high_low["confidence_ge_0_90"]["numerator"] == 1
    assert high_low["confidence_ge_0_90"]["denominator"] == 2
    assert high_low["confidence_lt_0_90"]["numerator"] == 1
    assert high_low["confidence_lt_0_90"]["denominator"] == 2


def test_full_calibration_separates_raw_vs_post_gate_and_subsets(tmp_path: Path) -> None:
    """compute_full_calibration_report must separate intrinsic raw vs post-gate and skip empty_malformed."""
    records = [
        {
            "case_id": "BASE-001",
            "base_id": "BASE-001",
            "category": "base",
            "abstained": False,
            "should_abstain": False,
            "confidence": 0.95,
            "is_correct": True,
            "raw_confidence": 0.95,
            "raw_is_correct": True,
            "gate_triggered": False,
        },
        {
            "case_id": "BASE-006",
            "base_id": "BASE-006",
            "category": "base",
            "abstained": True,
            "should_abstain": True,
            "confidence": 0.50,  # Forced by gate
            "is_correct": True,
            "raw_confidence": 0.90,  # Model originally answered with 0.90 and fake quote
            "raw_is_correct": False,
            "gate_triggered": True,
        },
        {
            "case_id": "VAR-095",
            "base_id": "BASE-001",
            "category": "empty_malformed",
            "abstained": True,
            "should_abstain": True,
            "confidence": 0.0,
            "is_correct": True,
            "gate_triggered": False,
        },
    ]

    report = compute_full_calibration_report(records, seed=42)
    assert report["evaluated_records_n"] == 2
    assert report["excluded_pre_api_validation_records_n"] == 1
    assert report["gate_overridden_count"] == 1
    assert report["answered_only_calibration"]["total_samples"] == 1
    assert report["abstained_only_calibration"]["total_samples"] == 1
    assert report["non_gate_overridden_calibration"]["total_samples"] == 1
    # Intrinsic raw model had 1/2 correct; post-gate had 2/2 correct
    assert report["intrinsic_raw_model_calibration"]["brier_score"] > report["post_gate_system_calibration"]["brier_score"]

    fig_path = tmp_path / "calibration_curve.png"
    plot_calibration_curve(report, fig_path)
    assert fig_path.exists()


def test_wilson_and_cluster_bootstrap_intervals() -> None:
    """Verify Wilson 95% CI and base_id cluster bootstrap CI."""
    low, high = wilson_ci_95(38, 50)
    assert 0.62 <= low <= 0.65
    assert 0.85 <= high <= 0.87

    records = [
        {"case_id": "BASE-001", "base_id": "BASE-001", "category": "base", "abstained": False, "should_abstain": False, "is_correct": True},
        {"case_id": "VAR-001", "base_id": "BASE-001", "category": "paraphrase_hard", "abstained": False, "should_abstain": False, "is_correct": True},
        {"case_id": "BASE-002", "base_id": "BASE-002", "category": "base", "abstained": False, "should_abstain": False, "is_correct": False},
        {"case_id": "VAR-002", "base_id": "BASE-002", "category": "paraphrase_hard", "abstained": False, "should_abstain": False, "is_correct": False},
    ]
    c_boot = cluster_bootstrap_ci_95(records, n_resamples=200, seed=42)
    assert c_boot["resampling_unit"] == "base_id"
    assert c_boot["n_clusters"] == 2
    assert c_boot["n_records"] == 4

    all_m = compute_all_metrics(records)
    assert all_m["overall_all_150"]["numerator"] == 2
    assert all_m["overall_all_150"]["denominator"] == 4
