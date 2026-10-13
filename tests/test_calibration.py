import json
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


def test_calibration_handles_missing_confidence_and_extreme_values() -> None:
    """Verify compute_ece_and_brier safely filters None/invalid confidence and accurately scores extreme 0/1 values."""
    # 2 valid extreme cases:
    # 1. conf=1.0, corr=0 (complete overconfidence): brier = (1 - 0)^2 = 1.0, gap = |0 - 1.0| = 1.0
    # 2. conf=0.0, corr=1 (complete underconfidence): brier = (0 - 1)^2 = 1.0, gap = |1 - 0.0| = 1.0
    # Plus None, nan, and invalid string entries that must be filtered out cleanly
    confs = [1.0, 0.0, None, float("nan"), "invalid_conf"]
    corrs = [0, 1, 1, 0, 1]

    res = compute_ece_and_brier(confs, corrs, n_bins=10, n_bootstrap=100, seed=42)
    assert res["total_samples"] == 2
    assert res["missing_or_invalid_samples"] == 3
    # Both samples have gap=1.0 -> ECE = (1/2)*1.0 + (1/2)*1.0 = 1.0
    assert res["ece"] == 1.0
    # Mean Brier score = (1.0 + 1.0) / 2 = 1.0
    assert res["brier_score"] == 1.0


def test_calibration_disaggregates_api_error_fallbacks() -> None:
    """compute_full_calibration_report must disaggregate API error fallbacks from genuine model predictions."""
    records = [
        # Genuine correct prediction
        {"case_id": "BASE-001", "category": "base", "confidence": 0.99, "is_correct": True, "abstained": False, "should_abstain": False},
        # Genuine correct prediction
        {"case_id": "BASE-002", "category": "base", "confidence": 1.00, "is_correct": True, "abstained": False, "should_abstain": False},
        # API token-exhaustion error fallback: confidence 0.0 assigned by handler, failed on answerable case
        {"case_id": "BASE-005", "category": "base", "confidence": 0.00, "is_correct": False, "abstained": True, "should_abstain": False, "error_fallback": True},
        # Pre-API malformed rejection
        {"case_id": "VAR-095", "category": "empty_malformed", "confidence": 1.00, "is_correct": True, "abstained": True, "should_abstain": True},
    ]

    report = compute_full_calibration_report(records, seed=42)
    assert report["evaluated_records_n"] == 3
    assert report["excluded_pre_api_validation_records_n"] == 1
    assert report["api_fallback_records_n"] == 1
    assert report["valid_model_records_n"] == 2

    # Genuine model predictions only (2 samples, confs 0.99 and 1.0, both correct)
    valid_cal = report["valid_model_predictions_only"]
    assert valid_cal["total_samples"] == 2
    assert valid_cal["brier_score"] < 0.001
    assert "PROVISIONAL / PIPELINE ARTIFACT DISCLOSURE" in report["pipeline_artifact_disclosure"]
    assert report["confidence_provenance"]["model_generated_count"] == 2
    assert report["confidence_provenance"]["api_error_fallback_count"] == 1
    assert report["confidence_provenance"]["pre_api_validation_count"] == 1


def test_confidence_provenance_and_single_bin_isolation_on_saved_records() -> None:
    """Verify confidence provenance and single-bin concentration on saved benchmark data."""
    res_dir = Path("results")
    scored = [
        json.loads(line)
        for line in (res_dir / "scored_grounded_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outputs = {
        r["case_id"]: r
        for r in (
            json.loads(line)
            for line in (res_dir / "outputs_grounded_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    report = compute_full_calibration_report(scored, outputs_by_id=outputs, seed=42)
    assert report["evaluated_records_n"] == 144
    assert report["excluded_pre_api_validation_records_n"] == 6
    assert report["api_fallback_records_n"] == 32
    assert report["valid_model_records_n"] == 112

    prov = report["confidence_provenance"]
    assert prov["model_generated_count"] == 112
    assert prov["api_error_fallback_count"] == 32
    assert prov["pre_api_validation_count"] == 6
    assert prov["gate_fallback_count"] == 0

    # Genuine model predictions: all 112 in single bin [0.9, 1.0]
    valid_cal = report["valid_model_predictions_only"]
    assert valid_cal["total_samples"] == 112
    non_empty_bins = [b for b in valid_cal["bins"] if b["count"] > 0]
    assert len(non_empty_bins) == 1
    assert non_empty_bins[0]["bin_range"] == "[0.9, 1.0]"
    assert non_empty_bins[0]["count"] == 112
    assert non_empty_bins[0]["accuracy"] == 1.0
