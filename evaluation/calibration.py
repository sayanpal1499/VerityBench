"""Phase 6 Confidence Calibration (ECE, Brier, Subsets, Raw vs Post-Gate, and Plots) (evaluation/calibration.py)."""

from pathlib import Path
import random
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _assign_bin_index(conf: float, n_bins: int = 10) -> int:
    """Assign a confidence value in [0.0, 1.0] to one of n_bins equal-width bins."""
    clamped = max(0.0, min(1.0, float(conf)))
    if clamped >= 1.0:
        return n_bins - 1
    return int(clamped * n_bins)


def _compute_ece_value_only(
    confidences: list[float], correctness: list[int], n_bins: int = 10
) -> float:
    """Compute scalar ECE across non-empty bins for fast bootstrap resampling."""
    total = len(confidences)
    if total == 0:
        return 0.0
    bin_confs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_corrs: list[list[int]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        b = _assign_bin_index(c, n_bins)
        bin_confs[b].append(c)
        bin_corrs[b].append(int(bool(y)))

    ece = 0.0
    for b in range(n_bins):
        cnt = len(bin_confs[b])
        if cnt == 0:
            continue
        mean_c = sum(bin_confs[b]) / cnt
        acc = sum(bin_corrs[b]) / cnt
        ece += (cnt / total) * abs(acc - mean_c)
    return ece


def compute_ece_and_brier(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute 10-bin ECE, Brier score, bootstrap 95% CI, bin counts, and >=0.9 vs <0.9 table."""
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness lists must have the same length.")

    total = len(confidences)
    if total == 0:
        return {
            "total_samples": 0,
            "ece": 0.0,
            "ece_bootstrap_ci_95": [0.0, 0.0],
            "brier_score": 0.0,
            "bins": [],
            "empty_bins": [f"[{i/n_bins:.1f}, {(i+1)/n_bins:.1f})" for i in range(n_bins)],
            "sparse_bins": [],
            "high_vs_low_confidence_table": {},
        }

    bin_confs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_corrs: list[list[int]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        b = _assign_bin_index(c, n_bins)
        bin_confs[b].append(float(c))
        bin_corrs[b].append(1 if y else 0)

    bins_out: list[dict[str, Any]] = []
    empty_bins: list[str] = []
    sparse_bins: list[str] = []
    ece = 0.0

    for b in range(n_bins):
        low_edge = b / n_bins
        high_edge = (b + 1) / n_bins
        bracket = "]" if b == n_bins - 1 else ")"
        label = f"[{low_edge:.1f}, {high_edge:.1f}{bracket}"
        cnt = len(bin_confs[b])
        if cnt == 0:
            empty_bins.append(label)
            bins_out.append({
                "bin_index": b,
                "bin_range": label,
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "abs_gap": None,
                "weighted_gap": 0.0,
            })
            continue

        if cnt < 5:
            sparse_bins.append(label)

        mean_c = sum(bin_confs[b]) / cnt
        acc = sum(bin_corrs[b]) / cnt
        gap = abs(acc - mean_c)
        w_gap = (cnt / total) * gap
        ece += w_gap

        bins_out.append({
            "bin_index": b,
            "bin_range": label,
            "count": cnt,
            "mean_confidence": round(mean_c, 4),
            "accuracy": round(acc, 4),
            "abs_gap": round(gap, 4),
            "weighted_gap": round(w_gap, 4),
        })

    brier = sum((float(c) - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correctness)) / total

    # Bootstrap 95% CI for ECE
    rng = random.Random(seed)
    boot_eces: list[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(total) for _ in range(total)]
        c_samp = [confidences[i] for i in idxs]
        y_samp = [correctness[i] for i in idxs]
        boot_eces.append(_compute_ece_value_only(c_samp, y_samp, n_bins=n_bins))
    boot_eces.sort()
    ci_low = boot_eces[int(0.025 * n_bootstrap)]
    ci_high = boot_eces[int(0.975 * n_bootstrap)]

    # Diagnostic split: confidence >= 0.90 vs < 0.90
    high_pairs = [(c, y) for c, y in zip(confidences, correctness) if c >= 0.90]
    low_pairs = [(c, y) for c, y in zip(confidences, correctness) if c < 0.90]
    high_corr = sum(1 for _, y in high_pairs if y)
    low_corr = sum(1 for _, y in low_pairs if y)

    return {
        "total_samples": total,
        "ece": round(ece, 4),
        "ece_bootstrap_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "brier_score": round(brier, 4),
        "bins": bins_out,
        "empty_bins": empty_bins,
        "sparse_bins": sparse_bins,
        "high_vs_low_confidence_table": {
            "confidence_ge_0_90": {
                "numerator": high_corr,
                "denominator": len(high_pairs),
                "accuracy": round(high_corr / len(high_pairs), 4) if high_pairs else None,
                "mean_confidence": (
                    round(sum(c for c, _ in high_pairs) / len(high_pairs), 4)
                    if high_pairs
                    else None
                ),
            },
            "confidence_lt_0_90": {
                "numerator": low_corr,
                "denominator": len(low_pairs),
                "accuracy": round(low_corr / len(low_pairs), 4) if low_pairs else None,
                "mean_confidence": (
                    round(sum(c for c, _ in low_pairs) / len(low_pairs), 4)
                    if low_pairs
                    else None
                ),
            },
        },
    }


def compute_full_calibration_report(
    scored_records: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Compute separated raw-model vs post-gate calibration, plus answered-only and abstained-only subsets."""
    eval_records = [r for r in scored_records if r.get("category") != "empty_malformed"]
    excluded_pre_api = len(scored_records) - len(eval_records)

    post_confs = [float(r["confidence"]) for r in eval_records]
    post_corrs = [1 if r["is_correct"] else 0 for r in eval_records]

    raw_confs = [float(r.get("raw_confidence", r["confidence"])) for r in eval_records]
    raw_corrs = [
        1 if r.get("raw_is_correct", r["is_correct"]) else 0 for r in eval_records
    ]

    gate_overridden = [r for r in eval_records if r.get("gate_triggered", False)]
    non_gate_records = [r for r in eval_records if not r.get("gate_triggered", False)]

    answered_records = [r for r in eval_records if not r.get("abstained", False)]
    abstained_records = [r for r in eval_records if r.get("abstained", False)]

    return {
        "evaluated_records_n": len(eval_records),
        "excluded_pre_api_validation_records_n": excluded_pre_api,
        "gate_fallback_disclosure": (
            f"{len(gate_overridden)} record(s) were overridden by the Evidence-Quote Gate and assigned "
            f"the fixed engineering fallback confidence of 0.50 in [0.5, 0.6). This 0.50 value is a "
            f"deterministic rule, NOT a model-measured probability; therefore intrinsic raw-model "
            f"calibration and post-gate system calibration are reported separately."
        ),
        "gate_overridden_count": len(gate_overridden),
        "intrinsic_raw_model_calibration": compute_ece_and_brier(
            raw_confs, raw_corrs, seed=seed
        ),
        "post_gate_system_calibration": compute_ece_and_brier(
            post_confs, post_corrs, seed=seed
        ),
        "non_gate_overridden_calibration": compute_ece_and_brier(
            [float(r["confidence"]) for r in non_gate_records],
            [1 if r["is_correct"] else 0 for r in non_gate_records],
            seed=seed,
        ),
        "answered_only_calibration": compute_ece_and_brier(
            [float(r["confidence"]) for r in answered_records],
            [1 if r["is_correct"] else 0 for r in answered_records],
            seed=seed,
        ),
        "abstained_only_calibration": compute_ece_and_brier(
            [float(r["confidence"]) for r in abstained_records],
            [1 if r["is_correct"] else 0 for r in abstained_records],
            seed=seed,
        ),
    }


def plot_calibration_curve(
    calibration_report: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot reliability diagram (top) and confidence bin count histogram (bottom)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    post_cal = calibration_report.get("post_gate_system_calibration", calibration_report)
    raw_cal = calibration_report.get("intrinsic_raw_model_calibration")

    bins = post_cal.get("bins", [])
    if not bins:
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # Diagonal perfect calibration line
    ax_top.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect Calibration (y = x)")

    post_xs = [b["mean_confidence"] for b in bins if b["count"] > 0 and b["mean_confidence"] is not None]
    post_ys = [b["accuracy"] for b in bins if b["count"] > 0 and b["accuracy"] is not None]
    ax_top.plot(
        post_xs,
        post_ys,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label=f"Post-Gate System (ECE={post_cal['ece']:.3f}, Brier={post_cal['brier_score']:.3f})",
    )

    if raw_cal and raw_cal.get("bins"):
        raw_bins = raw_cal["bins"]
        raw_xs = [b["mean_confidence"] for b in raw_bins if b["count"] > 0 and b["mean_confidence"] is not None]
        raw_ys = [b["accuracy"] for b in raw_bins if b["count"] > 0 and b["accuracy"] is not None]
        ax_top.plot(
            raw_xs,
            raw_ys,
            marker="s",
            linestyle=":",
            linewidth=1.8,
            color="#ff7f0e",
            label=f"Intrinsic Raw Model (ECE={raw_cal['ece']:.3f}, Brier={raw_cal['brier_score']:.3f})",
        )

    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.set_xlabel("Mean Predicted Confidence")
    ax_top.set_ylabel("Empirical Accuracy")
    ax_top.set_title("Reliability Diagram & Confidence Histogram (10 Equal-Width Bins)")
    ax_top.legend(loc="lower right", fontsize=8.5)
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # Bottom histogram of bin counts
    bin_centers = [(b["bin_index"] + 0.5) / len(bins) for b in bins]
    counts = [b["count"] for b in bins]
    ax_bot.bar(bin_centers, counts, width=0.08, color="#1f77b4", alpha=0.8, edgecolor="black")
    for cx, cnt in zip(bin_centers, counts):
        if cnt > 0:
            ax_bot.text(cx, cnt + 0.5, str(cnt), ha="center", va="bottom", fontsize=8)
    ax_bot.set_xlim(0.0, 1.0)
    ax_bot.set_xlabel("Confidence Bin")
    ax_bot.set_ylabel("Record Count")
    ax_bot.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
