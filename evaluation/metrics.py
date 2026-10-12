"""Phase 6 Aggregate & Per-Category Metrics, Wilson CIs, and Base-Case Cluster Bootstrap (evaluation/metrics.py)."""

import math
from pathlib import Path
import random
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.hallucination import compute_hallucination_rates
from evaluation.scoring import compute_abstention_metrics


def wilson_ci_95(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute the 95% Wilson score interval for a binomial proportion (k / n).

    Reported as a descriptive interval per category/slice.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
        / denom
    )
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (round(low, 4), round(high, 4))


def format_rate_with_ci(successes: int, total: int) -> dict[str, Any]:
    """Format a proportion with numerator, denominator, percentage string, and 95% Wilson CI."""
    rate = round(successes / total, 4) if total > 0 else 0.0
    ci_low, ci_high = wilson_ci_95(successes, total)
    pct_str = f"{successes}/{total} ({rate * 100:.1f}%)"
    return {
        "numerator": successes,
        "denominator": total,
        "rate": rate,
        "formatted": pct_str,
        "wilson_ci_95": [ci_low, ci_high],
    }


def cluster_bootstrap_ci_95(
    scored_records: list[dict[str, Any]],
    metric_key: str = "is_correct",
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a 95% cluster-bootstrap confidence interval resampling over unique base_id clusters.

    Because adversarial variants are derived from base cases, resampling by base_id
    preserves intra-cluster correlation and avoids pretending 150 linked records are
    independent draws.
    """
    clusters: dict[str, list[dict[str, Any]]] = {}
    for rec in scored_records:
        base_id = str(rec.get("base_id") or rec.get("case_id"))
        clusters.setdefault(base_id, []).append(rec)

    cluster_ids = sorted(clusters.keys())
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return {
            "resampling_unit": "base_id",
            "n_clusters": 0,
            "n_records": 0,
            "n_resamples": n_resamples,
            "ci_95": [0.0, 0.0],
        }

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_resamples):
        sampled_ids = [cluster_ids[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        num = 0
        den = 0
        for cid in sampled_ids:
            for r in clusters[cid]:
                num += 1 if r.get(metric_key, False) else 0
                den += 1
        boot_means.append(num / den if den > 0 else 0.0)

    boot_means.sort()
    ci_low = round(boot_means[int(0.025 * n_resamples)], 4)
    ci_high = round(boot_means[int(0.975 * n_resamples)], 4)
    return {
        "resampling_unit": "base_id",
        "n_clusters": n_clusters,
        "n_records": len(scored_records),
        "n_resamples": n_resamples,
        "seed": seed,
        "ci_95": [ci_low, ci_high],
        "note": (
            "Primary aggregate uncertainty interval resampled at the base_id cluster level "
            "so linked variants are not treated as independent observations."
        ),
    }


def compute_all_metrics(
    scored_records: list[dict[str, Any]],
    hallucination_records: Optional[list[dict[str, Any]]] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute overall, base-only, variant-only, per-category, long-context position, abstention, and hallucination metrics."""
    total_all = len(scored_records)
    correct_all = sum(1 for r in scored_records if r.get("is_correct", False))

    # Evaluated LLM records (excluding pre-API empty_malformed validation tests)
    llm_records = [r for r in scored_records if r.get("category") != "empty_malformed"]
    correct_llm = sum(1 for r in llm_records if r.get("is_correct", False))

    base_records = [r for r in scored_records if r.get("category") == "base"]
    correct_base = sum(1 for r in base_records if r.get("is_correct", False))

    variant_records = [r for r in scored_records if r.get("category") != "base"]
    correct_variants = sum(1 for r in variant_records if r.get("is_correct", False))

    # Per-category breakdown
    by_category: dict[str, dict[str, Any]] = {}
    categories_seen: list[str] = []
    for r in scored_records:
        cat = r.get("category", "base")
        if cat not in categories_seen:
            categories_seen.append(cat)

    for cat in categories_seen:
        cat_recs = [r for r in scored_records if r.get("category") == cat]
        cat_corr = sum(1 for r in cat_recs if r.get("is_correct", False))
        entry = format_rate_with_ci(cat_corr, len(cat_recs))
        if len(cat_recs) < 15:
            entry["small_sample_note"] = (
                f"Small sample size (n={len(cat_recs)}); Wilson 95% CI is wide and descriptive only."
            )
        by_category[cat] = entry

    # Long-context position breakdown (start / middle / end)
    long_ctx_by_pos: dict[str, dict[str, Any]] = {}
    for pos in ("start", "middle", "end"):
        pos_recs = [
            r
            for r in scored_records
            if r.get("category") == "long_context" and r.get("context_position") == pos
        ]
        if pos_recs:
            pos_corr = sum(1 for r in pos_recs if r.get("is_correct", False))
            long_ctx_by_pos[pos] = format_rate_with_ci(pos_corr, len(pos_recs))

    abstention = compute_abstention_metrics(scored_records)
    hallucination = (
        compute_hallucination_rates(hallucination_records)
        if hallucination_records is not None
        else {}
    )

    overall_entry = format_rate_with_ci(correct_all, total_all)
    overall_entry["cluster_bootstrap_ci_95"] = cluster_bootstrap_ci_95(
        scored_records, metric_key="is_correct", n_resamples=1000, seed=seed
    )

    llm_only_entry = format_rate_with_ci(correct_llm, len(llm_records))
    llm_only_entry["cluster_bootstrap_ci_95"] = cluster_bootstrap_ci_95(
        llm_records, metric_key="is_correct", n_resamples=1000, seed=seed
    )

    return {
        "overall_all_150": overall_entry,
        "overall_llm_evaluated_144": llm_only_entry,
        "base_cases_only": format_rate_with_ci(correct_base, len(base_records)),
        "adversarial_variants_only": format_rate_with_ci(
            correct_variants, len(variant_records)
        ),
        "by_category": by_category,
        "long_context_by_position": long_ctx_by_pos,
        "abstention": abstention,
        "hallucination": hallucination,
    }


def compare_ablation_metrics(
    grounded_metrics: dict[str, Any], naive_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compare grounded_v1 (with Evidence-Quote Gate) against naive_v0 ablation baseline."""
    g_acc = grounded_metrics["overall_all_150"]["rate"]
    n_acc = naive_metrics["overall_all_150"]["rate"]
    g_hal = (
        grounded_metrics.get("hallucination", {})
        .get("output_level_all", {})
        .get("rate", 0.0)
    )
    n_hal = (
        naive_metrics.get("hallucination", {})
        .get("output_level_all", {})
        .get("rate", 0.0)
    )
    g_abs_rec = (
        grounded_metrics.get("abstention", {})
        .get("abstention_recall", {})
        .get("rate", 0.0)
    )
    n_abs_rec = (
        naive_metrics.get("abstention", {})
        .get("abstention_recall", {})
        .get("rate", 0.0)
    )

    per_cat_diff: dict[str, dict[str, Any]] = {}
    all_cats = list(grounded_metrics.get("by_category", {}).keys())
    for cat in all_cats:
        g_c = grounded_metrics["by_category"].get(cat, {})
        n_c = naive_metrics.get("by_category", {}).get(cat, {})
        per_cat_diff[cat] = {
            "grounded_v1": g_c.get("formatted", "N/A"),
            "naive_v0": n_c.get("formatted", "N/A"),
            "delta_rate": round(g_c.get("rate", 0.0) - n_c.get("rate", 0.0), 4),
        }

    return {
        "overall_accuracy_delta": round(g_acc - n_acc, 4),
        "output_hallucination_rate_delta": round(g_hal - n_hal, 4),
        "abstention_recall_delta": round(g_abs_rec - n_abs_rec, 4),
        "per_category_comparison": per_cat_diff,
    }


def plot_accuracy_by_category(
    grounded_metrics: dict[str, Any],
    output_path: Path,
    naive_metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Render accuracy-by-category Matplotlib bar chart strictly from saved metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_cat = grounded_metrics.get("by_category", {})
    categories = list(by_cat.keys())
    g_rates = [by_cat[c]["rate"] * 100.0 for c in categories]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(categories)))

    if naive_metrics and "by_category" in naive_metrics:
        width = 0.38
        n_rates = [
            naive_metrics["by_category"].get(c, {}).get("rate", 0.0) * 100.0
            for c in categories
        ]
        ax.bar([i - width / 2 for i in x], g_rates, width=width, label="grounded_v1 (Gate ON)", color="#1f77b4")
        ax.bar([i + width / 2 for i in x], n_rates, width=width, label="naive_v0 (Ablation)", color="#ff7f0e", alpha=0.85)
        ax.legend()
    else:
        ax.bar(x, g_rates, width=0.55, color="#1f77b4", label="grounded_v1")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Accuracy by Evaluation Category (with Numerator/Denominator Counts)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, c in enumerate(categories):
        num = by_cat[c]["numerator"]
        den = by_cat[c]["denominator"]
        ax.text(
            i - (0.19 if naive_metrics else 0.0),
            g_rates[i] + 1.5,
            f"{num}/{den}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hallucination_rate_by_category(
    grounded_metrics: dict[str, Any],
    output_path: Path,
    naive_metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Render output-level hallucination rate by category strictly from saved metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g_hal_cats = grounded_metrics.get("hallucination", {}).get("by_category", {})
    categories = list(g_hal_cats.keys())
    if not categories:
        return
    g_rates = [g_hal_cats[c]["output_hallucination_rate"] * 100.0 for c in categories]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = list(range(len(categories)))

    if naive_metrics and "hallucination" in naive_metrics:
        n_hal_cats = naive_metrics["hallucination"].get("by_category", {})
        n_rates = [
            n_hal_cats.get(c, {}).get("output_hallucination_rate", 0.0) * 100.0
            for c in categories
        ]
        width = 0.38
        ax.bar([i - width / 2 for i in x], g_rates, width=width, label="grounded_v1 (Gate ON)", color="#2ca02c")
        ax.bar([i + width / 2 for i in x], n_rates, width=width, label="naive_v0 (Ablation)", color="#d62728", alpha=0.85)
        ax.legend()
    else:
        ax.bar(x, g_rates, width=0.55, color="#d62728", label="grounded_v1")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Output Hallucination Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Output-Level Hallucination Rate by Category")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
