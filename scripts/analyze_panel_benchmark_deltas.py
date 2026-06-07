from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_predictions.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_benchmark_delta_checks.md"
TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42

BASELINE_GROUPS = [
    "M1_operational",
    "M2_own_country_news",
    "M3_external_unweighted_events",
    "M4_total_import_network",
    "M6a_total_equal_placebo",
    "M6b_total_shuffled_placebo",
    "M6c_total_random_placebo",
]
FOCUS_GROUP = "M5_me_strict_network"


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS, parse_dates=["week"])


def paired_bootstrap_delta(
    focus: pd.DataFrame,
    baseline: pd.DataFrame,
    metric_fn,
    n_boot: int = 2000,
) -> tuple[float, float, float, float]:
    merged = focus.merge(
        baseline,
        on=["fold", "week", "ISO3", TARGET],
        suffixes=("_focus", "_baseline"),
    )
    y = merged[TARGET].to_numpy()
    focus_score = merged["predicted_probability_focus"].to_numpy()
    base_score = merged["predicted_probability_baseline"].to_numpy()
    point = metric_fn(y, focus_score) - metric_fn(y, base_score)

    rng = np.random.default_rng(RANDOM_SEED)
    deltas = []
    n = len(merged)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(metric_fn(y[idx], focus_score[idx]) - metric_fn(y[idx], base_score[idx]))

    low, high = np.percentile(deltas, [2.5, 97.5])
    p_gt_0 = float(np.mean(np.array(deltas) > 0))
    return float(point), float(low), float(high), p_gt_0


def run() -> None:
    predictions = load_predictions()
    predictions = predictions[predictions["model"].eq("random_forest")].copy()
    focus = predictions[predictions["feature_group"].eq(FOCUS_GROUP)]

    rows = []
    for baseline_group in BASELINE_GROUPS:
        baseline = predictions[predictions["feature_group"].eq(baseline_group)]
        pr_point, pr_low, pr_high, pr_p = paired_bootstrap_delta(
            focus, baseline, average_precision_score
        )
        roc_point, roc_low, roc_high, roc_p = paired_bootstrap_delta(
            focus, baseline, roc_auc_score
        )
        rows.append(
            {
                "focus_group": FOCUS_GROUP,
                "baseline_group": baseline_group,
                "pooled_pr_auc_delta": pr_point,
                "pr_delta_ci_low": pr_low,
                "pr_delta_ci_high": pr_high,
                "pr_delta_bootstrap_p_gt_0": pr_p,
                "pooled_roc_auc_delta": roc_point,
                "roc_delta_ci_low": roc_low,
                "roc_delta_ci_high": roc_high,
                "roc_delta_bootstrap_p_gt_0": roc_p,
            }
        )

    deltas = pd.DataFrame(rows).sort_values("pooled_pr_auc_delta", ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(TABLE_DIR / "panel_benchmark_m5_deltas.csv", index=False)

    content = f"""# Panel Benchmark Delta Checks

## Purpose

This diagnostic compares `{FOCUS_GROUP}` against operational, unweighted event, total-import network, and placebo alternatives using pooled out-of-sample predictions from the 2023, 2024, and 2025 rolling-origin test folds.

The bootstrap is paired at the country-week prediction level. It is a stability diagnostic, not a formal causal test.

## Random Forest M5 Delta Results

{deltas.to_markdown(index=False)}

## Reading

Positive PR-AUC deltas mean the machinery/electronics strict network model ranks abnormal country-weeks better than the comparison model on pooled test predictions. If intervals cross zero, the result should be framed as directional rather than conclusive.
"""
    REPORT.write_text(content, encoding="utf-8")
    print(deltas.to_string(index=False))
    print(f"Saved: {TABLE_DIR / 'panel_benchmark_m5_deltas.csv'}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    run()
