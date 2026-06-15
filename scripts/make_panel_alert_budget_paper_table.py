from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
INPUT = TABLE_DIR / "panel_alert_budget_event_network.csv"
OUTPUT = TABLE_DIR / "panel_alert_budget_paper_table.csv"
REPORT = PROJECT_ROOT / "reports" / "panel_alert_budget_paper_table.md"

DISPLAY_GROUPS = [
    "L1_operational",
    "L3_external_only",
    "L4_own_plus_external",
    "L5_own_external_total_network",
    "L8_own_external_total_equal_placebo",
    "L9_own_external_total_shuffled_placebo",
    "L10_own_external_total_random_placebo",
]

GROUP_LABELS = {
    "L1_operational": "Operational baseline",
    "L3_external_only": "External events only",
    "L4_own_plus_external": "Own + external events",
    "L5_own_external_total_network": "Own + external + true total network",
    "L8_own_external_total_equal_placebo": "Own + external + equal-weight placebo",
    "L9_own_external_total_shuffled_placebo": "Own + external + shuffled placebo",
    "L10_own_external_total_random_placebo": "Own + external + random placebo",
}


def interpretation(row: pd.Series) -> str:
    budget = int(row["budget"])
    if row["feature_group"] == "L5_own_external_total_network" and budget in {25, 50}:
        return "Best medium-budget alert ranking among tested true/placebo total-exposure groups"
    if row["feature_group"] == "L5_own_external_total_network" and budget == 10:
        return "Does not improve the very top alert budget"
    if "placebo" in row["feature_group"] and row["total_hit_delta_vs_operational"] > 0:
        return "Placebo also improves, so do not claim network-specific causality"
    if row["feature_group"] == "L1_operational":
        return "Reference"
    return "Diagnostic comparator"


def make_table(results: pd.DataFrame) -> pd.DataFrame:
    subset = results.loc[results["feature_group"].isin(DISPLAY_GROUPS)].copy()
    grouped = (
        subset.groupby(["budget", "feature_group"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            folds_with_more_hits=("hit_delta_vs_operational", lambda s: int((s > 0).sum())),
            total_alerts=("alerts", "sum"),
            total_hits=("hits", "sum"),
            baseline_total_hits=("baseline_hits", "sum"),
            total_false_alerts=("false_alerts", "sum"),
            mean_precision_at_budget=("precision_at_budget", "mean"),
            mean_recall_at_budget=("recall_at_budget", "mean"),
            mean_precision_delta_vs_operational=("precision_delta_vs_operational", "mean"),
            mean_recall_delta_vs_operational=("recall_delta_vs_operational", "mean"),
        )
    )
    grouped["total_hit_delta_vs_operational"] = grouped["total_hits"] - grouped["baseline_total_hits"]
    grouped["feature_label"] = grouped["feature_group"].map(GROUP_LABELS)
    grouped["interpretation"] = grouped.apply(interpretation, axis=1)
    grouped["display_order"] = grouped["feature_group"].map(
        {name: idx for idx, name in enumerate(DISPLAY_GROUPS)}
    )
    return grouped.sort_values(["budget", "display_order"]).drop(columns=["display_order"])


def write_report(table: pd.DataFrame) -> None:
    compact = table[
        [
            "budget",
            "feature_label",
            "total_hits",
            "baseline_total_hits",
            "total_hit_delta_vs_operational",
            "folds_with_more_hits",
            "mean_precision_at_budget",
            "mean_precision_delta_vs_operational",
            "interpretation",
        ]
    ].copy()
    content = f"""# Panel Alert-Budget Paper Table

## Purpose

This table is a paper-facing summary of the fixed alert-budget diagnostic. It is designed for applied benchmark writing, where a fixed top-k alert budget is a realistic secondary evaluation of warning-system utility.

## Table

{compact.to_markdown(index=False)}

## Suggested Caption

Fixed-budget alert-ranking performance for Random Forest models in the locked rolling-origin panel benchmark. Each test year contributes the top-k ranked country-week alerts. The true total-network event bundle improves medium-budget alert hits over the operational baseline at k=25 and k=50, but not at k=10. Placebo variants also improve in some settings, so this result should be interpreted as applied alert-ranking utility rather than proof that exact trade-network weights are uniquely predictive.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run scripts/analyze_panel_alert_budget_event_network.py first."
        )
    results = pd.read_csv(INPUT)
    table = make_table(results)
    table.to_csv(OUTPUT, index=False)
    write_report(table)
    print(f"Saved: {OUTPUT}")
    print(f"Report: {REPORT}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
