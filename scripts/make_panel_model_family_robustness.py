from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
MAIN_SUMMARY = TABLE_DIR / "panel_benchmark_summary.csv"
HGB_SUMMARY = TABLE_DIR / "panel_gradient_boosting_summary.csv"
OUTPUT = TABLE_DIR / "panel_model_family_robustness.csv"
REPORT = PROJECT_ROOT / "reports" / "panel_model_family_robustness.md"

FEATURE_ORDER = [
    "M1_operational",
    "M2_own_country_news",
    "M3_external_unweighted_events",
    "M4_total_import_network",
    "M5_me_strict_network",
    "M6a_total_equal_placebo",
    "M6b_total_shuffled_placebo",
    "M6c_total_random_placebo",
    "M6d_me_placebo_bundle",
    "M7_full_event_network",
]

FOCUS_GROUPS = [
    "M1_operational",
    "M3_external_unweighted_events",
    "M4_total_import_network",
    "M5_me_strict_network",
    "M6a_total_equal_placebo",
    "M6b_total_shuffled_placebo",
    "M6c_total_random_placebo",
]


def load_summaries() -> pd.DataFrame:
    main = pd.read_csv(MAIN_SUMMARY)
    hgb = pd.read_csv(HGB_SUMMARY)
    keep_cols = [
        "feature_group",
        "model",
        "folds",
        "mean_pr_auc",
        "std_pr_auc",
        "mean_roc_auc",
        "mean_f1",
        "mean_precision_at_25",
        "total_tp",
        "total_fp",
        "total_fn",
    ]
    combined = pd.concat([main[keep_cols], hgb[keep_cols]], ignore_index=True)
    combined = combined.loc[combined["feature_group"].isin(FEATURE_ORDER)].copy()
    combined["feature_order"] = combined["feature_group"].map(
        {feature: idx for idx, feature in enumerate(FEATURE_ORDER)}
    )
    return combined.sort_values(["model", "feature_order"]).drop(columns=["feature_order"])


def add_model_family_deltas(combined: pd.DataFrame) -> pd.DataFrame:
    baseline = combined.loc[
        combined["feature_group"].eq("M1_operational"),
        ["model", "mean_pr_auc", "mean_roc_auc"],
    ].rename(columns={"mean_pr_auc": "m1_mean_pr_auc", "mean_roc_auc": "m1_mean_roc_auc"})
    out = combined.merge(baseline, on="model", how="left")
    out["mean_pr_auc_delta_vs_m1"] = out["mean_pr_auc"] - out["m1_mean_pr_auc"]
    out["mean_roc_auc_delta_vs_m1"] = out["mean_roc_auc"] - out["m1_mean_roc_auc"]
    out["rank_within_model_pr_auc"] = (
        out.groupby("model")["mean_pr_auc"].rank(method="min", ascending=False).astype(int)
    )
    return out.sort_values(["model", "rank_within_model_pr_auc", "feature_group"])


def write_report(table: pd.DataFrame) -> None:
    focus = table.loc[table["feature_group"].isin(FOCUS_GROUPS)].copy()
    top_by_model = (
        table.sort_values(["model", "rank_within_model_pr_auc"])
        .groupby("model", as_index=False)
        .head(3)
        [
            [
                "model",
                "rank_within_model_pr_auc",
                "feature_group",
                "mean_pr_auc",
                "mean_pr_auc_delta_vs_m1",
                "mean_roc_auc",
                "mean_precision_at_25",
            ]
        ]
    )
    event_network = focus.loc[
        focus["feature_group"].isin(
            ["M3_external_unweighted_events", "M4_total_import_network", "M5_me_strict_network"]
        )
    ]
    event_network_summary = (
        event_network.groupby("feature_group", as_index=False)
        .agg(
            model_families=("model", "nunique"),
            families_positive_pr_delta=("mean_pr_auc_delta_vs_m1", lambda s: int((s > 0).sum())),
            mean_delta_across_families=("mean_pr_auc_delta_vs_m1", "mean"),
            best_delta=("mean_pr_auc_delta_vs_m1", "max"),
            worst_delta=("mean_pr_auc_delta_vs_m1", "min"),
        )
        .sort_values("mean_delta_across_families", ascending=False)
    )

    content = f"""# Panel Model-Family Robustness

## Purpose

This report consolidates the locked Logistic Regression, Random Forest, and dependency-safe sklearn HistGradientBoosting results for the 11-country panel benchmark. It addresses Gate 3: whether the benchmark includes a third model family without adding dependency risk.

## Top Feature Groups Within Each Model Family

{top_by_model.to_markdown(index=False)}

## Focus Feature Groups

{focus[["feature_group", "model", "rank_within_model_pr_auc", "mean_pr_auc", "std_pr_auc", "mean_pr_auc_delta_vs_m1", "mean_roc_auc", "mean_precision_at_25"]].to_markdown(index=False)}

## Event/Network Delta Consistency

{event_network_summary.to_markdown(index=False)}

## Reading

The third model family is implemented and reported using `HistGradientBoostingClassifier`, satisfying model-family breadth without introducing XGBoost or LightGBM dependencies. The substantive conclusion remains cautious: no model family establishes a decisive network-specific win. Random Forest favors machinery/electronics network exposure, HistGradientBoosting gives modest positive deltas for external events and ME network exposure but ranks equal-weight placebo highest, and Logistic Regression is dominated by several placebo/network alternatives with very small differences. Therefore, Gate 3 can be treated as satisfied for benchmark breadth, while Gate 1 and Gate 2 claims should remain modest.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    table = add_model_family_deltas(load_summaries())
    table.to_csv(OUTPUT, index=False)
    write_report(table)
    print(f"Saved: {OUTPUT}")
    print(f"Report: {REPORT}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
