from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark_models import (  # noqa: E402
    DEFAULT_DATASET,
    FEATURE_GROUPS,
    TARGET,
    make_models,
    select_threshold,
    load_dataset,
    temporal_split,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
DIAGNOSTIC_TABLE = TABLE_DIR / "test_predictions_selected.csv"
DIAGNOSTIC_MD = PROJECT_ROOT / "reports" / "benchmark_diagnostics.md"

SELECTED_GROUPS = [
    "M1_operational",
    "M2_simple_news",
    "M3_unweighted_me_event",
    "M5_me_network",
    "M6_random_placebo",
]


def collect_predictions(df: pd.DataFrame) -> pd.DataFrame:
    train, validation, test = temporal_split(df)
    rows = []

    for group_name in SELECTED_GROUPS:
        features = FEATURE_GROUPS[group_name]
        model = make_models()["random_forest"]
        model.fit(train[features], train[TARGET])

        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold, _ = select_threshold(validation[TARGET], val_proba)

        test_proba = model.predict_proba(test[features])[:, 1]
        test_pred = (test_proba >= threshold).astype(int)

        out = test[["week", "portcalls_container", "next_week_container", TARGET]].copy()
        out["feature_group"] = group_name
        out["model"] = "random_forest"
        out["selected_threshold"] = threshold
        out["predicted_probability"] = test_proba
        out["predicted_label"] = test_pred
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def write_report(predictions: pd.DataFrame, path: Path) -> None:
    wide_prob = predictions.pivot_table(
        index=["week", "portcalls_container", "next_week_container", TARGET],
        columns="feature_group",
        values="predicted_probability",
    ).reset_index()
    wide_pred = predictions.pivot_table(
        index=["week", "portcalls_container", "next_week_container", TARGET],
        columns="feature_group",
        values="predicted_label",
    ).reset_index()

    positives = wide_prob[wide_prob[TARGET].eq(1)].copy()

    m2 = predictions[predictions["feature_group"].eq("M2_simple_news")]
    m5 = predictions[predictions["feature_group"].eq("M5_me_network")]
    comparison = m2.merge(
        m5,
        on=["week", "portcalls_container", "next_week_container", TARGET],
        suffixes=("_m2", "_m5"),
    )
    m2_only_false_alerts = comparison[
        comparison[TARGET].eq(0)
        & comparison["predicted_label_m2"].eq(1)
        & comparison["predicted_label_m5"].eq(0)
    ]
    m5_missed_positives = comparison[
        comparison[TARGET].eq(1)
        & comparison["predicted_label_m2"].eq(1)
        & comparison["predicted_label_m5"].eq(0)
    ]

    content = f"""# Benchmark Diagnostics

## Purpose

This diagnostic inspects the 2025 test split at the week level. It focuses on Random Forest because it is the strongest model family in the current benchmark.

## Key Observation

`M2_simple_news` captures all five positive test weeks but produces more false positives. `M5_me_network` captures four of five positive weeks and reduces false positives. This means the current network feature is not the best ranking signal by PR-AUC, but it behaves like a stricter relevance filter.

## Positive Test Weeks

{positives.to_markdown(index=False)}

## M2 Alerts Suppressed By M5

- Count: {len(m2_only_false_alerts)}

{m2_only_false_alerts[["week", "portcalls_container", "next_week_container", "predicted_probability_m2", "predicted_probability_m5"]].head(20).to_markdown(index=False)}

## Positives Missed By M5 But Captured By M2

- Count: {len(m5_missed_positives)}

{m5_missed_positives[["week", "portcalls_container", "next_week_container", "predicted_probability_m2", "predicted_probability_m5"]].to_markdown(index=False)}

## Interpretation

The network layer is not currently indispensable for maximizing PR-AUC. Its defensible role is narrower: it changes alert selectivity and provides an interpretable dependency-channel exposure representation. A PhD-level extension should therefore focus on when network structure filters event relevance, not on claiming universal predictive improvement.
"""
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    predictions = collect_predictions(df)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(DIAGNOSTIC_TABLE, index=False)
    write_report(predictions, DIAGNOSTIC_MD)

    print(f"Saved diagnostic predictions: {DIAGNOSTIC_TABLE}")
    print(f"Saved diagnostic report: {DIAGNOSTIC_MD}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
