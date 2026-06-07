from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark_models import (  # noqa: E402
    DEFAULT_DATASET,
    FEATURE_GROUPS,
    TARGET,
    load_dataset,
    make_models,
    select_threshold,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_TABLE = TABLE_DIR / "rolling_origin_results.csv"
OUTPUT_MD = PROJECT_ROOT / "reports" / "rolling_origin_check.md"

WINDOWS = [
    (
        "test_2023",
        "2021-01-01",
        "2021-12-31",
        "2022-01-01",
        "2022-12-31",
        "2023-01-01",
        "2023-12-31",
    ),
    (
        "test_2024",
        "2021-01-01",
        "2022-12-31",
        "2023-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
    ),
    (
        "test_2025",
        "2021-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
        "2025-01-01",
        "2025-12-31",
    ),
]

GROUPS = [
    "M1_operational",
    "M2_simple_news",
    "M3_unweighted_me_event",
    "M4_total_import_network",
    "M5_me_network",
    "M6_random_placebo",
]


def split_window(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= train_start) & (df["week"] <= train_end)].copy()
    validation = df[(df["week"] >= val_start) & (df["week"] <= val_end)].copy()
    test = df[(df["week"] >= test_start) & (df["week"] <= test_end)].copy()
    return train, validation, test


def evaluate_group(
    group: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict | None:
    if (
        train[TARGET].nunique() < 2
        or validation[TARGET].nunique() < 2
        or test[TARGET].nunique() < 2
    ):
        return None

    features = FEATURE_GROUPS[group]
    model = make_models()["random_forest"]
    model.fit(train[features], train[TARGET])

    val_proba = model.predict_proba(validation[features])[:, 1]
    threshold, val_f1 = select_threshold(validation[TARGET], val_proba)

    test_proba = model.predict_proba(test[features])[:, 1]
    pred = (test_proba >= threshold).astype(int)
    y_test = test[TARGET].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "feature_group": group,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_positives": int(train[TARGET].sum()),
        "validation_positives": int(validation[TARGET].sum()),
        "test_positives": int(test[TARGET].sum()),
        "selected_threshold": threshold,
        "validation_f1": val_f1,
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "alerts": int(pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def write_report(results: pd.DataFrame, path: Path) -> None:
    mean_results = (
        results.groupby("feature_group", as_index=False)[
            ["pr_auc", "roc_auc", "precision", "recall", "f1"]
        ]
        .mean()
        .sort_values("pr_auc", ascending=False)
    )

    content = f"""# Rolling-Origin Check

## Purpose

This check evaluates whether the M1-M6 pattern is stable across multiple temporal test years rather than only the locked 2025 test split.

## Windows

- `test_2023`: train 2021, validation 2022, test 2023
- `test_2024`: train 2021-2022, validation 2023, test 2024
- `test_2025`: train 2021-2023, validation 2024, test 2025

## Mean Random Forest Results

{mean_results.to_markdown(index=False)}

## Window-Level Results

{results[["window", "feature_group", "test_positives", "pr_auc", "roc_auc", "alerts", "tp", "fp", "fn", "precision", "recall", "f1"]].sort_values(["window", "pr_auc"], ascending=[True, False]).to_markdown(index=False)}

## Interpretation

Simple news controls are the most stable event-informed winner in PR-AUC across these rolling-origin windows. Machinery/electronics network exposure does not show stable predictive superiority. This pushes the paper away from a performance-improvement story and toward a structural-audit story: the network layer should be used to assess whether NLP-derived signals are supply-chain-plausible, not assumed to improve prediction automatically.
"""
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    rows = []

    for window_name, *dates in WINDOWS:
        train, validation, test = split_window(df, *dates)
        for group in GROUPS:
            row = evaluate_group(group, train, validation, test)
            if row is not None:
                row["window"] = window_name
                rows.append(row)

    results = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_TABLE, index=False)
    write_report(results, OUTPUT_MD)

    print(f"Saved rolling-origin results: {OUTPUT_TABLE}")
    print(f"Saved rolling-origin report: {OUTPUT_MD}")
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
