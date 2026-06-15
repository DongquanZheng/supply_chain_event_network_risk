from __future__ import annotations

from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    FOLDS,
    TARGET,
    run_fold,
    summarize_metrics,
)


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_benchmark_results.md"
METRICS = TABLE_DIR / "panel32_benchmark_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_benchmark_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_benchmark_predictions.csv"
INTERPRETATION = TABLE_DIR / "panel32_benchmark_model_interpretation.csv"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, df: pd.DataFrame) -> None:
    top = summary.head(16)
    rf = summary.loc[summary["model"].eq("random_forest")].sort_values("mean_pr_auc", ascending=False)
    logistic = summary.loc[summary["model"].eq("logistic")].sort_values("mean_pr_auc", ascending=False)

    content = f"""# Panel32 Benchmark Results

## Dataset

- File: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}

## Evaluation

- Temporal rolling-origin validation.
- Test folds: 2023, 2024, and 2025.
- Thresholds are selected on the immediately preceding validation year only.
- Main ranking metric: PR-AUC.
- Models: balanced Logistic Regression and Random Forest.

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Random Forest

{rf[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Logistic Regression

{logistic[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Reading

This is the first LR/RF benchmark on the expanded32 exploratory panel. It is direct engineering progress for Gate 4 and an initial stress test for Gate 1/Gate 2 at broader country scope. Treat it as exploratory until advanced models, placebos, alert-budget tables, and LOCO transfer diagnostics are rerun on the same expanded panel.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    metric_rows = []
    prediction_rows = []
    interpretation_frames = []
    for fold in FOLDS:
        fold_metrics, fold_predictions, fold_interpretation = run_fold(fold, df)
        metric_rows.extend(fold_metrics)
        prediction_rows.extend(fold_predictions)
        interpretation_frames.extend(fold_interpretation)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    interpretation = pd.concat(interpretation_frames, ignore_index=True)
    summary = summarize_metrics(metrics)

    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    interpretation.to_csv(INTERPRETATION, index=False)
    write_report(metrics, summary, df)

    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved interpretation: {INTERPRETATION}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
