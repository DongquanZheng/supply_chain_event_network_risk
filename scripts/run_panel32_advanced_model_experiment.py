from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_panel_advanced_model_experiment as advanced  # noqa: E402
from scripts.run_panel_advanced_model_experiment import TARGET  # noqa: E402


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"


def load_panel32_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)


def write_panel32_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    key_deltas: pd.DataFrame,
) -> None:
    top = summary.head(25)
    model_sections = []
    for model_name in sorted(summary["model"].unique()):
        view = summary.loc[summary["model"].eq(model_name)].sort_values("mean_pr_auc", ascending=False)
        model_sections.append(
            f"""## {model_name}

{view[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}
"""
        )
    delta_summary = advanced.event_network_delta_summary(metrics)
    best_deltas = deltas.groupby("model", group_keys=False).head(5)

    content = f"""# Panel32 Advanced Model Experiment

## Purpose

This expanded32 research run tests stronger model families on the new 32-country panel while preserving the existing M1-M7 feature groups, rolling-origin temporal folds, validation-selected thresholds, and PR-AUC priority.

## Dataset And Protocol

- Dataset: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Test folds: 2023, 2024, 2025 rolling-origin
- Models attempted: ExtraTrees, sklearn GradientBoosting, XGBoost, LightGBM, CatBoost if installed
- Main metric: PR-AUC
- Thresholds: validation-selected only

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

{"".join(model_sections)}
## Mean Fold Deltas Versus M1

{delta_summary.to_markdown(index=False)}

## Best Pooled Bootstrap Deltas Versus M1

{best_deltas.to_markdown(index=False)}

## Key 5000-Draw Checks

{key_deltas.to_markdown(index=False)}

## Reading

This is the first advanced-model stress test on expanded32. A defensible improvement requires event/network groups to beat each model's own operational baseline and preferably separate from equal, shuffled, and random placebo variants. Treat positive results here as exploratory until alert-budget, LOCO transfer, severe-label, and concentration checks are rerun on the same expanded panel.
"""
    advanced.REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    advanced.load_dataset = load_panel32_dataset
    advanced.write_report = write_panel32_report
    advanced.REPORT = PROJECT_ROOT / "reports" / "panel32_advanced_model_experiment.md"
    advanced.METRICS = TABLE_DIR / "panel32_advanced_model_metrics_by_fold.csv"
    advanced.SUMMARY = TABLE_DIR / "panel32_advanced_model_summary.csv"
    advanced.PREDICTIONS = TABLE_DIR / "panel32_advanced_model_predictions.csv"
    advanced.DELTAS = TABLE_DIR / "panel32_advanced_model_bootstrap_deltas.csv"
    advanced.KEY_DELTAS = TABLE_DIR / "panel32_advanced_model_key_deltas.csv"
    advanced.run()


if __name__ == "__main__":
    run()
