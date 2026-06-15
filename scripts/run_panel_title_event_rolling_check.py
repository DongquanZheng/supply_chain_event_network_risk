from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    summarize_metrics,
)
from scripts.run_panel_title_event_benchmark import (  # noqa: E402
    OUTPUT_DATASET,
    TABLE_DIR,
    build_dataset,
    fit_model,
    make_feature_groups,
    make_models,
)


REPORT = PROJECT_ROOT / "reports" / "panel_title_event_rolling_check.md"


@dataclass(frozen=True)
class TitleFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


FOLDS = [
    TitleFold(
        "test_2024",
        "2023-01-01",
        "2023-07-01",
        "2023-07-01",
        "2024-01-01",
        "2024-01-01",
        "2025-01-01",
    ),
    TitleFold(
        "test_2025",
        "2023-01-01",
        "2024-01-01",
        "2024-01-01",
        "2025-01-01",
        "2025-01-01",
        "2026-01-01",
    ),
]


def split_fold(df: pd.DataFrame, fold: TitleFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= fold.train_start) & (df["week"] < fold.train_end)].copy()
    validation = df[(df["week"] >= fold.validation_start) & (df["week"] < fold.validation_end)].copy()
    test = df[(df["week"] >= fold.test_start) & (df["week"] < fold.test_end)].copy()
    return train, validation, test


def run_fold(df: pd.DataFrame, fold: TitleFold) -> list[dict]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    for group_name, features in feature_groups.items():
        for model_name, model in make_models().items():
            fit_model(model_name, model, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            rows.append(
                {
                    "fold": fold.name,
                    "feature_group": group_name,
                    "model": model_name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "train_positives": int(train[TARGET].sum()),
                    "validation_positives": int(validation[TARGET].sum()),
                    "test_positives": int(test[TARGET].sum()),
                    "selected_threshold": threshold,
                    "validation_f1": val_f1,
                    **scores,
                }
            )
    return rows


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary.head(18)
    m1 = metrics.loc[
        metrics["feature_group"].eq("S1_operational_2023_2025"),
        ["model", "fold", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": "m1_pr_auc", "roc_auc": "m1_roc_auc"})
    deltas = metrics.merge(m1, on=["model", "fold"], how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq("S1_operational_2023_2025")].copy()
    deltas["pr_auc_delta_vs_m1"] = deltas["pr_auc"] - deltas["m1_pr_auc"]
    deltas["roc_auc_delta_vs_m1"] = deltas["roc_auc"] - deltas["m1_roc_auc"]
    delta_summary = (
        deltas.groupby(["model", "feature_group"], as_index=False)
        .agg(
            mean_pr_auc_delta_vs_m1=("pr_auc_delta_vs_m1", "mean"),
            folds_beating_m1=("pr_auc_delta_vs_m1", lambda s: int((s > 0).sum())),
            mean_roc_auc_delta_vs_m1=("roc_auc_delta_vs_m1", "mean"),
        )
        .sort_values(["model", "mean_pr_auc_delta_vs_m1"], ascending=[True, False])
    )

    content = f"""# Panel Title-Level Event Rolling Check

## Purpose

This exploratory check tests whether strict title-level GDELT event features remain useful under more than one temporal split. The local candidate-document cache starts in 2023, so this is not the locked 2021-2025 rolling benchmark. It is a two-fold temporal stability diagnostic.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_title_event_benchmark_2023_2025.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Fold `test_2024`: train 2023H1, validate 2023H2, test 2024
- Fold `test_2025`: train 2023, validate 2024, test 2025
- Thresholds: validation-selected only
- Main metric: PR-AUC

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Deltas Versus Operational Baseline

{delta_summary.to_markdown(index=False)}

## Fold-Level Metrics

{metrics[["fold", "feature_group", "model", "pr_auc", "roc_auc", "f1", "precision", "recall", "test_positives"]].sort_values(["model", "feature_group", "fold"]).to_markdown(index=False)}

## Reading

This is useful for Gate 1 only if title-level event features improve over the operational baseline across both folds or across multiple model families. If the improvement is concentrated in one model or one fold, the result should remain exploratory.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DATASET.exists():
        df = pd.read_csv(OUTPUT_DATASET, parse_dates=["week"])
    else:
        df = build_dataset()

    rows = []
    for fold in FOLDS:
        rows.extend(run_fold(df, fold))
    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)

    metrics.to_csv(TABLE_DIR / "panel_title_event_rolling_metrics_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_title_event_rolling_summary.csv", index=False)
    write_report(df, metrics, summary)

    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved metrics: {TABLE_DIR / 'panel_title_event_rolling_metrics_by_fold.csv'}")
    print(f"Saved summary: {TABLE_DIR / 'panel_title_event_rolling_summary.csv'}")
    print(f"Saved report: {REPORT}")
    print(summary.head(18).to_string(index=False))


if __name__ == "__main__":
    run()
