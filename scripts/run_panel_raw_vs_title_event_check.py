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
    EXTERNAL_UNWEIGHTED_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
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
    make_models,
    title_columns,
)
from scripts.run_panel_title_event_rolling_check import FOLDS, split_fold  # noqa: E402


REPORT = PROJECT_ROOT / "reports" / "panel_raw_vs_title_event_check.md"
METRICS = TABLE_DIR / "panel_raw_vs_title_event_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel_raw_vs_title_event_summary.csv"
PREDICTIONS = TABLE_DIR / "panel_raw_vs_title_event_predictions.csv"


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    title_own = title_columns("own")
    title_external = title_columns("external")
    return {
        "C1_operational": base,
        "C2_raw_own_news": base + OWN_NEWS_FEATURES,
        "C3_raw_external_events": base + EXTERNAL_UNWEIGHTED_FEATURES,
        "C4_raw_own_external": base + OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES,
        "C5_title_own_events": base + title_own,
        "C6_title_external_events": base + title_external,
        "C7_title_own_external": base + title_own + title_external,
        "C8_raw_external_plus_title_external": base + EXTERNAL_UNWEIGHTED_FEATURES + title_external,
        "C9_raw_all_plus_title_external": base + OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES + title_external,
    }


def run_fold(df: pd.DataFrame, fold) -> list[dict]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    prediction_rows = []
    for group_name, features in feature_groups.items():
        for model_name, model in make_models().items():
            fit_model(model_name, model, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            predictions = test[["ISO3", "country", "week", TARGET]].copy()
            predictions["fold"] = fold.name
            predictions["feature_group"] = group_name
            predictions["model"] = model_name
            predictions["predicted_probability"] = test_proba
            predictions["selected_threshold"] = threshold
            predictions["validation_f1"] = val_f1
            prediction_rows.append(predictions)
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
    return rows, prediction_rows


def delta_summary(metrics: pd.DataFrame, baseline_group: str, label: str) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["feature_group"].eq(baseline_group),
        ["model", "fold", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": f"{label}_pr_auc", "roc_auc": f"{label}_roc_auc"})
    deltas = metrics.merge(baseline, on=["model", "fold"], how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq(baseline_group)].copy()
    deltas[f"pr_auc_delta_vs_{label}"] = deltas["pr_auc"] - deltas[f"{label}_pr_auc"]
    deltas[f"roc_auc_delta_vs_{label}"] = deltas["roc_auc"] - deltas[f"{label}_roc_auc"]
    return (
        deltas.groupby(["model", "feature_group"], as_index=False)
        .agg(
            **{
                f"mean_pr_auc_delta_vs_{label}": (f"pr_auc_delta_vs_{label}", "mean"),
                f"folds_beating_{label}": (f"pr_auc_delta_vs_{label}", lambda s: int((s > 0).sum())),
                f"mean_roc_auc_delta_vs_{label}": (f"roc_auc_delta_vs_{label}", "mean"),
            }
        )
        .sort_values(["model", f"mean_pr_auc_delta_vs_{label}"], ascending=[True, False])
    )


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary.head(20)
    versus_operational = delta_summary(metrics, "C1_operational", "operational")
    versus_raw_external = delta_summary(metrics, "C3_raw_external_events", "raw_external")
    rf_detail = metrics.loc[
        metrics["model"].eq("random_forest"),
        ["fold", "feature_group", "pr_auc", "roc_auc", "f1", "precision", "recall", "test_positives"],
    ].sort_values(["fold", "pr_auc"], ascending=[True, False])

    content = f"""# Panel Raw-vs-Title Event Check

## Purpose

This exploratory Gate 1 check compares the current raw GDELT event controls with stricter title-level event features under the same 2023-start two-fold temporal diagnostic. It tests whether title-level events are a cleaner replacement, a complement, or just a model-specific signal.

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

{versus_operational.to_markdown(index=False)}

## Deltas Versus Raw External Events

{versus_raw_external.to_markdown(index=False)}

## Random Forest Fold Detail

{rf_detail.to_markdown(index=False)}

## Reading

This check is still exploratory because the candidate-document cache starts in 2023. A publishable Gate 1 claim would require the event layer to beat the operational baseline under a predeclared comparison and preferably across model families. If title-level external events only help Random Forest, the paper should frame them as a promising event-definition refinement rather than a settled replacement for raw GDELT controls.
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
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(df, fold)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    write_report(df, metrics, summary)

    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
