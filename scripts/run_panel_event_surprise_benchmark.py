from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_EQUAL_PLACEBO_FEATURES,
    TOTAL_NETWORK_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    load_dataset,
    select_threshold,
    split_fold,
    summarize_metrics,
)
from scripts.run_panel_title_event_benchmark import fit_model, make_models  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_event_surprise_benchmark.md"
METRICS = TABLE_DIR / "panel_event_surprise_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel_event_surprise_summary.csv"
PREDICTIONS = TABLE_DIR / "panel_event_surprise_predictions.csv"
DELTAS = TABLE_DIR / "panel_event_surprise_bootstrap_deltas.csv"
BASELINE_GROUP = "Z1_operational"
RANDOM_SEED = 42
ROLLING_WINDOW_WEEKS = 12


def surprise_name(feature: str) -> str:
    return f"{feature}_surprise_z{ROLLING_WINDOW_WEEKS}w"


def add_surprise_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.sort_values(["ISO3", "week"]).copy()
    for feature in features:
        grouped = out.groupby("ISO3", sort=False)[feature]
        prior = grouped.shift(1)
        prior_mean = prior.groupby(out["ISO3"], sort=False).transform(
            lambda s: s.rolling(ROLLING_WINDOW_WEEKS, min_periods=4).mean()
        )
        prior_std = prior.groupby(out["ISO3"], sort=False).transform(
            lambda s: s.rolling(ROLLING_WINDOW_WEEKS, min_periods=4).std()
        )
        z = (out[feature] - prior_mean) / prior_std.replace(0, np.nan)
        out[surprise_name(feature)] = z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)
    return out.sort_values(["week", "ISO3"]).reset_index(drop=True)


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    own_external = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    own_external_z = [surprise_name(feature) for feature in own_external]
    total_network_z = [surprise_name(feature) for feature in TOTAL_NETWORK_FEATURES]
    total_equal_z = [surprise_name(feature) for feature in TOTAL_EQUAL_PLACEBO_FEATURES]
    me_network_z = [surprise_name(feature) for feature in ME_NETWORK_FEATURES]

    return {
        "Z1_operational": base,
        "Z2_raw_own_external": base + own_external,
        "Z3_surprise_own_external": base + own_external_z,
        "Z4_raw_plus_surprise_own_external": base + own_external + own_external_z,
        "Z5_surprise_own_external_total_network": base + own_external_z + total_network_z,
        "Z6_surprise_own_external_total_equal_placebo": base + own_external_z + total_equal_z,
        "Z7_surprise_own_external_me_network": base + own_external_z + me_network_z,
        "Z8_raw_plus_surprise_total_network": base
        + own_external
        + TOTAL_NETWORK_FEATURES
        + own_external_z
        + total_network_z,
    }


def run_fold(df: pd.DataFrame, fold) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    prediction_frames = []
    for group_name, features in feature_groups.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise KeyError(f"{group_name} missing features: {missing}")

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
            prediction_frames.append(predictions)

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
    return rows, prediction_frames


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

    delta_array = np.array(deltas)
    low, high = np.percentile(delta_array, [2.5, 97.5])
    p_gt_0 = float(np.mean(delta_array > 0))
    return float(point), float(low), float(high), p_gt_0


def bootstrap_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model_predictions in predictions.groupby("model"):
        baseline = model_predictions.loc[model_predictions["feature_group"].eq(BASELINE_GROUP)].copy()
        focus_groups = [
            group for group in sorted(model_predictions["feature_group"].unique()) if group != BASELINE_GROUP
        ]
        for focus_group in focus_groups:
            focus = model_predictions.loc[model_predictions["feature_group"].eq(focus_group)].copy()
            pr_point, pr_low, pr_high, pr_p = paired_bootstrap_delta(
                focus, baseline, average_precision_score
            )
            roc_point, roc_low, roc_high, roc_p = paired_bootstrap_delta(focus, baseline, roc_auc_score)
            rows.append(
                {
                    "model": model_name,
                    "focus_group": focus_group,
                    "baseline_group": BASELINE_GROUP,
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
    return pd.DataFrame(rows).sort_values(["model", "pooled_pr_auc_delta"], ascending=[True, False])


def delta_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["feature_group"].eq(BASELINE_GROUP),
        ["model", "fold", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": "baseline_pr_auc", "roc_auc": "baseline_roc_auc"})
    deltas = metrics.merge(baseline, on=["model", "fold"], how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq(BASELINE_GROUP)].copy()
    deltas["pr_auc_delta_vs_operational"] = deltas["pr_auc"] - deltas["baseline_pr_auc"]
    deltas["roc_auc_delta_vs_operational"] = deltas["roc_auc"] - deltas["baseline_roc_auc"]
    return (
        deltas.groupby(["model", "feature_group"], as_index=False)
        .agg(
            mean_pr_auc_delta_vs_operational=("pr_auc_delta_vs_operational", "mean"),
            folds_beating_operational=("pr_auc_delta_vs_operational", lambda s: int((s > 0).sum())),
            mean_roc_auc_delta_vs_operational=("roc_auc_delta_vs_operational", "mean"),
        )
        .sort_values(["model", "mean_pr_auc_delta_vs_operational"], ascending=[True, False])
    )


def write_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    top = summary.head(20)
    fold_view = metrics[
        ["fold", "feature_group", "model", "pr_auc", "roc_auc", "f1", "precision", "recall", "test_positives"]
    ].sort_values(["model", "feature_group", "fold"])
    delta_view = delta_summary(metrics)

    content = f"""# Panel Event-Surprise Benchmark

## Purpose

This exploratory Gate 1 check tests whether event anomalies are more stable than raw event levels. The current raw event layer is vulnerable to country scale and media-volume differences, so this script derives fold-safe 12-week surprise z-scores using only each country's prior event history.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_event_network_benchmark.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Surprise transform: current event value minus prior 12-week country mean, divided by prior 12-week country standard deviation; missing/zero-variance histories set to 0 and z-scores clipped to [-5, 5]
- Folds: locked rolling-origin test years 2023, 2024, and 2025
- Models: Logistic Regression, Random Forest, HistGradientBoosting
- Thresholds: validation-selected only
- Main metric: PR-AUC

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Mean Deltas Versus Operational

{delta_view.to_markdown(index=False)}

## Paired Bootstrap Deltas Versus Operational

{deltas.to_markdown(index=False)}

## Fold-Level Metrics

{fold_view.to_markdown(index=False)}

## Reading

This is exploratory unless a surprise specification beats `Z1_operational` across most rolling folds and has a clearly positive pooled PR-AUC delta. If the surprise feature helps only one model or is matched by the equal-weight placebo, it should be treated as a diagnostic refinement rather than a publishable breakthrough.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base_df = load_dataset()
    surprise_features = sorted(
        set(
            OWN_NEWS_FEATURES
            + EXTERNAL_UNWEIGHTED_FEATURES
            + TOTAL_NETWORK_FEATURES
            + TOTAL_EQUAL_PLACEBO_FEATURES
            + ME_NETWORK_FEATURES
        )
    )
    df = add_surprise_features(base_df, surprise_features)

    rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(df, fold)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    deltas = bootstrap_deltas(predictions)

    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    deltas.to_csv(DELTAS, index=False)
    write_report(df, metrics, summary, predictions, deltas)

    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
