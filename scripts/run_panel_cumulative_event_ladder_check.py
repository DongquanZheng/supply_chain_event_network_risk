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
    TOTAL_RANDOM_PLACEBO_FEATURES,
    TOTAL_SHUFFLED_PLACEBO_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    load_dataset,
    select_threshold,
    split_fold,
    summarize_metrics,
)
from scripts.run_panel_title_event_benchmark import fit_model, make_models  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_cumulative_event_ladder_check.md"
METRICS = TABLE_DIR / "panel_cumulative_event_ladder_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel_cumulative_event_ladder_summary.csv"
PREDICTIONS = TABLE_DIR / "panel_cumulative_event_ladder_predictions.csv"
DELTAS = TABLE_DIR / "panel_cumulative_event_ladder_bootstrap_deltas.csv"
RANDOM_SEED = 42
BASELINE_GROUP = "L1_operational"


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    own_external = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    return {
        "L1_operational": base,
        "L2_own_news": base + OWN_NEWS_FEATURES,
        "L3_external_only": base + EXTERNAL_UNWEIGHTED_FEATURES,
        "L4_own_plus_external": base + own_external,
        "L5_own_external_total_network": base + own_external + TOTAL_NETWORK_FEATURES,
        "L6_own_external_me_network": base + own_external + ME_NETWORK_FEATURES,
        "L7_own_external_total_me_network": base
        + own_external
        + TOTAL_NETWORK_FEATURES
        + ME_NETWORK_FEATURES,
        "L8_own_external_total_equal_placebo": base + own_external + TOTAL_EQUAL_PLACEBO_FEATURES,
        "L9_own_external_total_shuffled_placebo": base + own_external + TOTAL_SHUFFLED_PLACEBO_FEATURES,
        "L10_own_external_total_random_placebo": base + own_external + TOTAL_RANDOM_PLACEBO_FEATURES,
    }


def run_fold(df: pd.DataFrame, fold) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    prediction_frames = []
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
            roc_point, roc_low, roc_high, roc_p = paired_bootstrap_delta(
                focus, baseline, roc_auc_score
            )
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


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    top = summary.head(21)
    delta_view = deltas.sort_values(["model", "pooled_pr_auc_delta"], ascending=[True, False])
    fold_view = metrics[
        ["fold", "feature_group", "model", "pr_auc", "roc_auc", "f1", "precision", "recall", "test_positives"]
    ].sort_values(["model", "feature_group", "fold"])

    content = f"""# Panel Cumulative Event Ladder Check

## Purpose

This diagnostic tests whether the exploratory raw own+external event-layer signal survives in the full locked 2021-2025 panel benchmark. It is motivated by the finding that the current main benchmark's `M3_external_unweighted_events` is external-only, while the strongest exploratory signal came from the cumulative own+external event specification.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_event_network_benchmark.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Folds: locked rolling-origin test years 2023, 2024, and 2025
- Thresholds: validation-selected only
- Main metric: PR-AUC

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Paired Bootstrap Deltas Versus Operational

{delta_view.to_markdown(index=False)}

## Fold-Level Metrics

{fold_view.to_markdown(index=False)}

## Reading

If `L4_own_plus_external` beats `L1_operational` across most folds and has a positive paired bootstrap PR-AUC interval, Gate 1 becomes much stronger. If it fails in the locked benchmark, the 2023-start raw-vs-title result should remain exploratory and the paper should avoid a strong event-layer prediction claim.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

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
    write_report(df, metrics, summary, deltas)

    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(21).to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
