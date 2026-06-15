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
    FOLDS,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    load_dataset,
    select_threshold,
    split_fold,
    summarize_metrics,
)
from scripts.run_panel_cumulative_event_ladder_check import make_feature_groups  # noqa: E402
from scripts.run_panel_title_event_benchmark import fit_model, make_models  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_validation_selected_ladder.md"
ALL_METRICS = TABLE_DIR / "panel_validation_selected_ladder_all_candidates.csv"
SELECTIONS = TABLE_DIR / "panel_validation_selected_ladder_selections.csv"
SUMMARY = TABLE_DIR / "panel_validation_selected_ladder_summary.csv"
PREDICTIONS = TABLE_DIR / "panel_validation_selected_ladder_predictions.csv"
DELTAS = TABLE_DIR / "panel_validation_selected_ladder_bootstrap_deltas.csv"
BASELINE_GROUP = "L1_operational"
RANDOM_SEED = 42


SELECTION_POLICIES = {
    "VS_event_ladder": [
        "L1_operational",
        "L2_own_news",
        "L3_external_only",
        "L4_own_plus_external",
    ],
    "VS_true_network_ladder": [
        "L1_operational",
        "L2_own_news",
        "L3_external_only",
        "L4_own_plus_external",
        "L5_own_external_total_network",
        "L6_own_external_me_network",
        "L7_own_external_total_me_network",
    ],
    "VS_placebo_ladder": [
        "L1_operational",
        "L2_own_news",
        "L3_external_only",
        "L4_own_plus_external",
        "L8_own_external_total_equal_placebo",
        "L9_own_external_total_shuffled_placebo",
        "L10_own_external_total_random_placebo",
    ],
}


def run_fold(df: pd.DataFrame, fold) -> tuple[list[dict], pd.DataFrame, list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    prediction_frames = []
    for group_name, features in feature_groups.items():
        for model_name, model in make_models().items():
            fit_model(model_name, model, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            validation_pr_auc = average_precision_score(validation[TARGET].to_numpy(), val_proba)
            validation_roc_auc = roc_auc_score(validation[TARGET].to_numpy(), val_proba)
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
                    "validation_pr_auc": validation_pr_auc,
                    "validation_roc_auc": validation_roc_auc,
                    "selected_threshold": threshold,
                    "validation_f1": val_f1,
                    **scores,
                }
            )

            predictions = test[["ISO3", "country", "week", TARGET]].copy()
            predictions["fold"] = fold.name
            predictions["candidate_group"] = group_name
            predictions["model"] = model_name
            predictions["predicted_probability"] = test_proba
            predictions["selected_threshold"] = threshold
            predictions["validation_f1"] = val_f1
            predictions["validation_pr_auc"] = validation_pr_auc
            prediction_frames.append(predictions)

    metrics = pd.DataFrame(rows)
    return rows, metrics, prediction_frames


def choose_policy_predictions(
    all_metrics: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    selected_predictions = []

    for (fold, model), fold_model_metrics in all_metrics.groupby(["fold", "model"]):
        for policy_name, candidates in SELECTION_POLICIES.items():
            available = fold_model_metrics.loc[fold_model_metrics["feature_group"].isin(candidates)].copy()
            selected = available.sort_values(
                ["validation_pr_auc", "feature_group"], ascending=[False, True]
            ).iloc[0]
            selected_group = selected["feature_group"]
            selected_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "selection_policy": policy_name,
                    "selected_group": selected_group,
                    "selected_validation_pr_auc": selected["validation_pr_auc"],
                    "selected_test_pr_auc": selected["pr_auc"],
                    "selected_test_roc_auc": selected["roc_auc"],
                    "selected_test_f1": selected["f1"],
                    "selected_test_precision": selected["precision"],
                    "selected_test_recall": selected["recall"],
                    "selected_threshold": selected["selected_threshold"],
                    "validation_f1": selected["validation_f1"],
                }
            )

            pred = candidate_predictions.loc[
                candidate_predictions["fold"].eq(fold)
                & candidate_predictions["model"].eq(model)
                & candidate_predictions["candidate_group"].eq(selected_group)
            ].copy()
            pred["feature_group"] = policy_name
            pred["selected_candidate_group"] = selected_group
            selected_predictions.append(pred)

    return pd.DataFrame(selected_rows), pd.concat(selected_predictions, ignore_index=True)


def build_policy_metrics(selections: pd.DataFrame, all_metrics: pd.DataFrame) -> pd.DataFrame:
    base = all_metrics.loc[all_metrics["feature_group"].eq(BASELINE_GROUP)].copy()
    selected = selections.rename(
        columns={
            "selection_policy": "feature_group",
            "selected_group": "selected_candidate_group",
            "selected_test_pr_auc": "pr_auc",
            "selected_test_roc_auc": "roc_auc",
            "selected_test_f1": "f1",
            "selected_test_precision": "precision",
            "selected_test_recall": "recall",
        }
    )
    selected["roc_auc"] = selected["roc_auc"].astype(float)
    selected["pr_auc"] = selected["pr_auc"].astype(float)
    selected["f1"] = selected["f1"].astype(float)
    selected["precision"] = selected["precision"].astype(float)
    selected["recall"] = selected["recall"].astype(float)

    selected = selected.merge(
        all_metrics[
            [
                "fold",
                "model",
                "feature_group",
                "test_positives",
                "tp",
                "fp",
                "fn",
                "tn",
                "precision_at_10",
                "precision_at_25",
                "precision_at_50",
            ]
        ],
        left_on=["fold", "model", "selected_candidate_group"],
        right_on=["fold", "model", "feature_group"],
        suffixes=("", "_candidate"),
    )
    selected["feature_group"] = selected["feature_group"]

    return pd.concat(
        [
            base[
                [
                    "fold",
                    "feature_group",
                    "model",
                    "test_positives",
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "precision",
                    "recall",
                    "precision_at_10",
                    "precision_at_25",
                    "precision_at_50",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ]
            ],
            selected[
                [
                    "fold",
                    "feature_group",
                    "model",
                    "test_positives",
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "precision",
                    "recall",
                    "precision_at_10",
                    "precision_at_25",
                    "precision_at_50",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ]
            ],
        ],
        ignore_index=True,
    )


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


def bootstrap_deltas(predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, policy_name), focus in predictions.groupby(["model", "feature_group"]):
        baseline = baseline_predictions.loc[baseline_predictions["model"].eq(model_name)].copy()
        pr_point, pr_low, pr_high, pr_p = paired_bootstrap_delta(focus, baseline, average_precision_score)
        roc_point, roc_low, roc_high, roc_p = paired_bootstrap_delta(focus, baseline, roc_auc_score)
        rows.append(
            {
                "model": model_name,
                "selection_policy": policy_name,
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


def write_report(
    df: pd.DataFrame,
    all_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    selection_counts = (
        selections.groupby(["selection_policy", "model", "selected_group"], as_index=False)
        .size()
        .rename(columns={"size": "folds_selected"})
        .sort_values(["selection_policy", "model", "folds_selected"], ascending=[True, True, False])
    )
    baseline_compare = summary.sort_values(["model", "mean_pr_auc"], ascending=[True, False])
    validation_view = all_metrics[
        ["fold", "feature_group", "model", "validation_pr_auc", "pr_auc", "roc_auc"]
    ].sort_values(["model", "fold", "validation_pr_auc"], ascending=[True, True, False])

    content = f"""# Panel Validation-Selected Ladder

## Purpose

This exploratory pipeline diagnostic tests whether event and network feature groups help when the pipeline is allowed to select the feature ladder by validation-year PR-AUC, including the option to keep the operational baseline. It uses no test labels for feature-group selection.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_event_network_benchmark.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Folds: locked rolling-origin test years 2023, 2024, and 2025
- Selection metric: validation PR-AUC
- Thresholds: validation-selected F1 thresholds after the feature group is fit
- Main test metric: PR-AUC
- Policies:
  - `VS_event_ladder`: operational, own news, external, own+external
  - `VS_true_network_ladder`: event ladder plus true total and machinery/electronics network exposure groups
  - `VS_placebo_ladder`: event ladder plus total equal/shuffled/random placebo groups

## Selected Test Summary

{baseline_compare[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Selection Counts

{selection_counts.to_markdown(index=False)}

## Paired Bootstrap Deltas Versus Operational

{deltas.to_markdown(index=False)}

## Candidate Validation/Test Detail

{validation_view.to_markdown(index=False)}

## Reading

This is a validation-selected application-pipeline diagnostic, not a replacement for the fixed benchmark ladder. It would strengthen Gate 1 only if validation-selected event/network policies beat the operational baseline on held-out test PR-AUC and do so more convincingly than the placebo policy. If validation selects placebo groups or fails to improve test PR-AUC, the result should be treated as further evidence for cautious claims.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    rows = []
    candidate_prediction_frames = []
    for fold in FOLDS:
        fold_rows, _, fold_predictions = run_fold(df, fold)
        rows.extend(fold_rows)
        candidate_prediction_frames.extend(fold_predictions)

    all_metrics = pd.DataFrame(rows)
    candidate_predictions = pd.concat(candidate_prediction_frames, ignore_index=True)
    selections, selected_predictions = choose_policy_predictions(all_metrics, candidate_predictions)
    policy_metrics = build_policy_metrics(selections, all_metrics)
    summary = summarize_metrics(policy_metrics)

    baseline_predictions = candidate_predictions.loc[
        candidate_predictions["candidate_group"].eq(BASELINE_GROUP)
    ].rename(columns={"candidate_group": "feature_group"})
    deltas = bootstrap_deltas(selected_predictions, baseline_predictions)

    all_metrics.to_csv(ALL_METRICS, index=False)
    selections.to_csv(SELECTIONS, index=False)
    summary.to_csv(SUMMARY, index=False)
    selected_predictions.to_csv(PREDICTIONS, index=False)
    deltas.to_csv(DELTAS, index=False)
    write_report(df, all_metrics, selections, summary, deltas)

    print(f"Saved all candidate metrics: {ALL_METRICS}")
    print(f"Saved selections: {SELECTIONS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
