from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import TARGET, add_country_dummies, evaluate_predictions, select_threshold  # noqa: E402
from scripts.run_panel32_network_gated_conversion_main import (  # noqa: E402
    RANDOM_SEED,
    SEVERE_TARGET,
    fit_model,
    load_dataset,
    make_feature_groups,
    make_models,
    safe_ap,
    top_hits,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_network_gated_conversion_loco.md"
RESULTS = TABLE_DIR / "panel32_network_gated_conversion_loco_results.csv"
PREDICTIONS = TABLE_DIR / "panel32_network_gated_conversion_loco_predictions.csv"
SUMMARY = TABLE_DIR / "panel32_network_gated_conversion_loco_summary.csv"
DELTAS = TABLE_DIR / "panel32_network_gated_conversion_loco_deltas.csv"
DEPLOYMENT = TABLE_DIR / "panel32_network_gated_deployment_alert_output.csv"

LOCO_GROUPS = [
    "NG0_portwatch_operational",
    "NG1_portwatch_gdelt_additive",
    "NG2_portwatch_gdelt_wits_additive",
    "NG5_compact_network_gated_true",
    "NG6_equal_compact_gated_placebo",
    "NG6_random_compact_gated_placebo",
    "NG6_shuffled_compact_gated_placebo",
]


def split_loco(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2021-01-01") & (df["week"] < "2024-01-01")].copy()
    validation = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["ISO3"].eq(holdout)) & (df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, validation, test


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def score_holdout(df: pd.DataFrame, holdout: str) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_loco(df, holdout)
    if test.empty or validation.empty or train[TARGET].sum() < 5 or validation[TARGET].sum() < 5 or test[TARGET].sum() < 1:
        return [], []
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    model, fit_mode = make_models()["sklearn_gradient_boosting"]
    rows = []
    predictions = []

    for feature_group in LOCO_GROUPS:
        features = usable_features(train, groups[feature_group])
        model, fit_mode = make_models()["sklearn_gradient_boosting"]
        fit_model(model, fit_mode, train[features], train[TARGET])
        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        main_scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)

        pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
        pred["holdout_iso3"] = holdout
        pred["feature_group"] = feature_group
        pred["model"] = "sklearn_gradient_boosting"
        pred["predicted_probability"] = test_proba

        row = {
            "holdout_iso3": holdout,
            "holdout_country": test["country"].iloc[0],
            "feature_group": feature_group,
            "model": "sklearn_gradient_boosting",
            "feature_count": len(features),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_positives": int(train[TARGET].sum()),
            "validation_positives": int(validation[TARGET].sum()),
            "test_positives": int(test[TARGET].sum()),
            "test_severe_positives": int(test[SEVERE_TARGET].sum()),
            "selected_threshold": threshold,
            "validation_f1": val_f1,
            **main_scores,
            "severe_pr_auc": safe_ap(test[SEVERE_TARGET], pd.Series(test_proba, index=test.index)),
        }
        for k in [5, 10, 25]:
            row[f"main_top{k}_hits"] = top_hits(pred, TARGET, k)
            row[f"severe_top{k}_hits"] = top_hits(pred, SEVERE_TARGET, k)
        rows.append(row)
        predictions.append(pred)
    return rows, predictions


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "holdouts": ("holdout_iso3", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "median_main_pr_auc": ("pr_auc", "median"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "median_severe_pr_auc": ("severe_pr_auc", "median"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "feature_count": ("feature_count", "max"),
    }
    for prefix in ["main", "severe"]:
        for k in [5, 10, 25]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        results.groupby(["feature_group", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)
    )


def pooled_delta(
    predictions: pd.DataFrame,
    focus: str,
    baseline: str,
    target: str,
    n_boot: int = 1500,
) -> tuple[float, float, float, float]:
    focus_frame = predictions.loc[predictions["feature_group"].eq(focus)]
    baseline_frame = predictions.loc[predictions["feature_group"].eq(baseline)]
    merged = focus_frame.merge(
        baseline_frame,
        on=["ISO3", "country", "week", "holdout_iso3", TARGET, SEVERE_TARGET],
        suffixes=("_focus", "_baseline"),
    ).reset_index(drop=True)
    point = safe_ap(merged[target], merged["predicted_probability_focus"]) - safe_ap(
        merged[target], merged["predicted_probability_baseline"]
    )
    draws = []
    for seed in range(n_boot):
        sample = merged.sample(n=len(merged), replace=True, random_state=RANDOM_SEED + seed)
        if sample[target].nunique() < 2:
            continue
        draws.append(
            average_precision_score(sample[target], sample["predicted_probability_focus"])
            - average_precision_score(sample[target], sample["predicted_probability_baseline"])
        )
    series = pd.Series(draws)
    return point, float(series.quantile(0.025)), float(series.quantile(0.975)), float((series > 0).mean())


def hit_delta(predictions: pd.DataFrame, focus: str, baseline: str, target: str, k: int) -> int:
    delta = 0
    subset = predictions.loc[predictions["feature_group"].isin([focus, baseline])]
    for _, frame in subset.groupby("holdout_iso3"):
        focus_top = frame.loc[frame["feature_group"].eq(focus)].sort_values("predicted_probability", ascending=False).head(k)
        base_top = frame.loc[frame["feature_group"].eq(baseline)].sort_values("predicted_probability", ascending=False).head(k)
        delta += int(focus_top[target].sum()) - int(base_top[target].sum())
    return delta


def make_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    focus = "NG5_compact_network_gated_true"
    baselines = [
        "NG0_portwatch_operational",
        "NG1_portwatch_gdelt_additive",
        "NG2_portwatch_gdelt_wits_additive",
        "NG6_equal_compact_gated_placebo",
        "NG6_random_compact_gated_placebo",
        "NG6_shuffled_compact_gated_placebo",
    ]
    rows = []
    for baseline in baselines:
        for label_name, target in [("main", TARGET), ("severe", SEVERE_TARGET)]:
            pr = pooled_delta(predictions, focus, baseline, target)
            rows.append(
                {
                    "contrast": f"compact_gated_true_vs_{baseline}",
                    "focus_feature_group": focus,
                    "baseline_feature_group": baseline,
                    "label": label_name,
                    "pooled_pr_auc_delta": pr[0],
                    "ci_low": pr[1],
                    "ci_high": pr[2],
                    "p_gt_0": pr[3],
                    "top5_hit_delta": hit_delta(predictions, focus, baseline, target, 5),
                    "top10_hit_delta": hit_delta(predictions, focus, baseline, target, 10),
                    "top25_hit_delta": hit_delta(predictions, focus, baseline, target, 25),
                }
            )
    return pd.DataFrame(rows)


def make_deployment_output(summary: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("feature_group")
    rows = []
    mode_specs = [
        (
            "Unseen-country operational guardrail",
            "NG0_portwatch_operational",
            "Operational-only transfer reference and top-k guardrail.",
            "main/supporting",
        ),
        (
            "Unseen-country GDELT additive",
            "NG1_portwatch_gdelt_additive",
            "Tests whether event pressure transfers without WITS gating.",
            "supporting",
        ),
        (
            "Unseen-country WITS additive",
            "NG2_portwatch_gdelt_wits_additive",
            "Additive PortWatch+GDELT+WITS baseline.",
            "main baseline",
        ),
        (
            "Unseen-country compact gated conversion",
            "NG5_compact_network_gated_true",
            "Proposed Network-Gated Event Conversion score.",
            "main candidate",
        ),
        (
            "Unseen-country compact equal placebo",
            "NG6_equal_compact_gated_placebo",
            "Equal WITS-placebo guardrail for gated conversion.",
            "negative/placebo",
        ),
        (
            "Unseen-country compact random placebo",
            "NG6_random_compact_gated_placebo",
            "Random WITS-placebo guardrail for gated conversion.",
            "negative/placebo",
        ),
        (
            "Unseen-country compact shuffled placebo",
            "NG6_shuffled_compact_gated_placebo",
            "Shuffled WITS-placebo guardrail for gated conversion.",
            "negative/placebo",
        ),
    ]
    for mode, group, why, bucket in mode_specs:
        row = indexed.loc[group]
        rows.append(
            {
                "deployment_mode": mode,
                "score": group,
                "validation_design": "Expanded32 leave-one-country-out; train non-holdout 2021-2023, validate non-holdout 2024, test held-out 2025",
                "mean_main_pr_auc": row["mean_main_pr_auc"],
                "mean_severe_pr_auc": row["mean_severe_pr_auc"],
                "main_top5_top10_top25": f"{int(row['main_top5_hits'])}/{int(row['main_top10_hits'])}/{int(row['main_top25_hits'])}",
                "severe_top5_top10_top25": f"{int(row['severe_top5_hits'])}/{int(row['severe_top10_hits'])}/{int(row['severe_top25_hits'])}",
                "why_this_mode": why,
                "paper_bucket": bucket,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, deltas: pd.DataFrame, deployment: pd.DataFrame) -> None:
    content = f"""# Panel32 Network-Gated Event Conversion LOCO

## Purpose

This is the unseen-country transfer check for the main-paper **Network-Gated Event Conversion Model**. It uses only PortWatch, GDELT, and WITS-derived features. Each country is held out completely: models train on other countries in 2021-2023, thresholds are selected on other countries in 2024, and testing is on the held-out country in 2025.

## Policy Summary

{summary.to_markdown(index=False)}

## Compact Gated Deltas

{deltas.to_markdown(index=False)}

## Deployment-Aware Alert Output

{deployment.to_markdown(index=False)}

## Reading

This table asks whether the compact network-gated conversion score transfers to fully unseen countries. A promotable main claim requires the compact true-gated score to improve over operational, additive GDELT/WITS, and equal/random/shuffled WITS placebo versions, especially on PR-AUC and deployment-relevant top-k. If placebo or additive baselines match it, the model should be framed as mechanism and attribution evidence rather than a transfer-performance breakthrough.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    if RESULTS.exists() and PREDICTIONS.exists():
        existing_results = pd.read_csv(RESULTS)
        existing_predictions = pd.read_csv(PREDICTIONS, parse_dates=["week"])
        rows.extend(existing_results.to_dict("records"))
        prediction_frames.append(existing_predictions)
        completed = set(existing_results["holdout_iso3"].dropna().unique())
        print(f"Resuming from checkpoint with {len(completed)} completed holdouts", flush=True)
    else:
        completed = set()
    for holdout in sorted(df["ISO3"].unique()):
        if holdout in completed:
            continue
        holdout_rows, holdout_predictions = score_holdout(df, holdout)
        rows.extend(holdout_rows)
        prediction_frames.extend(holdout_predictions)
        if rows and prediction_frames:
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS, index=False)
        print(f"Finished {holdout}: {len(holdout_rows)} policy rows", flush=True)

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(results)
    deltas = make_deltas(predictions)
    deployment = make_deployment_output(summary, deltas)
    results.to_csv(RESULTS, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    summary.to_csv(SUMMARY, index=False)
    deltas.to_csv(DELTAS, index=False)
    deployment.to_csv(DEPLOYMENT, index=False)
    write_report(summary, deltas, deployment)
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved deployment output: {DEPLOYMENT}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
