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

import scripts.run_panel_advanced_model_experiment as advanced  # noqa: E402
import scripts.run_panel_benchmark_models as base  # noqa: E402
from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    INTERACTION_FEATURES,
    ME_NETWORK_FEATURES,
    ME_PLACEBO_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_EQUAL_PLACEBO_FEATURES,
    TOTAL_NETWORK_FEATURES,
    TOTAL_RANDOM_PLACEBO_FEATURES,
    TOTAL_SHUFFLED_PLACEBO_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    precision_at_k,
    select_threshold,
    split_fold,
)


BASE_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_hazard_memory_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_hazard_memory_benchmark.md"
METRICS = TABLE_DIR / "panel32_hazard_memory_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_hazard_memory_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_hazard_memory_predictions.csv"
ALERTS = TABLE_DIR / "panel32_hazard_memory_alert_budget.csv"
KEY = TABLE_DIR / "panel32_hazard_memory_key_contrasts.csv"

SEVERE_TARGET = "abnormal_next_week_container_2p0sigma"

HAZARD_FEATURES = [
    "haz_current_abnormal_flag",
    "haz_current_low_z",
    "haz_current_low_depth",
    "haz_abnormal_lag1",
    "haz_abnormal_lag2",
    "haz_abnormal_lag4",
    "haz_abnormal_count_4w",
    "haz_abnormal_count_8w",
    "haz_abnormal_count_12w",
    "haz_low_depth_mean_4w",
    "haz_low_depth_mean_8w",
    "haz_low_depth_max_12w",
    "haz_current_shortfall_flag",
    "haz_shortfall_streak_4w",
    "haz_shortfall_streak_8w",
    "haz_recovery_from_low_1w",
    "haz_recovery_from_low_4w",
    "haz_rebound_positive_1w",
    "haz_rebound_positive_4w",
    "haz_level_rank_12w",
    "haz_recent_min_ratio_8w",
]

HAZARD_MODIFIERS = [
    "haz_current_abnormal_flag",
    "haz_current_low_depth",
    "haz_abnormal_count_4w",
    "haz_abnormal_count_8w",
    "haz_shortfall_streak_4w",
]

INTERACTION_ROOTS = [
    "article_count",
    "external_article_count",
    "external_very_negative_article_share",
    "network_very_negative_exposure",
    "me_network_strict_very_negative_exposure",
    "me_equal_strict_very_negative_exposure",
    "shortfall_x_network_very_negative",
    "shortfall_x_me_network_strict",
]


def unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def rolling_rank_last(values: pd.Series, window: int) -> pd.Series:
    def rank_last(arr: np.ndarray) -> float:
        if len(arr) <= 1 or np.all(pd.isna(arr)):
            return 0.5
        series = pd.Series(arr)
        return float((series.rank(method="average").iloc[-1] - 1.0) / max(len(series) - 1, 1))

    return values.rolling(window, min_periods=2).apply(rank_last, raw=True)


def add_hazard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ISO3", "week"]).reset_index(drop=True).copy()
    std = out["rolling_std_container_12w"].replace(0, np.nan)
    out["haz_current_low_z"] = ((out["rolling_mean_container_12w"] - out["portcalls_container"]) / std).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    out["haz_current_low_depth"] = out["haz_current_low_z"].clip(lower=0.0)
    out["haz_current_abnormal_flag"] = out["portcalls_container"].lt(out["abnormal_threshold"]).astype(float)
    out["haz_current_shortfall_flag"] = out["operational_shortfall_12w"].gt(0).astype(float)

    grouped_abnormal = out.groupby("ISO3", sort=False)["haz_current_abnormal_flag"]
    grouped_low_depth = out.groupby("ISO3", sort=False)["haz_current_low_depth"]
    grouped_shortfall = out.groupby("ISO3", sort=False)["haz_current_shortfall_flag"]
    grouped_portcalls = out.groupby("ISO3", sort=False)["portcalls_container"]

    out["haz_abnormal_lag1"] = grouped_abnormal.shift(1)
    out["haz_abnormal_lag2"] = grouped_abnormal.shift(2)
    out["haz_abnormal_lag4"] = grouped_abnormal.shift(4)
    for window in [4, 8, 12]:
        out[f"haz_abnormal_count_{window}w"] = grouped_abnormal.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum()
        )
    out["haz_low_depth_mean_4w"] = grouped_low_depth.transform(lambda s: s.rolling(4, min_periods=1).mean())
    out["haz_low_depth_mean_8w"] = grouped_low_depth.transform(lambda s: s.rolling(8, min_periods=1).mean())
    out["haz_low_depth_max_12w"] = grouped_low_depth.transform(lambda s: s.rolling(12, min_periods=1).max())
    out["haz_shortfall_streak_4w"] = grouped_shortfall.transform(lambda s: s.rolling(4, min_periods=1).sum())
    out["haz_shortfall_streak_8w"] = grouped_shortfall.transform(lambda s: s.rolling(8, min_periods=1).sum())
    lag_low = grouped_low_depth.shift(1)
    out["haz_recovery_from_low_1w"] = (lag_low - out["haz_current_low_depth"]).clip(lower=0.0)
    lag4_low = grouped_low_depth.shift(4)
    out["haz_recovery_from_low_4w"] = (lag4_low - out["haz_current_low_depth"]).clip(lower=0.0)
    out["haz_rebound_positive_1w"] = (out["portcalls_container"] - out["lag_container_1w"]).clip(lower=0.0)
    out["haz_rebound_positive_4w"] = (out["portcalls_container"] - out["lag_container_4w"]).clip(lower=0.0)
    out["haz_level_rank_12w"] = grouped_portcalls.transform(lambda s: rolling_rank_last(s, 12))
    min_8w = grouped_portcalls.transform(lambda s: s.rolling(8, min_periods=1).min())
    out["haz_recent_min_ratio_8w"] = (
        out["portcalls_container"] / min_8w.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    threshold = out["rolling_mean_container_12w"] - 2.0 * out["rolling_std_container_12w"]
    out[SEVERE_TARGET] = (out["next_week_container"] < threshold).astype(int)

    interaction_data = {}
    for root in INTERACTION_ROOTS:
        if root not in out.columns:
            continue
        for modifier in HAZARD_MODIFIERS:
            interaction_data[f"{root}_x_{modifier}"] = out[root] * out[modifier]
    if interaction_data:
        out = pd.concat([out, pd.DataFrame(interaction_data, index=out.index)], axis=1)

    hazard_cols = [col for col in out.columns if col.startswith("haz_") or "_x_haz_" in col]
    out[hazard_cols] = out[hazard_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.sort_values(["week", "ISO3"]).reset_index(drop=True)


def load_or_build_dataset() -> pd.DataFrame:
    if OUTPUT_DATASET.exists():
        return pd.read_csv(OUTPUT_DATASET, parse_dates=["week"])
    base_df = pd.read_csv(BASE_DATASET, parse_dates=["week"])
    out = add_hazard_features(base_df)
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DATASET, index=False)
    return out


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base_features = OPERATIONAL_FEATURES + country_features
    own_external = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    all_placebos = TOTAL_EQUAL_PLACEBO_FEATURES + TOTAL_SHUFFLED_PLACEBO_FEATURES + TOTAL_RANDOM_PLACEBO_FEATURES
    m7 = (
        OWN_NEWS_FEATURES
        + EXTERNAL_UNWEIGHTED_FEATURES
        + TOTAL_NETWORK_FEATURES
        + ME_NETWORK_FEATURES
        + INTERACTION_FEATURES
    )
    hazard_interactions = [f"{root}_x_{modifier}" for root in INTERACTION_ROOTS for modifier in HAZARD_MODIFIERS]
    return {
        "P32HAZ0_operational": unique(base_features),
        "P32HAZ1_operational_hazard": unique(base_features + HAZARD_FEATURES),
        "P32HAZ2_raw_external_hazard": unique(base_features + own_external + HAZARD_FEATURES),
        "P32HAZ3_me_network_hazard": unique(base_features + ME_NETWORK_FEATURES + HAZARD_FEATURES),
        "P32HAZ4_me_placebo_hazard": unique(base_features + ME_PLACEBO_FEATURES + HAZARD_FEATURES),
        "P32HAZ5_full_event_network": unique(base_features + m7),
        "P32HAZ6_full_event_hazard": unique(base_features + m7 + HAZARD_FEATURES),
        "P32HAZ7_full_event_hazard_interactions": unique(base_features + m7 + HAZARD_FEATURES + hazard_interactions),
        "P32HAZ8_placebo_hazard": unique(base_features + all_placebos + ME_PLACEBO_FEATURES + HAZARD_FEATURES),
    }


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in frame.columns and pd.api.types.is_numeric_dtype(frame[f])]


def make_models(train_y: pd.Series) -> dict[str, tuple[object, str]]:
    models = {f"lr_rf::{name}": (model, "plain") for name, model in base.make_models().items()}
    models.update({f"advanced::{name}": pair for name, pair in advanced.make_models(train_y).items()})
    return models


def fit_model(model, fit_mode: str, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if fit_mode == "plain":
        model.fit(train_x, train_y)
    else:
        advanced.fit_model(model, fit_mode, train_x, train_y)


def topk_hits(frame: pd.DataFrame, score_col: str, target_col: str, k: int) -> int:
    return int(frame.sort_values(score_col, ascending=False).head(k)[target_col].sum())


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(country_features)
    rows = []
    predictions = []
    for group_name, raw_features in groups.items():
        features = usable_features(train, raw_features)
        for model_name, (model, fit_mode) in make_models(train[TARGET]).items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            main_scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            severe_pr_auc = average_precision_score(test[SEVERE_TARGET], test_proba)
            row = {
                "fold": fold.name,
                "feature_group": group_name,
                "model": model_name,
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
                "severe_pr_auc": severe_pr_auc,
            }
            pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
            pred["fold"] = fold.name
            pred["feature_group"] = group_name
            pred["model"] = model_name
            pred["predicted_probability"] = test_proba
            for k in [10, 25, 50, 100]:
                row[f"main_top{k}_hits"] = topk_hits(pred, "predicted_probability", TARGET, k)
                row[f"severe_top{k}_hits"] = topk_hits(pred, "predicted_probability", SEVERE_TARGET, k)
            rows.append(row)
            predictions.append(pred)
    return rows, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "folds": ("fold", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "mean_f1": ("f1", "mean"),
    }
    for prefix in ["main", "severe"]:
        for k in [10, 25, 50, 100]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        metrics.groupby(["feature_group", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits", "mean_severe_pr_auc"], ascending=False)
    )


def alert_budget(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, model, fold), frame in predictions.groupby(["feature_group", "model", "fold"], sort=False):
        for k in [10, 25, 50, 100]:
            top = frame.sort_values("predicted_probability", ascending=False).head(k)
            rows.append(
                {
                    "feature_group": group,
                    "model": model,
                    "fold": fold,
                    "top_k_per_fold": k,
                    "main_hits": int(top[TARGET].sum()),
                    "severe_hits": int(top[SEVERE_TARGET].sum()),
                    "alerts": len(top),
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby(["feature_group", "model", "top_k_per_fold"], as_index=False)
        .agg(main_hits=("main_hits", "sum"), severe_hits=("severe_hits", "sum"), alerts=("alerts", "sum"))
        .assign(main_precision=lambda d: d["main_hits"] / d["alerts"], severe_precision=lambda d: d["severe_hits"] / d["alerts"])
        .sort_values(["top_k_per_fold", "main_hits", "severe_hits"], ascending=[True, False, False])
    )


def key_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("advanced::sklearn_gradient_boosting", "P32HAZ5_full_event_network"),
        ("advanced::sklearn_gradient_boosting", "P32HAZ6_full_event_hazard"),
        ("advanced::sklearn_gradient_boosting", "P32HAZ7_full_event_hazard_interactions"),
        ("advanced::sklearn_gradient_boosting", "P32HAZ0_operational"),
        ("advanced::xgboost", "P32HAZ3_me_network_hazard"),
        ("advanced::xgboost", "P32HAZ0_operational"),
        ("advanced::extra_trees", "P32HAZ1_operational_hazard"),
        ("advanced::extra_trees", "P32HAZ8_placebo_hazard"),
        ("lr_rf::random_forest", "P32HAZ2_raw_external_hazard"),
        ("lr_rf::random_forest", "P32HAZ0_operational"),
    ]
    rows = []
    for model, group in specs:
        match = summary.loc[summary["model"].eq(model) & summary["feature_group"].eq(group)]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, alerts: pd.DataFrame, key: pd.DataFrame) -> None:
    top = summary.head(25)
    top25 = alerts.loc[alerts["top_k_per_fold"].eq(25)].head(20)
    content = f"""# Panel32 Hazard-Memory Benchmark

## Purpose

This branch adds leakage-safe operational hazard-memory features to the expanded32 panel: current low-activity depth, lagged abnormal flags, recent abnormal counts, shortfall streaks, recovery/rebound, local level rank, and interactions with event/network signals. It tests whether the next candidate signal should come from operational state memory rather than more selector tuning over M1-M7.

## Dataset

- File: `data/processed/multicountry32_hazard_memory_benchmark.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Current abnormal rows: {int(df["haz_current_abnormal_flag"].sum())}
- Rows with prior 4-week abnormal history: {int(df["haz_abnormal_count_4w"].gt(0).sum())}

## Top Mean PR-AUC Results

{top[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "severe_top50_hits"]].to_markdown(index=False)}

## Top-25 Alert Leaders

{top25[["feature_group", "model", "main_hits", "severe_hits", "alerts", "main_precision", "severe_precision"]].to_markdown(index=False)}

## Key Contrasts

{key[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "severe_top10_hits", "severe_top25_hits"]].to_markdown(index=False)}

## Reading

Hazard memory is useful only if it improves over the current expanded32 fixed GB full-event/network reference (`0.2095` main PR-AUC, main top-25 `33`, severe top-25 `23`) or creates a new severe/top-k candidate without relying on placebo-like features. If hazard variants do not beat that reference, the next expanded32 breakthrough should come from physical chokepoint/route, LOCO transfer design, or new operational congestion/schedule data.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_dataset()
    rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(fold, df)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(metrics)
    alerts = alert_budget(predictions)
    key = key_contrasts(summary)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    alerts.to_csv(ALERTS, index=False)
    key.to_csv(KEY, index=False)
    write_report(df, summary, alerts, key)
    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved alerts: {ALERTS}")
    print(f"Saved key contrasts: {KEY}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
