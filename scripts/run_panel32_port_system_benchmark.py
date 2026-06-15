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

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    INTERACTION_FEATURES,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_NETWORK_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)
from scripts.run_panel32_hazard_memory_benchmark import HAZARD_FEATURES  # noqa: E402
from scripts.run_panel32_portwatch_chokepoint_benchmark import (  # noqa: E402
    SEVERE_TARGET,
    fit_model,
    load_or_build_dataset as load_chokepoint_dataset,
    make_models,
    prefixed_cols,
    topk_hits,
    unique,
)


PORT_SYSTEM_WEEKLY = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_country_weekly.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_port_system_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_port_system_benchmark.md"
METRICS = TABLE_DIR / "panel32_port_system_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_port_system_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_port_system_predictions.csv"
ALERTS = TABLE_DIR / "panel32_port_system_alert_budget.csv"
KEY = TABLE_DIR / "panel32_port_system_key_contrasts.csv"
FAILURE_COUNTRIES = TABLE_DIR / "panel32_transfer_failure_country_diagnostics.csv"


PORT_SYSTEM_DISTRIBUTION = [
    "ps_top_ports",
    "ps_selected_container_coverage",
    "ps_selected_import_share_coverage",
    "ps_container_ports_active",
    "ps_container_top1_share",
    "ps_container_top3_share",
    "ps_container_hhi",
    "ps_container_active_ports",
    "ps_trade_container_top1_share",
    "ps_trade_container_top3_share",
    "ps_trade_container_hhi",
    "ps_trade_container_active_ports",
]

PORT_SYSTEM_ANOMALY = [
    "ps_container_neg_z52_max",
    "ps_container_neg_z52_sum",
    "ps_container_neg_z52_weighted",
    "ps_container_neg_z52_gt1_ports",
    "ps_container_neg_z52_gt2_ports",
    "ps_trade_container_neg_z52_max",
    "ps_trade_container_neg_z52_weighted",
    "ps_import_export_balance_weighted",
    "ps_top3_static_ports_neg_z52",
]

PORT_SYSTEM_LEVELS = [
    "ps_container_calls_selected_log",
    "ps_container_calls_selected_change_4w",
    "ps_total_calls_selected_log",
    "ps_total_calls_selected_change_4w",
    "ps_container_trade_selected_log",
    "ps_container_trade_selected_change_4w",
    "ps_container_ports_active_change_4w",
    "ps_container_neg_z52_weighted_change_4w",
    "ps_container_hhi_change_4w",
    "ps_trade_container_hhi_change_4w",
]

INTERACTION_ROOTS = [
    "operational_shortfall_12w",
    "negative_trend_4w",
    "network_very_negative_exposure",
    "me_network_strict_very_negative_exposure",
    "haz_current_low_depth",
    "haz_abnormal_count_4w",
    "pwc_route_physical_stress",
    "pwc_global_red_event_count",
]

PORT_MODIFIERS = [
    "ps_container_neg_z52_weighted",
    "ps_container_neg_z52_max",
    "ps_container_neg_z52_gt2_ports",
    "ps_container_hhi",
    "ps_container_top1_share",
    "ps_trade_container_neg_z52_weighted",
]


def load_port_system_weekly() -> pd.DataFrame:
    if not PORT_SYSTEM_WEEKLY.exists():
        raise FileNotFoundError(
            f"Missing {PORT_SYSTEM_WEEKLY}. Run scripts/fetch_portwatch_port_system_panel32.py first."
        )
    return pd.read_csv(PORT_SYSTEM_WEEKLY, parse_dates=["week"])


def add_port_system_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ps_cols = [col for col in out.columns if col.startswith("ps_")]
    out[ps_cols] = out[ps_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    interaction_data = {}
    for root in INTERACTION_ROOTS:
        if root not in out.columns:
            continue
        for modifier in PORT_MODIFIERS:
            if modifier in out.columns:
                interaction_data[f"{root}_x_{modifier}"] = out[root] * out[modifier]
    if interaction_data:
        out = pd.concat([out, pd.DataFrame(interaction_data, index=out.index)], axis=1)
    ps_cols = [col for col in out.columns if col.startswith("ps_") or "_x_ps_" in col]
    out[ps_cols] = out[ps_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.sort_values(["week", "ISO3"]).reset_index(drop=True)


def load_or_build_dataset() -> pd.DataFrame:
    if OUTPUT_DATASET.exists():
        return pd.read_csv(OUTPUT_DATASET, parse_dates=["week"])
    base = load_chokepoint_dataset()
    port_system = load_port_system_weekly()
    ps_cols = [col for col in port_system.columns if col.startswith("ps_")]
    out = base.merge(port_system[["ISO3", "week"] + ps_cols], on=["ISO3", "week"], how="left")
    missing = out.loc[out["ps_top_ports"].isna(), ["ISO3", "week"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(f"Missing port-system rows after merge:\n{missing.head(20)}")
    out = add_port_system_interactions(out)
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DATASET, index=False)
    return out


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def ps_interaction_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "_x_ps_" in col]


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base_features = OPERATIONAL_FEATURES + country_features
    raw = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    full_event_network = raw + TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES + INTERACTION_FEATURES
    route = prefixed_cols(df, "pwc_route_")
    route_delta = prefixed_cols(df, "pwc_route_minus_")
    global_cp = prefixed_cols(df, "pwc_global_")
    route_stress = unique(
        [
            col
            for col in route + route_delta + global_cp
            if any(key in col for key in ["active", "event", "red", "alert", "neg_z52", "physical_stress"])
        ]
    )
    distribution = available(df, PORT_SYSTEM_DISTRIBUTION)
    anomaly = available(df, PORT_SYSTEM_ANOMALY)
    levels = available(df, PORT_SYSTEM_LEVELS)
    port_system = unique(distribution + anomaly)
    port_full = unique(distribution + anomaly + levels)
    interactions = ps_interaction_cols(df)
    return {
        "P32PS0_full_event_network": unique(base_features + full_event_network),
        "P32PS1_operational_port_system": unique(base_features + port_system),
        "P32PS2_raw_port_system": unique(base_features + raw + port_system),
        "P32PS3_full_event_port_system": unique(base_features + full_event_network + port_full),
        "P32PS4_full_event_hazard_port_system": unique(
            base_features + full_event_network + HAZARD_FEATURES + port_full
        ),
        "P32PS5_chokepoint_port_system": unique(base_features + raw + route_stress + port_full),
        "P32PS6_full_physical_port_system": unique(
            base_features + full_event_network + HAZARD_FEATURES + route_stress + port_full
        ),
        "P32PS7_port_system_interactions": unique(
            base_features + full_event_network + HAZARD_FEATURES + route_stress + port_system + interactions
        ),
    }


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in frame.columns and pd.api.types.is_numeric_dtype(frame[f])]


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
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
            pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
            pred["fold"] = fold.name
            pred["feature_group"] = group_name
            pred["model"] = model_name
            pred["predicted_probability"] = test_proba
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
            for k in [10, 25, 50, 100]:
                row[f"main_top{k}_hits"] = topk_hits(pred, TARGET, k)
                row[f"severe_top{k}_hits"] = topk_hits(pred, SEVERE_TARGET, k)
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
        .assign(
            main_precision=lambda d: d["main_hits"] / d["alerts"],
            severe_precision=lambda d: d["severe_hits"] / d["alerts"],
        )
        .sort_values(["top_k_per_fold", "main_hits", "severe_hits"], ascending=[True, False, False])
    )


def key_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("advanced::sklearn_gradient_boosting", "P32PS0_full_event_network"),
        ("advanced::sklearn_gradient_boosting", "P32PS3_full_event_port_system"),
        ("advanced::sklearn_gradient_boosting", "P32PS4_full_event_hazard_port_system"),
        ("advanced::sklearn_gradient_boosting", "P32PS6_full_physical_port_system"),
        ("advanced::sklearn_gradient_boosting", "P32PS7_port_system_interactions"),
        ("advanced::extra_trees", "P32PS2_raw_port_system"),
        ("advanced::extra_trees", "P32PS6_full_physical_port_system"),
        ("advanced::xgboost", "P32PS3_full_event_port_system"),
        ("fast::random_forest", "P32PS2_raw_port_system"),
        ("fast::random_forest", "P32PS0_full_event_network"),
    ]
    rows = []
    for model, group in specs:
        match = summary.loc[summary["model"].eq(model) & summary["feature_group"].eq(group)]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return pd.DataFrame(rows)


def failure_country_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if not FAILURE_COUNTRIES.exists():
        return pd.DataFrame()
    failure = pd.read_csv(FAILURE_COUNTRIES)
    priority = failure.loc[failure["router_outcome_regime"].ne("router_pr_auc_and_top25_win"), "ISO3"].tolist()
    candidates = [
        ("advanced::sklearn_gradient_boosting", "P32PS0_full_event_network"),
        ("advanced::sklearn_gradient_boosting", "P32PS3_full_event_port_system"),
        ("advanced::sklearn_gradient_boosting", "P32PS6_full_physical_port_system"),
        ("advanced::sklearn_gradient_boosting", "P32PS7_port_system_interactions"),
        ("advanced::extra_trees", "P32PS6_full_physical_port_system"),
    ]
    rows = []
    for model, group in candidates:
        frame = predictions.loc[
            predictions["model"].eq(model)
            & predictions["feature_group"].eq(group)
            & predictions["ISO3"].isin(priority)
        ].copy()
        if frame.empty:
            continue
        for iso3, country_frame in frame.groupby("ISO3"):
            rows.append(
                {
                    "feature_group": group,
                    "model": model,
                    "ISO3": iso3,
                    "country": country_frame["country"].iloc[0],
                    "rows": len(country_frame),
                    "main_positives": int(country_frame[TARGET].sum()),
                    "severe_positives": int(country_frame[SEVERE_TARGET].sum()),
                    "main_top10_hits": int(country_frame.sort_values("predicted_probability", ascending=False).head(10)[TARGET].sum()),
                    "main_top25_hits": int(country_frame.sort_values("predicted_probability", ascending=False).head(25)[TARGET].sum()),
                    "severe_top10_hits": int(country_frame.sort_values("predicted_probability", ascending=False).head(10)[SEVERE_TARGET].sum()),
                    "severe_top25_hits": int(country_frame.sort_values("predicted_probability", ascending=False).head(25)[SEVERE_TARGET].sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, alerts: pd.DataFrame, key: pd.DataFrame, failure_table: pd.DataFrame) -> None:
    top = summary.head(25)
    top25 = alerts.loc[alerts["top_k_per_fold"].eq(25)].head(20)
    failure_preview = failure_table.head(40) if not failure_table.empty else pd.DataFrame()
    content = f"""# Panel32 Port-System Benchmark

## Purpose

This experiment tests whether the expanded32 PortWatch selected-port daily data add direct operational value beyond the current expanded32 full-event/network, hazard, and official chokepoint references. The new features describe selected-port concentration, localized port-call shortfalls, trade-flow balance, and interactions with hazard/event/route stress.

## Dataset

- File: `data/processed/multicountry32_port_system_benchmark.csv`
- Port-system feature cache: `data/interim/portwatch_port_system_panel32_country_weekly.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}
- Port-system feature columns: {len([col for col in df.columns if col.startswith("ps_") or "_x_ps_" in col])}

## Evaluation

- Temporal rolling-origin validation with test years 2023, 2024, and 2025.
- Thresholds are selected on the immediately preceding validation year only.
- Main ranking metric: PR-AUC; alert-budget top-k hits are secondary.
- Model families follow the expanded32 official chokepoint branch.

## Top Mean PR-AUC Results

{top[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "severe_top50_hits"]].to_markdown(index=False)}

## Top-25 Alert Leaders

{top25[["feature_group", "model", "main_hits", "severe_hits", "alerts", "main_precision", "severe_precision"]].to_markdown(index=False)}

## Key Contrasts

{key[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "severe_top10_hits", "severe_top25_hits"]].to_markdown(index=False)}

## Priority-Country Preview

{failure_preview.to_markdown(index=False)}

## Reading

Promote this branch only if port-system candidates improve over the expanded32 full-event/network reference (`0.2095` main PR-AUC, main top-25 `33`, severe top-25 `23`) or materially improve severe/failure-country alerting. If port-system features do not improve the frontier, keep the cache as mechanism data and move to more direct waiting-time, AIS queue, schedule, closure/labor, or route-duration evidence.
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
    failure_table = failure_country_table(predictions)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    alerts.to_csv(ALERTS, index=False)
    key.to_csv(KEY, index=False)
    if not failure_table.empty:
        failure_table.to_csv(TABLE_DIR / "panel32_port_system_failure_country_alerts.csv", index=False)
    write_report(df, summary, alerts, key, failure_table)
    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
