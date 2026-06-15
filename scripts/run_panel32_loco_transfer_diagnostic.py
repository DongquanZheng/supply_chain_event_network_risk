from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_panel32_portwatch_chokepoint_benchmark as pwc  # noqa: E402
from scripts.run_panel32_hazard_memory_benchmark import HAZARD_FEATURES  # noqa: E402
from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    INTERACTION_FEATURES,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_NETWORK_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_loco_transfer_diagnostic.md"
RESULTS = TABLE_DIR / "panel32_loco_transfer_results.csv"
SUMMARY = TABLE_DIR / "panel32_loco_transfer_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_loco_transfer_predictions.csv"
ALERTS = TABLE_DIR / "panel32_loco_transfer_alerts.csv"
COUNTRY_ALERTS = TABLE_DIR / "panel32_loco_transfer_country_alerts.csv"

SEVERE_TARGET = pwc.SEVERE_TARGET
RANDOM_SEED = 42


def unique(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def split_loco(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2021-01-01") & (df["week"] < "2024-01-01")].copy()
    validation = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["ISO3"].eq(holdout)) & (df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, validation, test


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    raw = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    m7 = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES + TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES + INTERACTION_FEATURES
    route = pwc.prefixed_cols(df, "pwc_route_")
    route_delta = pwc.prefixed_cols(df, "pwc_route_minus_")
    global_cp = pwc.prefixed_cols(df, "pwc_global_")
    return {
        "OP": unique(base),
        "FULL": unique(base + m7),
        "FULL_HAZARD": unique(base + m7 + HAZARD_FEATURES),
        "GLOBAL_CP": unique(base + raw + global_cp),
        "EVENT_ROUTE": unique(base + m7 + route + route_delta),
        "RANDOM_PLACEBO": unique(base + raw + pwc.prefixed_cols(df, "pwc_random_")),
    }


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def fit_gb(train_x: pd.DataFrame, train_y: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=16,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )
    weights = compute_sample_weight(class_weight="balanced", y=train_y)
    model.fit(train_x, train_y, sample_weight=weights)
    return model


def fit_et(train_x: pd.DataFrame, train_y: pd.Series) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=90,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    return model


def rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(pct=True, method="average").to_numpy()


def blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    return left_weight * rank01(left) + (1.0 - left_weight) * rank01(right)


def score_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "gb":
        model = fit_gb(train[features], train[TARGET])
    elif model_name == "et":
        model = fit_et(train[features], train[TARGET])
    else:
        raise ValueError(model_name)
    return model.predict_proba(validation[features])[:, 1], model.predict_proba(test[features])[:, 1]


def run_holdout(df: pd.DataFrame, holdout: str) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_loco(df, holdout)
    if test.empty or validation.empty or train[TARGET].sum() < 5 or validation[TARGET].sum() < 5 or test[TARGET].sum() < 1:
        return [], []
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)

    candidate_specs = {
        "OP_gb": ("OP", "gb"),
        "FULL_gb": ("FULL", "gb"),
        "FULL_HAZARD_gb": ("FULL_HAZARD", "gb"),
        "GLOBAL_CP_gb": ("GLOBAL_CP", "gb"),
        "EVENT_ROUTE_et": ("EVENT_ROUTE", "et"),
        "RANDOM_PLACEBO_et": ("RANDOM_PLACEBO", "et"),
    }
    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for name, (group_name, model_name) in candidate_specs.items():
        features = usable_features(train, groups[group_name])
        val_scores[name], test_scores[name] = score_candidate(train, validation, test, features, model_name)

    val_scores["BLEND_full_eventroute_80_20"] = blend(val_scores["FULL_gb"], val_scores["EVENT_ROUTE_et"], 0.8)
    test_scores["BLEND_full_eventroute_80_20"] = blend(test_scores["FULL_gb"], test_scores["EVENT_ROUTE_et"], 0.8)
    val_scores["BLEND_full_hazard_80_20"] = blend(val_scores["FULL_gb"], val_scores["FULL_HAZARD_gb"], 0.8)
    test_scores["BLEND_full_hazard_80_20"] = blend(test_scores["FULL_gb"], test_scores["FULL_HAZARD_gb"], 0.8)
    val_scores["BLEND_full_random_80_20"] = blend(val_scores["FULL_gb"], val_scores["RANDOM_PLACEBO_et"], 0.8)
    test_scores["BLEND_full_random_80_20"] = blend(test_scores["FULL_gb"], test_scores["RANDOM_PLACEBO_et"], 0.8)

    val_metric_rows = []
    for candidate, val_score in val_scores.items():
        ordered = pd.DataFrame({TARGET: validation[TARGET].to_numpy(), "score": val_score}).sort_values(
            "score", ascending=False
        )
        val_metric_rows.append(
            {
                "candidate": candidate,
                "validation_pr_auc": average_precision_score(validation[TARGET], val_score),
                "validation_top10_hits": int(ordered.head(10)[TARGET].sum()),
                "validation_top25_hits": int(ordered.head(25)[TARGET].sum()),
            }
        )
    val_metrics = pd.DataFrame(val_metric_rows)
    selector_map = {
        "LOCOSEL_best_val_pr_auc": val_metrics.sort_values(
            ["validation_pr_auc", "validation_top25_hits"], ascending=False
        ).iloc[0]["candidate"],
        "LOCOSEL_best_val_top25": val_metrics.sort_values(
            ["validation_top25_hits", "validation_pr_auc"], ascending=False
        ).iloc[0]["candidate"],
    }

    policies = {
        "LOCO0_operational_gb": "OP_gb",
        "LOCO1_full_event_gb": "FULL_gb",
        "LOCO2_full_hazard_gb": "FULL_HAZARD_gb",
        "LOCO3_global_cp_gb": "GLOBAL_CP_gb",
        "LOCO4_event_route_et": "EVENT_ROUTE_et",
        "LOCO5_blend_full_eventroute": "BLEND_full_eventroute_80_20",
        "LOCO6_blend_full_hazard": "BLEND_full_hazard_80_20",
        "LOCO7_blend_full_random_placebo": "BLEND_full_random_80_20",
        **selector_map,
    }

    rows = []
    prediction_frames = []
    for policy, candidate in policies.items():
        val_score = val_scores[candidate]
        test_score = test_scores[candidate]
        threshold, val_f1 = select_threshold(validation[TARGET], val_score)
        main_scores = evaluate_predictions(test[TARGET].to_numpy(), test_score, threshold)
        severe_pr_auc = average_precision_score(test[SEVERE_TARGET], test_score)
        row = {
            "holdout_iso3": holdout,
            "holdout_country": test["country"].iloc[0],
            "policy": policy,
            "selected_candidate": candidate,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_positives": int(train[TARGET].sum()),
            "validation_positives": int(validation[TARGET].sum()),
            "test_positives": int(test[TARGET].sum()),
            "test_severe_positives": int(test[SEVERE_TARGET].sum()),
            "validation_pr_auc": float(
                val_metrics.loc[val_metrics["candidate"].eq(candidate), "validation_pr_auc"].iloc[0]
            ),
            "validation_top25_hits": int(
                val_metrics.loc[val_metrics["candidate"].eq(candidate), "validation_top25_hits"].iloc[0]
            ),
            "selected_threshold": threshold,
            "validation_f1": val_f1,
            **main_scores,
            "severe_pr_auc": severe_pr_auc,
        }
        pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
        pred["holdout_iso3"] = holdout
        pred["policy"] = policy
        pred["selected_candidate"] = candidate
        pred["predicted_probability"] = test_score
        for k in [5, 10, 25]:
            top = pred.sort_values("predicted_probability", ascending=False).head(k)
            row[f"main_top{k}_hits"] = int(top[TARGET].sum())
            row[f"severe_top{k}_hits"] = int(top[SEVERE_TARGET].sum())
        rows.append(row)
        prediction_frames.append(pred)
    return rows, prediction_frames


def alert_budget(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, holdout), frame in predictions.groupby(["policy", "holdout_iso3"], sort=False):
        for k in [5, 10, 25]:
            top = frame.sort_values("predicted_probability", ascending=False).head(k)
            rows.append(
                {
                    "policy": policy,
                    "holdout_iso3": holdout,
                    "top_k": k,
                    "main_hits": int(top[TARGET].sum()),
                    "severe_hits": int(top[SEVERE_TARGET].sum()),
                    "alerts": len(top),
                    "test_positives": int(frame[TARGET].sum()),
                    "test_severe_positives": int(frame[SEVERE_TARGET].sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("policy", as_index=False)
        .agg(
            holdouts=("holdout_iso3", "nunique"),
            mean_main_pr_auc=("pr_auc", "mean"),
            median_main_pr_auc=("pr_auc", "median"),
            mean_severe_pr_auc=("severe_pr_auc", "mean"),
            median_severe_pr_auc=("severe_pr_auc", "median"),
            mean_roc_auc=("roc_auc", "mean"),
            total_tp=("tp", "sum"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
        )
        .sort_values(["mean_main_pr_auc", "mean_severe_pr_auc"], ascending=False)
    )
    alert_summary = (
        alerts.groupby(["policy", "top_k"], as_index=False)
        .agg(main_hits=("main_hits", "sum"), severe_hits=("severe_hits", "sum"), alerts=("alerts", "sum"))
        .assign(main_precision=lambda d: d["main_hits"] / d["alerts"], severe_precision=lambda d: d["severe_hits"] / d["alerts"])
    )
    wide_hits = alert_summary.pivot(index="policy", columns="top_k", values=["main_hits", "severe_hits"]).reset_index()
    wide_hits.columns = [
        "policy" if col[0] == "policy" else f"{col[0]}_top{col[1]}" for col in wide_hits.columns.to_flat_index()
    ]
    return summary.merge(wide_hits, on="policy", how="left")


def country_alert_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, frame in predictions.groupby("policy", sort=False):
        for k in [10, 25]:
            top_frames = []
            for _, holdout_frame in frame.groupby("holdout_iso3", sort=False):
                top_frames.append(holdout_frame.sort_values("predicted_probability", ascending=False).head(k))
            top = pd.concat(top_frames, ignore_index=True)
            grouped = (
                top.groupby(["ISO3", "country"], as_index=False)
                .agg(alerts=("week", "size"), main_hits=(TARGET, "sum"), severe_hits=(SEVERE_TARGET, "sum"))
                .assign(policy=policy, top_k=k)
            )
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def write_report(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    alerts: pd.DataFrame,
    country_alerts: pd.DataFrame,
) -> None:
    top = summary.head(14)
    top25 = alerts.loc[alerts["top_k"].eq(25)].groupby("policy", as_index=False).agg(
        main_hits=("main_hits", "sum"), severe_hits=("severe_hits", "sum"), alerts=("alerts", "sum")
    )
    concentration = (
        country_alerts.loc[country_alerts["top_k"].eq(25)]
        .groupby("policy", as_index=False)
        .agg(
            alert_countries=("ISO3", "nunique"),
            hit_countries=("main_hits", lambda s: int((s > 0).sum())),
            max_alert_share=("alerts", lambda s: float(s.max() / s.sum())),
        )
    )
    content = f"""# Panel32 Leave-One-Country-Out Transfer Diagnostic

## Purpose

This experiment tests expanded32 transfer behavior when each country is held out completely. Models train on 2021-2023 rows from the other countries, validate on 2024 rows from the other countries, and test on the held-out country in 2025.

## Candidate Families

- Operational GB.
- Full event/network GB reference.
- Full event/network + hazard GB.
- Global official chokepoint GB.
- Event-route official chokepoint ExtraTrees.
- Fixed rank blends: full+event-route, full+hazard, and full+random-placebo.
- Validation-selected best PR-AUC and best top-25 candidates.

## Top Policy Summary

{top.to_markdown(index=False)}

## Top-25 Alert Budget

{top25.sort_values("main_hits", ascending=False).to_markdown(index=False)}

## Top-25 Country Concentration

{concentration.sort_values("max_alert_share").to_markdown(index=False)}

## Holdout Detail

{results.sort_values(["holdout_iso3", "policy"]).to_markdown(index=False)}

## Reading

Treat this as transfer evidence only. A useful expanded32 transfer candidate should beat the full-event GB reference on holdout mean PR-AUC or top-k hits and should not simply mirror the random-placebo blend. If no candidate clears that bar, the next transfer improvement needs stronger new data or a different hierarchical transfer model.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pwc.load_or_build_dataset()
    rows = []
    prediction_frames = []
    for holdout in sorted(df["ISO3"].unique()):
        holdout_rows, holdout_predictions = run_holdout(df, holdout)
        rows.extend(holdout_rows)
        prediction_frames.extend(holdout_predictions)
        if rows and prediction_frames:
            partial_results = pd.DataFrame(rows)
            partial_predictions = pd.concat(prediction_frames, ignore_index=True)
            partial_results.to_csv(RESULTS, index=False)
            partial_predictions.to_csv(PREDICTIONS, index=False)
        print(f"Finished {holdout}: {len(holdout_rows)} policy rows", flush=True)
    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    alerts = alert_budget(predictions)
    summary = summarize(results, alerts)
    country_alerts = country_alert_table(predictions)

    results.to_csv(RESULTS, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    alerts.to_csv(ALERTS, index=False)
    summary.to_csv(SUMMARY, index=False)
    country_alerts.to_csv(COUNTRY_ALERTS, index=False)
    write_report(summary, results, alerts, country_alerts)

    print(f"Saved results: {RESULTS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
