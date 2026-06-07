from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3, ISO3_TO_GDELT
from src.portwatch import fetch_country_daily
from src.wits import build_partner_dependency_weights, fetch_partner_trade


EVENT_PATH = PROJECT_ROOT / "data" / "interim" / "gkg_partner_event_features_2021-01-01_2025-12-31.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "multicountry_container_m1_m2_panel.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_m1_m2_sanity.md"

TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42

OPERATIONAL = [
    "portcalls_container",
    "lag_container_1w",
    "lag_container_2w",
    "lag_container_4w",
    "rolling_mean_container_4w",
    "rolling_std_container_4w",
    "rolling_mean_container_8w",
    "rolling_std_container_8w",
    "rolling_mean_container_12w",
    "rolling_std_container_12w",
    "rolling_change_container_4w",
    "month",
    "quarter",
]

EVENT_CONTROLS = [
    "article_count",
    "avg_tone",
    "negative_article_share",
    "very_negative_article_share",
    "trade_transport_count",
    "risk_theme_count",
]

EXTERNAL_EVENT_CONTROLS = [
    "external_article_count",
    "external_avg_tone",
    "external_negative_article_share",
    "external_very_negative_article_share",
    "external_trade_transport_count",
    "external_risk_theme_count",
]

NETWORK_EVENT_CONTROLS = [
    "network_negative_exposure",
    "network_very_negative_exposure",
    "network_trade_transport_exposure",
    "network_risk_theme_exposure",
    "network_partner_article_count",
    "equal_negative_exposure",
    "equal_very_negative_exposure",
    "equal_trade_transport_exposure",
    "equal_risk_theme_exposure",
]


def build_country_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = (
        daily.assign(week=daily["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby(["ISO3", "country", "week"], as_index=False)
        .agg(
            portcalls_container=("portcalls_container", "sum"),
            days_observed=("date", "nunique"),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["days_observed"].eq(7)].copy()

    weekly["next_week_container"] = weekly["portcalls_container"].shift(-1)
    weekly["rolling_mean_12w"] = weekly["portcalls_container"].rolling(12).mean()
    weekly["rolling_std_12w"] = weekly["portcalls_container"].rolling(12).std()
    weekly["abnormal_threshold"] = weekly["rolling_mean_12w"] - 1.5 * weekly["rolling_std_12w"]
    weekly[TARGET] = (weekly["next_week_container"] < weekly["abnormal_threshold"]).astype(int)

    for lag in [1, 2, 4]:
        weekly[f"lag_container_{lag}w"] = weekly["portcalls_container"].shift(lag)

    for window in [4, 8, 12]:
        weekly[f"rolling_mean_container_{window}w"] = weekly["portcalls_container"].rolling(window).mean()
        weekly[f"rolling_std_container_{window}w"] = weekly["portcalls_container"].rolling(window).std()

    weekly["rolling_change_container_4w"] = weekly["portcalls_container"] - weekly["portcalls_container"].shift(4)
    weekly["month"] = weekly["week"].dt.month
    weekly["quarter"] = weekly["week"].dt.quarter

    required = [
        "next_week_container",
        "lag_container_1w",
        "lag_container_2w",
        "lag_container_4w",
        "rolling_mean_container_4w",
        "rolling_std_container_4w",
        "rolling_mean_container_8w",
        "rolling_std_container_8w",
        "rolling_mean_container_12w",
        "rolling_std_container_12w",
        "rolling_change_container_4w",
    ]
    return weekly.dropna(subset=required).reset_index(drop=True)


def build_panel_dataset() -> pd.DataFrame:
    countries = sorted(set(GDELT_TO_ISO3.values()))
    weekly_frames = []
    for iso3 in countries:
        daily = fetch_country_daily(iso3, timeout=90)
        weekly_frames.append(build_country_weekly(daily))

    operational = pd.concat(weekly_frames, ignore_index=True)
    events = pd.read_csv(EVENT_PATH, parse_dates=["event_week"])
    events["ISO3"] = events["code"].map(GDELT_TO_ISO3)
    events = events.dropna(subset=["ISO3"]).copy()
    external_events = build_external_event_controls(events)
    network_events = build_total_import_network_controls(events)

    dataset = (
        operational.merge(
            events[["event_week", "ISO3"] + EVENT_CONTROLS],
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .merge(
            external_events,
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .merge(
            network_events,
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .sort_values(["week", "ISO3"])
        .reset_index(drop=True)
    )
    dataset["gdelt_code"] = dataset["ISO3"].map(ISO3_TO_GDELT)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)
    return dataset


def build_external_event_controls(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    countries = sorted(events["ISO3"].dropna().unique())
    for week, week_events in events.groupby("event_week"):
        for target_iso3 in countries:
            external = week_events.loc[week_events["ISO3"].ne(target_iso3)]
            article_count = external["article_count"].sum()
            rows.append(
                {
                    "event_week": week,
                    "ISO3": target_iso3,
                    "external_article_count": article_count,
                    "external_avg_tone": (
                        (external["avg_tone"] * external["article_count"]).sum()
                        / article_count
                    ),
                    "external_negative_article_share": (
                        (external["negative_article_share"] * external["article_count"]).sum()
                        / article_count
                    ),
                    "external_very_negative_article_share": (
                        (external["very_negative_article_share"] * external["article_count"]).sum()
                        / article_count
                    ),
                    "external_trade_transport_count": external["trade_transport_count"].sum(),
                    "external_risk_theme_count": external["risk_theme_count"].sum(),
                }
            )
    return pd.DataFrame(rows)


def build_total_import_network_controls(events: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(events["ISO3"].dropna().unique())
    weight_frames = []
    for target_iso3 in countries:
        partners = [iso3 for iso3 in countries if iso3 != target_iso3]
        trade = fetch_partner_trade(target_iso3, year=2023)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        weights = weights.rename(columns={"partner_iso3": "partner_iso3"})
        weights["ISO3"] = target_iso3
        weights["equal_weight"] = 1 / len(partners)
        weight_frames.append(weights[["ISO3", "partner_iso3", "import_dependency_share", "equal_weight"]])

    weights = pd.concat(weight_frames, ignore_index=True)
    partner_events = events.rename(columns={"ISO3": "partner_iso3"}).copy()
    joined = weights.merge(partner_events, on="partner_iso3", how="inner")

    joined["network_negative_contribution"] = (
        joined["import_dependency_share"] * joined["negative_article_share"]
    )
    joined["network_very_negative_contribution"] = (
        joined["import_dependency_share"] * joined["very_negative_article_share"]
    )
    joined["network_trade_transport_contribution"] = (
        joined["import_dependency_share"] * joined["trade_transport_count"]
    )
    joined["network_risk_theme_contribution"] = (
        joined["import_dependency_share"] * joined["risk_theme_count"]
    )
    joined["equal_negative_contribution"] = (
        joined["equal_weight"] * joined["negative_article_share"]
    )
    joined["equal_very_negative_contribution"] = (
        joined["equal_weight"] * joined["very_negative_article_share"]
    )
    joined["equal_trade_transport_contribution"] = (
        joined["equal_weight"] * joined["trade_transport_count"]
    )
    joined["equal_risk_theme_contribution"] = (
        joined["equal_weight"] * joined["risk_theme_count"]
    )

    return (
        joined.groupby(["event_week", "ISO3"], as_index=False)
        .agg(
            network_negative_exposure=("network_negative_contribution", "sum"),
            network_very_negative_exposure=("network_very_negative_contribution", "sum"),
            network_trade_transport_exposure=("network_trade_transport_contribution", "sum"),
            network_risk_theme_exposure=("network_risk_theme_contribution", "sum"),
            network_partner_article_count=("article_count", "sum"),
            equal_negative_exposure=("equal_negative_contribution", "sum"),
            equal_very_negative_exposure=("equal_very_negative_contribution", "sum"),
            equal_trade_transport_exposure=("equal_trade_transport_contribution", "sum"),
            equal_risk_theme_exposure=("equal_risk_theme_contribution", "sum"),
        )
        .sort_values(["event_week", "ISO3"])
    )


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= "2021-01-01") & (df["week"] < "2024-01-01")].copy()
    val = df[(df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, val, test


def add_country_dummies(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    countries = sorted(train["ISO3"].unique())
    frames = []
    for frame in [train, val, test]:
        out = frame.copy()
        for country in countries:
            out[f"country_{country}"] = (out["ISO3"] == country).astype(int)
        frames.append(out)
    return frames[0], frames[1], frames[2], [f"country_{country}" for country in countries]


def make_models() -> dict[str, object]:
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def select_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(name: str, features: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for model_name, model in make_models().items():
        model.fit(train[features], train[TARGET])
        val_proba = model.predict_proba(val[features])[:, 1]
        threshold = select_threshold(val[TARGET], val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        pred = (test_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test[TARGET], pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "feature_group": name,
                "model": model_name,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "train_positives": int(train[TARGET].sum()),
                "validation_positives": int(val[TARGET].sum()),
                "test_positives": int(test[TARGET].sum()),
                "selected_threshold": threshold,
                "roc_auc": roc_auc_score(test[TARGET], test_proba),
                "pr_auc": average_precision_score(test[TARGET], test_proba),
                "precision": precision_score(test[TARGET], pred, zero_division=0),
                "recall": recall_score(test[TARGET], pred, zero_division=0),
                "f1": f1_score(test[TARGET], pred, zero_division=0),
                "accuracy": (pred == test[TARGET].to_numpy()).mean(),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return rows


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["ISO3", "country"], as_index=False)
        .agg(
            rows=("week", "size"),
            positives=(TARGET, "sum"),
            min_week=("week", "min"),
            max_week=("week", "max"),
        )
        .assign(positive_rate=lambda d: d["positives"] / d["rows"])
        .sort_values("ISO3")
    )


def write_report(dataset: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    content = f"""# Panel M1/M2 Sanity Benchmark

## Purpose

This is a first multi-country panel check. It tests whether a country-week container disruption prediction task is viable before adding network exposure.

## Dataset

- File: `data/processed/multicountry_container_m1_m2_panel.csv`
- Countries: {dataset["ISO3"].nunique()}
- Rows: {len(dataset)}
- Positive labels: {int(dataset[TARGET].sum())}
- Positive rate: {dataset[TARGET].mean():.3f}
- Week range: {dataset["week"].min().date()} to {dataset["week"].max().date()}

## Split

- Train: 2021-2023
- Validation: 2024
- Test: 2025
- Threshold selection: validation F1

## Results

{metrics[["feature_group", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "fp", "tp"]].to_markdown(index=False)}

## Country Coverage

{summary.to_markdown(index=False)}

## Interpretation

If M2 improves over M1 in this panel setting, the panel benchmark is viable for adding network exposure. If M2 does not improve, the immediate priority should be improving event/NLP features before constructing full country-specific trade-network exposure.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    dataset = build_panel_dataset()
    train, val, test = temporal_split(dataset)
    train, val, test, country_features = add_country_dummies(train, val, test)

    feature_sets = {
        "P1_operational_panel": OPERATIONAL + country_features,
        "P2_operational_plus_country_events": OPERATIONAL + country_features + EVENT_CONTROLS,
        "P3_operational_plus_external_events": OPERATIONAL + country_features + EXTERNAL_EVENT_CONTROLS,
        "P4_operational_plus_country_and_external_events": OPERATIONAL
        + country_features
        + EVENT_CONTROLS
        + EXTERNAL_EVENT_CONTROLS,
        "P5_operational_plus_total_import_network": OPERATIONAL
        + country_features
        + NETWORK_EVENT_CONTROLS[:5],
        "P5b_operational_plus_equal_weight_placebo": OPERATIONAL
        + country_features
        + [
            "equal_negative_exposure",
            "equal_very_negative_exposure",
            "equal_trade_transport_exposure",
            "equal_risk_theme_exposure",
            "network_partner_article_count",
        ],
        "P6_operational_plus_external_and_network": OPERATIONAL
        + country_features
        + EXTERNAL_EVENT_CONTROLS
        + NETWORK_EVENT_CONTROLS,
        "P7_operational_plus_all_events_and_network": OPERATIONAL
        + country_features
        + EVENT_CONTROLS
        + EXTERNAL_EVENT_CONTROLS
        + NETWORK_EVENT_CONTROLS,
    }

    rows = []
    for name, features in feature_sets.items():
        rows.extend(evaluate(name, features, train, val, test))

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    summary = country_summary(dataset)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLE_DIR / "panel_m1_m2_sanity_metrics.csv"
    summary_path = TABLE_DIR / "panel_m1_m2_country_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(dataset, metrics, summary)

    print(f"Saved dataset: {DATASET_PATH}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Report: {REPORT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run()
