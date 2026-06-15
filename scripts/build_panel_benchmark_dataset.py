from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3, ISO3_TO_GDELT
from src.portwatch import fetch_country_daily
from src.wits import (
    build_partner_dependency_weights,
    fetch_partner_trade,
    fetch_partner_trade_by_product,
)


EVENT_PATH = PROJECT_ROOT / "data" / "interim" / "gkg_partner_event_features_2021-01-01_2025-12-31.csv"
ME_STRICT_EVENT_PATH = PROJECT_ROOT / "data" / "interim" / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "panel_benchmark_dataset_summary.md"

TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42
ME_PRODUCT_CODE = "84-85_MachElec"

EVENT_COLUMNS = [
    "article_count",
    "avg_tone",
    "negative_article_share",
    "very_negative_article_share",
    "trade_transport_count",
    "risk_theme_count",
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
    weekly["year"] = weekly["week"].dt.year

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


def build_operational_panel(countries: list[str]) -> pd.DataFrame:
    frames = []
    for iso3 in countries:
        daily = fetch_country_daily(iso3, timeout=90)
        frames.append(build_country_weekly(daily))
    return pd.concat(frames, ignore_index=True)


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_week"])
    events["ISO3"] = events["code"].map(GDELT_TO_ISO3)
    return events.dropna(subset=["ISO3"]).copy()


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
                    "external_avg_tone": (external["avg_tone"] * external["article_count"]).sum() / article_count,
                    "external_negative_article_share": (
                        external["negative_article_share"] * external["article_count"]
                    ).sum()
                    / article_count,
                    "external_very_negative_article_share": (
                        external["very_negative_article_share"] * external["article_count"]
                    ).sum()
                    / article_count,
                    "external_trade_transport_count": external["trade_transport_count"].sum(),
                    "external_risk_theme_count": external["risk_theme_count"].sum(),
                }
            )
    return pd.DataFrame(rows)


def build_dependency_weights(countries: list[str]) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(RANDOM_SEED)

    for target_iso3 in countries:
        partners = [iso3 for iso3 in countries if iso3 != target_iso3]
        trade = fetch_partner_trade(target_iso3, year=2023)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        weights = weights.rename(columns={"partner_iso3": "partner_iso3"})
        weights["ISO3"] = target_iso3
        weights["equal_weight"] = 1 / len(partners)
        weights["shuffled_weight"] = rng.permutation(weights["import_dependency_share"].to_numpy())
        raw_random = rng.random(len(weights))
        weights["random_weight"] = raw_random / raw_random.sum()
        frames.append(
            weights[
                [
                    "ISO3",
                    "partner_iso3",
                    "import_dependency_share",
                    "equal_weight",
                    "shuffled_weight",
                    "random_weight",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True)


def build_me_dependency_weights(countries: list[str]) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(RANDOM_SEED + 7)

    for target_iso3 in countries:
        partners = [iso3 for iso3 in countries if iso3 != target_iso3]
        trade = fetch_partner_trade_by_product(target_iso3, year=2023, product=ME_PRODUCT_CODE)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        weights["ISO3"] = target_iso3
        weights["me_equal_weight"] = 1 / len(weights)
        weights["me_shuffled_weight"] = rng.permutation(weights["import_dependency_share"].to_numpy())
        raw_random = rng.random(len(weights))
        weights["me_random_weight"] = raw_random / raw_random.sum()
        frames.append(
            weights[
                [
                    "ISO3",
                    "partner_iso3",
                    "import_dependency_share",
                    "me_equal_weight",
                    "me_shuffled_weight",
                    "me_random_weight",
                ]
            ].rename(columns={"import_dependency_share": "me_dependency_share"})
        )

    return pd.concat(frames, ignore_index=True)


def weighted_event_controls(events: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    partner_events = events.rename(columns={"ISO3": "partner_iso3"}).copy()
    joined = weights.merge(partner_events, on="partner_iso3", how="inner")

    specs = {
        "network": "import_dependency_share",
        "equal": "equal_weight",
        "shuffled": "shuffled_weight",
        "random": "random_weight",
    }
    signal_cols = [
        "negative_article_share",
        "very_negative_article_share",
        "trade_transport_count",
        "risk_theme_count",
    ]
    for prefix, weight_col in specs.items():
        for signal in signal_cols:
            joined[f"{prefix}_{signal}_contribution"] = joined[weight_col] * joined[signal]

    agg = {
        "network_partner_article_count": ("article_count", "sum"),
    }
    for prefix in specs:
        for signal in signal_cols:
            out_name = signal.replace("article_share", "exposure").replace("_count", "_exposure")
            agg[f"{prefix}_{out_name}"] = (f"{prefix}_{signal}_contribution", "sum")

    return (
        joined.groupby(["event_week", "ISO3"], as_index=False)
        .agg(**agg)
        .sort_values(["event_week", "ISO3"])
    )


def load_me_strict_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_week"])
    events["ISO3"] = events["code"].map(GDELT_TO_ISO3)
    return events.dropna(subset=["ISO3"]).copy()


def build_external_me_controls(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    countries = sorted(events["ISO3"].dropna().unique())
    signal = "machinery_electronics_disruption_very_negative_share"
    count = "machinery_electronics_disruption_article_count"

    for week, week_events in events.groupby("event_week"):
        for target_iso3 in countries:
            external = week_events.loc[week_events["ISO3"].ne(target_iso3)]
            denom = external[count].sum()
            weighted_signal = (
                (external[signal] * external[count]).sum() / denom if denom else 0
            )
            rows.append(
                {
                    "event_week": week,
                    "ISO3": target_iso3,
                    "external_me_strict_very_negative_exposure": weighted_signal,
                    "external_me_strict_article_count": denom,
                }
            )

    return pd.DataFrame(rows)


def weighted_me_strict_controls(events: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    partner_events = events.rename(columns={"ISO3": "partner_iso3"}).copy()
    joined = weights.merge(partner_events, on="partner_iso3", how="inner")

    signal = "machinery_electronics_disruption_very_negative_share"
    count = "machinery_electronics_disruption_article_count"
    specs = {
        "me_network": "me_dependency_share",
        "me_equal": "me_equal_weight",
        "me_shuffled": "me_shuffled_weight",
        "me_random": "me_random_weight",
    }
    for prefix, weight_col in specs.items():
        joined[f"{prefix}_strict_contribution"] = joined[weight_col] * joined[signal]

    return (
        joined.groupby(["event_week", "ISO3"], as_index=False)
        .agg(
            me_network_strict_very_negative_exposure=("me_network_strict_contribution", "sum"),
            me_equal_strict_very_negative_exposure=("me_equal_strict_contribution", "sum"),
            me_shuffled_strict_very_negative_exposure=("me_shuffled_strict_contribution", "sum"),
            me_random_strict_very_negative_exposure=("me_random_strict_contribution", "sum"),
            me_network_strict_article_count=(count, "sum"),
        )
        .sort_values(["event_week", "ISO3"])
    )


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9
    std = out["rolling_std_container_12w"].replace(0, np.nan)
    out["operational_shortfall_12w"] = (out["rolling_mean_container_12w"] - out["portcalls_container"]) / std
    out["negative_trend_4w"] = -out["rolling_change_container_4w"] / std
    out["network_minus_equal_very_negative"] = (
        out["network_very_negative_exposure"] - out["equal_very_negative_exposure"]
    )
    out["network_to_equal_very_negative_ratio"] = (
        out["network_very_negative_exposure"] / (out["equal_very_negative_exposure"] + eps)
    )
    out["shortfall_x_network_very_negative"] = (
        out["operational_shortfall_12w"] * out["network_very_negative_exposure"]
    )
    out["trend_x_network_very_negative"] = (
        out["negative_trend_4w"] * out["network_very_negative_exposure"]
    )
    out["shortfall_x_external_very_negative"] = (
        out["operational_shortfall_12w"] * out["external_very_negative_article_share"]
    )
    out["trend_x_external_very_negative"] = (
        out["negative_trend_4w"] * out["external_very_negative_article_share"]
    )
    if "me_network_strict_very_negative_exposure" in out:
        out["me_network_minus_equal_strict"] = (
            out["me_network_strict_very_negative_exposure"]
            - out["me_equal_strict_very_negative_exposure"]
        )
        out["shortfall_x_me_network_strict"] = (
            out["operational_shortfall_12w"]
            * out["me_network_strict_very_negative_exposure"]
        )
        out["trend_x_me_network_strict"] = (
            out["negative_trend_4w"]
            * out["me_network_strict_very_negative_exposure"]
        )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    countries = sorted(set(GDELT_TO_ISO3.values()))
    operational = build_operational_panel(countries)
    events = load_events(Path(args.events))
    me_events = load_me_strict_events(Path(args.me_events))
    external_events = build_external_event_controls(events)
    weights = build_dependency_weights(countries)
    weighted_events = weighted_event_controls(events, weights)
    external_me_events = build_external_me_controls(me_events)
    me_weights = build_me_dependency_weights(countries)
    weighted_me_events = weighted_me_strict_controls(me_events, me_weights)

    dataset = (
        operational.merge(
            events[["event_week", "ISO3"] + EVENT_COLUMNS],
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
            weighted_events,
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .merge(
            me_events[
                [
                    "event_week",
                    "ISO3",
                    "machinery_electronics_disruption_very_negative_share",
                    "machinery_electronics_disruption_article_count",
                ]
            ].rename(
                columns={
                    "machinery_electronics_disruption_very_negative_share": "own_me_strict_very_negative_share",
                    "machinery_electronics_disruption_article_count": "own_me_strict_article_count",
                }
            ),
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .merge(
            external_me_events,
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .merge(
            weighted_me_events,
            left_on=["week", "ISO3"],
            right_on=["event_week", "ISO3"],
            how="inner",
        )
        .drop(columns=["event_week"])
        .sort_values(["week", "ISO3"])
        .reset_index(drop=True)
    )
    dataset["gdelt_code"] = dataset["ISO3"].map(ISO3_TO_GDELT)
    dataset = add_interactions(dataset)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    summary_file = output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output
    write_summary(Path(args.summary), dataset, summary_file)
    print(f"Saved dataset: {output}")
    print(f"Rows: {len(dataset)}")
    print(f"Positive labels: {int(dataset[TARGET].sum())}")
    print(f"Positive rate: {dataset[TARGET].mean():.3f}")
    return dataset


def write_summary(path: Path, dataset: pd.DataFrame, dataset_file: Path | None = None) -> None:
    country_summary = (
        dataset.groupby(["ISO3", "country"], as_index=False)
        .agg(rows=("week", "size"), positives=(TARGET, "sum"))
        .assign(positive_rate=lambda d: d["positives"] / d["rows"])
    )
    year_summary = (
        dataset.groupby(dataset["week"].dt.year)
        .agg(rows=("week", "size"), positives=(TARGET, "sum"))
        .reset_index()
        .rename(columns={"week": "year"})
    )
    year_summary["positive_rate"] = year_summary["positives"] / year_summary["rows"]
    missing = dataset.isna().sum()
    missing = missing[missing.gt(0)]
    missing_text = missing.to_markdown() if len(missing) else "No missing values."

    content = f"""# Panel Benchmark Dataset Summary

## Dataset

- File: `{dataset_file.as_posix() if dataset_file else "data/processed/multicountry_container_event_network_benchmark.csv"}`
- Countries: {dataset["ISO3"].nunique()}
- Rows: {len(dataset)}
- Positive labels: {int(dataset[TARGET].sum())}
- Positive rate: {dataset[TARGET].mean():.3f}
- Week range: {dataset["week"].min().date()} to {dataset["week"].max().date()}
- Target: `{TARGET}`, defined as next-week container port calls below the current rolling 12-week mean minus 1.5 standard deviations.

## Feature Blocks

- Operational container time-series features
- Own-country GDELT event controls
- External unweighted partner-event controls
- Total-import network-weighted partner-event exposure
- Equal, shuffled, and random-weight placebo exposure
- Machinery/electronics strict event-network exposure
- Operational vulnerability and event-network interaction features

## Country Summary

{country_summary.to_markdown(index=False)}

## Year Summary

{year_summary.to_markdown(index=False)}

## Missingness

{missing_text}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(EVENT_PATH))
    parser.add_argument("--me-events", default=str(ME_STRICT_EVENT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
