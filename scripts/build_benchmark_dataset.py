from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3, JAPAN
from src.network_exposure import attach_iso3_codes, compute_network_exposure
from src.portwatch import fetch_country_daily
from src.wits import (
    build_partner_dependency_weights,
    fetch_partner_trade,
    fetch_partner_trade_by_product,
)


DEFAULT_GENERAL_GKG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_event_features_2024-10-01_2025-03-31.csv"
)
DEFAULT_STRICT_GKG = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_channel_specific_strict_event_features_2024-10-01_2025-03-31.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "japan_container_event_network_benchmark.csv"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "benchmark_dataset_summary.md"

ME_PRODUCT_CODE = "84-85_MachElec"
TARGET_COL = "abnormal_next_week_container"
RANDOM_SEED = 42


def build_weekly_container_base(daily: pd.DataFrame) -> pd.DataFrame:
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
    weekly["abnormal_threshold"] = (
        weekly["rolling_mean_12w"] - 1.5 * weekly["rolling_std_12w"]
    )
    weekly[TARGET_COL] = (
        weekly["next_week_container"] < weekly["abnormal_threshold"]
    ).astype(int)

    for lag in [1, 2, 4]:
        weekly[f"lag_container_{lag}w"] = weekly["portcalls_container"].shift(lag)

    for window in [4, 8, 12]:
        weekly[f"rolling_mean_container_{window}w"] = (
            weekly["portcalls_container"].rolling(window).mean()
        )
        weekly[f"rolling_std_container_{window}w"] = (
            weekly["portcalls_container"].rolling(window).std()
        )

    weekly["rolling_change_container_4w"] = (
        weekly["portcalls_container"] - weekly["portcalls_container"].shift(4)
    )
    weekly["month"] = weekly["week"].dt.month
    weekly["quarter"] = weekly["week"].dt.quarter

    required = [
        "next_week_container",
        "rolling_mean_12w",
        "rolling_std_12w",
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


def load_partner_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_week"])
    events = attach_iso3_codes(events, GDELT_TO_ISO3)
    events = events.rename(columns={"iso3": "partner_iso3"})
    return events.loc[events["partner_iso3"].ne(JAPAN.iso3)].copy()


def build_simple_news_controls(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["tone_x_articles"] = out["avg_tone"] * out["article_count"]

    grouped = (
        out.groupby("event_week", as_index=False)
        .agg(
            news_article_count=("article_count", "sum"),
            tone_x_articles=("tone_x_articles", "sum"),
            unweighted_negative_exposure=("negative_article_share", "mean"),
            unweighted_very_negative_exposure=("very_negative_article_share", "mean"),
            news_trade_transport_count=("trade_transport_count", "sum"),
            news_risk_theme_count=("risk_theme_count", "sum"),
        )
        .sort_values("event_week")
    )
    grouped["news_avg_tone"] = grouped["tone_x_articles"] / grouped["news_article_count"]
    return grouped.drop(columns=["tone_x_articles"])


def build_total_network_exposure(
    events: pd.DataFrame,
    partner_iso3: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade = fetch_partner_trade(JAPAN.wits_reporter, year=2023)
    weights = build_partner_dependency_weights(trade, partner_iso3=partner_iso3)
    event_for_network = events.rename(columns={"partner_iso3": "iso3"})
    _, exposure = compute_network_exposure(event_for_network, weights)
    exposure = exposure.rename(
        columns={
            "network_negative_exposure": "total_network_negative_exposure",
            "network_very_negative_exposure": "total_network_very_negative_exposure",
            "network_risk_theme_exposure": "total_network_risk_theme_exposure",
            "network_trade_transport_exposure": "total_network_trade_transport_exposure",
            "partner_article_count": "total_network_partner_article_count",
        }
    )
    return weights, exposure


def build_me_weights(partner_iso3: list[str]) -> pd.DataFrame:
    trade = fetch_partner_trade_by_product(
        JAPAN.wits_reporter,
        year=2023,
        product=ME_PRODUCT_CODE,
    )
    weights = build_partner_dependency_weights(trade, partner_iso3=partner_iso3)
    return weights.rename(columns={"import_dependency_share": "me_dependency_share"})


def build_me_strict_exposure(
    strict_events: pd.DataFrame,
    me_weights: pd.DataFrame,
) -> pd.DataFrame:
    me = strict_events.merge(
        me_weights[["partner_iso3", "me_dependency_share"]],
        on="partner_iso3",
        how="inner",
    )

    weight_table = (
        me[["partner_iso3", "me_dependency_share"]]
        .drop_duplicates()
        .sort_values("partner_iso3")
        .reset_index(drop=True)
    )
    n_partners = len(weight_table)
    rng = np.random.default_rng(RANDOM_SEED)
    weight_table["me_equal_weight"] = 1 / n_partners
    weight_table["me_shuffled_weight"] = rng.permutation(
        weight_table["me_dependency_share"].to_numpy()
    )
    random_raw = rng.random(n_partners)
    weight_table["me_random_weight"] = random_raw / random_raw.sum()

    me = me.merge(
        weight_table[
            [
                "partner_iso3",
                "me_equal_weight",
                "me_shuffled_weight",
                "me_random_weight",
            ]
        ],
        on="partner_iso3",
        how="left",
    )

    signal = "machinery_electronics_disruption_very_negative_share"
    me["me_strict_network_contribution"] = me[signal] * me["me_dependency_share"]
    me["me_strict_equal_contribution"] = me[signal] * me["me_equal_weight"]
    me["me_strict_shuffled_contribution"] = me[signal] * me["me_shuffled_weight"]
    me["me_strict_random_contribution"] = me[signal] * me["me_random_weight"]

    return (
        me.groupby("event_week", as_index=False)
        .agg(
            me_strict_network_exposure=("me_strict_network_contribution", "sum"),
            me_strict_unweighted_exposure=(signal, "mean"),
            me_strict_equal_exposure=("me_strict_equal_contribution", "sum"),
            me_strict_shuffled_exposure=("me_strict_shuffled_contribution", "sum"),
            me_strict_random_exposure=("me_strict_random_contribution", "sum"),
            me_strict_article_count=(
                "machinery_electronics_disruption_article_count",
                "sum",
            ),
        )
        .sort_values("event_week")
    )


def write_summary(path: Path, dataset: pd.DataFrame, feature_groups: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = dataset.isna().sum()
    missing = missing[missing.gt(0)].sort_values(ascending=False)
    missing_text = "\n".join(f"- `{k}`: {v}" for k, v in missing.items()) or "- None"

    feature_text = []
    for group, cols in feature_groups.items():
        feature_text.append(f"### {group}\n")
        feature_text.extend(f"- `{col}`" for col in cols)
        feature_text.append("")

    content = f"""# Benchmark Dataset Summary

## Dataset

- File: `data/processed/japan_container_event_network_benchmark.csv`
- Target country: Japan
- Primary target: `{TARGET_COL}`
- Target definition: next-week `portcalls_container` below current rolling 12-week mean minus 1.5 rolling standard deviations.
- Date range: {dataset["week"].min().date()} to {dataset["week"].max().date()}
- Weekly observations: {len(dataset)}
- Positive labels: {int(dataset[TARGET_COL].sum())}
- Positive rate: {dataset[TARGET_COL].mean():.3f}

## Important Status Note

This dataset uses the currently cached GDELT window only. It is suitable for pipeline validation and exploratory modeling, but it is not yet sufficient for final IEEE BigData-style benchmark claims if the number of positive labels is small.

## Missingness

{missing_text}

## Feature Groups

{chr(10).join(feature_text)}
"""
    path.write_text(content, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    daily = fetch_country_daily(JAPAN.iso3)
    container_base = build_weekly_container_base(daily)

    general_events = load_partner_events(Path(args.general_gkg))
    strict_events = load_partner_events(Path(args.strict_gkg))

    partner_iso3 = sorted(
        set(general_events["partner_iso3"]).intersection(strict_events["partner_iso3"])
    )

    simple_news = build_simple_news_controls(general_events)
    _, total_network = build_total_network_exposure(general_events, partner_iso3)
    me_weights = build_me_weights(partner_iso3)
    me_exposure = build_me_strict_exposure(strict_events, me_weights)

    dataset = (
        container_base.merge(simple_news, left_on="week", right_on="event_week", how="inner")
        .drop(columns=["event_week"])
        .merge(total_network, left_on="week", right_on="event_week", how="inner")
        .drop(columns=["event_week"])
        .merge(me_exposure, left_on="week", right_on="event_week", how="inner")
        .drop(columns=["event_week"])
        .sort_values("week")
        .reset_index(drop=True)
    )

    feature_groups = {
        "M1 operational": [
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
        ],
        "M2 simple news controls": [
            "news_article_count",
            "news_avg_tone",
            "unweighted_negative_exposure",
            "unweighted_very_negative_exposure",
            "news_trade_transport_count",
            "news_risk_theme_count",
        ],
        "M3 unweighted machinery/electronics event signal": [
            "me_strict_unweighted_exposure",
            "me_strict_article_count",
        ],
        "M4 total-import network": [
            "total_network_negative_exposure",
            "total_network_very_negative_exposure",
            "total_network_risk_theme_exposure",
            "total_network_trade_transport_exposure",
        ],
        "M5 machinery/electronics strict network": [
            "me_strict_network_exposure",
        ],
        "M6 placebo network": [
            "me_strict_equal_exposure",
            "me_strict_shuffled_exposure",
            "me_strict_random_exposure",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    write_summary(Path(args.summary), dataset, feature_groups)

    print(f"Saved dataset: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Positive labels: {int(dataset[TARGET_COL].sum())}")
    print(f"Positive rate: {dataset[TARGET_COL].mean():.3f}")
    print(f"Saved summary: {args.summary}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-gkg", default=str(DEFAULT_GENERAL_GKG))
    parser.add_argument("--strict-gkg", default=str(DEFAULT_STRICT_GKG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
