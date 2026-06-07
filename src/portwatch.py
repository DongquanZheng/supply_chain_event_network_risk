from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


PORTWATCH_REG_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Trade_Data_REG/FeatureServer/0/query"
)


def fetch_country_daily(iso3: str, timeout: int = 60) -> pd.DataFrame:
    rows = []
    offset = 0

    while True:
        params = {
            "where": f"ISO3='{iso3}'",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 1000,
            "orderByFields": "date ASC",
        }
        response = requests.get(PORTWATCH_REG_URL, params=params, timeout=timeout)
        response.raise_for_status()

        features = response.json().get("features", [])
        if not features:
            break

        rows.extend(feature["attributes"] for feature in features)
        if len(features) < 1000:
            break

        offset += len(features)

    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def build_weekly_operational_base(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        daily.assign(week=lambda d: d["date"].dt.to_period("W-SUN").apply(lambda x: x.start_time))
        .groupby(["ISO3", "country", "week"], as_index=False)
        .agg(
            portcalls=("portcalls", "sum"),
            portcalls_container=("portcalls_container", "sum"),
            import_value=("import", "sum"),
            export_value=("export", "sum"),
            shipment=("shipment", "sum"),
            days_observed=("date", "nunique"),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )

    weekly = weekly.loc[weekly["days_observed"].eq(7)].copy()
    weekly["next_week_portcalls"] = weekly["portcalls"].shift(-1)
    weekly["rolling_mean_12w"] = weekly["portcalls"].shift(1).rolling(12).mean()
    weekly["rolling_std_12w"] = weekly["portcalls"].shift(1).rolling(12).std()
    weekly["abnormal_threshold"] = (
        weekly["rolling_mean_12w"] - 1.5 * weekly["rolling_std_12w"]
    )
    weekly["abnormal_next_week"] = (
        weekly["next_week_portcalls"] < weekly["abnormal_threshold"]
    ).astype(int)

    weekly["lag_portcalls_1w"] = weekly["portcalls"].shift(1)
    weekly["lag_portcalls_2w"] = weekly["portcalls"].shift(2)
    weekly["lag_portcalls_4w"] = weekly["portcalls"].shift(4)
    weekly["rolling_mean_4w"] = weekly["portcalls"].shift(1).rolling(4).mean()
    weekly["rolling_mean_8w"] = weekly["portcalls"].shift(1).rolling(8).mean()
    weekly["rolling_std_4w"] = weekly["portcalls"].shift(1).rolling(4).std()
    weekly["rolling_std_8w"] = weekly["portcalls"].shift(1).rolling(8).std()
    weekly["rolling_change_4w"] = weekly["lag_portcalls_1w"] - weekly["lag_portcalls_4w"]
    weekly["month"] = weekly["week"].dt.month
    weekly["quarter"] = weekly["week"].dt.quarter

    required = [
        "next_week_portcalls",
        "rolling_mean_12w",
        "rolling_std_12w",
        "lag_portcalls_1w",
        "lag_portcalls_2w",
        "lag_portcalls_4w",
        "rolling_mean_4w",
        "rolling_mean_8w",
        "rolling_std_4w",
        "rolling_std_8w",
        "rolling_change_4w",
    ]
    return weekly.dropna(subset=required).reset_index(drop=True)


def load_operational_base(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week"])

