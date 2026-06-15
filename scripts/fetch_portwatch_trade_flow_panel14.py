from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portwatch import fetch_country_daily  # noqa: E402


PANEL14 = PROJECT_ROOT / "data" / "processed" / "multicountry14_container_event_network_benchmark.csv"
DAILY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_trade_flow_panel14_daily.csv"
WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_trade_flow_panel14_weekly.csv"
REPORT = PROJECT_ROOT / "reports" / "portwatch_trade_flow_panel14.md"

FLOW_COLS = [
    "portcalls",
    "portcalls_container",
    "portcalls_cargo",
    "import",
    "export",
    "shipment",
    "import_container",
    "export_container",
    "import_cargo",
    "export_cargo",
]
YOY_COLS = [
    "portcalls_container_30MA_yoy_doy",
    "shipment_30MA_yoy_doy",
    "import_container_30MA_yoy_doy",
    "export_container_30MA_yoy_doy",
]


def panel_countries() -> list[str]:
    df = pd.read_csv(PANEL14, usecols=["ISO3"])
    return sorted(df["ISO3"].dropna().unique())


def fetch_daily(countries: list[str], force: bool) -> pd.DataFrame:
    if DAILY_OUT.exists() and not force:
        return pd.read_csv(DAILY_OUT, parse_dates=["date"])
    frames = []
    for iso3 in countries:
        daily = fetch_country_daily(iso3, timeout=120)
        frames.append(daily)
        print(f"Fetched {iso3}: {len(daily)} daily rows")
    out = pd.concat(frames, ignore_index=True).sort_values(["ISO3", "date"]).reset_index(drop=True)
    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DAILY_OUT, index=False)
    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_trade_flow_features(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.sort_values(["ISO3", "week"]).copy()
    out["tf_trade_total"] = out["tf_import"] + out["tf_export"]
    out["tf_container_trade_total"] = out["tf_import_container"] + out["tf_export_container"]
    out["tf_import_export_balance"] = safe_divide(
        out["tf_import"] - out["tf_export"], out["tf_trade_total"]
    ).fillna(0.0)
    out["tf_container_import_export_balance"] = safe_divide(
        out["tf_import_container"] - out["tf_export_container"], out["tf_container_trade_total"]
    ).fillna(0.0)
    out["tf_shipment_per_portcall"] = safe_divide(out["tf_shipment"], out["tf_portcalls"]).fillna(0.0)
    out["tf_trade_per_portcall"] = safe_divide(out["tf_trade_total"], out["tf_portcalls"]).fillna(0.0)
    out["tf_container_trade_per_container_call"] = safe_divide(
        out["tf_container_trade_total"], out["tf_portcalls_container"]
    ).fillna(0.0)
    out["tf_container_share_of_calls"] = safe_divide(out["tf_portcalls_container"], out["tf_portcalls"]).fillna(0.0)
    out["tf_cargo_share_of_calls"] = safe_divide(out["tf_portcalls_cargo"], out["tf_portcalls"]).fillna(0.0)

    base_cols = [
        "tf_import",
        "tf_export",
        "tf_shipment",
        "tf_import_container",
        "tf_export_container",
        "tf_trade_total",
        "tf_container_trade_total",
        "tf_shipment_per_portcall",
        "tf_trade_per_portcall",
        "tf_container_trade_per_container_call",
    ]
    for col in base_cols:
        out[f"{col}_log"] = np.log1p(out[col].clip(lower=0))
        grouped = out.groupby("ISO3", sort=False)[col]
        prior_mean = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).mean())
        prior_std = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).std())
        out[f"{col}_z52"] = safe_divide(out[col] - prior_mean, prior_std).replace([np.inf, -np.inf], 0).fillna(0.0)
        out[f"{col}_change_4w"] = grouped.transform(lambda s: s - s.shift(4)).fillna(0.0)
        out[f"{col}_persist4"] = grouped.transform(lambda s: s.rolling(4, min_periods=1).mean()).fillna(0.0)

    tf_cols = [col for col in out.columns if col.startswith("tf_")]
    out[tf_cols] = out[tf_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def build_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in FLOW_COLS + YOY_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    weekly = (
        out.assign(week=out["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby(["ISO3", "country", "week"], as_index=False)
        .agg(
            **{f"tf_{col}": (col, "sum") for col in FLOW_COLS},
            **{f"tf_{col}": (col, "mean") for col in YOY_COLS},
            tf_days_observed=("date", "nunique"),
        )
        .sort_values(["ISO3", "week"])
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["tf_days_observed"].eq(7)].copy()
    return add_trade_flow_features(weekly)


def write_report(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    coverage = (
        weekly.groupby(["ISO3", "country"], as_index=False)
        .agg(
            weeks=("week", "nunique"),
            first_week=("week", "min"),
            last_week=("week", "max"),
            mean_trade_total=("tf_trade_total", "mean"),
            mean_shipment=("tf_shipment", "mean"),
        )
        .sort_values("ISO3")
    )
    content = f"""# PortWatch Trade-Flow Panel14

## Purpose

This cache adds PortWatch trade-flow and shipment fields that were available in the ArcGIS daily service but not used in the main container-call benchmark. The features are intended as operational demand/recovery/backlog proxies, not as new event data.

## Outputs

- Daily cache: `data/interim/portwatch_trade_flow_panel14_daily.csv`
- Weekly cache: `data/interim/portwatch_trade_flow_panel14_weekly.csv`

## Coverage

- Daily rows: {len(daily)}
- Weekly complete rows: {len(weekly)}
- Countries: {weekly["ISO3"].nunique()}
- Week range: {weekly["week"].min().date()} to {weekly["week"].max().date()}

{coverage.to_markdown(index=False)}

## Notes

All downstream benchmarks must use current-or-prior-week features only. The weekly features include current-week trade flow, 4-week changes, 4-week persistence, and prior-52-week z-scores.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    countries = panel_countries()
    daily = fetch_daily(countries, force=args.force)
    weekly = build_weekly(daily)
    WEEKLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(WEEKLY_OUT, index=False)
    write_report(daily, weekly)
    print(f"Saved daily: {DAILY_OUT}")
    print(f"Saved weekly: {WEEKLY_OUT}")
    print(f"Saved report: {REPORT}")
    print(weekly.groupby('ISO3').size().to_string())


if __name__ == "__main__":
    run(parse_args())
