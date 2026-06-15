from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services"
PORT_META_URL = f"{BASE_URL}/PortWatch_ports_database/FeatureServer/0/query"
DAILY_PORTS_URL = f"{BASE_URL}/Daily_Ports_Data/FeatureServer/0/query"
PANEL14 = PROJECT_ROOT / "data" / "processed" / "multicountry14_container_event_network_benchmark.csv"
META_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel14_metadata.csv"
DAILY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel14_daily.csv"
PORT_WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel14_port_weekly.csv"
COUNTRY_WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel14_country_weekly.csv"
REPORT = PROJECT_ROOT / "reports" / "portwatch_port_system_panel14.md"

NUMERIC_DAILY_COLS = [
    "portcalls_container",
    "portcalls",
    "import_container",
    "export_container",
    "import",
    "export",
]


def panel_countries() -> list[str]:
    return sorted(pd.read_csv(PANEL14, usecols=["ISO3"])["ISO3"].dropna().unique())


def fetch_arcgis_offset(
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    order_by: str | None = None,
    page_size: int = 1000,
    timeout: int = 120,
) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        if order_by:
            params["orderByFields"] = order_by
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        features = payload.get("features", [])
        if not features:
            break
        rows.extend(feature["attributes"] for feature in features)
        if len(features) < page_size and not payload.get("exceededTransferLimit", False):
            break
        offset += len(features)
    return pd.DataFrame(rows)


def load_or_fetch_metadata(force: bool) -> pd.DataFrame:
    if META_OUT.exists() and not force:
        return pd.read_csv(META_OUT)
    out_fields = (
        "portid,portname,country,ISO3,lat,lon,vessel_count_total,vessel_count_container,"
        "share_country_maritime_import,share_country_maritime_export"
    )
    meta = fetch_arcgis_offset(PORT_META_URL, out_fields=out_fields, order_by="ISO3 ASC")
    meta = meta.loc[meta["ISO3"].isin(panel_countries())].copy()
    for col in [
        "vessel_count_total",
        "vessel_count_container",
        "share_country_maritime_import",
        "share_country_maritime_export",
    ]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0.0)
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META_OUT, index=False)
    return meta


def select_ports(meta: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames = []
    for iso3, frame in meta.groupby("ISO3", sort=False):
        candidates = frame.loc[frame["vessel_count_container"].gt(0)].copy()
        if candidates.empty:
            candidates = frame.copy()
        selected = candidates.sort_values(
            ["vessel_count_container", "share_country_maritime_import", "vessel_count_total"],
            ascending=[False, False, False],
        ).head(top_n)
        total_container = frame["vessel_count_container"].sum()
        total_import_share = frame["share_country_maritime_import"].sum()
        selected = selected.copy()
        selected["selected_rank"] = range(1, len(selected) + 1)
        selected["selected_top_n"] = top_n
        selected["selected_container_coverage"] = (
            selected["vessel_count_container"].sum() / total_container if total_container else np.nan
        )
        selected["selected_import_share_coverage"] = (
            selected["share_country_maritime_import"].sum() / total_import_share if total_import_share else np.nan
        )
        frames.append(selected)
        print(f"Selected {iso3}: {len(selected)} ports")
    return pd.concat(frames, ignore_index=True)


def fetch_daily_for_ports(portids: list[str], force: bool) -> pd.DataFrame:
    if DAILY_OUT.exists() and not force:
        return pd.read_csv(DAILY_OUT, parse_dates=["date"])
    out_fields = (
        "date,portid,portname,country,ISO3,portcalls_container,portcalls,"
        "import_container,export_container,import,export"
    )
    frames = []
    for portid in sorted(portids):
        frame = fetch_arcgis_offset(
            DAILY_PORTS_URL,
            where=f"portid='{portid}'",
            out_fields=out_fields,
            order_by="date ASC",
            timeout=180,
        )
        frames.append(frame)
        print(f"Fetched {portid}: {len(frame)} daily rows")
    daily = pd.concat(frames, ignore_index=True).sort_values(["ISO3", "portid", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_OUT, index=False)
    return daily


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def add_port_anomalies(port_weekly: pd.DataFrame) -> pd.DataFrame:
    out = port_weekly.sort_values(["portid", "week"]).copy()
    for col in [
        "psp_portcalls_container",
        "psp_portcalls",
        "psp_import_container",
        "psp_export_container",
        "psp_trade_container",
    ]:
        grouped = out.groupby("portid", sort=False)[col]
        mean = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).mean())
        std = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).std())
        z = safe_divide(out[col] - mean, std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out[f"{col}_z52"] = z
        out[f"{col}_neg_z52"] = (-z).clip(lower=0.0)
        out[f"{col}_pos_z52"] = z.clip(lower=0.0)
        out[f"{col}_change_4w"] = grouped.transform(lambda s: s - s.shift(4)).fillna(0.0)
    psp_cols = [col for col in out.columns if col.startswith("psp_")]
    out[psp_cols] = out[psp_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def build_port_weekly(daily: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in NUMERIC_DAILY_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    weekly = (
        out.assign(week=out["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby(["ISO3", "country", "portid", "portname", "week"], as_index=False)
        .agg(
            psp_portcalls_container=("portcalls_container", "sum"),
            psp_portcalls=("portcalls", "sum"),
            psp_import_container=("import_container", "sum"),
            psp_export_container=("export_container", "sum"),
            psp_import=("import", "sum"),
            psp_export=("export", "sum"),
            psp_days_observed=("date", "nunique"),
        )
        .sort_values(["ISO3", "portid", "week"])
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["psp_days_observed"].eq(7)].copy()
    weekly["psp_trade_container"] = weekly["psp_import_container"] + weekly["psp_export_container"]
    weekly["psp_trade_total"] = weekly["psp_import"] + weekly["psp_export"]
    weekly["psp_container_import_export_balance"] = safe_divide(
        weekly["psp_import_container"] - weekly["psp_export_container"], weekly["psp_trade_container"]
    ).fillna(0.0)
    static_cols = [
        "portid",
        "selected_rank",
        "vessel_count_container",
        "vessel_count_total",
        "share_country_maritime_import",
        "share_country_maritime_export",
        "selected_container_coverage",
        "selected_import_share_coverage",
    ]
    weekly = weekly.merge(selected[static_cols], on="portid", how="left")
    for col in static_cols[1:]:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0.0)
    total_weight = weekly.groupby("ISO3")["vessel_count_container"].transform("sum")
    weekly["psp_static_container_weight"] = safe_divide(weekly["vessel_count_container"], total_weight).fillna(0.0)
    return add_port_anomalies(weekly)


def concentration_features(frame: pd.DataFrame, value_col: str, prefix: str) -> dict:
    values = frame[value_col].clip(lower=0.0).to_numpy(dtype=float)
    total = float(values.sum())
    if total <= 0:
        return {
            f"{prefix}_top1_share": 0.0,
            f"{prefix}_top3_share": 0.0,
            f"{prefix}_hhi": 0.0,
            f"{prefix}_active_ports": 0,
        }
    shares = np.sort(values / total)[::-1]
    return {
        f"{prefix}_top1_share": float(shares[0]),
        f"{prefix}_top3_share": float(shares[:3].sum()),
        f"{prefix}_hhi": float(np.square(shares).sum()),
        f"{prefix}_active_ports": int(np.count_nonzero(values > 0)),
    }


def build_country_weekly(port_weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (iso3, country, week), frame in port_weekly.groupby(["ISO3", "country", "week"], sort=False):
        weighted_neg = frame["psp_portcalls_container_neg_z52"] * frame["psp_static_container_weight"]
        weighted_trade_neg = frame["psp_trade_container_neg_z52"] * frame["psp_static_container_weight"]
        row = {
            "ISO3": iso3,
            "country": country,
            "week": week,
            "ps_top_ports": frame["portid"].nunique(),
            "ps_selected_container_coverage": frame["selected_container_coverage"].max(),
            "ps_selected_import_share_coverage": frame["selected_import_share_coverage"].max(),
            "ps_container_calls_selected": frame["psp_portcalls_container"].sum(),
            "ps_total_calls_selected": frame["psp_portcalls"].sum(),
            "ps_container_trade_selected": frame["psp_trade_container"].sum(),
            "ps_container_ports_active": int(frame["psp_portcalls_container"].gt(0).sum()),
            "ps_container_neg_z52_max": frame["psp_portcalls_container_neg_z52"].max(),
            "ps_container_neg_z52_sum": frame["psp_portcalls_container_neg_z52"].sum(),
            "ps_container_neg_z52_weighted": weighted_neg.sum(),
            "ps_container_neg_z52_gt1_ports": int(frame["psp_portcalls_container_neg_z52"].gt(1.0).sum()),
            "ps_container_neg_z52_gt2_ports": int(frame["psp_portcalls_container_neg_z52"].gt(2.0).sum()),
            "ps_trade_container_neg_z52_max": frame["psp_trade_container_neg_z52"].max(),
            "ps_trade_container_neg_z52_weighted": weighted_trade_neg.sum(),
            "ps_import_export_balance_weighted": (
                frame["psp_container_import_export_balance"] * frame["psp_static_container_weight"]
            ).sum(),
            "ps_top3_static_ports_neg_z52": frame.nsmallest(3, "selected_rank")["psp_portcalls_container_neg_z52"].sum(),
        }
        row.update(concentration_features(frame, "psp_portcalls_container", "ps_container"))
        row.update(concentration_features(frame, "psp_trade_container", "ps_trade_container"))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["ISO3", "week"]).reset_index(drop=True)
    ps_cols = [col for col in out.columns if col.startswith("ps_")]
    for col in ps_cols:
        if col.endswith("_selected") or col.endswith("_calls_selected") or col.endswith("_trade_selected"):
            out[f"{col}_log"] = np.log1p(out[col].clip(lower=0.0))
        grouped = out.groupby("ISO3", sort=False)[col]
        if pd.api.types.is_numeric_dtype(out[col]):
            out[f"{col}_change_4w"] = grouped.transform(lambda s: s - s.shift(4)).fillna(0.0)
    ps_cols = [col for col in out.columns if col.startswith("ps_")]
    out[ps_cols] = out[ps_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def write_report(meta: pd.DataFrame, selected: pd.DataFrame, daily: pd.DataFrame, country_weekly: pd.DataFrame) -> None:
    selected_summary = (
        selected.groupby(["ISO3", "country"], as_index=False)
        .agg(
            selected_ports=("portid", "nunique"),
            selected_container_coverage=("selected_container_coverage", "max"),
            selected_import_share_coverage=("selected_import_share_coverage", "max"),
            total_panel_ports=("portid", "size"),
        )
        .sort_values("ISO3")
    )
    coverage = (
        country_weekly.groupby(["ISO3", "country"], as_index=False)
        .agg(
            weeks=("week", "nunique"),
            first_week=("week", "min"),
            last_week=("week", "max"),
            mean_active_container_ports=("ps_container_ports_active", "mean"),
            max_neg_ports_gt2=("ps_container_neg_z52_gt2_ports", "max"),
        )
        .sort_values("ISO3")
    )
    top_ports = selected.sort_values(["ISO3", "selected_rank"])[
        ["ISO3", "country", "selected_rank", "portid", "portname", "vessel_count_container", "share_country_maritime_import"]
    ]
    content = f"""# PortWatch Port-System Panel14

## Purpose

This cache adds high-frequency port-level PortWatch data for the major container ports in each panel14 country. It is designed to capture port-system concentration and localized port-level stress that country-level aggregate port calls may hide.

## Source

- Metadata service: `PortWatch_ports_database`
- Daily service: `Daily_Ports_Data`

## Outputs

- Selected port metadata: `data/interim/portwatch_port_system_panel14_metadata.csv`
- Daily selected-port cache: `data/interim/portwatch_port_system_panel14_daily.csv`
- Port-week cache: `data/interim/portwatch_port_system_panel14_port_weekly.csv`
- Country-week feature cache: `data/interim/portwatch_port_system_panel14_country_weekly.csv`

## Coverage

- Panel metadata rows before top-port selection: {len(meta)}
- Selected ports: {selected["portid"].nunique()}
- Daily rows: {len(daily)}
- Country-week rows: {len(country_weekly)}
- Week range: {country_weekly["week"].min().date()} to {country_weekly["week"].max().date()}

## Selected-Port Coverage

{selected_summary.to_markdown(index=False)}

## Weekly Coverage

{coverage.to_markdown(index=False)}

## Selected Ports

{top_ports.to_markdown(index=False)}

## Notes

The selected ports are the top ports per country by historical container-vessel count in `PortWatch_ports_database`. Downstream features emphasize distributional structure and localized port-level negative anomalies: active selected container ports, top-port shares, HHI, max/sum/weighted port-level negative z-scores, and import/export balance.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    meta = load_or_fetch_metadata(args.force)
    selected = select_ports(meta, args.top_n)
    selected.to_csv(META_OUT, index=False)
    daily = fetch_daily_for_ports(selected["portid"].dropna().astype(str).unique().tolist(), args.force)
    port_weekly = build_port_weekly(daily, selected)
    country_weekly = build_country_weekly(port_weekly)
    port_weekly.to_csv(PORT_WEEKLY_OUT, index=False)
    country_weekly.to_csv(COUNTRY_WEEKLY_OUT, index=False)
    write_report(meta, selected, daily, country_weekly)
    print(f"Saved metadata: {META_OUT}")
    print(f"Saved daily: {DAILY_OUT}")
    print(f"Saved port-weekly: {PORT_WEEKLY_OUT}")
    print(f"Saved country-weekly: {COUNTRY_WEEKLY_OUT}")
    print(f"Saved report: {REPORT}")
    print(country_weekly.head().to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
