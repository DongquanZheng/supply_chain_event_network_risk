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


BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
DAILY_CHOKEPOINT_URL = f"{BASE_URL}/Daily_Chokepoints_Data/FeatureServer/0/query"
CHOKEPOINT_META_URL = f"{BASE_URL}/PortWatch_chokepoints_database/FeatureServer/0/query"
DISRUPTION_URL = f"{BASE_URL}/portwatch_disruptions_database/FeatureServer/0/query"
DISRUPTION_PORT_URL = f"{BASE_URL}/disruptions_with_ports/FeatureServer/0/query"

PANEL14 = PROJECT_ROOT / "data" / "processed" / "multicountry14_container_event_network_benchmark.csv"
DAILY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_daily.csv"
META_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_metadata.csv"
DISRUPTIONS_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_disruptions_official.csv"
DISRUPTION_PORTS_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_disruptions_with_ports.csv"
WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_weekly.csv"
ROUTE_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_route_exposure_panel14_weekly.csv"
ROUTE_WEIGHTS_OUT = PROJECT_ROOT / "reports" / "tables" / "panel14_portwatch_chokepoint_route_weights.csv"
REPORT = PROJECT_ROOT / "reports" / "portwatch_chokepoint_panel.md"

RANDOM_SEED = 42

VESSEL_COLS = [
    "n_container",
    "n_dry_bulk",
    "n_general_cargo",
    "n_roro",
    "n_tanker",
    "n_cargo",
    "n_total",
    "capacity_container",
    "capacity_dry_bulk",
    "capacity_general_cargo",
    "capacity_roro",
    "capacity_tanker",
    "capacity_cargo",
    "capacity",
]

ALERT_SCORES = {
    "GREEN": 0.0,
    "YELLOW": 1.0,
    "ORANGE": 2.0,
    "RED": 3.0,
}

# Transparent physical-route exposure priors over official PortWatch chokepoints.
# These are scenario/exposure weights, not causal or estimated route shares.
ROUTE_WEIGHTS = {
    "ARE": {
        "chokepoint6": 0.35,  # Strait of Hormuz
        "chokepoint4": 0.20,  # Bab el-Mandeb Strait
        "chokepoint1": 0.20,  # Suez Canal
        "chokepoint7": 0.15,  # Cape of Good Hope
        "chokepoint5": 0.10,  # Malacca Strait
    },
    "AUS": {
        "chokepoint5": 0.28,
        "chokepoint15": 0.14,
        "chokepoint19": 0.10,
        "chokepoint16": 0.08,
        "chokepoint1": 0.08,
        "chokepoint2": 0.05,
    },
    "CHN": {
        "chokepoint5": 0.28,
        "chokepoint11": 0.15,
        "chokepoint14": 0.10,
        "chokepoint6": 0.10,
        "chokepoint1": 0.12,
        "chokepoint4": 0.10,
        "chokepoint7": 0.08,
        "chokepoint2": 0.04,
    },
    "DEU": {
        "chokepoint1": 0.30,
        "chokepoint4": 0.18,
        "chokepoint8": 0.14,
        "chokepoint9": 0.10,
        "chokepoint10": 0.06,
        "chokepoint3": 0.05,
        "chokepoint7": 0.12,
    },
    "IDN": {
        "chokepoint5": 0.25,
        "chokepoint15": 0.18,
        "chokepoint16": 0.12,
        "chokepoint19": 0.12,
        "chokepoint20": 0.10,
        "chokepoint1": 0.08,
        "chokepoint6": 0.08,
    },
    "JPN": {
        "chokepoint5": 0.24,
        "chokepoint11": 0.16,
        "chokepoint12": 0.14,
        "chokepoint13": 0.08,
        "chokepoint6": 0.12,
        "chokepoint1": 0.12,
        "chokepoint4": 0.08,
        "chokepoint2": 0.04,
    },
    "KOR": {
        "chokepoint5": 0.24,
        "chokepoint12": 0.18,
        "chokepoint11": 0.14,
        "chokepoint6": 0.12,
        "chokepoint1": 0.12,
        "chokepoint4": 0.08,
        "chokepoint17": 0.05,
        "chokepoint2": 0.04,
    },
    "MYS": {
        "chokepoint5": 0.40,
        "chokepoint19": 0.10,
        "chokepoint15": 0.08,
        "chokepoint1": 0.14,
        "chokepoint4": 0.12,
        "chokepoint6": 0.10,
    },
    "NLD": {
        "chokepoint1": 0.30,
        "chokepoint4": 0.18,
        "chokepoint8": 0.15,
        "chokepoint9": 0.12,
        "chokepoint10": 0.05,
        "chokepoint7": 0.12,
        "chokepoint2": 0.04,
    },
    "SAU": {
        "chokepoint1": 0.28,
        "chokepoint4": 0.24,
        "chokepoint6": 0.22,
        "chokepoint7": 0.12,
        "chokepoint5": 0.08,
    },
    "SGP": {
        "chokepoint5": 0.45,
        "chokepoint1": 0.18,
        "chokepoint4": 0.14,
        "chokepoint6": 0.12,
        "chokepoint19": 0.04,
        "chokepoint15": 0.04,
    },
    "THA": {
        "chokepoint5": 0.32,
        "chokepoint1": 0.16,
        "chokepoint4": 0.12,
        "chokepoint6": 0.10,
        "chokepoint14": 0.08,
        "chokepoint19": 0.06,
    },
    "USA": {
        "chokepoint2": 0.28,
        "chokepoint22": 0.10,
        "chokepoint23": 0.08,
        "chokepoint24": 0.05,
        "chokepoint1": 0.12,
        "chokepoint4": 0.08,
        "chokepoint6": 0.06,
        "chokepoint8": 0.05,
        "chokepoint7": 0.08,
    },
    "VNM": {
        "chokepoint5": 0.30,
        "chokepoint11": 0.12,
        "chokepoint14": 0.12,
        "chokepoint1": 0.14,
        "chokepoint4": 0.10,
        "chokepoint6": 0.08,
        "chokepoint15": 0.05,
    },
}


def fetch_arcgis(url: str, *, where: str = "1=1", order_by: str | None = None, timeout: int = 120) -> pd.DataFrame:
    id_response = requests.get(url, params={"where": where, "returnIdsOnly": "true", "f": "json"}, timeout=timeout)
    id_response.raise_for_status()
    id_payload = id_response.json()
    object_ids = id_payload.get("objectIds") or []
    object_id_field = id_payload.get("objectIdFieldName", "ObjectId")
    if object_ids:
        rows = []
        id_chunk_size = 100
        for start in range(0, len(object_ids), id_chunk_size):
            chunk = object_ids[start : start + id_chunk_size]
            params = {
                "objectIds": ",".join(str(value) for value in chunk),
                "outFields": "*",
                "returnGeometry": "false",
                "f": "json",
            }
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            rows.extend(feature["attributes"] for feature in payload.get("features", []))
        out = pd.DataFrame(rows)
        if order_by:
            sort_cols = [part.strip().split()[0] for part in order_by.split(",")]
            sort_cols = [col for col in sort_cols if col in out.columns]
            if sort_cols:
                out = out.sort_values(sort_cols).reset_index(drop=True)
        elif object_id_field in out.columns:
            out = out.sort_values(object_id_field).reset_index(drop=True)
        return out

    rows = []
    offset = 0
    page_size = 1000
    while True:
        params = {
            "where": where,
            "outFields": "*",
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


def fetch_daily_chokepoints(portids: list[str], timeout: int = 120) -> pd.DataFrame:
    frames = []
    for portid in sorted(portids):
        frame = fetch_arcgis_offset(
            DAILY_CHOKEPOINT_URL,
            where=f"portid='{portid}'",
            order_by="date ASC",
            timeout=timeout,
        )
        frames.append(frame)
        print(f"Fetched {portid}: {len(frame)} daily rows")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "portid"]).reset_index(drop=True)


def fetch_arcgis_offset(url: str, *, where: str = "1=1", order_by: str | None = None, timeout: int = 120) -> pd.DataFrame:
    rows = []
    offset = 0
    page_size = 1000
    while True:
        params = {
            "where": where,
            "outFields": "*",
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


def parse_arcgis_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any() and numeric.dropna().abs().median() > 10_000_000_000:
            out[col] = pd.to_datetime(numeric, unit="ms", errors="coerce")
        else:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def panel_countries() -> list[str]:
    return sorted(pd.read_csv(PANEL14, usecols=["ISO3"])["ISO3"].dropna().unique())


def load_or_fetch(force: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if DAILY_OUT.exists() and META_OUT.exists() and DISRUPTIONS_OUT.exists() and DISRUPTION_PORTS_OUT.exists() and not force:
        daily = pd.read_csv(DAILY_OUT, parse_dates=["date"])
        meta = pd.read_csv(META_OUT)
        disruptions = pd.read_csv(DISRUPTIONS_OUT, parse_dates=["fromdate", "todate", "editdate"])
        disruption_ports = pd.read_csv(DISRUPTION_PORTS_OUT, parse_dates=["fromdate", "todate"])
        return daily, meta, disruptions, disruption_ports

    meta = fetch_arcgis(CHOKEPOINT_META_URL, order_by="portid ASC")
    daily = fetch_daily_chokepoints(meta["portid"].dropna().astype(str).tolist())
    disruptions = fetch_arcgis(DISRUPTION_URL, order_by="fromdate ASC")
    disruption_ports = fetch_arcgis(DISRUPTION_PORT_URL, order_by="eventid ASC")

    daily = parse_arcgis_dates(daily, ["date"])
    disruptions = parse_arcgis_dates(disruptions, ["fromdate", "todate", "editdate"])
    disruption_ports = parse_arcgis_dates(disruption_ports, ["fromdate", "todate"])

    for path, frame in [
        (DAILY_OUT, daily),
        (META_OUT, meta),
        (DISRUPTIONS_OUT, disruptions),
        (DISRUPTION_PORTS_OUT, disruption_ports),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return daily, meta, disruptions, disruption_ports


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def add_prior_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.sort_values(["portid", "week"]).copy()
    roots = [
        "cpw_n_container",
        "cpw_n_total",
        "cpw_capacity_container",
        "cpw_capacity",
        "cpw_container_utilization",
        "cpw_total_utilization",
    ]
    for col in roots:
        grouped = out.groupby("portid", sort=False)[col]
        mean = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).mean())
        std = grouped.transform(lambda s: s.shift(1).rolling(52, min_periods=8).std())
        z = safe_divide(out[col] - mean, std).replace([np.inf, -np.inf], 0).fillna(0.0)
        out[f"{col}_z52"] = z
        out[f"{col}_pos_z52"] = z.clip(lower=0)
        out[f"{col}_neg_z52"] = (-z).clip(lower=0)
        out[f"{col}_change_4w"] = grouped.transform(lambda s: s - s.shift(4)).fillna(0.0)
    cpw_cols = [col for col in out.columns if col.startswith("cpw_")]
    out[cpw_cols] = out[cpw_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def active_disruption_grid(disruptions: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.Series(weekly["week"].drop_duplicates().sort_values())
    if disruptions.empty:
        return pd.DataFrame({"week": [], "portid": []})
    rows = []
    max_week_end = weeks.max() + pd.Timedelta(days=6)
    for event in disruptions.to_dict("records"):
        affected = str(event.get("affectedports", "") or "")
        portids = [item.strip() for item in affected.split(";") if item.strip().startswith("chokepoint")]
        if not portids:
            continue
        start = pd.to_datetime(event.get("fromdate"), errors="coerce")
        if pd.isna(start):
            continue
        end = pd.to_datetime(event.get("todate"), errors="coerce")
        if pd.isna(end):
            end = max_week_end
        alert = str(event.get("alertlevel", "") or "").upper()
        severity = str(event.get("severitytext", "") or "").upper()
        score = ALERT_SCORES.get(alert, 0.0)
        active_weeks = weeks.loc[(weeks + pd.Timedelta(days=6) >= start) & (weeks <= end)]
        for week in active_weeks:
            overlap_start = max(week, start.normalize())
            overlap_end = min(week + pd.Timedelta(days=6), end.normalize())
            active_days = max(0, int((overlap_end - overlap_start).days) + 1)
            for portid in portids:
                rows.append(
                    {
                        "week": week,
                        "portid": portid,
                        "eventid": event.get("eventid"),
                        "eventname": event.get("eventname"),
                        "eventtype": event.get("eventtype"),
                        "alertlevel": alert,
                        "severitytext": severity,
                        "alert_score": score,
                        "active_days": active_days,
                        "is_red": float(alert == "RED"),
                        "is_orange": float(alert == "ORANGE"),
                    }
                )
    if not rows:
        return pd.DataFrame({"week": [], "portid": []})
    return pd.DataFrame(rows)


def build_weekly(daily: pd.DataFrame, disruptions: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in VESSEL_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    weekly = (
        out.assign(week=out["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby(["portid", "portname", "week"], as_index=False)
        .agg(
            **{f"cpw_{col}": (col, "sum") for col in VESSEL_COLS if col.startswith("n_")},
            **{f"cpw_{col}": (col, "mean") for col in VESSEL_COLS if col.startswith("capacity")},
            cpw_days_observed=("date", "nunique"),
        )
        .sort_values(["portid", "week"])
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["cpw_days_observed"].eq(7)].copy()
    weekly["cpw_container_utilization"] = safe_divide(weekly["cpw_n_container"], weekly["cpw_capacity_container"]).fillna(0.0)
    weekly["cpw_total_utilization"] = safe_divide(weekly["cpw_n_total"], weekly["cpw_capacity"]).fillna(0.0)
    weekly = add_prior_anomalies(weekly)

    active = active_disruption_grid(disruptions, weekly)
    if not active.empty:
        event_weekly = (
            active.groupby(["week", "portid"], as_index=False)
            .agg(
                cpw_active_event_count=("eventid", "nunique"),
                cpw_red_event_count=("is_red", "sum"),
                cpw_orange_event_count=("is_orange", "sum"),
                cpw_max_alert_score=("alert_score", "max"),
                cpw_active_event_days=("active_days", "sum"),
            )
        )
    else:
        event_weekly = pd.DataFrame(columns=["week", "portid"])
    weekly = weekly.merge(event_weekly, on=["week", "portid"], how="left")
    event_cols = [
        "cpw_active_event_count",
        "cpw_red_event_count",
        "cpw_orange_event_count",
        "cpw_max_alert_score",
        "cpw_active_event_days",
    ]
    for col in event_cols:
        if col not in weekly.columns:
            weekly[col] = 0.0
    weekly[event_cols] = weekly[event_cols].fillna(0.0)
    weekly["cpw_active_red"] = weekly["cpw_red_event_count"].gt(0).astype(float)
    weekly["cpw_active_any_disruption"] = weekly["cpw_active_event_count"].gt(0).astype(float)
    cpw_cols = [col for col in weekly.columns if col.startswith("cpw_")]
    weekly[cpw_cols] = weekly[cpw_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weekly


def route_weight_frame(countries: list[str], portids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1701)
    frames = []
    portids = sorted(portids)
    for iso3 in countries:
        raw = ROUTE_WEIGHTS.get(iso3, {})
        frame = pd.DataFrame({"ISO3": iso3, "portid": portids})
        frame["route_weight"] = frame["portid"].map(raw).fillna(0.0)
        if frame["route_weight"].sum() <= 0:
            frame["route_weight"] = 1.0 / len(portids)
        else:
            frame["route_weight"] = frame["route_weight"] / frame["route_weight"].sum()
        frame["route_equal_weight"] = 1.0 / len(portids)
        random_weight = rng.random(len(portids))
        frame["route_random_weight"] = random_weight / random_weight.sum()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def weighted_route_features(weekly: pd.DataFrame, weights: pd.DataFrame, prefix: str, weight_col: str) -> pd.DataFrame:
    value_cols = [
        "cpw_n_container_neg_z52",
        "cpw_n_total_neg_z52",
        "cpw_capacity_container_neg_z52",
        "cpw_capacity_neg_z52",
        "cpw_container_utilization_pos_z52",
        "cpw_total_utilization_pos_z52",
        "cpw_n_container_change_4w",
        "cpw_n_total_change_4w",
        "cpw_active_event_count",
        "cpw_red_event_count",
        "cpw_orange_event_count",
        "cpw_max_alert_score",
        "cpw_active_event_days",
        "cpw_active_red",
        "cpw_active_any_disruption",
    ]
    joined = weights[["ISO3", "portid", weight_col]].merge(weekly[["week", "portid"] + value_cols], on="portid", how="inner")
    agg_cols = {}
    for col in value_cols:
        out_col = f"{prefix}_{col.replace('cpw_', '')}"
        joined[out_col] = joined[col] * joined[weight_col]
        agg_cols[out_col] = (out_col, "sum")
    return joined.groupby(["ISO3", "week"], as_index=False).agg(**agg_cols)


def build_route_panel(weekly: pd.DataFrame) -> pd.DataFrame:
    countries = panel_countries()
    weights = route_weight_frame(countries, weekly["portid"].drop_duplicates().tolist())
    ROUTE_WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(ROUTE_WEIGHTS_OUT, index=False)

    frames = []
    for prefix, weight_col in [
        ("pwc_route", "route_weight"),
        ("pwc_equal", "route_equal_weight"),
        ("pwc_random", "route_random_weight"),
    ]:
        frames.append(weighted_route_features(weekly, weights, prefix, weight_col))
    route = frames[0]
    for frame in frames[1:]:
        route = route.merge(frame, on=["ISO3", "week"], how="outer")

    global_cols = [
        "cpw_n_container_neg_z52",
        "cpw_n_total_neg_z52",
        "cpw_capacity_container_neg_z52",
        "cpw_capacity_neg_z52",
        "cpw_active_event_count",
        "cpw_red_event_count",
        "cpw_max_alert_score",
        "cpw_active_event_days",
        "cpw_active_red",
    ]
    global_week = (
        weekly.groupby("week", as_index=False)[global_cols]
        .agg(
            {
                "cpw_n_container_neg_z52": "mean",
                "cpw_n_total_neg_z52": "mean",
                "cpw_capacity_container_neg_z52": "mean",
                "cpw_capacity_neg_z52": "mean",
                "cpw_active_event_count": "sum",
                "cpw_red_event_count": "sum",
                "cpw_max_alert_score": "max",
                "cpw_active_event_days": "sum",
                "cpw_active_red": "max",
            }
        )
        .rename(columns={col: f"pwc_global_{col.replace('cpw_', '')}" for col in global_cols})
    )
    route = route.merge(global_week, on="week", how="left")
    pwc_cols = [col for col in route.columns if col.startswith("pwc_")]
    route[pwc_cols] = route[pwc_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return route.sort_values(["ISO3", "week"]).reset_index(drop=True)


def write_report(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    route: pd.DataFrame,
    meta: pd.DataFrame,
    disruptions: pd.DataFrame,
    disruption_ports: pd.DataFrame,
) -> None:
    coverage = (
        weekly.groupby(["portid", "portname"], as_index=False)
        .agg(
            weeks=("week", "nunique"),
            first_week=("week", "min"),
            last_week=("week", "max"),
            active_event_weeks=("cpw_active_any_disruption", "sum"),
            red_event_weeks=("cpw_active_red", "sum"),
            mean_container_calls=("cpw_n_container", "mean"),
        )
        .sort_values("portid")
    )
    route_cov = (
        route.groupby("ISO3", as_index=False)
        .agg(
            weeks=("week", "nunique"),
            route_red_weeks=("pwc_route_active_red", "sum"),
            route_event_days=("pwc_route_active_event_days", "sum"),
        )
        .sort_values("ISO3")
    )
    content = f"""# PortWatch Chokepoint Panel

## Purpose

This cache adds official PortWatch chokepoint and disruption data. It is a physical-exposure data expansion, distinct from the earlier GDELT chokepoint proxy.

## Sources

- `Daily_Chokepoints_Data`
- `PortWatch_chokepoints_database`
- `portwatch_disruptions_database`
- `disruptions_with_ports`

## Outputs

- Daily chokepoint cache: `data/interim/portwatch_chokepoint_daily.csv`
- Weekly chokepoint cache: `data/interim/portwatch_chokepoint_weekly.csv`
- Country-week route exposure cache: `data/interim/portwatch_chokepoint_route_exposure_panel14_weekly.csv`
- Route weights: `reports/tables/panel14_portwatch_chokepoint_route_weights.csv`

## Coverage

- Daily rows: {len(daily)}
- Chokepoints in metadata: {meta["portid"].nunique() if "portid" in meta.columns else len(meta)}
- Weekly chokepoint rows: {len(weekly)}
- Route country-week rows: {len(route)}
- Official disruption events: {len(disruptions)}
- Event-port rows: {len(disruption_ports)}
- Daily date range: {daily["date"].min().date()} to {daily["date"].max().date()}
- Weekly range: {weekly["week"].min().date()} to {weekly["week"].max().date()}

## Chokepoint Weekly Coverage

{coverage.to_markdown(index=False)}

## Panel14 Route Exposure Coverage

{route_cov.to_markdown(index=False)}

## Notes

Route weights are transparent exposure priors over official PortWatch chokepoints. They are intended for mechanism, scenario, and prediction tests, not causal attribution.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    daily, meta, disruptions, disruption_ports = load_or_fetch(args.force)
    weekly = build_weekly(daily, disruptions)
    route = build_route_panel(weekly)
    WEEKLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(WEEKLY_OUT, index=False)
    route.to_csv(ROUTE_OUT, index=False)
    write_report(daily, weekly, route, meta, disruptions, disruption_ports)
    print(f"Saved daily: {DAILY_OUT}")
    print(f"Saved weekly: {WEEKLY_OUT}")
    print(f"Saved route panel: {ROUTE_OUT}")
    print(f"Saved report: {REPORT}")
    print(route.groupby("ISO3").size().to_string())


if __name__ == "__main__":
    run(parse_args())
