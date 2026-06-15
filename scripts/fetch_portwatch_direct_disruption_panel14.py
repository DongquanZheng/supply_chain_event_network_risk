from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_portwatch_chokepoint_panel import (  # noqa: E402
    DISRUPTIONS_OUT,
    DISRUPTION_PORTS_OUT,
)


PANEL14 = PROJECT_ROOT / "data" / "processed" / "multicountry14_container_event_network_benchmark.csv"
WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel14_total_dependency_weights_2023.csv"
WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_direct_disruption_country_weekly.csv"
PANEL_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_direct_disruption_panel14_weekly.csv"
REPORT = PROJECT_ROOT / "reports" / "portwatch_direct_disruption_panel14.md"

ALERT_SCORES = {"GREEN": 0.0, "YELLOW": 1.0, "ORANGE": 2.0, "RED": 3.0}
EVENT_TYPES = ["TC", "FL", "EQ", "OT", "DR", "VO", "WF"]
RANDOM_SEED = 42

COUNTRY_TO_ISO3 = {
    "Australia": "AUS",
    "China": "CHN",
    "Hong Kong SAR": "CHN",
    "Hong Kong Special Administrative Region": "CHN",
    "Macao SAR": "CHN",
    "Macao Special Administrative Region": "CHN",
    "Taiwan Province of China": "CHN",
    "Germany": "DEU",
    "Indonesia": "IDN",
    "Japan": "JPN",
    "Korea": "KOR",
    "Malaysia": "MYS",
    "Netherlands": "NLD",
    "Saudi Arabia": "SAU",
    "Singapore": "SGP",
    "Thailand": "THA",
    "United Arab Emirates": "ARE",
    "United States": "USA",
    "Vietnam": "VNM",
}


def ensure_source_files() -> None:
    if DISRUPTIONS_OUT.exists() and DISRUPTION_PORTS_OUT.exists():
        return
    raise FileNotFoundError(
        "Missing PortWatch disruption source caches. Run scripts/fetch_portwatch_chokepoint_panel.py first."
    )


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL14, parse_dates=["week"], low_memory=False)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_source_files()
    disruptions = pd.read_csv(DISRUPTIONS_OUT, parse_dates=["fromdate", "todate"], low_memory=False)
    ports = pd.read_csv(DISRUPTION_PORTS_OUT, parse_dates=["fromdate", "todate"], low_memory=False)
    return disruptions, ports


def week_grid(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(panel["week"].drop_duplicates().sort_values().reset_index(drop=True))


def expand_active_event_weeks(panel: pd.DataFrame, disruptions: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    weeks = week_grid(panel)
    max_week_end = weeks.max() + pd.Timedelta(days=6)
    meta_cols = ["eventid", "eventtype", "alertlevel", "severitytext", "affectedports", "n_affectedports"]
    meta = disruptions[[col for col in meta_cols if col in disruptions.columns]].drop_duplicates("eventid")
    events = ports.merge(meta, on="eventid", how="left", suffixes=("", "_official"))
    events["source_iso3"] = events["country"].map(COUNTRY_TO_ISO3)
    events = events.loc[events["source_iso3"].isin(panel["ISO3"].unique())].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for event in events.to_dict("records"):
        start = pd.to_datetime(event.get("fromdate"), errors="coerce")
        if pd.isna(start):
            continue
        end = pd.to_datetime(event.get("todate"), errors="coerce")
        if pd.isna(end):
            end = max_week_end
        if end < start:
            end = start
        active_weeks = weeks.loc[(weeks + pd.Timedelta(days=6) >= start) & (weeks <= end)]
        alert = str(event.get("alertlevel") or "").upper()
        event_type = str(event.get("eventtype") or "OT").upper()
        if event_type not in EVENT_TYPES:
            event_type = "OT"
        for week in active_weeks:
            overlap_start = max(week, start.normalize())
            overlap_end = min(week + pd.Timedelta(days=6), end.normalize())
            active_days = max(1, int((overlap_end - overlap_start).days) + 1)
            rows.append(
                {
                    "ISO3": event["source_iso3"],
                    "week": week,
                    "eventid": event.get("eventid"),
                    "portid": event.get("portid"),
                    "portname": event.get("portname"),
                    "eventname": event.get("eventname"),
                    "eventtype": event_type,
                    "alertlevel": alert,
                    "alert_score": ALERT_SCORES.get(alert, 0.0),
                    "active_days": active_days,
                    "distance_km": pd.to_numeric(event.get("distance_km"), errors="coerce"),
                    "is_red": float(alert == "RED"),
                    "is_orange": float(alert == "ORANGE"),
                }
            )
    return pd.DataFrame(rows)


def add_rolls(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["ISO3", "week"]).copy()
    grouped = out.groupby("ISO3", sort=False)
    for col in value_cols:
        out[f"{col}_roll4w"] = grouped[col].transform(lambda s: s.rolling(4, min_periods=1).sum())
        out[f"{col}_roll8w"] = grouped[col].transform(lambda s: s.rolling(8, min_periods=1).sum())
        out[f"{col}_lag1w"] = grouped[col].shift(1).fillna(0.0)
    return out


def build_country_weekly(panel: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    base = panel[["ISO3", "week"]].drop_duplicates().sort_values(["ISO3", "week"])
    if active.empty:
        out = base.copy()
        value_cols = [
            "pdd_own_active_event_count",
            "pdd_own_red_event_count",
            "pdd_own_active_event_days",
            "pdd_own_affected_port_count",
            "pdd_own_max_alert_score",
        ]
        for col in value_cols:
            out[col] = 0.0
        return add_rolls(out, value_cols)

    type_frames = []
    for event_type in EVENT_TYPES:
        tmp = (
            active.loc[active["eventtype"].eq(event_type)]
            .groupby(["ISO3", "week"], as_index=False)["eventid"]
            .nunique()
            .rename(columns={"eventid": f"pdd_own_type_{event_type.lower()}_event_count"})
        )
        type_frames.append(tmp)

    weekly = (
        active.groupby(["ISO3", "week"], as_index=False)
        .agg(
            pdd_own_active_event_count=("eventid", "nunique"),
            pdd_own_red_event_count=("is_red", "sum"),
            pdd_own_orange_event_count=("is_orange", "sum"),
            pdd_own_active_event_days=("active_days", "sum"),
            pdd_own_affected_port_count=("portid", "nunique"),
            pdd_own_max_alert_score=("alert_score", "max"),
            pdd_own_min_distance_km=("distance_km", "min"),
        )
        .sort_values(["ISO3", "week"])
    )
    for frame in type_frames:
        weekly = weekly.merge(frame, on=["ISO3", "week"], how="left")
    out = base.merge(weekly, on=["ISO3", "week"], how="left")
    pdd_cols = [col for col in out.columns if col.startswith("pdd_own_")]
    out[pdd_cols] = out[pdd_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    count_cols = [col for col in pdd_cols if col != "pdd_own_min_distance_km"]
    return add_rolls(out, count_cols)


def weighted_exposure(country_weekly: pd.DataFrame, weights: pd.DataFrame, prefix: str, weight_col: str) -> pd.DataFrame:
    source_cols = [col for col in country_weekly.columns if col.startswith("pdd_own_")]
    source = country_weekly[["ISO3", "week"] + source_cols].rename(columns={"ISO3": "partner_iso3"})
    joined = weights[["ISO3", "partner_iso3", weight_col]].merge(source, on="partner_iso3", how="left")
    joined[source_cols] = joined[source_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    agg = {}
    weighted_cols = {}
    for col in source_cols:
        out_col = f"{prefix}_{col.replace('pdd_own_', '')}"
        weighted_cols[out_col] = joined[col] * joined[weight_col]
        agg[out_col] = (out_col, "sum")
    joined = pd.concat([joined[["ISO3", "week"]], pd.DataFrame(weighted_cols, index=joined.index)], axis=1)
    return joined.groupby(["ISO3", "week"], as_index=False).agg(**agg)


def build_panel_features(panel: pd.DataFrame, country_weekly: pd.DataFrame) -> pd.DataFrame:
    weights = pd.read_csv(WEIGHTS)
    frames = [
        country_weekly,
        weighted_exposure(country_weekly, weights, "pdd_network", "import_dependency_share"),
        weighted_exposure(country_weekly, weights, "pdd_equal", "equal_weight"),
        weighted_exposure(country_weekly, weights, "pdd_random", "random_weight"),
        weighted_exposure(country_weekly, weights, "pdd_shuffled", "shuffled_weight"),
    ]
    global_cols = [col for col in country_weekly.columns if col.startswith("pdd_own_") and "min_distance" not in col]
    global_week = country_weekly.groupby("week", as_index=False)[global_cols].sum()
    global_week = global_week.rename(columns={col: f"pdd_global_{col.replace('pdd_own_', '')}" for col in global_cols})

    out = panel[["ISO3", "week"]].drop_duplicates().sort_values(["ISO3", "week"]).copy()
    for frame in frames:
        out = out.merge(frame, on=["ISO3", "week"], how="left")
    out = out.merge(global_week, on="week", how="left")
    pdd_cols = [col for col in out.columns if col.startswith("pdd_")]
    out[pdd_cols] = out[pdd_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.sort_values(["ISO3", "week"]).reset_index(drop=True)


def write_report(
    panel: pd.DataFrame,
    disruptions: pd.DataFrame,
    ports: pd.DataFrame,
    active: pd.DataFrame,
    country_weekly: pd.DataFrame,
    panel_features: pd.DataFrame,
) -> None:
    coverage = (
        country_weekly.groupby("ISO3", as_index=False)
        .agg(
            nonzero_weeks=("pdd_own_active_event_count", lambda s: int((s > 0).sum())),
            red_weeks=("pdd_own_red_event_count", lambda s: int((s > 0).sum())),
            total_events=("pdd_own_active_event_count", "sum"),
            max_week_events=("pdd_own_active_event_count", "max"),
        )
        .sort_values(["nonzero_weeks", "total_events"], ascending=False)
    )
    port_country_coverage = (
        ports.assign(source_iso3=ports["country"].map(COUNTRY_TO_ISO3))
        .dropna(subset=["source_iso3"])
        .groupby(["source_iso3", "country"], as_index=False)
        .agg(rows=("eventid", "size"), events=("eventid", "nunique"), ports=("portid", "nunique"))
        .sort_values(["source_iso3", "rows"], ascending=[True, False])
    )
    content = f"""# PortWatch Direct Port-Disruption Panel14

## Purpose

This dataset converts the official IMF PortWatch `disruptions_with_ports` layer into country-week features for the 14-country benchmark. It is a direct operational-event layer: affected ports are mapped to their country, then partner-country events are propagated through total-import dependency weights with equal/random/shuffled placebos.

## Sources

- `portwatch_disruptions_database`
- `disruptions_with_ports`
- Existing panel: `data/processed/multicountry14_container_event_network_benchmark.csv`
- Existing weights: `data/interim/panel14_total_dependency_weights_2023.csv`

## Coverage

- Panel rows: {len(panel)}
- Panel countries: {panel["ISO3"].nunique()}
- Panel weeks: {panel["week"].nunique()} from {panel["week"].min().date()} to {panel["week"].max().date()}
- Official disruption events: {len(disruptions)}
- Disruption-port rows: {len(ports)}
- Active country-week event rows after panel-country mapping: {len(active)}
- Country-week output rows: {len(country_weekly)}
- Panel feature rows: {len(panel_features)}

## Direct Country Coverage

{coverage.to_markdown(index=False)}

## Port-Country Mapping Coverage

{port_country_coverage.to_markdown(index=False)}

## Interpretation

These are official disruption/event records tied to affected ports, but coverage is uneven across the 14-country panel. Direct features should be treated as high-precision event evidence; network/equal/random/shuffled features are exposure mappings for benchmarking and placebo checks, not causal effects.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(content, encoding="utf-8")


def main() -> None:
    panel = load_panel()
    disruptions, ports = load_sources()
    active = expand_active_event_weeks(panel, disruptions, ports)
    country_weekly = build_country_weekly(panel, active)
    panel_features = build_panel_features(panel, country_weekly)

    WEEKLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    country_weekly.to_csv(WEEKLY_OUT, index=False)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel_features.to_csv(PANEL_OUT, index=False)
    write_report(panel, disruptions, ports, active, country_weekly, panel_features)
    print(f"Wrote {WEEKLY_OUT} rows={len(country_weekly)}")
    print(f"Wrote {PANEL_OUT} rows={len(panel_features)}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
