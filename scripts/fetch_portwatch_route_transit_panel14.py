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
CONNECTION_URL = f"{BASE_URL}/Spillover_Simulator_Maritime_Connections/FeatureServer/0/query"
IMPACT_URL = f"{BASE_URL}/spillovers_port_level_impact/FeatureServer/0/query"

PANEL14 = PROJECT_ROOT / "data" / "processed" / "multicountry14_container_event_network_benchmark.csv"
DEPENDENCY_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel14_total_dependency_weights_2023.csv"

CONNECTION_EDGES_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_route_transit_connections_panel14.csv"
IMPACT_EDGES_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_route_transit_impact_edges_panel14.csv"
PAIR_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_route_transit_country_pair_panel14.csv"
REPORT = PROJECT_ROOT / "reports" / "portwatch_route_transit_panel14.md"

RANDOM_SEED = 42


def panel_countries() -> list[str]:
    return sorted(pd.read_csv(PANEL14, usecols=["ISO3"])["ISO3"].dropna().unique())


def arcgis_where_for_panel(countries: list[str]) -> str:
    quoted = ", ".join(f"'{country}'" for country in countries)
    return f"from_iso3 IN ({quoted}) AND to_iso3 IN ({quoted}) AND from_iso3 <> to_iso3"


def fetch_arcgis_offset(url: str, *, where: str, order_by: str | None = None, timeout: int = 120) -> pd.DataFrame:
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


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(values.mean()) if values.notna().any() else np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def aggregate_impact_edges(impact: pd.DataFrame) -> pd.DataFrame:
    data = impact.copy()
    data["average_transit_days"] = pd.to_numeric(data["average_transit_days"], errors="coerce")
    data["daily_capacity_at_risk"] = pd.to_numeric(data["daily_capacity_at_risk"], errors="coerce").fillna(0.0)
    data["relative_capacity_at_risk"] = pd.to_numeric(data["relative_capacity_at_risk"], errors="coerce").fillna(0.0)
    data["impact_weight"] = data["daily_capacity_at_risk"].clip(lower=0.0)
    no_capacity = data["impact_weight"].le(0)
    data.loc[no_capacity, "impact_weight"] = data.loc[no_capacity, "relative_capacity_at_risk"].clip(lower=0.0)

    rows = []
    for (partner, target), group in data.groupby(["from_iso3", "to_iso3"], sort=False):
        weights = group["impact_weight"]
        transit = group["average_transit_days"]
        total_capacity = float(group["daily_capacity_at_risk"].sum())
        total_relative = float(group["relative_capacity_at_risk"].sum())
        denom = float(weights.sum())
        if denom > 0:
            fast_share = float(weights.loc[transit.le(7)].sum() / denom)
            medium_share = float(weights.loc[transit.gt(7) & transit.le(21)].sum() / denom)
            long_share = float(weights.loc[transit.gt(21)].sum() / denom)
        else:
            fast_share = medium_share = long_share = 0.0
        rows.append(
            {
                "partner_iso3": partner,
                "ISO3": target,
                "impact_edge_count": len(group),
                "impact_origin_ports": group["from_portid"].nunique(),
                "impact_destination_ports": group["to_portid"].nunique(),
                "impact_transit_days_mean": float(transit.mean()),
                "impact_transit_days_median": float(transit.median()),
                "impact_transit_days_weighted": weighted_mean(transit, weights),
                "daily_capacity_at_risk_sum": total_capacity,
                "relative_capacity_at_risk_sum": total_relative,
                "impact_fast_share_le7d": fast_share,
                "impact_medium_share_8_21d": medium_share,
                "impact_long_share_gt21d": long_share,
            }
        )
    return pd.DataFrame(rows)


def aggregate_connection_edges(connections: pd.DataFrame) -> pd.DataFrame:
    data = connections.copy()
    data["transit_days"] = pd.to_numeric(data["transit_days"], errors="coerce")
    data["distance"] = pd.to_numeric(data["distance"], errors="coerce")
    return (
        data.groupby(["from_iso3", "to_iso3"], as_index=False)
        .agg(
            connection_edge_count=("from_id", "size"),
            connection_origin_ports=("from_id", "nunique"),
            connection_destination_ports=("to_id", "nunique"),
            connection_transit_days_mean=("transit_days", "mean"),
            connection_transit_days_median=("transit_days", "median"),
            connection_distance_mean=("distance", "mean"),
            connection_distance_median=("distance", "median"),
        )
        .rename(columns={"from_iso3": "partner_iso3", "to_iso3": "ISO3"})
    )


def add_weights_and_lags(pair: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    weights = pd.read_csv(DEPENDENCY_WEIGHTS)
    weights = weights.loc[weights["ISO3"].isin(countries) & weights["partner_iso3"].isin(countries)].copy()
    weights = weights.loc[weights["ISO3"].ne(weights["partner_iso3"])].copy()
    out = weights.merge(pair, on=["ISO3", "partner_iso3"], how="left")
    out["route_pair_observed"] = out["impact_edge_count"].notna() | out["connection_edge_count"].notna()

    transit_candidates = [
        "impact_transit_days_weighted",
        "impact_transit_days_median",
        "connection_transit_days_median",
        "connection_transit_days_mean",
    ]
    out["route_transit_days"] = np.nan
    for col in transit_candidates:
        out["route_transit_days"] = out["route_transit_days"].fillna(out[col])
    target_median = out.groupby("ISO3")["route_transit_days"].transform("median")
    global_median = float(out["route_transit_days"].median())
    out["route_transit_days"] = out["route_transit_days"].fillna(target_median).fillna(global_median)
    out["route_lag_weeks"] = np.rint(out["route_transit_days"] / 7.0).clip(0, 4).astype(int)

    for col in [
        "daily_capacity_at_risk_sum",
        "relative_capacity_at_risk_sum",
        "impact_fast_share_le7d",
        "impact_medium_share_8_21d",
        "impact_long_share_gt21d",
        "impact_edge_count",
        "connection_edge_count",
        "connection_distance_mean",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["route_capacity_share"] = out.groupby("ISO3")["relative_capacity_at_risk_sum"].transform(
        lambda s: s.clip(lower=0.0) / s.clip(lower=0.0).sum() if s.clip(lower=0.0).sum() > 0 else 1.0 / len(s)
    )
    out["route_hybrid_weight_raw"] = 0.7 * out["import_dependency_share"] + 0.3 * out["route_capacity_share"]
    out["route_hybrid_weight"] = out.groupby("ISO3")["route_hybrid_weight_raw"].transform(lambda s: s / s.sum())
    rng = np.random.default_rng(RANDOM_SEED + 177)
    random_values = rng.random(len(out))
    out["route_random_weight_raw"] = random_values
    out["route_random_weight"] = out.groupby("ISO3")["route_random_weight_raw"].transform(lambda s: s / s.sum())
    out = out.drop(columns=["route_hybrid_weight_raw", "route_random_weight_raw"])
    return out.sort_values(["ISO3", "partner_iso3"]).reset_index(drop=True)


def load_or_fetch(force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    countries = panel_countries()
    where = arcgis_where_for_panel(countries)
    if CONNECTION_EDGES_OUT.exists() and IMPACT_EDGES_OUT.exists() and not force:
        connections = pd.read_csv(CONNECTION_EDGES_OUT)
        impact = pd.read_csv(IMPACT_EDGES_OUT)
        return connections, impact

    connections = fetch_arcgis_offset(CONNECTION_URL, where=where, order_by="from_iso3 ASC, to_iso3 ASC")
    impact = fetch_arcgis_offset(IMPACT_URL, where=where, order_by="from_iso3 ASC, to_iso3 ASC")
    for path, frame in [(CONNECTION_EDGES_OUT, connections), (IMPACT_EDGES_OUT, impact)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return connections, impact


def write_report(pair: pd.DataFrame, connections: pd.DataFrame, impact: pd.DataFrame) -> None:
    lag_dist = pair["route_lag_weeks"].value_counts().sort_index().rename_axis("route_lag_weeks").reset_index(name="pairs")
    observed = pair["route_pair_observed"].value_counts().rename_axis("route_pair_observed").reset_index(name="pairs")
    content = f"""# PortWatch Route Transit Panel14

## Purpose

This cache adds official PortWatch route/transit spillover structure to the 14-country benchmark. It is designed for route-lagged event exposure: partner-country event signals can be shifted by estimated maritime transit time before entering a target-country alert model.

## Sources

- ArcGIS service: `Spillover_Simulator_Maritime_Connections`
- ArcGIS service: `spillovers_port_level_impact`
- Query scope: panel14 country pairs only, `from_iso3 != to_iso3`
- Downloaded/updated: {pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

## Outputs

- `data/interim/portwatch_route_transit_connections_panel14.csv`
- `data/interim/portwatch_route_transit_impact_edges_panel14.csv`
- `data/interim/portwatch_route_transit_country_pair_panel14.csv`

## Coverage

- Raw connection edges: {len(connections)}
- Raw impact edges: {len(impact)}
- Country-pair rows after merging WITS dependency weights: {len(pair)}
- Countries: {pair["ISO3"].nunique()}
- Partners: {pair["partner_iso3"].nunique()}
- Observed route pairs:

{observed.to_markdown(index=False)}

## Route Lag Distribution

{lag_dist.to_markdown(index=False)}

## Interpretation Rules

Use these variables as exposure mapping, not causal proof. The primary candidate weight is `route_hybrid_weight`, a transparent blend of WITS import dependency and PortWatch route capacity share. Placebo checks should use `equal_weight` and `route_random_weight` under the same lagging protocol.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refetch PortWatch ArcGIS layers even if local caches exist.")
    return parser.parse_args()


def run() -> None:
    countries = panel_countries()
    connections, impact = load_or_fetch(parse_args().force)
    pair = aggregate_impact_edges(impact).merge(
        aggregate_connection_edges(connections), on=["ISO3", "partner_iso3"], how="outer"
    )
    pair = add_weights_and_lags(pair, countries)
    PAIR_OUT.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(PAIR_OUT, index=False)
    write_report(pair, connections, impact)
    print(f"Saved route pair table: {PAIR_OUT}")
    print(f"Saved report: {REPORT}")
    print(pair[["ISO3", "partner_iso3", "route_transit_days", "route_lag_weeks", "route_hybrid_weight"]].head().to_string(index=False))


if __name__ == "__main__":
    run()
