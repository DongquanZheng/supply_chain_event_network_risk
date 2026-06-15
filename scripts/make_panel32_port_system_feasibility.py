from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fetch_portwatch_port_system_panel14 import fetch_arcgis_offset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services"
PORT_META_URL = f"{BASE_URL}/PortWatch_ports_database/FeatureServer/0/query"

PANEL32 = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
SEVERE_GUARDRAILS = PROJECT_ROOT / "reports" / "tables" / "panel32_deployment_severe_guardrail_countries.csv"

META_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_metadata.csv"
FEASIBILITY_OUT = PROJECT_ROOT / "reports" / "tables" / "panel32_port_system_feasibility.csv"
SELECTED_PORTS_OUT = PROJECT_ROOT / "reports" / "tables" / "panel32_port_system_selected_ports.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "portwatch_port_system_panel32_feasibility.md"

TOP_N = 12


def panel_countries() -> pd.DataFrame:
    panel = pd.read_csv(PANEL32, usecols=["ISO3", "country"])
    return panel.drop_duplicates().sort_values("ISO3").reset_index(drop=True)


def fetch_metadata(force: bool = False) -> pd.DataFrame:
    if META_OUT.exists() and not force:
        return pd.read_csv(META_OUT)
    out_fields = (
        "portid,portname,country,ISO3,lat,lon,vessel_count_total,vessel_count_container,"
        "share_country_maritime_import,share_country_maritime_export"
    )
    meta = fetch_arcgis_offset(PORT_META_URL, out_fields=out_fields, order_by="ISO3 ASC")
    panel_iso3 = panel_countries()["ISO3"].tolist()
    meta = meta.loc[meta["ISO3"].isin(panel_iso3)].copy()
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


def select_ports(meta: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    rows = []
    for iso3, frame in meta.groupby("ISO3", sort=False):
        candidates = frame.loc[frame["vessel_count_container"].gt(0)].copy()
        if candidates.empty:
            candidates = frame.copy()
        selected = candidates.sort_values(
            ["vessel_count_container", "share_country_maritime_import", "vessel_count_total"],
            ascending=[False, False, False],
        ).head(top_n)
        selected = selected.copy()
        total_container = frame["vessel_count_container"].sum()
        total_import_share = frame["share_country_maritime_import"].sum()
        selected["selected_rank"] = range(1, len(selected) + 1)
        selected["selected_top_n"] = top_n
        selected["selected_container_coverage"] = (
            selected["vessel_count_container"].sum() / total_container if total_container else np.nan
        )
        selected["selected_import_share_coverage"] = (
            selected["share_country_maritime_import"].sum() / total_import_share if total_import_share else np.nan
        )
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def failure_priority() -> pd.DataFrame:
    if not SEVERE_GUARDRAILS.exists():
        return pd.DataFrame(columns=["ISO3", "priority_reason"])
    guard = pd.read_csv(SEVERE_GUARDRAILS)
    pr_fail = guard.loc[
        guard["LOCK4_70haz30op_vs_LOCK0_op_main_pr_auc_delta"].lt(0)
        | guard["LOCK4_70haz30op_vs_LOCK0_op_severe_pr_auc_delta"].lt(0)
        | guard["LOCK4_70haz30op_vs_LOCK0_op_severe_top25_delta"].lt(0)
    ].copy()
    reasons = []
    for _, row in pr_fail.iterrows():
        flags = []
        if row["LOCK4_70haz30op_vs_LOCK0_op_main_pr_auc_delta"] < 0:
            flags.append("main_pr_auc_loss")
        if row["LOCK4_70haz30op_vs_LOCK0_op_severe_pr_auc_delta"] < 0:
            flags.append("severe_pr_auc_loss")
        if row["LOCK4_70haz30op_vs_LOCK0_op_severe_top25_delta"] < 0:
            flags.append("severe_top25_loss")
        reasons.append({"ISO3": row["ISO3"], "priority_reason": "+".join(flags)})
    return pd.DataFrame(reasons)


def build_feasibility(meta: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    panel = panel_countries()
    total = meta.groupby("ISO3").agg(
        total_panel_ports=("portid", "nunique"),
        total_container_vessel_count=("vessel_count_container", "sum"),
        total_vessel_count=("vessel_count_total", "sum"),
        total_import_share=("share_country_maritime_import", "sum"),
    )
    selected_summary = selected.groupby("ISO3").agg(
        selected_ports=("portid", "nunique"),
        selected_container_vessel_count=("vessel_count_container", "sum"),
        selected_import_share=("share_country_maritime_import", "sum"),
        selected_container_coverage=("selected_container_coverage", "max"),
        selected_import_share_coverage=("selected_import_share_coverage", "max"),
        top_port=("portname", "first"),
        top_port_container_vessels=("vessel_count_container", "first"),
    )
    out = panel.merge(total, on="ISO3", how="left").merge(selected_summary, on="ISO3", how="left")
    out = out.merge(failure_priority(), on="ISO3", how="left")
    out["priority_reason"] = out["priority_reason"].fillna("")
    out["metadata_ready"] = out["total_panel_ports"].fillna(0).gt(0)
    out["selected_daily_fetch_candidate"] = (
        out["metadata_ready"]
        & out["selected_ports"].fillna(0).gt(0)
        & out["selected_container_vessel_count"].fillna(0).gt(0)
    )
    out["priority_failure_country"] = out["priority_reason"].ne("")
    return out.sort_values(["priority_failure_country", "ISO3"], ascending=[False, True]).reset_index(drop=True)


def write_report(feasibility: pd.DataFrame, selected: pd.DataFrame) -> None:
    priority = feasibility.loc[feasibility["priority_failure_country"]].copy()
    low_coverage = feasibility.loc[
        feasibility["selected_daily_fetch_candidate"]
        & feasibility["selected_container_coverage"].lt(0.80)
    ].copy()
    top_ports = selected.loc[selected["selected_rank"].le(3), [
        "ISO3",
        "country",
        "selected_rank",
        "portid",
        "portname",
        "vessel_count_container",
        "share_country_maritime_import",
    ]]
    content = f"""# Panel32 PortWatch Port-System Feasibility

## Purpose

This feasibility check asks whether the public PortWatch port-level metadata can support an expanded32 direct operational data branch before fetching daily port time series. It focuses especially on countries where the locked transfer router fails or has severe-label caveats.

## Source

- PortWatch ArcGIS service: `PortWatch_ports_database`
- Candidate downstream daily service: `Daily_Ports_Data`

## Coverage Summary

- Panel32 countries: `{feasibility['ISO3'].nunique()}`
- Countries with PortWatch port metadata: `{int(feasibility['metadata_ready'].sum())}`
- Countries with selected container-port candidates: `{int(feasibility['selected_daily_fetch_candidate'].sum())}`
- Selected ports if fetching top `{TOP_N}` per country: `{int(selected['portid'].nunique())}`
- Priority failure/caveat countries covered by metadata: `{int(priority['metadata_ready'].sum())}/{len(priority)}`

## Priority Failure/Caveat Countries

{priority[['ISO3','country','priority_reason','total_panel_ports','selected_ports','selected_container_coverage','selected_import_share_coverage','top_port']].to_markdown(index=False)}

## Low Selected-Port Container Coverage

{low_coverage[['ISO3','country','selected_container_coverage','selected_import_share_coverage','selected_ports','total_panel_ports','top_port']].to_markdown(index=False)}

## Top Selected Ports Snapshot

{top_ports.head(96).to_markdown(index=False)}

## Reading

If priority countries have metadata and plausible selected-port coverage, the next concrete branch should fetch `Daily_Ports_Data` for the selected expanded32 ports, build weekly port-system anomaly/concentration features, and test them as direct operational data against the locked transfer/deployment references. If coverage is poor for a failure country, prefer other direct feeds such as AIS queue/waiting-time, labor/closure, schedule/blank-sailing, or route-duration data.
"""
    REPORT_OUT.write_text(content, encoding="utf-8")


def run() -> None:
    meta = fetch_metadata()
    selected = select_ports(meta)
    feasibility = build_feasibility(meta, selected)
    FEASIBILITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    feasibility.to_csv(FEASIBILITY_OUT, index=False)
    selected.to_csv(SELECTED_PORTS_OUT, index=False)
    write_report(feasibility, selected)
    print(f"Saved metadata: {META_OUT}")
    print(f"Saved feasibility: {FEASIBILITY_OUT}")
    print(f"Saved selected ports: {SELECTED_PORTS_OUT}")
    print(f"Saved report: {REPORT_OUT}")
    print(feasibility[["ISO3", "country", "priority_reason", "selected_daily_fetch_candidate", "selected_container_coverage"]].to_string(index=False))


if __name__ == "__main__":
    run()
