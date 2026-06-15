from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fetch_portwatch_port_system_panel14 import (
    DAILY_PORTS_URL,
    build_country_weekly,
    build_port_weekly,
    fetch_arcgis_offset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTED_PORTS = PROJECT_ROOT / "reports" / "tables" / "panel32_port_system_selected_ports.csv"
FEASIBILITY = PROJECT_ROOT / "reports" / "tables" / "panel32_port_system_feasibility.csv"

DAILY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_daily.csv"
DAILY_PORT_DIR = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_daily_ports"
PORT_WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_port_weekly.csv"
COUNTRY_WEEKLY_OUT = PROJECT_ROOT / "data" / "interim" / "portwatch_port_system_panel32_country_weekly.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "portwatch_port_system_panel32.md"


def port_cache_path(portid: str) -> Path:
    safe = str(portid).replace("/", "_").replace("\\", "_")
    return DAILY_PORT_DIR / f"{safe}.csv"


def fetch_one_port(portid: str) -> pd.DataFrame:
    out_fields = (
        "date,portid,portname,country,ISO3,portcalls_container,portcalls,"
        "import_container,export_container,import,export"
    )
    return fetch_arcgis_offset(
        DAILY_PORTS_URL,
        where=f"portid='{portid}'",
        out_fields=out_fields,
        order_by="date ASC",
        timeout=180,
    )


def read_cached_ports(portids: list[str]) -> list[pd.DataFrame]:
    frames = []
    for portid in portids:
        path = port_cache_path(portid)
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["date"]))
    return frames


def fetch_daily_for_ports(selected: pd.DataFrame, force: bool, max_missing_ports: int | None) -> tuple[pd.DataFrame, bool]:
    if DAILY_OUT.exists() and not force:
        return pd.read_csv(DAILY_OUT, parse_dates=["date"]), True
    DAILY_PORT_DIR.mkdir(parents=True, exist_ok=True)
    portids = sorted(selected["portid"].dropna().unique())
    if force:
        for path in DAILY_PORT_DIR.glob("*.csv"):
            path.unlink()
    missing = [portid for portid in portids if not port_cache_path(portid).exists()]
    fetch_batch = missing if max_missing_ports is None else missing[:max_missing_ports]
    for idx, portid in enumerate(fetch_batch, start=1):
        frame = fetch_one_port(portid)
        frame["date"] = pd.to_datetime(frame["date"])
        frame.to_csv(port_cache_path(portid), index=False)
        print(
            f"Fetched missing port {idx}/{len(fetch_batch)} "
            f"({len(missing)} total missing before batch); {portid}: {len(frame)} daily rows"
        )
    remaining = [portid for portid in portids if not port_cache_path(portid).exists()]
    frames = read_cached_ports(portids)
    if not frames:
        return pd.DataFrame(), False
    daily = pd.concat(frames, ignore_index=True).sort_values(["ISO3", "portid", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    complete = len(remaining) == 0
    if complete:
        DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(DAILY_OUT, index=False)
    else:
        print(f"Partial cache only: {len(portids) - len(remaining)}/{len(portids)} ports fetched; {len(remaining)} remaining")
    return daily, complete


def write_report(selected: pd.DataFrame, daily: pd.DataFrame, port_weekly: pd.DataFrame, country_weekly: pd.DataFrame) -> None:
    feasibility = pd.read_csv(FEASIBILITY)
    coverage = country_weekly.groupby(["ISO3", "country"]).agg(
        weeks=("week", "nunique"),
        first_week=("week", "min"),
        last_week=("week", "max"),
        mean_active_container_ports=("ps_container_ports_active", "mean"),
        max_neg_ports_gt2=("ps_container_neg_z52_gt2_ports", "max"),
        mean_selected_container_calls=("ps_container_calls_selected", "mean"),
    ).reset_index()
    priority = feasibility.loc[feasibility["priority_failure_country"].astype(bool), [
        "ISO3",
        "country",
        "priority_reason",
        "selected_ports",
        "selected_container_coverage",
        "selected_import_share_coverage",
        "top_port",
    ]]
    content = f"""# PortWatch Port-System Panel32

## Purpose

This cache extends the public PortWatch port-level daily data branch from panel14 to expanded32. It provides direct operational port-system features: selected-port concentration, localized negative portcall anomalies, trade-flow balance, and country-week aggregation.

## Source

- Metadata selection: `reports/tables/panel32_port_system_selected_ports.csv`
- Daily service: `Daily_Ports_Data`

## Outputs

- Daily selected-port cache: `{DAILY_OUT.relative_to(PROJECT_ROOT)}`
- Port-week cache: `{PORT_WEEKLY_OUT.relative_to(PROJECT_ROOT)}`
- Country-week feature cache: `{COUNTRY_WEEKLY_OUT.relative_to(PROJECT_ROOT)}`

## Coverage

- Selected ports: `{selected['portid'].nunique()}`
- Daily rows: `{len(daily)}`
- Port-week rows: `{len(port_weekly)}`
- Country-week rows: `{len(country_weekly)}`
- Countries with country-week rows: `{country_weekly['ISO3'].nunique()}`
- Week range: `{country_weekly['week'].min()}` to `{country_weekly['week'].max()}`

## Priority Failure/Caveat Country Coverage

{priority.to_markdown(index=False)}

## Weekly Coverage

{coverage.to_markdown(index=False)}

## Reading

This is now a real expanded32 direct-operational data cache. The next benchmark step should merge `portwatch_port_system_panel32_country_weekly.csv` into the expanded32 PortWatch chokepoint/hazard dataset and test whether selected-port anomalies improve the deployment-mode failure countries, severe top-k, or transfer PR-AUC. Do not claim value until that temporal/LOCO benchmark is run.
"""
    REPORT_OUT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-missing-ports", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not SELECTED_PORTS.exists():
        raise FileNotFoundError(f"Run scripts/make_panel32_port_system_feasibility.py first: {SELECTED_PORTS}")
    selected = pd.read_csv(SELECTED_PORTS)
    daily, complete = fetch_daily_for_ports(selected, force=args.force, max_missing_ports=args.max_missing_ports)
    if not complete and not args.allow_partial:
        print("Daily port cache is incomplete. Re-run this script to resume; weekly outputs will be built after completion.")
        return
    port_weekly = build_port_weekly(daily, selected)
    country_weekly = build_country_weekly(port_weekly)
    PORT_WEEKLY_OUT.parent.mkdir(parents=True, exist_ok=True)
    port_weekly.to_csv(PORT_WEEKLY_OUT, index=False)
    country_weekly.to_csv(COUNTRY_WEEKLY_OUT, index=False)
    write_report(selected, daily, port_weekly, country_weekly)
    print(f"Saved daily: {DAILY_OUT}")
    print(f"Saved port weekly: {PORT_WEEKLY_OUT}")
    print(f"Saved country weekly: {COUNTRY_WEEKLY_OUT}")
    print(f"Saved report: {REPORT_OUT}")
    print(country_weekly.groupby('ISO3')['week'].nunique().to_string())


if __name__ == "__main__":
    run()
