from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_panel_benchmark_dataset import build_country_weekly  # noqa: E402
from scripts.build_expanded14_panel_benchmark_dataset import EXTRA_GDELT_TO_ISO3  # noqa: E402
from src.config import GDELT_TO_ISO3  # noqa: E402
from src.portwatch import fetch_country_daily  # noqa: E402
from src.wits import build_partner_dependency_weights, fetch_partner_trade, fetch_partner_trade_by_product  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel20_country_expansion_feasibility.md"
OUTPUT = TABLE_DIR / "panel20_country_expansion_feasibility.csv"
ME_PRODUCT_CODE = "84-85_MachElec"

BASE14 = {
    **GDELT_TO_ISO3,
    **EXTRA_GDELT_TO_ISO3,
}

EXTRA_CANDIDATES = [
    {"ISO3": "BEL", "country": "Belgium", "gdelt_code": "BE", "region": "Europe"},
    {"ISO3": "CAN", "country": "Canada", "gdelt_code": "CA", "region": "North America"},
    {"ISO3": "ESP", "country": "Spain", "gdelt_code": "SP", "region": "Europe"},
    {"ISO3": "FRA", "country": "France", "gdelt_code": "FR", "region": "Europe"},
    {"ISO3": "GBR", "country": "United Kingdom", "gdelt_code": "UK", "region": "Europe"},
    {"ISO3": "ITA", "country": "Italy", "gdelt_code": "IT", "region": "Europe"},
    {"ISO3": "TUR", "country": "Turkey", "gdelt_code": "TU", "region": "Europe/West Asia"},
    {"ISO3": "IND", "country": "India", "gdelt_code": "IN", "region": "South Asia"},
    {"ISO3": "BRA", "country": "Brazil", "gdelt_code": "BR", "region": "Latin America"},
    {"ISO3": "MEX", "country": "Mexico", "gdelt_code": "MX", "region": "North America"},
    {"ISO3": "PAN", "country": "Panama", "gdelt_code": "PM", "region": "Latin America"},
    {"ISO3": "CHL", "country": "Chile", "gdelt_code": "CI", "region": "Latin America"},
    {"ISO3": "ZAF", "country": "South Africa", "gdelt_code": "SF", "region": "Africa"},
    {"ISO3": "PHL", "country": "Philippines", "gdelt_code": "RP", "region": "Southeast Asia"},
    {"ISO3": "EGY", "country": "Egypt", "gdelt_code": "EG", "region": "North Africa"},
    {"ISO3": "PAK", "country": "Pakistan", "gdelt_code": "PK", "region": "South Asia"},
    {"ISO3": "POL", "country": "Poland", "gdelt_code": "PL", "region": "Europe"},
    {"ISO3": "SWE", "country": "Sweden", "gdelt_code": "SW", "region": "Europe"},
]


def portwatch_check(iso3: str, timeout: int) -> dict:
    try:
        daily = fetch_country_daily(iso3, timeout=timeout)
        weekly = build_country_weekly(daily)
        locked = weekly.loc[(weekly["week"] >= "2021-01-01") & (weekly["week"] < "2026-01-01")]
        return {
            "portwatch_status": "ok",
            "portwatch_daily_rows": len(daily),
            "portwatch_weekly_rows": len(weekly),
            "locked_weekly_rows": len(locked),
            "locked_positive_labels": int(locked["abnormal_next_week_container"].sum()),
            "locked_positive_rate": float(locked["abnormal_next_week_container"].mean()) if len(locked) else 0.0,
            "portwatch_week_min": weekly["week"].min().date().isoformat() if len(weekly) else "",
            "portwatch_week_max": weekly["week"].max().date().isoformat() if len(weekly) else "",
            "portwatch_error": "",
        }
    except Exception as exc:
        return {
            "portwatch_status": "failed",
            "portwatch_daily_rows": 0,
            "portwatch_weekly_rows": 0,
            "locked_weekly_rows": 0,
            "locked_positive_labels": 0,
            "locked_positive_rate": 0.0,
            "portwatch_week_min": "",
            "portwatch_week_max": "",
            "portwatch_error": f"{type(exc).__name__}: {exc}",
        }


def wits_check(target_iso3: str, partners: list[str], timeout: int, product: str | None = None) -> dict:
    prefix = "wits_me" if product else "wits_total"
    try:
        if product:
            trade = fetch_partner_trade_by_product(target_iso3, year=2023, product=product, timeout=timeout)
        else:
            trade = fetch_partner_trade(target_iso3, year=2023, timeout=timeout)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        positive = weights.loc[weights["import_dependency_share"].gt(0)]
        return {
            f"{prefix}_status": "ok",
            f"{prefix}_mapped_partners": int(len(weights)),
            f"{prefix}_positive_partners": int(len(positive)),
            f"{prefix}_top_partner": positive.iloc[0]["partner_iso3"] if len(positive) else "",
            f"{prefix}_top_share": float(positive.iloc[0]["import_dependency_share"]) if len(positive) else 0.0,
            f"{prefix}_error": "",
        }
    except Exception as exc:
        return {
            f"{prefix}_status": "failed",
            f"{prefix}_mapped_partners": 0,
            f"{prefix}_positive_partners": 0,
            f"{prefix}_top_partner": "",
            f"{prefix}_top_share": 0.0,
            f"{prefix}_error": f"{type(exc).__name__}: {exc}",
        }


def assess_candidate(candidate: dict, partners: list[str], args: argparse.Namespace) -> dict:
    iso3 = candidate["ISO3"]
    row = {
        **candidate,
        "gdelt_mapping_status": "manual_fips_candidate",
        "already_in_base14": iso3 in set(BASE14.values()),
    }
    row.update(portwatch_check(iso3, timeout=args.portwatch_timeout))
    if args.portwatch_only:
        row.update(
            {
                "wits_total_status": "not_checked",
                "wits_total_mapped_partners": 0,
                "wits_total_positive_partners": 0,
                "wits_total_top_partner": "",
                "wits_total_top_share": 0.0,
                "wits_total_error": "",
                "wits_me_status": "not_checked",
                "wits_me_mapped_partners": 0,
                "wits_me_positive_partners": 0,
                "wits_me_top_partner": "",
                "wits_me_top_share": 0.0,
                "wits_me_error": "",
            }
        )
    else:
        row.update(wits_check(iso3, partners, timeout=args.wits_timeout))
        row.update(wits_check(iso3, partners, timeout=args.wits_timeout, product=ME_PRODUCT_CODE))

    blockers = []
    if row["portwatch_status"] != "ok" or row["locked_weekly_rows"] < args.min_locked_weeks:
        blockers.append("PortWatch coverage")
    if row["locked_positive_labels"] < args.min_positives:
        blockers.append("too few positive labels")
    if row["wits_total_status"] not in {"ok", "not_checked"} or row["wits_total_positive_partners"] < (
        1 if not args.portwatch_only else 0
    ):
        blockers.append("WITS total weights")
    if row["wits_me_status"] not in {"ok", "not_checked"} or row["wits_me_positive_partners"] < (
        1 if not args.portwatch_only else 0
    ):
        blockers.append("WITS ME weights")
    row["expansion_status"] = "ready_for_event_cache" if not blockers else "blocked"
    row["blockers"] = "; ".join(blockers) if blockers else "none"
    return row


def build_feasibility(args: argparse.Namespace) -> pd.DataFrame:
    current_iso3 = sorted(set(BASE14.values()))
    candidate_iso3 = [row["ISO3"] for row in EXTRA_CANDIDATES]
    partners = sorted(set(current_iso3 + candidate_iso3))
    rows = []
    for candidate in EXTRA_CANDIDATES:
        rows.append(assess_candidate(candidate, [p for p in partners if p != candidate["ISO3"]], args))
    out = pd.DataFrame(rows)
    return out.sort_values(["expansion_status", "locked_positive_labels", "locked_weekly_rows"], ascending=[False, False, False])


def write_report(feasibility: pd.DataFrame, args: argparse.Namespace) -> None:
    ready = feasibility.loc[feasibility["expansion_status"].eq("ready_for_event_cache")]
    content = f"""# Panel20+ Country Expansion Feasibility

## Purpose

This audit tests whether a broader country set can plausibly extend the current 14-country exploratory panel toward a 20+ country applied benchmark. It does not fetch GDELT event caches or rebuild the panel. It checks whether each candidate has enough PortWatch operational labels and WITS network weights to justify the next, more expensive GDELT/cache step.

## Settings

- Minimum locked weeks: {args.min_locked_weeks}
- Minimum positive labels: {args.min_positives}
- WITS checked: {not args.portwatch_only}

## Summary

- Candidates tested: {len(feasibility)}
- Ready for event-cache build: {len(ready)}
- Ready ISO3: {", ".join(ready["ISO3"].tolist()) if len(ready) else "none"}

## Candidate Table

{feasibility.to_markdown(index=False)}

## Reading

Countries marked `ready_for_event_cache` are not yet in the benchmark. They are candidates for the next expensive step: extend the GDELT country-code mapping, fetch expanded raw/strict/direct-title event caches, rebuild WITS weights over the expanded country set, regenerate the panel, and rerun temporal/placebo benchmarks.

Manual GDELT mappings in this file should be verified before BigQuery cache generation. Do not use this feasibility table as a paper result; use it as Gate 4 engineering evidence.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portwatch-timeout", type=int, default=90)
    parser.add_argument("--wits-timeout", type=int, default=60)
    parser.add_argument("--min-locked-weeks", type=int, default=200)
    parser.add_argument("--min-positives", type=int, default=10)
    parser.add_argument("--portwatch-only", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    feasibility = build_feasibility(args)
    feasibility.to_csv(OUTPUT, index=False)
    write_report(feasibility, args)
    print(f"Saved table: {OUTPUT}")
    print(f"Saved report: {REPORT}")
    print(feasibility.to_string(index=False))


if __name__ == "__main__":
    run()
