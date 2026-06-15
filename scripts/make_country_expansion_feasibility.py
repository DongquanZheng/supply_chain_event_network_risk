from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_panel_benchmark_dataset import build_country_weekly  # noqa: E402
from src.config import GDELT_TO_ISO3  # noqa: E402
from src.portwatch import fetch_country_daily  # noqa: E402
from src.wits import build_partner_dependency_weights, fetch_partner_trade, fetch_partner_trade_by_product  # noqa: E402


RAW_GDELT = PROJECT_ROOT / "data" / "interim" / "gkg_partner_event_features_2021-01-01_2025-12-31.csv"
ME_GDELT = PROJECT_ROOT / "data" / "interim" / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "country_expansion_feasibility.md"
OUTPUT = TABLE_DIR / "country_expansion_feasibility.csv"
ME_PRODUCT_CODE = "84-85_MachElec"


CANDIDATES = [
    {
        "ISO3": "SGP",
        "country": "Singapore",
        "gdelt_code": "SN",
        "priority": "early_candidate",
    },
    {
        "ISO3": "MYS",
        "country": "Malaysia",
        "gdelt_code": "MY",
        "priority": "early_candidate",
    },
    {
        "ISO3": "NLD",
        "country": "Netherlands",
        "gdelt_code": "NL",
        "priority": "early_candidate",
    },
]


def gdelt_week_counts(path: Path, code: str) -> int | None:
    if not path.exists():
        return None
    data = pd.read_csv(path, usecols=["event_week", "code"], parse_dates=["event_week"])
    return int(data.loc[data["code"].eq(code), "event_week"].nunique())


def portwatch_check(iso3: str) -> dict:
    try:
        daily = fetch_country_daily(iso3, timeout=90)
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


def wits_check(target_iso3: str, partners: list[str], product: str | None = None) -> dict:
    try:
        if product:
            trade = fetch_partner_trade_by_product(target_iso3, year=2023, product=product)
        else:
            trade = fetch_partner_trade(target_iso3, year=2023)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        positive_weights = weights.loc[weights["import_dependency_share"].gt(0)]
        prefix = "wits_me" if product else "wits_total"
        return {
            f"{prefix}_status": "ok",
            f"{prefix}_mapped_partners": int(len(weights)),
            f"{prefix}_positive_partners": int(len(positive_weights)),
            f"{prefix}_top_partner": positive_weights.iloc[0]["partner_iso3"] if len(positive_weights) else "",
            f"{prefix}_top_share": float(positive_weights.iloc[0]["import_dependency_share"]) if len(positive_weights) else 0.0,
            f"{prefix}_error": "",
        }
    except Exception as exc:
        prefix = "wits_me" if product else "wits_total"
        return {
            f"{prefix}_status": "failed",
            f"{prefix}_mapped_partners": 0,
            f"{prefix}_positive_partners": 0,
            f"{prefix}_top_partner": "",
            f"{prefix}_top_share": 0.0,
            f"{prefix}_error": f"{type(exc).__name__}: {exc}",
        }


def build_feasibility() -> pd.DataFrame:
    current = sorted(set(GDELT_TO_ISO3.values()))
    candidate_iso3 = [row["ISO3"] for row in CANDIDATES]
    expanded_partners = sorted(set(current + candidate_iso3))

    rows = []
    for candidate in CANDIDATES:
        iso3 = candidate["ISO3"]
        partners = [partner for partner in expanded_partners if partner != iso3]
        row = dict(candidate)
        row.update(portwatch_check(iso3))
        row["raw_gdelt_cached_weeks"] = gdelt_week_counts(RAW_GDELT, candidate["gdelt_code"])
        row["me_gdelt_cached_weeks"] = gdelt_week_counts(ME_GDELT, candidate["gdelt_code"])
        row.update(wits_check(iso3, partners))
        row.update(wits_check(iso3, partners, product=ME_PRODUCT_CODE))

        blockers = []
        if row["portwatch_status"] != "ok" or row["locked_weekly_rows"] < 200:
            blockers.append("PortWatch coverage")
        if not row["raw_gdelt_cached_weeks"]:
            blockers.append("raw GDELT cache")
        if not row["me_gdelt_cached_weeks"]:
            blockers.append("ME strict GDELT cache")
        if row["wits_total_status"] != "ok" or row["wits_total_positive_partners"] == 0:
            blockers.append("WITS total weights")
        if row["wits_me_status"] != "ok" or row["wits_me_positive_partners"] == 0:
            blockers.append("WITS ME weights")
        row["expansion_status"] = "ready_after_cache_rebuild" if blockers == ["raw GDELT cache", "ME strict GDELT cache"] else "blocked"
        row["blockers"] = "; ".join(blockers) if blockers else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(feasibility: pd.DataFrame) -> None:
    content = f"""# Country Expansion Feasibility

## Purpose

This active research expansion checks whether the first three non-panel candidates can enter the 11-country benchmark after rebuilding event/network caches. It does not expand the locked dataset by itself; it identifies which data sources are ready and which must be fetched.

## Candidates

{feasibility.to_markdown(index=False)}

## Reading

If a country has usable PortWatch rows and WITS total/ME weights but zero cached GDELT weeks, the next concrete action is to extend `GDELT_TO_ISO3`, query GDELT GKG for the expanded country-code set, rebuild raw and ME-strict event caches, then regenerate the panel dataset and rerun rolling benchmarks.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    feasibility = build_feasibility()
    feasibility.to_csv(OUTPUT, index=False)
    write_report(feasibility)
    print(f"Saved table: {OUTPUT}")
    print(f"Saved report: {REPORT}")
    print(feasibility.to_string(index=False))


if __name__ == "__main__":
    run()
