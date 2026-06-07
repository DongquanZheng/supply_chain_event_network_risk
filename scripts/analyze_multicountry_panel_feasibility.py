from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3
from src.portwatch import fetch_country_daily


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "multicountry_panel_feasibility.md"


def build_weekly_container_target(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = (
        daily.assign(week=daily["date"].dt.to_period("W-SUN").dt.start_time)
        .groupby(["ISO3", "country", "week"], as_index=False)
        .agg(
            portcalls_container=("portcalls_container", "sum"),
            days_observed=("date", "nunique"),
        )
        .sort_values(["ISO3", "week"])
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["days_observed"].eq(7)].copy()

    frames = []
    for _, group in weekly.groupby("ISO3"):
        group = group.sort_values("week").copy()
        group["next_week_container"] = group["portcalls_container"].shift(-1)
        group["rolling_mean_12w"] = group["portcalls_container"].rolling(12).mean()
        group["rolling_std_12w"] = group["portcalls_container"].rolling(12).std()
        group["abnormal_threshold"] = (
            group["rolling_mean_12w"] - 1.5 * group["rolling_std_12w"]
        )
        group["abnormal_next_week_container"] = (
            group["next_week_container"] < group["abnormal_threshold"]
        ).astype(int)
        frames.append(group)

    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(
        subset=["next_week_container", "rolling_mean_12w", "rolling_std_12w"]
    ).reset_index(drop=True)


def run() -> None:
    iso3_codes = sorted(set(GDELT_TO_ISO3.values()))
    frames = []
    failed = []

    for iso3 in iso3_codes:
        try:
            daily = fetch_country_daily(iso3, timeout=90)
            if not daily.empty:
                frames.append(build_weekly_container_target(daily))
        except Exception as exc:  # pragma: no cover - diagnostic script
            failed.append({"iso3": iso3, "error": f"{type(exc).__name__}: {exc}"})

    panel = pd.concat(frames, ignore_index=True)
    summary = (
        panel.groupby(["ISO3", "country"], as_index=False)
        .agg(
            rows=("week", "size"),
            positive_labels=("abnormal_next_week_container", "sum"),
            min_week=("week", "min"),
            max_week=("week", "max"),
        )
        .sort_values("rows", ascending=False)
    )
    summary["positive_rate"] = summary["positive_labels"] / summary["rows"]

    year_summary = (
        panel.groupby(panel["week"].dt.year)
        .agg(rows=("week", "size"), positive_labels=("abnormal_next_week_container", "sum"))
        .reset_index()
        .rename(columns={"week": "year"})
    )
    year_summary["positive_rate"] = year_summary["positive_labels"] / year_summary["rows"]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLE_DIR / "multicountry_panel_feasibility_by_country.csv"
    year_path = TABLE_DIR / "multicountry_panel_feasibility_by_year.csv"
    summary.to_csv(summary_path, index=False)
    year_summary.to_csv(year_path, index=False)

    failed_text = (
        pd.DataFrame(failed).to_markdown(index=False) if failed else "No PortWatch fetch failures."
    )
    content = f"""# Multi-Country Panel Feasibility

## Purpose

This diagnostic checks whether a multi-country panel is a plausible next direction for making the network layer empirically meaningful.

## Scope

- Countries: mapped GDELT/ISO3 countries currently used in the Japan-centered network experiments.
- Target: next-week abnormal weekly container port calls, using the same rolling 12-week threshold logic.

## Overall

- Countries with data: {summary["ISO3"].nunique()}
- Panel rows: {len(panel)}
- Positive labels: {int(panel["abnormal_next_week_container"].sum())}
- Positive rate: {panel["abnormal_next_week_container"].mean():.3f}
- Week range: {panel["week"].min().date()} to {panel["week"].max().date()}

## By Country

{summary.to_markdown(index=False)}

## By Year

{year_summary.to_markdown(index=False)}

## Fetch Failures

{failed_text}

## Interpretation

If the panel has enough positive labels across countries, network structure should be developed in a panel setting rather than by adding more Japan-only transformations. A panel allows the same event environment to be mapped through different country-specific dependency structures.
"""
    REPORT.write_text(content, encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {year_path}")
    print(f"Report: {REPORT}")
    print(summary.to_string(index=False))
    print(year_summary.to_string(index=False))


if __name__ == "__main__":
    run()
