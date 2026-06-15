from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3, ISO3_TO_GDELT  # noqa: E402


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
FEASIBILITY = PROJECT_ROOT / "reports" / "tables" / "multicountry_panel_feasibility_by_country.csv"
EVENTS = PROJECT_ROOT / "data" / "interim" / "gkg_partner_event_features_2021-01-01_2025-12-31.csv"
ME_EVENTS = PROJECT_ROOT / "data" / "interim" / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv"
TOTAL_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_total_dependency_weights_2023.csv"
ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT = TABLE_DIR / "panel_country_scope_audit.csv"
REPORT = PROJECT_ROOT / "reports" / "panel_country_scope_audit.md"

TARGET = "abnormal_next_week_container"

EARLY_PORTWATCH_FEASIBLE_NOT_INCLUDED = [
    {
        "ISO3": "SGP",
        "country": "Singapore",
        "reason": "PortWatch was feasible in early tests, but the locked 2021-2025 GDELT/WITS event-network cache does not include this ISO3/GDELT code.",
    },
    {
        "ISO3": "MYS",
        "country": "Malaysia",
        "reason": "PortWatch was feasible in early tests, but the locked 2021-2025 GDELT/WITS event-network cache does not include this ISO3/GDELT code.",
    },
    {
        "ISO3": "NLD",
        "country": "Netherlands",
        "reason": "PortWatch was feasible in early tests, but the locked 2021-2025 GDELT/WITS event-network cache does not include this ISO3/GDELT code.",
    },
]


def load_event_coverage(path: Path, label: str) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_week"])
    events["ISO3"] = events["code"].map(GDELT_TO_ISO3)
    coverage = (
        events.dropna(subset=["ISO3"])
        .groupby("ISO3", as_index=False)
        .agg(
            **{
                f"{label}_event_weeks": ("event_week", "nunique"),
                f"{label}_first_event_week": ("event_week", "min"),
                f"{label}_last_event_week": ("event_week", "max"),
            }
        )
    )
    return coverage


def partner_counts(path: Path, label: str) -> pd.DataFrame:
    weights = pd.read_csv(path)
    return (
        weights.groupby("ISO3", as_index=False)
        .agg(**{f"{label}_partner_count": ("partner_iso3", "nunique")})
    )


def build_audit() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET, parse_dates=["week"])
    feasibility = pd.read_csv(FEASIBILITY, parse_dates=["min_week", "max_week"])
    benchmark = (
        dataset.groupby(["ISO3", "country"], as_index=False)
        .agg(
            benchmark_rows=("week", "size"),
            benchmark_positives=(TARGET, "sum"),
            benchmark_first_week=("week", "min"),
            benchmark_last_week=("week", "max"),
        )
    )
    benchmark["benchmark_positive_rate"] = benchmark["benchmark_positives"] / benchmark["benchmark_rows"]

    event_coverage = load_event_coverage(EVENTS, "raw_gdelt")
    me_event_coverage = load_event_coverage(ME_EVENTS, "me_strict")
    total_partners = partner_counts(TOTAL_WEIGHTS, "total_network")
    me_partners = partner_counts(ME_WEIGHTS, "me_network")

    included = (
        feasibility.merge(benchmark, on=["ISO3", "country"], how="left")
        .merge(event_coverage, on="ISO3", how="left")
        .merge(me_event_coverage, on="ISO3", how="left")
        .merge(total_partners, on="ISO3", how="left")
        .merge(me_partners, on="ISO3", how="left")
    )
    included["gdelt_code"] = included["ISO3"].map(ISO3_TO_GDELT)
    included["scope_status"] = "included_locked_panel"
    included["scope_decision"] = (
        "Included: complete PortWatch weekly container panel, mapped GDELT country code, "
        "cached 2021-2025 raw and machinery/electronics event features, and WITS 2023 total/ME network weights."
    )

    excluded = pd.DataFrame(EARLY_PORTWATCH_FEASIBLE_NOT_INCLUDED)
    excluded["gdelt_code"] = excluded["ISO3"].map(ISO3_TO_GDELT)
    excluded["scope_status"] = "excluded_requires_new_event_network_cache"
    excluded["scope_decision"] = excluded["reason"]
    for col in included.columns:
        if col not in excluded.columns:
            excluded[col] = pd.NA
    excluded = excluded[included.columns]

    excluded_non_empty = excluded.dropna(axis=1, how="all")
    audit = pd.concat([included, excluded_non_empty], ignore_index=True)
    ordered = [
        "ISO3",
        "country",
        "gdelt_code",
        "scope_status",
        "scope_decision",
        "rows",
        "positive_labels",
        "positive_rate",
        "min_week",
        "max_week",
        "benchmark_rows",
        "benchmark_positives",
        "benchmark_positive_rate",
        "benchmark_first_week",
        "benchmark_last_week",
        "raw_gdelt_event_weeks",
        "me_strict_event_weeks",
        "total_network_partner_count",
        "me_network_partner_count",
    ]
    return audit[ordered].sort_values(["scope_status", "ISO3"]).reset_index(drop=True)


def write_report(audit: pd.DataFrame) -> None:
    included = audit.loc[audit["scope_status"].eq("included_locked_panel")].copy()
    excluded = audit.loc[~audit["scope_status"].eq("included_locked_panel")].copy()
    criteria = [
        "complete country-level PortWatch weekly container data with enough positive labels",
        "explicit GDELT country-code to ISO3 mapping",
        "cached weekly 2021-2025 GDELT event features for every country-week",
        "cached machinery/electronics strict GDELT event features",
        "WITS 2023 total-import and machinery/electronics dependency weights against the same country set",
        "balanced panel support for temporal rolling-origin validation",
    ]
    criteria_text = "\n".join(f"- {item}" for item in criteria)

    content = f"""# Panel Country Scope Audit

## Purpose

This audit addresses Gate 4: whether the current 11-country panel should be expanded or justified as a curated benchmark sample.

## Inclusion Criteria

{criteria_text}

## Current Locked Panel

- Countries: {included["ISO3"].nunique()}
- Benchmark rows: {int(included["benchmark_rows"].sum())}
- Benchmark positives: {int(included["benchmark_positives"].sum())}
- Benchmark positive rate: {included["benchmark_positives"].sum() / included["benchmark_rows"].sum():.3f}
- Benchmark week range: {pd.to_datetime(included["benchmark_first_week"]).min().date()} to {pd.to_datetime(included["benchmark_last_week"]).max().date()}
- Raw GDELT event weeks per included country: {int(included["raw_gdelt_event_weeks"].min())} to {int(included["raw_gdelt_event_weeks"].max())}
- Total-network partner count per included country: {int(included["total_network_partner_count"].min())} to {int(included["total_network_partner_count"].max())}

## Included Countries

{included[["ISO3", "country", "gdelt_code", "benchmark_rows", "benchmark_positives", "benchmark_positive_rate", "raw_gdelt_event_weeks", "me_strict_event_weeks", "total_network_partner_count", "me_network_partner_count"]].to_markdown(index=False)}

## Feasible But Not Included In Locked Cache

{excluded[["ISO3", "country", "scope_status", "scope_decision"]].to_markdown(index=False)}

## Interpretation

The 11-country panel is defensible as a curated applied benchmark sample, not as a global port-disruption benchmark. The sample is controlled by a strict intersection of PortWatch operational coverage, explicit GDELT country-code mapping, cached 2021-2025 event features, WITS 2023 dependency weights, and balanced temporal validation.

Expanding to 20+ countries is possible in principle, but it is not a drop-in change. It requires selecting additional countries, validating GDELT country-code mappings, querying and caching new GDELT event features, rebuilding WITS dependency weights over the expanded country set, regenerating network exposure features, and rerunning all placebo/rolling-origin diagnostics. Until that work is done, claims should be scoped to an 11-country curated benchmark.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    audit.to_csv(OUTPUT, index=False)
    write_report(audit)
    print(f"Saved: {OUTPUT}")
    print(f"Report: {REPORT}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    run()
