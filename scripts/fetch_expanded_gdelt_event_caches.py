from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from google.cloud import bigquery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gkg_bigquery import build_country_week_event_query  # noqa: E402


DEFAULT_PROJECT = "supply-chain-network-risk"
DEFAULT_CODES = ["US", "CH", "KS", "AS", "AE", "SA", "VM", "TH", "ID", "GM", "JA", "SN", "MY", "NL"]
DEFAULT_RAW_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_event_features_2021-01-01_2025-12-31_expanded14.csv"
)
DEFAULT_ME_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31_expanded14.csv"
)


def quoted(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def build_me_strict_query(start_date: str, end_date: str, gdelt_codes: list[str]) -> str:
    code_sql = quoted(gdelt_codes)
    return f"""
WITH base AS (
  SELECT
    DATE_TRUNC(DATE(_PARTITIONTIME), WEEK(MONDAY)) AS event_week,
    DocumentIdentifier,
    V2Themes,
    AllNames,
    V2Organizations,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    REGEXP_EXTRACT_ALL(V2Locations, r'#([A-Z]{{2}})#') AS codes
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    _PARTITIONDATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
    AND V2Locations IS NOT NULL
    AND V2Tone IS NOT NULL
),
expanded AS (
  SELECT
    event_week,
    code,
    DocumentIdentifier,
    V2Themes,
    AllNames,
    V2Organizations,
    tone,
    REGEXP_CONTAINS(
      LOWER(CONCAT(
        DocumentIdentifier, ' ',
        IFNULL(V2Themes, ''), ' ',
        IFNULL(AllNames, ''), ' ',
        IFNULL(V2Organizations, '')
      )),
      r'(semiconductor|semiconductors|chip|chips|electronics|electronic|machinery|machine|machines|factory|factories|manufactur|industrial equipment|supply chain)'
    ) AS machinery_electronics_flag,
    REGEXP_CONTAINS(
      LOWER(CONCAT(
        DocumentIdentifier, ' ',
        IFNULL(V2Themes, ''), ' ',
        IFNULL(AllNames, ''), ' ',
        IFNULL(V2Organizations, '')
      )),
      r'(disruption|disruptions|delay|delays|shortage|shortages|congestion|strike|strikes|attack|attacks|blocked|blockage|reroute|rerouting|shutdown|halt|crisis|conflict|war|earthquake|flood|storm|typhoon|fire|explosion)'
    ) AS disruption_flag
  FROM base, UNNEST(codes) AS code
  WHERE code IN ({code_sql})
),
deduped AS (
  SELECT DISTINCT
    event_week,
    code,
    DocumentIdentifier,
    tone,
    machinery_electronics_flag,
    disruption_flag
  FROM expanded
)
SELECT
  event_week,
  code,
  COUNT(*) AS article_count,
  AVG(CASE
    WHEN machinery_electronics_flag AND disruption_flag AND tone < -5 THEN 1 ELSE 0
  END) AS machinery_electronics_disruption_very_negative_share,
  SUM(CASE
    WHEN machinery_electronics_flag AND disruption_flag THEN 1 ELSE 0
  END) AS machinery_electronics_disruption_article_count
FROM deduped
GROUP BY event_week, code
ORDER BY event_week, code
"""


def complete_week_code_grid(df: pd.DataFrame, start_date: str, end_date: str, gdelt_codes: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["event_week"] = pd.to_datetime(out["event_week"])
    weeks = pd.date_range(
        pd.to_datetime(start_date).to_period("W-SUN").start_time,
        pd.to_datetime(end_date).to_period("W-SUN").start_time,
        freq="W-MON",
    )
    grid = pd.MultiIndex.from_product([weeks, gdelt_codes], names=["event_week", "code"]).to_frame(index=False)
    out = grid.merge(out, on=["event_week", "code"], how="left")
    numeric = [col for col in out.columns if col not in {"event_week", "code"}]
    out[numeric] = out[numeric].fillna(0)
    return out.sort_values(["event_week", "code"]).reset_index(drop=True)


def estimate_query_gb(client: bigquery.Client, query: str) -> float:
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    return job.total_bytes_processed / 1e9


def run_query(client: bigquery.Client, query: str, maximum_gb_billed: float) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(maximum_gb_billed * 1e9),
        use_query_cache=True,
    )
    return client.query(query, job_config=job_config).to_dataframe()


def maybe_fetch(
    client: bigquery.Client,
    query: str,
    output: Path,
    start_date: str,
    end_date: str,
    gdelt_codes: list[str],
    maximum_gb_billed: float,
    dry_run: bool,
    force: bool,
) -> None:
    estimated = estimate_query_gb(client, query)
    print(f"{output.name}: estimated GB = {estimated:.2f}")
    if dry_run:
        return
    if output.exists() and not force:
        print(f"Using existing cache: {output}")
        return
    result = run_query(client, query, maximum_gb_billed=maximum_gb_billed)
    result = complete_week_code_grid(result, start_date, end_date, gdelt_codes)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Rows: {len(result)}")
    print(f"Saved: {output}")


def run(args: argparse.Namespace) -> None:
    gdelt_codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    client = bigquery.Client(project=args.project)

    if args.kind in {"raw", "both"}:
        raw_query = build_country_week_event_query(args.start_date, args.end_date, gdelt_codes)
        maybe_fetch(
            client,
            raw_query,
            Path(args.raw_output),
            args.start_date,
            args.end_date,
            gdelt_codes,
            args.maximum_gb_billed,
            args.dry_run,
            args.force,
        )

    if args.kind in {"me", "both"}:
        me_query = build_me_strict_query(args.start_date, args.end_date, gdelt_codes)
        maybe_fetch(
            client,
            me_query,
            Path(args.me_output),
            args.start_date,
            args.end_date,
            gdelt_codes,
            args.maximum_gb_billed,
            args.dry_run,
            args.force,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES))
    parser.add_argument("--kind", choices=["raw", "me", "both"], default="both")
    parser.add_argument("--maximum-gb-billed", type=float, default=1500)
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--me-output", default=str(DEFAULT_ME_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
