from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from google.cloud import bigquery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_expanded_gdelt_event_caches import complete_week_code_grid  # noqa: E402


DEFAULT_PROJECT = "supply-chain-network-risk"
DEFAULT_CODES = ["US", "CH", "KS", "AS", "AE", "SA", "VM", "TH", "ID", "GM", "JA", "SN", "MY", "NL"]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_strict_taxonomy_event_features_2021-01-01_2025-12-31_expanded14.csv"
)


CATEGORY_PATTERNS = {
    "maritime": (
        r"(port|ports|terminal|terminals|shipping|ship|ships|vessel|vessels|"
        r"freight|cargo|container|containers|maritime|logistics|suez|panama canal|"
        r"panama-canal|red sea|red-sea|strait|canal)"
    ),
    "trade_policy": (
        r"(tariff|tariffs|sanction|sanctions|export control|export ban|import ban|"
        r"customs|trade war|trade restriction|trade restrictions|embargo|blacklist)"
    ),
    "weather": (
        r"(typhoon|hurricane|cyclone|storm|flood|flooding|earthquake|tsunami|"
        r"wildfire|landslide|extreme weather|heavy rain|drought)"
    ),
    "conflict_security": (
        r"(attack|attacks|war|conflict|armed conflict|missile|drone|piracy|houthi|"
        r"terror|security threat|blockade|military)"
    ),
    "energy_transport": (
        r"(oil|gas|fuel|energy|lng|crude|petroleum|tanker|tankers|refinery|"
        r"pipeline|electricity|power grid|coal)"
    ),
    "manufacturing_electronics": (
        r"(semiconductor|semiconductors|chip|chips|electronics|electronic|"
        r"machinery|machine|machines|factory|factories|manufactur|industrial equipment|"
        r"supply chain|supply-chain|automotive|battery|batteries)"
    ),
}

DISRUPTION_PATTERN = (
    r"(disruption|disruptions|delay|delays|delayed|shortage|shortages|congestion|"
    r"strike|strikes|walkout|attack|attacks|blocked|blockage|reroute|rerouting|"
    r"shutdown|shut down|halt|halts|crisis|collapse|closed|closure|suspended|"
    r"fire|explosion|accident|grounded|stuck)"
)


def quoted(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def sql_flag(name: str, pattern: str, require_disruption: bool = True) -> str:
    category_flag = f"REGEXP_CONTAINS(text_blob, r'{pattern}')"
    if require_disruption:
        return f"({category_flag} AND REGEXP_CONTAINS(text_blob, r'{DISRUPTION_PATTERN}')) AS {name}_flag"
    return f"{category_flag} AS {name}_flag"


def category_selects(name: str) -> list[str]:
    return [
        f"SUM(CASE WHEN {name}_flag THEN 1 ELSE 0 END) AS {name}_article_count",
        f"AVG(CASE WHEN {name}_flag THEN 1 ELSE 0 END) AS {name}_article_share",
        f"AVG(CASE WHEN {name}_flag AND tone < -5 THEN 1 ELSE 0 END) AS {name}_very_negative_share",
        (
            f"SUM(CASE WHEN {name}_flag THEN GREATEST(-tone, 0) ELSE 0 END) "
            f"AS {name}_negative_severity"
        ),
    ]


def build_query(start_date: str, end_date: str, gdelt_codes: list[str]) -> str:
    code_sql = quoted(gdelt_codes)
    flag_sql = ",\n    ".join(
        sql_flag(name, pattern, require_disruption=(name not in {"trade_policy", "weather", "conflict_security"}))
        for name, pattern in CATEGORY_PATTERNS.items()
    )
    select_sql = ",\n  ".join(
        item
        for name in CATEGORY_PATTERNS
        for item in category_selects(name)
    )

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
    tone,
    LOWER(CONCAT(
      IFNULL(DocumentIdentifier, ''), ' ',
      IFNULL(V2Themes, ''), ' ',
      IFNULL(AllNames, ''), ' ',
      IFNULL(V2Organizations, '')
    )) AS text_blob
  FROM base, UNNEST(codes) AS code
  WHERE code IN ({code_sql})
),
flagged AS (
  SELECT DISTINCT
    event_week,
    code,
    DocumentIdentifier,
    tone,
    {flag_sql}
  FROM expanded
)
SELECT
  event_week,
  code,
  COUNT(*) AS strict_candidate_article_count,
  {select_sql}
FROM flagged
GROUP BY event_week, code
ORDER BY event_week, code
"""


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


def run(args: argparse.Namespace) -> None:
    gdelt_codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    client = bigquery.Client(project=args.project)
    query = build_query(args.start_date, args.end_date, gdelt_codes)
    estimated = estimate_query_gb(client, query)
    print(f"{Path(args.output).name}: estimated GB = {estimated:.2f}")
    if args.dry_run:
        return

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Using existing cache: {output}")
        return

    result = run_query(client, query, args.maximum_gb_billed)
    result = complete_week_code_grid(result, args.start_date, args.end_date, gdelt_codes)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Rows: {len(result)}")
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES))
    parser.add_argument("--maximum-gb-billed", type=float, default=1500)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
