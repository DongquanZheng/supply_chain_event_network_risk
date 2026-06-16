"""Fetch sampled GDELT candidate documents for the expanded32 text NLP branch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from google.cloud import bigquery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.panel32_config import PANEL32_GDELT_CODES  # noqa: E402


DEFAULT_PROJECT = "supply-chain-network-risk"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_nlp_candidate_docs_2021-01-01_2025-12-31_expanded32.csv"
)


def quoted(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def build_query(start_date: str, end_date: str, max_docs_per_code_week: int, gdelt_codes: list[str]) -> str:
    code_sql = quoted(gdelt_codes)
    return f"""
WITH base AS (
  SELECT
    DATE_TRUNC(DATE(_PARTITIONTIME), WEEK(MONDAY)) AS event_week,
    SourceCommonName,
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
    AND DocumentIdentifier IS NOT NULL
    AND (
      REGEXP_CONTAINS(
        LOWER(CONCAT(
          DocumentIdentifier, ' ',
          IFNULL(V2Themes, ''), ' ',
          IFNULL(AllNames, ''), ' ',
          IFNULL(V2Organizations, '')
        )),
        r'(supply|shipping|freight|cargo|container|port|vessel|semiconductor|chip|electronics|machinery|factory|manufactur|export|import|trade|tariff|sanction|strike|disruption|delay|shortage|congestion|logistics|transport)'
      )
      OR V2Themes LIKE '%ECON_TRADE%'
      OR V2Themes LIKE '%WB_133_TRANSPORT%'
      OR V2Themes LIKE '%WB_135_TRANSPORT%'
      OR V2Themes LIKE '%WB_167_PORTS%'
    )
),
expanded AS (
  SELECT
    event_week,
    code,
    SourceCommonName,
    DocumentIdentifier,
    V2Themes,
    AllNames,
    V2Organizations,
    tone
  FROM base, UNNEST(codes) AS code
  WHERE code IN ({code_sql})
),
deduped AS (
  SELECT DISTINCT
    event_week,
    code,
    SourceCommonName,
    DocumentIdentifier,
    V2Themes,
    AllNames,
    V2Organizations,
    tone
  FROM expanded
),
sampled AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY event_week, code
      ORDER BY ABS(FARM_FINGERPRINT(DocumentIdentifier))
    ) AS sample_rank
  FROM deduped
)
SELECT
  event_week,
  code,
  SourceCommonName,
  DocumentIdentifier,
  V2Themes,
  AllNames,
  V2Organizations,
  tone
FROM sampled
WHERE sample_rank <= {max_docs_per_code_week}
ORDER BY event_week, code, sample_rank
"""


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Using cached file: {output}")
        return

    codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    client = bigquery.Client(project=args.project)
    query = build_query(args.start_date, args.end_date, args.max_docs_per_code_week, codes)
    dry_job = client.query(query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    estimated_gb = dry_job.total_bytes_processed / 1e9
    print(f"Estimated GB: {estimated_gb:.2f}")
    if args.dry_run:
        return

    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(args.maximum_gb_billed * 1e9),
        use_query_cache=True,
    )
    result = client.query(query, job_config=job_config).to_dataframe()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Rows: {len(result):,}")
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--codes", default=",".join(PANEL32_GDELT_CODES))
    parser.add_argument("--max-docs-per-code-week", type=int, default=40)
    parser.add_argument("--maximum-gb-billed", type=float, default=1500)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
