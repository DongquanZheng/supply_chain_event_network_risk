from __future__ import annotations

from google.cloud import bigquery
import pandas as pd


def make_client(project: str) -> bigquery.Client:
    return bigquery.Client(project=project)


def _quoted_codes(gdelt_codes: list[str]) -> str:
    return ", ".join(f"'{code}'" for code in gdelt_codes)


def build_country_week_event_query(
    start_date: str,
    end_date: str,
    gdelt_codes: list[str],
) -> str:
    code_sql = _quoted_codes(gdelt_codes)

    return f"""
WITH base AS (
  SELECT
    DATE_TRUNC(DATE(_PARTITIONTIME), WEEK(MONDAY)) AS event_week,
    DocumentIdentifier,
    V2Themes,
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
    tone
  FROM base, UNNEST(codes) AS code
  WHERE code IN ({code_sql})
),
deduped AS (
  SELECT DISTINCT
    event_week,
    code,
    DocumentIdentifier,
    V2Themes,
    tone
  FROM expanded
)
SELECT
  event_week,
  code,
  COUNT(*) AS article_count,
  AVG(tone) AS avg_tone,
  AVG(CASE WHEN tone < 0 THEN 1 ELSE 0 END) AS negative_article_share,
  AVG(CASE WHEN tone < -5 THEN 1 ELSE 0 END) AS very_negative_article_share,
  SUM(CASE
    WHEN V2Themes LIKE '%ECON_TRADE%'
      OR V2Themes LIKE '%WB_133_TRANSPORT%'
      OR V2Themes LIKE '%WB_135_TRANSPORT%'
      OR LOWER(DocumentIdentifier) LIKE '%shipping%'
      OR LOWER(DocumentIdentifier) LIKE '%freight%'
      OR LOWER(DocumentIdentifier) LIKE '%cargo%'
      OR LOWER(DocumentIdentifier) LIKE '%container%'
      OR LOWER(DocumentIdentifier) LIKE '%supply-chain%'
      OR LOWER(DocumentIdentifier) LIKE '%suez%'
      OR LOWER(DocumentIdentifier) LIKE '%panama-canal%'
      OR LOWER(DocumentIdentifier) LIKE '%red-sea%'
    THEN 1 ELSE 0 END
  ) AS trade_transport_count,
  SUM(CASE
    WHEN V2Themes LIKE '%CRISISLEX%'
      OR V2Themes LIKE '%ARMEDCONFLICT%'
      OR V2Themes LIKE '%PROTEST%'
      OR V2Themes LIKE '%STRIKE%'
      OR V2Themes LIKE '%SANCTION%'
      OR V2Themes LIKE '%NATURAL_DISASTER%'
      OR V2Themes LIKE '%MANMADE_DISASTER%'
      OR V2Themes LIKE '%USPEC_UNCERTAINTY%'
      OR LOWER(DocumentIdentifier) LIKE '%delay%'
      OR LOWER(DocumentIdentifier) LIKE '%disruption%'
      OR LOWER(DocumentIdentifier) LIKE '%strike%'
      OR LOWER(DocumentIdentifier) LIKE '%attack%'
      OR LOWER(DocumentIdentifier) LIKE '%sanction%'
      OR LOWER(DocumentIdentifier) LIKE '%blocked%'
      OR LOWER(DocumentIdentifier) LIKE '%reroute%'
    THEN 1 ELSE 0 END
  ) AS risk_theme_count
FROM deduped
GROUP BY event_week, code
ORDER BY event_week, article_count DESC
"""


def estimate_query_gb(client: bigquery.Client, query: str) -> float:
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    return job.total_bytes_processed / 1e9


def run_query(
    client: bigquery.Client,
    query: str,
    maximum_gb_billed: float,
) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=int(maximum_gb_billed * 1e9),
        use_query_cache=True,
    )
    return client.query(query, job_config=job_config).to_dataframe()

