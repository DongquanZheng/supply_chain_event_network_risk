from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd
import requests


WITS_TRADE_URL = (
    "https://wits.worldbank.org/API/V1/SDMX/V21/"
    "datasource/tradestats-trade/reporter/{reporter}/year/{year}/"
    "partner/all/product/{product}/indicator/{indicator}"
)


def fetch_partner_trade(
    reporter: str,
    year: int,
    indicator: str = "MPRT-TRD-VL",
    timeout: int = 60,
) -> pd.DataFrame:
    url = WITS_TRADE_URL.format(
        reporter=reporter,
        year=year,
        product="Total",
        indicator=indicator,
    )
    return _fetch_partner_trade_url(url, timeout=timeout)


def fetch_partner_trade_by_product(
    reporter: str,
    year: int,
    product: str,
    indicator: str = "MPRT-TRD-VL",
    timeout: int = 60,
) -> pd.DataFrame:
    url = WITS_TRADE_URL.format(
        reporter=reporter,
        year=year,
        product=product,
        indicator=indicator,
    )
    return _fetch_partner_trade_url(url, timeout=timeout)


def _fetch_partner_trade_url(url: str, timeout: int = 60) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows = []

    for series in root.iter():
        if not series.tag.endswith("Series"):
            continue

        attrs = series.attrib
        for obs in series:
            if obs.tag.endswith("Obs"):
                rows.append(
                    {
                        "reporter": attrs.get("REPORTER"),
                        "partner_iso3": attrs.get("PARTNER"),
                        "year": int(obs.attrib["TIME_PERIOD"]),
                        "indicator": attrs.get("INDICATOR"),
                        "value_thousand_usd": float(obs.attrib["OBS_VALUE"]),
                    }
                )

    return pd.DataFrame(rows)


def build_partner_dependency_weights(
    trade_df: pd.DataFrame,
    partner_iso3: list[str],
    normalize_within_subset: bool = True,
) -> pd.DataFrame:
    weights = trade_df.loc[trade_df["partner_iso3"].isin(partner_iso3)].copy()

    if normalize_within_subset:
        denominator = weights["value_thousand_usd"].sum()
    else:
        denominator = trade_df.loc[
            trade_df["partner_iso3"].ne("WLD"),
            "value_thousand_usd",
        ].sum()

    weights["import_dependency_share"] = weights["value_thousand_usd"] / denominator
    return weights.sort_values("import_dependency_share", ascending=False).reset_index(drop=True)
