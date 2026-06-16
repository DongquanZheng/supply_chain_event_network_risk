from __future__ import annotations

from src.config import GDELT_TO_ISO3 as BASE_GDELT_TO_ISO3


PANEL14_EXTRA_GDELT_TO_ISO3 = {
    "SN": "SGP",
    "MY": "MYS",
    "NL": "NLD",
}


PANEL32_GDELT_TO_ISO3 = {
    **BASE_GDELT_TO_ISO3,
    **PANEL14_EXTRA_GDELT_TO_ISO3,
    "BE": "BEL",
    "CA": "CAN",
    "SP": "ESP",
    "FR": "FRA",
    "UK": "GBR",
    "IT": "ITA",
    "TU": "TUR",
    "IN": "IND",
    "BR": "BRA",
    "MX": "MEX",
    "PM": "PAN",
    "CI": "CHL",
    "SF": "ZAF",
    "RP": "PHL",
    "EG": "EGY",
    "PK": "PAK",
    "PL": "POL",
    "SW": "SWE",
}

PANEL32_GDELT_CODES = sorted(PANEL32_GDELT_TO_ISO3)
PANEL32_ISO3_TO_GDELT = {iso3: code for code, iso3 in PANEL32_GDELT_TO_ISO3.items()}
