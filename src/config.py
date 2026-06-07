from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetConfig:
    name: str
    iso3: str
    gdelt_code: str
    wits_reporter: str


JAPAN = TargetConfig(
    name="Japan",
    iso3="JPN",
    gdelt_code="JA",
    wits_reporter="JPN",
)

CHINA = TargetConfig(
    name="China",
    iso3="CHN",
    gdelt_code="CH",
    wits_reporter="CHN",
)


GDELT_TO_ISO3 = {
    "JA": "JPN",
    "CH": "CHN",
    "KS": "KOR",
    "US": "USA",
    "AS": "AUS",
    "AE": "ARE",
    "SA": "SAU",
    "VM": "VNM",
    "TH": "THA",
    "ID": "IDN",
    "GM": "DEU",
}


ISO3_TO_GDELT = {iso3: gdelt for gdelt, iso3 in GDELT_TO_ISO3.items()}


OPERATIONAL_FEATURES = [
    "lag_portcalls_1w",
    "lag_portcalls_2w",
    "lag_portcalls_4w",
    "rolling_mean_4w",
    "rolling_mean_8w",
    "rolling_std_4w",
    "rolling_std_8w",
    "rolling_change_4w",
    "month",
    "quarter",
]

