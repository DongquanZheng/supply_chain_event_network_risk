from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_panel_benchmark_dataset as panel_builder  # noqa: E402
from src.panel32_config import PANEL32_GDELT_TO_ISO3  # noqa: E402


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_event_features_2021-01-01_2025-12-31_expanded32.csv"
)
DEFAULT_ME_EVENTS = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31_expanded32.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multicountry32_container_event_network_benchmark.csv"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "panel32_benchmark_dataset_summary.md"


def apply_expanded_mapping() -> None:
    panel_builder.GDELT_TO_ISO3.update(PANEL32_GDELT_TO_ISO3)
    panel_builder.ISO3_TO_GDELT.clear()
    panel_builder.ISO3_TO_GDELT.update(
        {iso3: gdelt for gdelt, iso3 in panel_builder.GDELT_TO_ISO3.items()}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--me-events", default=str(DEFAULT_ME_EVENTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def run() -> None:
    apply_expanded_mapping()
    panel_builder.build_dataset(parse_args())


if __name__ == "__main__":
    run()
