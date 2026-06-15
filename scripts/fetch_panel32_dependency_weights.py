from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import scripts.build_panel_benchmark_dataset as panel_builder  # noqa: E402
from scripts.build_expanded32_panel_benchmark_dataset import apply_expanded_mapping  # noqa: E402


PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
TOTAL_OUT = PROJECT_ROOT / "data" / "interim" / "panel32_total_dependency_weights_2023.csv"
ME_OUT = PROJECT_ROOT / "data" / "interim" / "panel32_me_dependency_weights_2023.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "panel32_dependency_weights_2023.md"


def summarize(weights: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    return (
        weights.groupby("ISO3")[weight_col]
        .agg(
            partner_count="count",
            top1_share="max",
            top3_share=lambda s: float(s.nlargest(3).sum()),
            hhi=lambda s: float((s**2).sum()),
            weight_sum="sum",
        )
        .reset_index()
        .sort_values("ISO3")
    )


def write_report(total: pd.DataFrame, me: pd.DataFrame) -> None:
    total_summary = summarize(total, "import_dependency_share")
    me_summary = summarize(me, "me_dependency_share")
    content = f"""# Panel32 WITS Dependency Weights 2023

## Purpose

This cache materializes the WITS 2023 dependency weights used for expanded32 mechanism diagnostics. It does not change benchmark model outputs; it makes trade-concentration descriptors reproducible for country-level failure analysis.

## Outputs

- `{TOTAL_OUT.relative_to(PROJECT_ROOT)}`
- `{ME_OUT.relative_to(PROJECT_ROOT)}`

## Coverage

| cache | countries | rows | min partners | max partners | min weight sum | max weight sum |
|:--|--:|--:|--:|--:|--:|--:|
| total imports | {total['ISO3'].nunique()} | {len(total)} | {int(total_summary['partner_count'].min())} | {int(total_summary['partner_count'].max())} | {total_summary['weight_sum'].min():.6f} | {total_summary['weight_sum'].max():.6f} |
| machinery/electronics | {me['ISO3'].nunique()} | {len(me)} | {int(me_summary['partner_count'].min())} | {int(me_summary['partner_count'].max())} | {me_summary['weight_sum'].min():.6f} | {me_summary['weight_sum'].max():.6f} |

## Total-Import Concentration Snapshot

{total_summary.sort_values('hhi', ascending=False).head(10).to_markdown(index=False)}

## Machinery/Electronics Concentration Snapshot

{me_summary.sort_values('hhi', ascending=False).head(10).to_markdown(index=False)}

## Reading

Use these files for expanded32 mechanism and failure-mode diagnostics. Because weights are normalized within the 32-country partner subset, interpret concentration as panel-relative dependency concentration, not as full-world import concentration.
"""
    REPORT_OUT.write_text(content, encoding="utf-8")


def run() -> None:
    panel = pd.read_csv(PANEL_PATH, usecols=["ISO3"])
    countries = sorted(panel["ISO3"].dropna().unique())
    apply_expanded_mapping()
    total = panel_builder.build_dependency_weights(countries)
    me = panel_builder.build_me_dependency_weights(countries)
    total.to_csv(TOTAL_OUT, index=False)
    me.to_csv(ME_OUT, index=False)
    write_report(total, me)
    print(f"Saved total weights: {TOTAL_OUT}")
    print(f"Saved ME weights: {ME_OUT}")
    print(f"Saved report: {REPORT_OUT}")


if __name__ == "__main__":
    run()
