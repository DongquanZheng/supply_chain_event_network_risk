"""Build a submission-readiness audit for the main IEEE BigData paper line.

This script is reporting-only. It does not train models and does not read any
fourth-source data. It checks whether the current PortWatch + GDELT + WITS
paper deliverables are present and records remaining writing/release work.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "main_paper_submission_readiness_audit.md"
CSV_PATH = TABLE_DIR / "main_paper_submission_readiness_audit.csv"


@dataclass(frozen=True)
class Requirement:
    deliverable_id: str
    requirement: str
    evidence_paths: tuple[str, ...]
    status_if_present: str
    remaining_action: str


REQUIREMENTS = (
    Requirement(
        "D1",
        "Main scripts are triaged into main/supporting/archive and non-main experiments are archived.",
        (
            "scripts/_archive_experiments",
            "docs/agent_harness/04_MODULE_MAP.md",
        ),
        "complete_for_current_main_line",
        "Do one final pass only if new scripts are added before submission.",
    ),
    Requirement(
        "D2",
        "Main-paper claim-evidence matrix exists and separates allowed claims from guardrails.",
        (
            "reports/main_paper_claim_evidence_matrix.md",
            "reports/tables/main_paper_claim_evidence_matrix.csv",
        ),
        "complete",
        "Keep synchronized if any result tables are regenerated.",
    ),
    Requirement(
        "D3",
        "Expanded32 main result tables are confirmed for operational, GDELT, WITS, gated, guarded, temporal, LOCO, top-k, severe, and shortfall rows.",
        (
            "reports/tables/main_paper_consolidated_model_ladder.csv",
            "reports/tables/main_paper_deployment_checks.csv",
            "reports/tables/ieee_bigdata_table2_main_model_ladder.md",
            "reports/tables/ieee_bigdata_table4_deployment_subgroups.md",
        ),
        "complete",
        "Prune columns for IEEE page limits during final formatting.",
    ),
    Requirement(
        "D4",
        "True/equal/random/shuffled WITS audit table is confirmed.",
        (
            "reports/tables/main_paper_network_audit_checks.csv",
            "reports/tables/ieee_bigdata_table3_network_audit.md",
        ),
        "complete",
        "Keep language guarded: true WITS does not consistently beat placebos.",
    ),
    Requirement(
        "D5",
        "APRS table and weight-sensitivity analysis are complete.",
        (
            "reports/main_paper_aprs.md",
            "reports/tables/main_paper_aprs_scores.csv",
            "reports/tables/main_paper_aprs_weight_sensitivity.csv",
            "scripts/make_main_paper_aprs_tables.py",
        ),
        "complete_reporting_metric",
        "Decide whether APRS appears as main Table 6 or an appendix reliability table.",
    ),
    Requirement(
        "D6",
        "GDELT conversion audit of 20-40 cases is complete.",
        (
            "reports/tables/gdelt_conversion_audit.csv",
            "reports/gdelt_conversion_audit.md",
            "reports/tables/ieee_bigdata_table5_gdelt_conversion_audit.md",
        ),
        "complete_metadata_derived",
        "Optional: add a small source/full-text appendix audit if article access is available.",
    ),
    Requirement(
        "D7",
        "Main figures/tables registry is updated and paper artifacts exist.",
        (
            "docs/agent_harness/05_FIGURE_TABLE_REGISTRY.md",
            "scripts/make_ieee_bigdata_paper_artifacts.py",
            "reports/figures/ieee_bigdata_fig1_framework.png",
            "reports/figures/ieee_bigdata_fig2_temporal_ladder.png",
            "reports/figures/ieee_bigdata_fig3_network_audit_deltas.png",
            "reports/figures/ieee_bigdata_fig4_shortfall_conversion.png",
            "reports/figures/ieee_bigdata_fig5_gdelt_conversion_audit.png",
        ),
        "complete",
        "Check figure readability once placed in IEEE two-column layout.",
    ),
    Requirement(
        "D8",
        "Paper skeleton and draft include claims, limitations, and overclaim guardrails.",
        (
            "paper/ieee_bigdata_skeleton.md",
            "paper/ieee_bigdata_draft.md",
        ),
        "complete_draft_ready",
        "Convert markdown draft into IEEE style and tighten result prose.",
    ),
)


TABLE_CHECKS = (
    "reports/tables/main_paper_claim_evidence_matrix.csv",
    "reports/tables/main_paper_consolidated_model_ladder.csv",
    "reports/tables/main_paper_network_audit_checks.csv",
    "reports/tables/main_paper_deployment_checks.csv",
    "reports/tables/main_paper_aprs_scores.csv",
    "reports/tables/main_paper_aprs_weight_sensitivity.csv",
    "reports/tables/gdelt_conversion_audit.csv",
)


FIGURE_CHECKS = tuple(
    f"reports/figures/ieee_bigdata_fig{i}_{name}.png"
    for i, name in (
        (1, "framework"),
        (2, "temporal_ladder"),
        (3, "network_audit_deltas"),
        (4, "shortfall_conversion"),
        (5, "gdelt_conversion_audit"),
    )
)


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def csv_row_count(path: str) -> int:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def file_size(path: str) -> int:
    return (ROOT / path).stat().st_size


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for req in REQUIREMENTS:
        missing = [path for path in req.evidence_paths if not exists(path)]
        status = "missing_evidence" if missing else req.status_if_present
        rows.append(
            {
                "deliverable_id": req.deliverable_id,
                "requirement": req.requirement,
                "status": status,
                "evidence_paths": "; ".join(req.evidence_paths),
                "missing_paths": "; ".join(missing),
                "remaining_action": req.remaining_action,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]]) -> None:
    completed = sum(1 for row in rows if not row["status"].startswith("missing"))
    table_rows = [(path, csv_row_count(path)) for path in TABLE_CHECKS if exists(path)]
    figures = [(path, file_size(path)) for path in FIGURE_CHECKS if exists(path)]

    lines = [
        "# Main Paper Submission Readiness Audit",
        "",
        "Date: 2026-06-15",
        "",
        "Scope: PortWatch + GDELT + WITS only. This audit is reporting-only; it does not train models or add data sources.",
        "",
        "## Overall Status",
        "",
        f"- Deliverables with current evidence: `{completed}/{len(rows)}`.",
        "- Main-paper status: ready for IEEE-style writing and formatting, not for stronger predictive overclaims.",
        "- Remaining work is writing/layout/source-review polish, not another main-paper model branch.",
        "",
        "## Deliverable Audit",
        "",
        "| ID | Requirement | Status | Evidence | Remaining action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {deliverable_id} | {requirement} | {status} | {evidence_paths} | {remaining_action} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Table Row Checks",
            "",
            "| Table | Rows |",
            "| --- | ---: |",
        ]
    )
    for path, count in table_rows:
        lines.append(f"| `{path}` | {count} |")

    lines.extend(
        [
            "",
            "## Figure File Checks",
            "",
            "| Figure | Bytes |",
            "| --- | ---: |",
        ]
    )
    for path, size in figures:
        lines.append(f"| `{path}` | {size} |")

    lines.extend(
        [
            "",
            "## Claim Readiness",
            "",
            "- Main evidence: expanded32 temporal ladder, LOCO check, WITS true/equal/random/shuffled audit, AA9 guarded integration, severe/top-k/shortfall subgroup checks, APRS reliability analysis, and 40-case GDELT conversion audit.",
            "- Supporting evidence: port-system vulnerability and earlier 11/14-country diagnostics can support motivation or appendix only.",
            "- Negative/archive evidence: failed wide/temporal/selective/smoothed gated variants, overlay/ranker selectors, and fourth-source probes should not be used as main-paper claims.",
            "- Guardrail: WITS is an audit/attribution layer; current evidence does not show that network gating universally beats additive or placebo controls.",
            "",
            "## Next Step",
            "",
            "Tighten `paper/ieee_bigdata_draft.md` into IEEE conference style, place Tables 1-6 and Figures 1-5, and decide whether an optional source/full-text appendix audit is worth doing. Do not open a new main-paper model branch unless the user explicitly reopens research expansion.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
