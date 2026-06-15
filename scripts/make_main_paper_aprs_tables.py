"""Build APRS tables for the main PortWatch+GDELT+WITS paper line.

APRS is a reporting-only metric: it combines existing predictive, network-audit,
and guardrail evidence without retraining models or adding data sources.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"

MODEL_LADDER = TABLE_DIR / "main_paper_consolidated_model_ladder.csv"
NETWORK_AUDIT = TABLE_DIR / "main_paper_network_audit_checks.csv"

OUT_SCORES = TABLE_DIR / "main_paper_aprs_scores.csv"
OUT_SENSITIVITY = TABLE_DIR / "main_paper_aprs_weight_sensitivity.csv"
OUT_REPORT = ROOT / "reports" / "main_paper_aprs.md"

MATERIAL_PR_AUC_DELTA = 0.05

WEIGHT_SCENARIOS = {
    "recommended_0p5_0p3_0p2": (0.50, 0.30, 0.20),
    "balanced_equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "performance_heavy": (0.60, 0.20, 0.20),
    "audit_heavy": (0.25, 0.50, 0.25),
    "guardrail_heavy": (0.25, 0.25, 0.50),
}

TEMPORAL_EVALUATIONS = {
    "temporal_main_ladder",
    "temporal_conversion_propensity",
    "temporal_guarded_integration",
}

AUDIT_SCOPE_MAP = {
    "temporal_compact_gated": "temporal_known_country",
    "temporal_conversion_propensity": "temporal_known_country",
    "temporal_guarded_AA9": "temporal_known_country",
    "loco_compact_gated": "loco_transfer",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def minmax(value: float, values: list[float]) -> float:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return 1.0
    return (value - lo) / (hi - lo)


def evaluation_scope(evaluation: str) -> str | None:
    if evaluation in TEMPORAL_EVALUATIONS:
        return "temporal_known_country"
    if evaluation == "loco_transfer":
        return "loco_transfer"
    return None


def build_audit_index(audit_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in audit_rows:
        scope = AUDIT_SCOPE_MAP.get(row["evaluation"])
        if scope is None:
            continue
        baseline = row["baseline_id"].lower()
        if not any(token in baseline for token in ("placebo", "equal", "random", "shuffled")):
            continue
        key = (scope, row["focus_id"])
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows in grouped.items():
        # "Best placebo" means the placebo that is hardest to beat; because
        # deltas are focus minus baseline, this is the minimum true-vs-placebo
        # delta among equal/random/shuffled controls.
        worst = min(rows, key=lambda r: to_float(r["pooled_pr_auc_delta"]))
        delta = to_float(worst["pooled_pr_auc_delta"])
        out[key] = {
            "best_placebo_baseline": worst["baseline_id"],
            "true_vs_best_placebo_pr_auc_delta": delta,
            "true_vs_best_placebo_top25_delta": to_float(worst.get("top25_hit_delta")),
            "audit_reliability_score": clamp(0.5 + delta / MATERIAL_PR_AUC_DELTA),
            "audit_note": "true-vs-best-placebo separation from network audit table",
        }
    return out


def model_label(row: dict[str, str]) -> str:
    return f"{row['artifact_id']}::{row['model']}"


def main() -> None:
    ladder_rows = read_csv(MODEL_LADDER)
    audit_index = build_audit_index(read_csv(NETWORK_AUDIT))

    candidates: list[dict[str, object]] = []
    for row in ladder_rows:
        scope = evaluation_scope(row["evaluation"])
        if scope is None:
            continue
        candidates.append(
            {
                "evaluation_scope": scope,
                "evaluation": row["evaluation"],
                "paper_bucket": row["paper_bucket"],
                "is_deployable_candidate": "no"
                if row["paper_bucket"] == "placebo_audit"
                else "yes",
                "artifact_id": row["artifact_id"],
                "model": row["model"],
                "candidate_id": model_label(row),
                "mean_main_pr_auc": to_float(row["mean_main_pr_auc"]),
                "main_top25_hits": to_float(row["main_top25_hits"]),
                "mean_severe_pr_auc": to_float(row["mean_severe_pr_auc"]),
                "severe_top25_hits": to_float(row["severe_top25_hits"]),
            }
        )

    by_scope: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        by_scope.setdefault(str(row["evaluation_scope"]), []).append(row)

    score_rows: list[dict[str, object]] = []
    for scope, rows in by_scope.items():
        pr_values = [float(r["mean_main_pr_auc"]) for r in rows]
        top25_values = [float(r["main_top25_hits"]) for r in rows]
        severe_top25_values = [float(r["severe_top25_hits"]) for r in rows]
        for row in rows:
            audit = audit_index.get((scope, str(row["artifact_id"])))
            if audit is None:
                audit = {
                    "best_placebo_baseline": "",
                    "true_vs_best_placebo_pr_auc_delta": "",
                    "true_vs_best_placebo_top25_delta": "",
                    "audit_reliability_score": 0.0,
                    "audit_note": "no true-vs-placebo audit contrast for this row",
                }
            predictive_score = minmax(float(row["mean_main_pr_auc"]), pr_values)
            main_top25_score = minmax(float(row["main_top25_hits"]), top25_values)
            severe_top25_score = minmax(float(row["severe_top25_hits"]), severe_top25_values)
            guardrail_score = 0.5 * main_top25_score + 0.5 * severe_top25_score
            alpha, beta, gamma = WEIGHT_SCENARIOS["recommended_0p5_0p3_0p2"]
            aprs = (
                alpha * predictive_score
                + beta * float(audit["audit_reliability_score"])
                + gamma * guardrail_score
            )
            score_rows.append(
                {
                    **row,
                    "predictive_performance_score": round(predictive_score, 6),
                    "audit_reliability_score": round(float(audit["audit_reliability_score"]), 6),
                    "guardrail_robustness_score": round(guardrail_score, 6),
                    "recommended_aprs": round(aprs, 6),
                    "best_placebo_baseline": audit["best_placebo_baseline"],
                    "true_vs_best_placebo_pr_auc_delta": audit[
                        "true_vs_best_placebo_pr_auc_delta"
                    ],
                    "true_vs_best_placebo_top25_delta": audit[
                        "true_vs_best_placebo_top25_delta"
                    ],
                    "audit_note": audit["audit_note"],
                }
            )

    sensitivity_rows: list[dict[str, object]] = []
    for row in score_rows:
        for scenario, (alpha, beta, gamma) in WEIGHT_SCENARIOS.items():
            aprs = (
                alpha * float(row["predictive_performance_score"])
                + beta * float(row["audit_reliability_score"])
                + gamma * float(row["guardrail_robustness_score"])
            )
            sensitivity_rows.append(
                {
                    "evaluation_scope": row["evaluation_scope"],
                    "weight_scenario": scenario,
                    "alpha_predictive": alpha,
                    "beta_audit": beta,
                    "gamma_guardrail": gamma,
                    "candidate_id": row["candidate_id"],
                    "paper_bucket": row["paper_bucket"],
                    "is_deployable_candidate": row["is_deployable_candidate"],
                    "evaluation": row["evaluation"],
                    "aprs": round(aprs, 6),
                    "predictive_performance_score": row["predictive_performance_score"],
                    "audit_reliability_score": row["audit_reliability_score"],
                    "guardrail_robustness_score": row["guardrail_robustness_score"],
                }
            )

    for (scope, scenario), rows in group_rows(
        sensitivity_rows, "evaluation_scope", "weight_scenario"
    ).items():
        ranked = sorted(rows, key=lambda r: float(r["aprs"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_scope_scenario"] = rank

    score_fieldnames = [
        "evaluation_scope",
        "evaluation",
        "paper_bucket",
        "is_deployable_candidate",
        "artifact_id",
        "model",
        "candidate_id",
        "mean_main_pr_auc",
        "main_top25_hits",
        "mean_severe_pr_auc",
        "severe_top25_hits",
        "predictive_performance_score",
        "audit_reliability_score",
        "guardrail_robustness_score",
        "recommended_aprs",
        "best_placebo_baseline",
        "true_vs_best_placebo_pr_auc_delta",
        "true_vs_best_placebo_top25_delta",
        "audit_note",
    ]
    sensitivity_fieldnames = [
        "evaluation_scope",
        "weight_scenario",
        "rank_within_scope_scenario",
        "alpha_predictive",
        "beta_audit",
        "gamma_guardrail",
        "candidate_id",
        "paper_bucket",
        "is_deployable_candidate",
        "evaluation",
        "aprs",
        "predictive_performance_score",
        "audit_reliability_score",
        "guardrail_robustness_score",
    ]
    write_csv(OUT_SCORES, score_rows, score_fieldnames)
    write_csv(OUT_SENSITIVITY, sensitivity_rows, sensitivity_fieldnames)
    OUT_REPORT.write_text(build_report(score_rows, sensitivity_rows), encoding="utf-8")


def group_rows(rows: list[dict[str, object]], *keys: str) -> dict[tuple[str, ...], list[dict[str, object]]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def top_rows(
    rows: list[dict[str, object]],
    scope: str,
    scenario: str,
    limit: int = 5,
    deployable_only: bool = False,
) -> list[dict[str, object]]:
    scoped = [
        r
        for r in rows
        if r["evaluation_scope"] == scope and r["weight_scenario"] == scenario
        and (not deployable_only or r.get("is_deployable_candidate") == "yes")
    ]
    return sorted(scoped, key=lambda r: float(r["aprs"]), reverse=True)[:limit]


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = [format_cell(row.get(key, "")) for _, key in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "/")


def build_report(
    score_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> str:
    temporal_top = top_rows(
        sensitivity_rows,
        "temporal_known_country",
        "recommended_0p5_0p3_0p2",
        deployable_only=True,
    )
    loco_top = top_rows(
        sensitivity_rows,
        "loco_transfer",
        "recommended_0p5_0p3_0p2",
        deployable_only=True,
    )
    placebo_stress_top = [
        r
        for r in sensitivity_rows
        if r["weight_scenario"] == "recommended_0p5_0p3_0p2"
        and r["is_deployable_candidate"] == "no"
    ]
    placebo_stress_top = sorted(
        placebo_stress_top,
        key=lambda r: (str(r["evaluation_scope"]), -float(r["aprs"])),
    )
    scenario_winners = []
    deployable_scenario_winners = []
    for key, rows in group_rows(
        sensitivity_rows, "evaluation_scope", "weight_scenario"
    ).items():
        ranked = sorted(rows, key=lambda r: int(r["rank_within_scope_scenario"]))
        scenario_winners.append(
            {
                "evaluation_scope": key[0],
                "weight_scenario": key[1],
                "winner": ranked[0]["candidate_id"],
                "aprs": ranked[0]["aprs"],
            }
        )
        deployable_ranked = sorted(
            [r for r in rows if r["is_deployable_candidate"] == "yes"],
            key=lambda r: float(r["aprs"]),
            reverse=True,
        )
        deployable_scenario_winners.append(
            {
                "evaluation_scope": key[0],
                "weight_scenario": key[1],
                "winner": deployable_ranked[0]["candidate_id"],
                "aprs": deployable_ranked[0]["aprs"],
            }
        )
    scenario_winners = sorted(
        scenario_winners, key=lambda r: (r["evaluation_scope"], r["weight_scenario"])
    )
    deployable_scenario_winners = sorted(
        deployable_scenario_winners,
        key=lambda r: (r["evaluation_scope"], r["weight_scenario"]),
    )

    audited_rows = [
        r
        for r in score_rows
        if str(r["best_placebo_baseline"]).strip()
    ]
    audited_rows = sorted(
        audited_rows,
        key=lambda r: (
            str(r["evaluation_scope"]),
            float(r["true_vs_best_placebo_pr_auc_delta"]),
        ),
        reverse=True,
    )

    return "\n".join(
        [
            "# Main-Paper APRS Audit",
            "",
            "APRS (Audited Predictive Reliability Score) is a reporting-only metric for the current PortWatch + GDELT + WITS main-paper line. It does not retrain models and does not add a fourth data source.",
            "",
            "Definition used here:",
            "",
            "```text",
            "APRS = alpha * PredictivePerformance",
            "     + beta  * AuditReliability",
            "     + gamma * GuardrailRobustness",
            "```",
            "",
            "- `PredictivePerformance`: min-max normalized mean PR-AUC within the same evaluation scope.",
            "- `AuditReliability`: `0.5 + (true WITS - best placebo PR-AUC delta) / 0.05`, clipped to `[0, 1]`; rows without a true-vs-placebo audit contrast receive `0` for this component.",
            "- `GuardrailRobustness`: average of min-max normalized main top-25 hits and severe top-25 hits within the same evaluation scope.",
            "- Recommended weights follow the framework note: `alpha=0.5`, `beta=0.3`, `gamma=0.2`.",
            "",
            "This design intentionally penalizes unaudited or placebo-competitive rows. APRS is therefore a reliability summary, not a device for forcing the full framework to win raw PR-AUC.",
            "",
            "## Recommended APRS Leaders: Deployable Candidates",
            "",
            "Temporal known-country monitoring:",
            "",
            markdown_table(
                temporal_top,
                [
                    ("Rank", "rank_within_scope_scenario"),
                    ("Candidate", "candidate_id"),
                    ("Bucket", "paper_bucket"),
                    ("APRS", "aprs"),
                    ("Predictive", "predictive_performance_score"),
                    ("Audit", "audit_reliability_score"),
                    ("Guardrail", "guardrail_robustness_score"),
                ],
            ),
            "",
            "LOCO transfer:",
            "",
            markdown_table(
                loco_top,
                [
                    ("Rank", "rank_within_scope_scenario"),
                    ("Candidate", "candidate_id"),
                    ("Bucket", "paper_bucket"),
                    ("APRS", "aprs"),
                    ("Predictive", "predictive_performance_score"),
                    ("Audit", "audit_reliability_score"),
                    ("Guardrail", "guardrail_robustness_score"),
                ],
            ),
            "",
            "Placebo controls are not deployable candidates. They are retained below as stress-test rows; a placebo leading the all-row ranking is evidence against promoting true WITS as a universal predictor.",
            "",
            "## Placebo Stress Rows",
            "",
            markdown_table(
                placebo_stress_top,
                [
                    ("Scope", "evaluation_scope"),
                    ("Candidate", "candidate_id"),
                    ("APRS", "aprs"),
                    ("Predictive", "predictive_performance_score"),
                    ("Audit", "audit_reliability_score"),
                    ("Guardrail", "guardrail_robustness_score"),
                ],
            ),
            "",
            "## True-vs-Placebo Audit Component",
            "",
            markdown_table(
                audited_rows,
                [
                    ("Scope", "evaluation_scope"),
                    ("Candidate", "candidate_id"),
                    ("Best placebo", "best_placebo_baseline"),
                    ("PR-AUC delta", "true_vs_best_placebo_pr_auc_delta"),
                    ("Top-25 delta", "true_vs_best_placebo_top25_delta"),
                    ("Audit score", "audit_reliability_score"),
                ],
            ),
            "",
            "## Weight Sensitivity Winners: Deployable Candidates",
            "",
            markdown_table(
                deployable_scenario_winners,
                [
                    ("Scope", "evaluation_scope"),
                    ("Weights", "weight_scenario"),
                    ("Winner", "winner"),
                    ("APRS", "aprs"),
                ],
            ),
            "",
            "## Weight Sensitivity Winners: All Rows Including Placebos",
            "",
            markdown_table(
                scenario_winners,
                [
                    ("Scope", "evaluation_scope"),
                    ("Weights", "weight_scenario"),
                    ("Winner", "winner"),
                    ("APRS", "aprs"),
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- APRS supports the paper's reliability-aware framing: a candidate can have good PR-AUC yet lose reliability score if it is not structurally audited or if a placebo network is stronger.",
            "- In temporal monitoring, guarded or conversion-aware rows can rank well when prediction, audit separation, and top-k/severe guardrails are combined.",
            "- In LOCO transfer, deployable APRS favors WITS additive under the recommended weights, while all-row APRS is led by equal-placebo. This is a warning against claiming transfer dominance for exact true WITS gating.",
            "- The true compact gated LOCO model receives a low audit component because equal/random placebos remain competitive or stronger; this reinforces the limitation that WITS is stronger as audit/attribution than as a universal transfer predictor.",
            "- APRS should be presented as task-specific evaluation for event-informed supply-chain risk prediction, not as a universal ML metric.",
            "",
            "## Outputs",
            "",
            f"- `{OUT_SCORES.relative_to(ROOT)}`",
            f"- `{OUT_SENSITIVITY.relative_to(ROOT)}`",
        ]
    )


if __name__ == "__main__":
    main()
