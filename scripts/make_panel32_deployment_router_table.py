from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

TEMPORAL_SUMMARY = TABLE_DIR / "panel32_validation_policy_selector_summary.csv"
TRANSFER_SUMMARY = TABLE_DIR / "panel32_locked_transfer_router_summary.csv"
TRANSFER_DELTAS = TABLE_DIR / "panel32_locked_transfer_router_deltas.csv"
FAILURE_COUNTRIES = TABLE_DIR / "panel32_transfer_failure_country_diagnostics.csv"
OPS4_SUMMARY = TABLE_DIR / "panel32_ops4_w20_stress_summary.csv"
OPS4_TOPK = TABLE_DIR / "panel32_ops4_w20_topk_bootstrap.csv"

OUT_CSV = TABLE_DIR / "panel32_deployment_router_table.csv"
OUT_MD = PROJECT_ROOT / "reports" / "panel32_deployment_router_table.md"


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def temporal_row(summary: pd.DataFrame) -> dict[str, str]:
    row = summary.loc[summary["policy"].eq("P32REF_gb_m7_full")].iloc[0]
    return {
        "Deployment mode": "Known-country temporal monitoring",
        "Recommended score": "P32REF_gb_m7_full",
        "Validation design": "Rolling temporal folds over expanded32",
        "Main evidence": (
            f"mean PR-AUC {fmt(row['mean_main_pr_auc'])}; "
            f"top-10/top-25/top-50 {int(row['main_top10_hits'])}/"
            f"{int(row['main_top25_hits'])}/{int(row['main_top50_hits'])}"
        ),
        "Severe-label evidence": (
            f"mean severe PR-AUC {fmt(row['mean_severe_pr_auc'])}; "
            f"severe top-10/top-25 {int(row['severe_top10_hits'])}/{int(row['severe_top25_hits'])}"
        ),
        "Why this mode": "Current expanded32 temporal reference; validation selectors did not beat it.",
        "Claim status": "Reference policy, not proof of network-weight superiority.",
    }


def transfer_row(summary: pd.DataFrame, policy: str, mode: str, why: str, status: str) -> dict[str, str]:
    row = summary.loc[summary["policy"].eq(policy)].iloc[0]
    return {
        "Deployment mode": mode,
        "Recommended score": policy,
        "Validation design": "Locked leave-one-country-out transfer; train 2021-2023, validate 2024, holdout-country test 2025",
        "Main evidence": (
            f"mean PR-AUC {fmt(row['mean_main_pr_auc'])}; "
            f"median PR-AUC {fmt(row['median_main_pr_auc'])}; "
            f"top-5/top-10/top-25 {int(row['main_top5_hits'])}/"
            f"{int(row['main_top10_hits'])}/{int(row['main_top25_hits'])}"
        ),
        "Severe-label evidence": (
            f"mean severe PR-AUC {fmt(row['mean_severe_pr_auc'])}; "
            f"severe top-5/top-10/top-25 {int(row['severe_top5_hits'])}/"
            f"{int(row['severe_top10_hits'])}/{int(row['severe_top25_hits'])}"
        ),
        "Why this mode": why,
        "Claim status": status,
    }


def ops4_row(summary: pd.DataFrame) -> dict[str, str]:
    row = summary.loc[summary["policy"].eq("OPS4FIX_w20")].iloc[0]
    return {
        "Deployment mode": "Unseen-country exploratory top-25 overlay",
        "Recommended score": "OPS4FIX_w20",
        "Validation design": (
            "Locked leave-one-country-out transfer; train 2021-2023, validate 2024, "
            "holdout-country test 2025; fixed 20% PS4-XGB / 80% operational rank overlay"
        ),
        "Main evidence": (
            f"mean PR-AUC {fmt(row['main_1p5_mean_pr_auc'])}; "
            f"top-10/top-25 {int(row['main_1p5_top10_hits'])}/{int(row['main_1p5_top25_hits'])}"
        ),
        "Severe-label evidence": (
            f"mean severe PR-AUC {fmt(row['sigma_2p0_mean_pr_auc'])}; "
            f"severe top-10/top-25 {int(row['sigma_2p0_top10_hits'])}/{int(row['sigma_2p0_top25_hits'])}"
        ),
        "Why this mode": "Fixed overlay improves main top-25 over operational in point estimate and preserves operational severe top-25.",
        "Claim status": "Exploratory only; later locked selector and 200-seed random-overlay check failed promotion guard.",
    }


def delta_note(deltas: pd.DataFrame, focus: str, baseline: str) -> str:
    row = deltas.loc[deltas["focus_policy"].eq(focus) & deltas["baseline_policy"].eq(baseline)]
    if row.empty:
        return ""
    row = row.iloc[0]
    return (
        f"{focus} vs {baseline}: pooled main PR-AUC delta {fmt(row['pooled_main_pr_auc_delta'])} "
        f"[{fmt(row['main_ci_low'])}, {fmt(row['main_ci_high'])}], "
        f"top-25 delta {int(row['main_top25_hit_delta'])}."
    )


def failure_note(country: pd.DataFrame) -> str:
    failures = country.loc[
        country["LOCK4_vs_LOCK0_pr_auc_delta"].lt(0),
        ["ISO3", "country", "LOCK4_vs_LOCK0_pr_auc_delta", "LOCK4_vs_LOCK0_top25_delta"],
    ].sort_values("LOCK4_vs_LOCK0_pr_auc_delta")
    labels = ", ".join(failures["ISO3"].tolist())
    return (
        f"Failure-mode diagnostic: 70/30 router loses PR-AUC to operational in "
        f"{len(failures)}/32 holdouts ({labels}). These countries are the priority list "
        f"for new direct operational data."
    )


def ops4_note(topk: pd.DataFrame, baseline: str, label: str = "main_1p5") -> str:
    row = topk.loc[
        topk["focus_policy"].eq("OPS4FIX_w20")
        & topk["baseline_policy"].eq(baseline)
        & topk["label"].eq(label)
        & topk["top_k_per_holdout"].eq(25)
    ]
    if row.empty:
        return ""
    row = row.iloc[0]
    return (
        f"OPS4FIX_w20 vs {baseline} ({label} top-25): delta {int(row['observed_hit_delta'])} "
        f"[{int(row['bootstrap_ci_low'])}, {int(row['bootstrap_ci_high'])}]."
    )


def run() -> None:
    temporal = pd.read_csv(TEMPORAL_SUMMARY)
    transfer = pd.read_csv(TRANSFER_SUMMARY)
    deltas = pd.read_csv(TRANSFER_DELTAS)
    country = pd.read_csv(FAILURE_COUNTRIES)
    ops4_summary = pd.read_csv(OPS4_SUMMARY)
    ops4_topk = pd.read_csv(OPS4_TOPK)

    rows = [
        temporal_row(temporal),
        transfer_row(
            transfer,
            "LOCK2_haz",
            "Unseen-country PR-AUC transfer",
            "Pure hazard has the cleanest positive pooled PR-AUC delta versus operational.",
            "Strong transfer PR-AUC evidence; top-k still trails operational.",
        ),
        transfer_row(
            transfer,
            "LOCK4_70haz30op",
            "Unseen-country balanced transfer",
            "Best holdout mean main/severe PR-AUC and recovers some top-k hits versus pure hazard.",
            "Balanced candidate; not a universal win because operational still leads top-25.",
        ),
        ops4_row(ops4_summary),
        transfer_row(
            transfer,
            "LOCK0_op",
            "Unseen-country operational top-k guardrail",
            "Operational score remains the top-25 and severe top-25 transfer reference.",
            "Guardrail reference; OPS4FIX_w20 improves main top-25 point estimate but not decisively versus operational.",
        ),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    delta_notes = "\n".join(
        [
            f"- {delta_note(deltas, 'LOCK2_haz', 'LOCK0_op')}",
            f"- {delta_note(deltas, 'LOCK4_70haz30op', 'LOCK0_op')}",
            f"- {delta_note(deltas, 'LOCK4_70haz30op', 'LOCK2_haz')}",
            f"- {ops4_note(ops4_topk, 'PSL3_70haz30op_gb', 'main_1p5')}",
            f"- {ops4_note(ops4_topk, 'PSL3_70haz30op_gb', 'sigma_2p0')}",
            f"- {ops4_note(ops4_topk, 'PSL0_op_gb', 'main_1p5')}",
            f"- {failure_note(country)}",
        ]
    )
    content = f"""# Panel32 Deployment Router Table

## Purpose

This table separates deployment modes instead of forcing one universal expanded32 score. Known-country temporal monitoring and fully unseen-country transfer use different validation designs and should be reported separately.

## Table

{out.to_markdown(index=False)}

## Key Guardrails

{delta_notes}

## Caption

Expanded32 deployment-mode policy summary. Known-country monitoring uses the strongest temporal expanded32 reference, while fully unseen-country deployment separates PR-AUC transfer, balanced transfer, top-k candidate, and operational guardrail objectives. This table should not be interpreted as evidence that one score dominates all metrics or that network exposure is causally predictive.
"""
    OUT_MD.write_text(content, encoding="utf-8")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved markdown: {OUT_MD}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
