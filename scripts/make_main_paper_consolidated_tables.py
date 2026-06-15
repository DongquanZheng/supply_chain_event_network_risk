"""Build consolidated main-paper result tables from existing reports.

This script does not train models. It reads previously generated
PortWatch/GDELT/WITS result CSVs and creates paper-facing summary tables for
the Network-Gated Event Conversion main line.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
REPORT_PATH = ROOT / "reports" / "main_paper_consolidated_results.md"


def read_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def to_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_bucket(row: pd.Series) -> str:
    name = str(row.get("artifact_id", ""))
    if "placebo" in name.lower():
        return "placebo_audit"
    if "gated" in name.lower() or "AA9" in name:
        return "proposed_or_guarded"
    if "wits" in name.lower():
        return "network_additive"
    if "gdelt" in name.lower():
        return "event_additive"
    if "operational" in name.lower():
        return "operational"
    return "supporting"


def build_model_ladder() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    ng = read_table("panel32_network_gated_conversion_summary.csv")
    ng = ng[ng["model"].eq("sklearn_gradient_boosting")].copy()
    keep_ng = [
        "NG0_portwatch_operational",
        "NG1_portwatch_gdelt_additive",
        "NG2_portwatch_gdelt_wits_additive",
        "NG5_compact_network_gated_true",
        "NG6_equal_compact_gated_placebo",
        "NG6_random_compact_gated_placebo",
        "NG6_shuffled_compact_gated_placebo",
    ]
    ng = ng[ng["feature_group"].isin(keep_ng)].copy()
    ng["evaluation"] = "temporal_main_ladder"
    ng["artifact_id"] = ng["feature_group"]
    ng["source_table"] = "panel32_network_gated_conversion_summary.csv"
    rows.append(ng)

    loco = read_table("panel32_network_gated_conversion_loco_summary.csv")
    keep_loco = [
        "NG0_portwatch_operational",
        "NG1_portwatch_gdelt_additive",
        "NG2_portwatch_gdelt_wits_additive",
        "NG5_compact_network_gated_true",
        "NG6_equal_compact_gated_placebo",
        "NG6_random_compact_gated_placebo",
        "NG6_shuffled_compact_gated_placebo",
    ]
    loco = loco[loco["feature_group"].isin(keep_loco)].copy()
    loco["evaluation"] = "loco_transfer"
    loco["artifact_id"] = loco["feature_group"]
    loco["source_table"] = "panel32_network_gated_conversion_loco_summary.csv"
    rows.append(loco)

    aa = read_table("panel32_country_shared_alert_allocation_policy_summary.csv")
    keep_aa = [
        "AA0_fixed_operational",
        "AA1_fixed_gdelt_additive",
        "AA2_fixed_true_wits_additive",
        "AA3_fixed_true_gated",
        "AA4_fixed_equal_gated_placebo",
        "AA5_fixed_random_gated_placebo",
        "AA6_fixed_shuffled_gated_placebo",
        "AA9_gated_if_validation_safe_else_additive",
    ]
    aa = aa[aa["policy"].isin(keep_aa)].copy()
    aa["evaluation"] = "temporal_guarded_integration"
    aa["artifact_id"] = aa["policy"]
    aa["feature_group"] = aa["policy"]
    aa["source_table"] = "panel32_country_shared_alert_allocation_policy_summary.csv"
    rows.append(aa)

    cp = read_table("panel32_gdelt_conversion_propensity_summary.csv")
    cp = cp[cp["model"].eq("sklearn_gradient_boosting")].copy()
    keep_cp = [
        "GCL0_portwatch_operational_calibrated",
        "GCL1_portwatch_gdelt_additive_calibrated",
        "GCL2_true_wits_additive_calibrated",
        "GCL3_gdelt_conversion_propensity",
        "GCL4_true_wits_conversion_propensity_gated",
        "GCL4_equal_conversion_propensity_gated_placebo",
        "GCL4_random_conversion_propensity_gated_placebo",
        "GCL4_shuffled_conversion_propensity_gated_placebo",
    ]
    cp = cp[cp["feature_group"].isin(keep_cp)].copy()
    cp["evaluation"] = "temporal_conversion_propensity"
    cp["artifact_id"] = cp["feature_group"]
    cp["source_table"] = "panel32_gdelt_conversion_propensity_summary.csv"
    rows.append(cp)

    out = pd.concat(rows, ignore_index=True, sort=False)
    numeric_cols = [
        "mean_main_pr_auc",
        "median_main_pr_auc",
        "mean_severe_pr_auc",
        "median_severe_pr_auc",
        "mean_roc_auc",
        "main_top10_hits",
        "main_top25_hits",
        "main_top50_hits",
        "severe_top10_hits",
        "severe_top25_hits",
        "severe_top50_hits",
    ]
    out = to_float_columns(out, numeric_cols)
    out["paper_bucket"] = out.apply(add_bucket, axis=1)
    columns = [
        "evaluation",
        "paper_bucket",
        "artifact_id",
        "model",
        "mean_main_pr_auc",
        "median_main_pr_auc",
        "mean_severe_pr_auc",
        "median_severe_pr_auc",
        "main_top10_hits",
        "main_top25_hits",
        "main_top50_hits",
        "severe_top10_hits",
        "severe_top25_hits",
        "severe_top50_hits",
        "source_table",
    ]
    return out[[c for c in columns if c in out.columns]].sort_values(
        ["evaluation", "paper_bucket", "artifact_id"]
    )


def build_network_audit() -> pd.DataFrame:
    specs = [
        (
            "temporal_compact_gated",
            "panel32_network_gated_conversion_key_deltas.csv",
            "feature_group",
            "NG5_compact_network_gated_true",
        ),
        (
            "loco_compact_gated",
            "panel32_network_gated_conversion_loco_deltas.csv",
            "feature_group",
            "NG5_compact_network_gated_true",
        ),
        (
            "temporal_guarded_AA9",
            "panel32_country_shared_alert_allocation_policy_deltas.csv",
            "policy",
            "AA9_gated_if_validation_safe_else_additive",
        ),
        (
            "temporal_conversion_propensity",
            "panel32_gdelt_conversion_propensity_deltas.csv",
            "feature_group",
            "GCL4_true_wits_conversion_propensity_gated",
        ),
    ]
    pieces: list[pd.DataFrame] = []
    for evaluation, filename, id_type, focus_id in specs:
        df = read_table(filename).copy()
        focus_col = "focus_policy" if id_type == "policy" else "focus_feature_group"
        base_col = "baseline_policy" if id_type == "policy" else "baseline_feature_group"
        df = df[df[focus_col].eq(focus_id) & df["label"].eq("main")].copy()
        df["evaluation"] = evaluation
        df["focus_id"] = df[focus_col]
        df["baseline_id"] = df[base_col]
        df["source_table"] = filename
        pieces.append(df)

    out = pd.concat(pieces, ignore_index=True, sort=False)
    out = to_float_columns(
        out,
        ["pooled_pr_auc_delta", "ci_low", "ci_high", "p_gt_0", "top10_hit_delta", "top25_hit_delta", "top50_hit_delta"],
    )
    out["audit_result"] = out.apply(classify_audit_result, axis=1)
    columns = [
        "evaluation",
        "contrast",
        "focus_id",
        "baseline_id",
        "pooled_pr_auc_delta",
        "ci_low",
        "ci_high",
        "p_gt_0",
        "top10_hit_delta",
        "top25_hit_delta",
        "top50_hit_delta",
        "audit_result",
        "source_table",
    ]
    return out[columns].sort_values(["evaluation", "contrast"])


def classify_audit_result(row: pd.Series) -> str:
    delta = row.get("pooled_pr_auc_delta")
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    top25 = row.get("top25_hit_delta")
    if pd.notna(delta) and pd.notna(ci_low) and ci_low > 0:
        return "positive_pr_auc_interval"
    if pd.notna(delta) and pd.notna(ci_high) and ci_high < 0:
        return "negative_pr_auc_interval"
    if pd.notna(top25) and top25 > 0:
        return "top25_positive_pr_auc_uncertain"
    if pd.notna(delta) and delta > 0:
        return "directional_pr_auc_positive"
    return "weak_or_negative"


def build_deployment_checks() -> pd.DataFrame:
    target = read_table("panel32_country_shared_alert_allocation_stress_target_sensitivity.csv")
    target = target[
        target["policy"].isin(
            [
                "AA9_gated_if_validation_safe_else_additive",
                "AA2_fixed_true_wits_additive",
                "AA0_fixed_operational",
                "AA3_fixed_true_gated",
                "AA6_fixed_shuffled_gated_placebo",
            ]
        )
    ].copy()
    target["evaluation"] = "target_sensitivity"
    target["subgroup_variable"] = "target_label"
    target["subgroup"] = target["label"]
    target = target.rename(
        columns={
            "mean_fold_pr_auc": "main_or_label_pr_auc",
            "top25_hits": "top25_hits",
        }
    )
    target["source_table"] = "panel32_country_shared_alert_allocation_stress_target_sensitivity.csv"

    sub = read_table("panel32_country_shared_alert_allocation_stress_subgroups.csv")
    sub = sub[
        sub["subgroup_variable"].eq("shortfall_regime")
        & sub["policy"].isin(
            [
                "AA9_gated_if_validation_safe_else_additive",
                "AA2_fixed_true_wits_additive",
                "AA3_fixed_true_gated",
                "AA6_fixed_shuffled_gated_placebo",
            ]
        )
    ].copy()
    sub["evaluation"] = "shortfall_subgroup"
    sub = sub.rename(
        columns={
            "main_pooled_pr_auc": "main_or_label_pr_auc",
            "main_top25_hits_within_subgroup": "top25_hits",
        }
    )
    sub["source_table"] = "panel32_country_shared_alert_allocation_stress_subgroups.csv"

    out = pd.concat([target, sub], ignore_index=True, sort=False)
    out = to_float_columns(out, ["main_or_label_pr_auc", "top25_hits", "rows", "positives"])
    columns = [
        "evaluation",
        "subgroup_variable",
        "subgroup",
        "policy",
        "rows",
        "positives",
        "main_or_label_pr_auc",
        "top25_hits",
        "source_table",
    ]
    return out[[c for c in columns if c in out.columns]].sort_values(
        ["evaluation", "subgroup_variable", "subgroup", "policy"]
    )


def write_markdown(
    ladder: pd.DataFrame, audit: pd.DataFrame, deployment: pd.DataFrame
) -> None:
    aa9 = ladder[ladder["artifact_id"].eq("AA9_gated_if_validation_safe_else_additive")].iloc[0]
    compact = ladder[
        (ladder["evaluation"].eq("temporal_main_ladder"))
        & (ladder["artifact_id"].eq("NG5_compact_network_gated_true"))
    ].iloc[0]
    loco = ladder[
        (ladder["evaluation"].eq("loco_transfer"))
        & (ladder["artifact_id"].eq("NG5_compact_network_gated_true"))
    ].iloc[0]
    cp = ladder[
        (ladder["evaluation"].eq("temporal_conversion_propensity"))
        & (ladder["artifact_id"].eq("GCL4_true_wits_conversion_propensity_gated"))
    ].iloc[0]

    lines = [
        "# Main-Paper Consolidated Results",
        "",
        "Date: 2026-06-15",
        "",
        "Scope: PortWatch + GDELT + WITS only. This report consolidates existing result tables; it does not train a new model.",
        "",
        "## Paper-Facing Read",
        "",
        f"- Temporal compact gated conversion has mean PR-AUC `{compact['mean_main_pr_auc']:.4f}` and top-25 `{int(compact['main_top25_hits'])}`.",
        f"- LOCO compact gated conversion has mean PR-AUC `{loco['mean_main_pr_auc']:.4f}` and top-25 `{int(loco['main_top25_hits'])}`; it does not beat WITS additive/equal-placebo transfer references.",
        f"- AA9 guarded integration has mean PR-AUC `{aa9['mean_main_pr_auc']:.4f}`, top-25 `{int(aa9['main_top25_hits'])}`, and severe top-25 `{int(aa9['severe_top25_hits'])}`.",
        f"- Fold-aware conversion-propensity gated has mean PR-AUC `{cp['mean_main_pr_auc']:.4f}` and top-25 `{int(cp['main_top25_hits'])}`; it is top-k/mechanism evidence, not a PR-AUC winner.",
        "",
        "## Generated Tables",
        "",
        "- `reports/tables/main_paper_consolidated_model_ladder.csv`",
        "- `reports/tables/main_paper_network_audit_checks.csv`",
        "- `reports/tables/main_paper_deployment_checks.csv`",
        "",
        "## Claim Impact",
        "",
        "- Strengthens: operational-vulnerability-conditioned event conversion; guarded deployment logic; network audit as attribution/placebo discipline.",
        "- Weakens or limits: any claim that true WITS gating universally beats additive or placebo models.",
        "- Still missing: manual 20-40 case GDELT conversion audit.",
        "",
        "## Top Audit Rows",
        "",
        audit.head(12).to_markdown(index=False),
        "",
        "## Guardrail",
        "",
        "Do not use this report to claim that WITS proves causal propagation, that GDELT measures true disruption, or that AA9 is universally superior.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ladder = build_model_ladder()
    audit = build_network_audit()
    deployment = build_deployment_checks()

    ladder.to_csv(TABLE_DIR / "main_paper_consolidated_model_ladder.csv", index=False)
    audit.to_csv(TABLE_DIR / "main_paper_network_audit_checks.csv", index=False)
    deployment.to_csv(TABLE_DIR / "main_paper_deployment_checks.csv", index=False)
    write_markdown(ladder, audit, deployment)

    print("Wrote main-paper consolidated result artifacts:")
    print(TABLE_DIR / "main_paper_consolidated_model_ladder.csv")
    print(TABLE_DIR / "main_paper_network_audit_checks.csv")
    print(TABLE_DIR / "main_paper_deployment_checks.csv")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
