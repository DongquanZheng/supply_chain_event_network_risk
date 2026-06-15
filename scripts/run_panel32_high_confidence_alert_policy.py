"""Evaluate a validation-safe high-confidence alert policy for panel32.

The policy is intentionally narrow and main-paper scoped: it reuses existing
out-of-sample scores from PortWatch + GDELT + WITS experiments, applies a fixed
top-k alert budget, and avoids test-label rule selection.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"

CANDIDATE_PREDICTIONS = (
    TABLE_DIR / "panel32_country_shared_alert_allocation_candidate_predictions.csv"
)
AA_SELECTIONS = TABLE_DIR / "panel32_country_shared_alert_allocation_selections.csv"
CONVERSION_PREDICTIONS = TABLE_DIR / "panel32_gdelt_conversion_propensity_predictions.csv"
MAIN_APRS = TABLE_DIR / "main_paper_aprs_scores.csv"

OUT_SUMMARY = TABLE_DIR / "panel32_high_confidence_alert_policy_summary.csv"
OUT_BY_FOLD = TABLE_DIR / "panel32_high_confidence_alert_policy_by_fold.csv"
OUT_ALERTS = TABLE_DIR / "panel32_high_confidence_alert_policy_alerts.csv"
OUT_PLACEBO = TABLE_DIR / "panel32_high_confidence_alert_policy_placebo_checks.csv"
OUT_REPORT = ROOT / "reports" / "panel32_high_confidence_alert_policy.md"

MODEL = "sklearn_gradient_boosting"
TOP_KS = (10, 25, 50)
AA9_REFERENCE_APRS = 0.843069
AA9_REFERENCE_PR_AUC = 0.1994398242864377
AA9_REFERENCE_TOP25 = 28
AA9_REFERENCE_SEVERE_TOP25 = 18
MATERIAL_PR_AUC_DELTA = 0.05

WEIGHTS = {
    "proposed_high_confidence_policy": {
        "aa_component": 0.40,
        "conversion_component": 0.40,
        "vulnerability_component": 0.20,
    }
}

AA_POLICY_COLUMNS = {
    "AA0_fixed_operational": "CS_OP",
    "AA1_fixed_gdelt_additive": "CS_GDELT",
    "AA2_fixed_true_wits_additive": "CS_TRUE_ADD",
    "AA3_fixed_true_gated": "CS_TRUE_GATED",
    "AA4_fixed_equal_gated_placebo": "CS_EQUAL_GATED",
    "AA5_fixed_random_gated_placebo": "CS_RANDOM_GATED",
    "AA6_fixed_shuffled_gated_placebo": "CS_SHUFFLED_GATED",
}

CONVERSION_FEATURES = {
    "GCL4_true_wits_conversion_propensity_gated": (
        "GCL4_true_wits_conversion_propensity_gated"
    ),
    "GCL4_equal_conversion_propensity_gated_placebo": (
        "GCL4_equal_conversion_propensity_gated_placebo"
    ),
    "GCL4_random_conversion_propensity_gated_placebo": (
        "GCL4_random_conversion_propensity_gated_placebo"
    ),
    "GCL4_shuffled_conversion_propensity_gated_placebo": (
        "GCL4_shuffled_conversion_propensity_gated_placebo"
    ),
}

PROPOSED_VARIANTS = {
    "proposed_high_confidence_true": {
        "aa_component": "AA9_score",
        "conversion_component": "GCL4_true_wits_conversion_propensity_gated",
        "paper_bucket": "proposed_policy",
        "is_deployable_candidate": "yes",
    },
    "proposed_high_confidence_equal_placebo": {
        "aa_component": "CS_EQUAL_GATED",
        "conversion_component": "GCL4_equal_conversion_propensity_gated_placebo",
        "paper_bucket": "placebo_audit",
        "is_deployable_candidate": "no",
    },
    "proposed_high_confidence_random_placebo": {
        "aa_component": "CS_RANDOM_GATED",
        "conversion_component": "GCL4_random_conversion_propensity_gated_placebo",
        "paper_bucket": "placebo_audit",
        "is_deployable_candidate": "no",
    },
    "proposed_high_confidence_shuffled_placebo": {
        "aa_component": "CS_SHUFFLED_GATED",
        "conversion_component": "GCL4_shuffled_conversion_propensity_gated_placebo",
        "paper_bucket": "placebo_audit",
        "is_deployable_candidate": "no",
    },
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def minmax(value: float, values: list[float]) -> float:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return 1.0
    return (value - lo) / (hi - lo)


def rank_pct_by_fold(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("fold")[column].rank(method="average", pct=True)


def read_candidate_predictions() -> pd.DataFrame:
    raw = pd.read_csv(CANDIDATE_PREDICTIONS)
    keys = [
        "ISO3",
        "country",
        "week",
        "abnormal_next_week_container",
        "abnormal_next_week_container_2p0sigma",
        "operational_shortfall_12w",
        "fold",
        "split",
    ]
    wide = (
        raw.pivot_table(
            index=keys,
            columns="candidate",
            values="predicted_probability",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def read_conversion_predictions() -> pd.DataFrame:
    raw = pd.read_csv(CONVERSION_PREDICTIONS)
    raw = raw[
        (raw["model"] == MODEL)
        & (raw["feature_group"].isin(CONVERSION_FEATURES.values()))
    ].copy()
    wide = (
        raw.pivot_table(
            index=["ISO3", "country", "week", "fold"],
            columns="feature_group",
            values="predicted_probability",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def aa9_score_from_validation_selection(df: pd.DataFrame) -> pd.Series:
    selections = pd.read_csv(AA_SELECTIONS)
    aa9 = selections[selections["policy"] == "AA9_gated_if_validation_safe_else_additive"]
    by_fold = dict(zip(aa9["fold"], aa9["spec_label"]))
    score = pd.Series(index=df.index, dtype=float)
    for fold, spec_label in by_fold.items():
        mask = df["fold"] == fold
        if "gated" in str(spec_label).lower():
            score.loc[mask] = df.loc[mask, "CS_TRUE_GATED"]
        else:
            score.loc[mask] = df.loc[mask, "CS_TRUE_ADD"]
    if score.isna().any():
        missing = sorted(df.loc[score.isna(), "fold"].unique().tolist())
        raise ValueError(f"Missing AA9 validation selections for folds: {missing}")
    return score


def build_test_frame() -> pd.DataFrame:
    candidates = read_candidate_predictions()
    conversion = read_conversion_predictions()
    test = candidates[candidates["split"] == "test"].copy()
    test = test.merge(conversion, on=["ISO3", "country", "week", "fold"], how="left")
    missing_conversion = test[list(CONVERSION_FEATURES.values())].isna().any(axis=None)
    if missing_conversion:
        raise ValueError("Conversion-propensity predictions are missing for test rows.")

    test["AA9_score"] = aa9_score_from_validation_selection(test)
    test["operational_shortfall_positive"] = test["operational_shortfall_12w"].clip(lower=0)

    for policy, column in AA_POLICY_COLUMNS.items():
        test[f"score::{policy}"] = test[column]
    test["score::AA9_gated_if_validation_safe_else_additive"] = test["AA9_score"]

    for feature_name, column in CONVERSION_FEATURES.items():
        test[f"score::{feature_name}"] = test[column]

    for variant, spec in PROPOSED_VARIANTS.items():
        aa_rank = rank_pct_by_fold(test, spec["aa_component"])
        cp_rank = rank_pct_by_fold(test, spec["conversion_component"])
        vuln_rank = rank_pct_by_fold(test, "operational_shortfall_positive")
        weights = WEIGHTS["proposed_high_confidence_policy"]
        test[f"score::{variant}"] = (
            weights["aa_component"] * aa_rank
            + weights["conversion_component"] * cp_rank
            + weights["vulnerability_component"] * vuln_rank
        )
        test[f"component_rank_aa::{variant}"] = aa_rank
        test[f"component_rank_conversion::{variant}"] = cp_rank
        test[f"component_rank_vulnerability::{variant}"] = vuln_rank

    return test


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, score))


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def top_hits(df: pd.DataFrame, score_col: str, label_col: str, k: int) -> int:
    top = df.sort_values(score_col, ascending=False).head(k)
    return int(top[label_col].sum())


def max_country_topk(df: pd.DataFrame, score_col: str, k: int) -> int:
    top = df.sort_values(score_col, ascending=False).head(k)
    if top.empty:
        return 0
    return int(top["ISO3"].value_counts().max())


def metrics_for_fold(df: pd.DataFrame, policy: str, score_col: str) -> dict[str, object]:
    y = df["abnormal_next_week_container"].astype(int).to_numpy()
    severe_y = df["abnormal_next_week_container_2p0sigma"].astype(int).to_numpy()
    score = df[score_col].astype(float).to_numpy()
    pred_at_median = (score >= np.median(score)).astype(int)

    row: dict[str, object] = {
        "policy": policy,
        "model": MODEL,
        "fold": df["fold"].iloc[0],
        "n_rows": len(df),
        "positives": int(y.sum()),
        "severe_positives": int(severe_y.sum()),
        "main_pr_auc": safe_average_precision(y, score),
        "severe_pr_auc": safe_average_precision(severe_y, score),
        "roc_auc": safe_roc_auc(y, score),
        "f1_at_median": float(f1_score(y, pred_at_median, zero_division=0)),
        "max_country_top25": max_country_topk(df, score_col, 25),
    }
    for k in TOP_KS:
        row[f"main_top{k}_hits"] = top_hits(
            df, score_col, "abnormal_next_week_container", k
        )
        row[f"severe_top{k}_hits"] = top_hits(
            df, score_col, "abnormal_next_week_container_2p0sigma", k
        )
        row[f"main_top{k}_precision"] = row[f"main_top{k}_hits"] / k
        row[f"severe_top{k}_precision"] = row[f"severe_top{k}_hits"] / k
    return row


def summarize_metrics(by_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in by_fold.groupby("policy", sort=False):
        row = {
            "policy": policy,
            "model": MODEL,
            "folds": int(group["fold"].nunique()),
            "mean_main_pr_auc": float(group["main_pr_auc"].mean()),
            "median_main_pr_auc": float(group["main_pr_auc"].median()),
            "mean_severe_pr_auc": float(group["severe_pr_auc"].mean()),
            "median_severe_pr_auc": float(group["severe_pr_auc"].median()),
            "mean_roc_auc": float(group["roc_auc"].mean()),
            "mean_f1_at_median": float(group["f1_at_median"].mean()),
            "max_country_top25_max": int(group["max_country_top25"].max()),
        }
        for k in TOP_KS:
            row[f"main_top{k}_hits"] = int(group[f"main_top{k}_hits"].sum())
            row[f"severe_top{k}_hits"] = int(group[f"severe_top{k}_hits"].sum())
            row[f"main_top{k}_precision"] = row[f"main_top{k}_hits"] / (
                k * row["folds"]
            )
            row[f"severe_top{k}_precision"] = row[f"severe_top{k}_hits"] / (
                k * row["folds"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paper_bucket(policy: str) -> str:
    if "placebo" in policy or policy.startswith("AA4") or policy.startswith("AA5") or policy.startswith("AA6"):
        return "placebo_audit"
    if policy.startswith("AA0"):
        return "operational"
    if policy.startswith("AA1"):
        return "event_additive"
    if policy.startswith("AA2") or policy.startswith("GCL2"):
        return "network_additive"
    if policy.startswith("AA3") or policy.startswith("GCL4_true"):
        return "network_gated"
    if policy.startswith("AA9"):
        return "guarded_integration"
    if policy.startswith("proposed_high_confidence_true"):
        return "proposed_policy"
    return "comparison"


def deployable(policy: str) -> str:
    return "no" if paper_bucket(policy) == "placebo_audit" else "yes"


def add_aprs(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    pr_values = summary["mean_main_pr_auc"].astype(float).tolist()
    top25_values = summary["main_top25_hits"].astype(float).tolist()
    severe_top25_values = summary["severe_top25_hits"].astype(float).tolist()

    proposed_true = summary.loc[
        summary["policy"] == "proposed_high_confidence_true"
    ].iloc[0]
    proposed_placebos = summary[
        summary["policy"].isin(
            [
                "proposed_high_confidence_equal_placebo",
                "proposed_high_confidence_random_placebo",
                "proposed_high_confidence_shuffled_placebo",
            ]
        )
    ].copy()
    best_placebo = proposed_placebos.sort_values(
        "mean_main_pr_auc", ascending=False
    ).iloc[0]
    proposed_delta = float(proposed_true["mean_main_pr_auc"]) - float(
        best_placebo["mean_main_pr_auc"]
    )
    proposed_top25_delta = int(proposed_true["main_top25_hits"]) - int(
        best_placebo["main_top25_hits"]
    )
    proposed_audit = clamp(0.5 + proposed_delta / MATERIAL_PR_AUC_DELTA)

    main_aprs = pd.read_csv(MAIN_APRS)
    aa9 = main_aprs[
        main_aprs["artifact_id"] == "AA9_gated_if_validation_safe_else_additive"
    ].iloc[0]

    aprs_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        predictive = minmax(float(row["mean_main_pr_auc"]), pr_values)
        top25_score = minmax(float(row["main_top25_hits"]), top25_values)
        severe_top25_score = minmax(
            float(row["severe_top25_hits"]), severe_top25_values
        )
        guardrail = 0.5 * top25_score + 0.5 * severe_top25_score
        audit = 0.0
        best_placebo_name = ""
        pr_delta: object = ""
        top25_delta: object = ""
        audit_note = "no true-vs-placebo audit contrast for this row"

        if row["policy"] == "proposed_high_confidence_true":
            audit = proposed_audit
            best_placebo_name = str(best_placebo["policy"])
            pr_delta = proposed_delta
            top25_delta = proposed_top25_delta
            audit_note = "fixed true high-confidence policy vs best matched placebo"
        elif row["policy"] == "AA9_gated_if_validation_safe_else_additive":
            audit = float(aa9["audit_reliability_score"])
            best_placebo_name = str(aa9["best_placebo_baseline"])
            pr_delta = aa9["true_vs_best_placebo_pr_auc_delta"]
            top25_delta = aa9["true_vs_best_placebo_top25_delta"]
            audit_note = "reference AA9 audit score from main APRS table"

        aprs = 0.50 * predictive + 0.30 * audit + 0.20 * guardrail
        aprs_rows.append(
            {
                "policy": row["policy"],
                "paper_bucket": paper_bucket(str(row["policy"])),
                "is_deployable_candidate": deployable(str(row["policy"])),
                "predictive_performance_score": predictive,
                "audit_reliability_score": audit,
                "guardrail_robustness_score": guardrail,
                "recommended_aprs": aprs,
                "reference_aa9_aprs": AA9_REFERENCE_APRS
                if row["policy"] == "AA9_gated_if_validation_safe_else_additive"
                else "",
                "best_placebo_baseline": best_placebo_name,
                "true_vs_best_placebo_pr_auc_delta": pr_delta,
                "true_vs_best_placebo_top25_delta": top25_delta,
                "audit_note": audit_note,
            }
        )
    aprs_df = pd.DataFrame(aprs_rows)
    return summary.merge(aprs_df, on="policy", how="left").sort_values(
        "recommended_aprs", ascending=False
    )


def build_metrics(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [
        *AA_POLICY_COLUMNS.keys(),
        "AA9_gated_if_validation_safe_else_additive",
        "GCL4_true_wits_conversion_propensity_gated",
        "GCL4_equal_conversion_propensity_gated_placebo",
        "GCL4_random_conversion_propensity_gated_placebo",
        "GCL4_shuffled_conversion_propensity_gated_placebo",
        *PROPOSED_VARIANTS.keys(),
    ]
    rows: list[dict[str, object]] = []
    for policy in policies:
        score_col = f"score::{policy}"
        for _, fold_df in test.groupby("fold", sort=True):
            rows.append(metrics_for_fold(fold_df, policy, score_col))
    by_fold = pd.DataFrame(rows)
    summary = add_aprs(summarize_metrics(by_fold))
    return by_fold, summary


def build_alerts(test: pd.DataFrame) -> pd.DataFrame:
    score_col = "score::proposed_high_confidence_true"
    alert_rows: list[pd.DataFrame] = []
    for fold, fold_df in test.groupby("fold", sort=True):
        top = fold_df.sort_values(score_col, ascending=False).head(50).copy()
        top.insert(0, "alert_rank", range(1, len(top) + 1))
        top.insert(0, "policy", "proposed_high_confidence_true")
        top["proposed_score"] = top[score_col]
        top["aa9_score"] = top["AA9_score"]
        top["conversion_score"] = top["GCL4_true_wits_conversion_propensity_gated"]
        top["aa_component_rank"] = top[
            "component_rank_aa::proposed_high_confidence_true"
        ]
        top["conversion_component_rank"] = top[
            "component_rank_conversion::proposed_high_confidence_true"
        ]
        top["vulnerability_component_rank"] = top[
            "component_rank_vulnerability::proposed_high_confidence_true"
        ]
        alert_rows.append(top)
    alerts = pd.concat(alert_rows, ignore_index=True)
    keep = [
        "policy",
        "fold",
        "alert_rank",
        "ISO3",
        "country",
        "week",
        "abnormal_next_week_container",
        "abnormal_next_week_container_2p0sigma",
        "operational_shortfall_12w",
        "proposed_score",
        "aa9_score",
        "conversion_score",
        "aa_component_rank",
        "conversion_component_rank",
        "vulnerability_component_rank",
    ]
    return alerts[keep]


def build_placebo_checks(summary: pd.DataFrame) -> pd.DataFrame:
    true = summary[summary["policy"] == "proposed_high_confidence_true"].iloc[0]
    rows: list[dict[str, object]] = []
    for placebo in [
        "proposed_high_confidence_equal_placebo",
        "proposed_high_confidence_random_placebo",
        "proposed_high_confidence_shuffled_placebo",
    ]:
        base = summary[summary["policy"] == placebo].iloc[0]
        rows.append(
            {
                "focus_policy": "proposed_high_confidence_true",
                "placebo_policy": placebo,
                "main_pr_auc_delta": float(true["mean_main_pr_auc"])
                - float(base["mean_main_pr_auc"]),
                "main_top10_hit_delta": int(true["main_top10_hits"])
                - int(base["main_top10_hits"]),
                "main_top25_hit_delta": int(true["main_top25_hits"])
                - int(base["main_top25_hits"]),
                "main_top50_hit_delta": int(true["main_top50_hits"])
                - int(base["main_top50_hits"]),
                "severe_top25_hit_delta": int(true["severe_top25_hits"])
                - int(base["severe_top25_hits"]),
                "placebo_recommended_aprs": float(base["recommended_aprs"]),
                "true_recommended_aprs": float(true["recommended_aprs"]),
                "audit_result": "true_beats_placebo"
                if float(true["mean_main_pr_auc"]) > float(base["mean_main_pr_auc"])
                else "placebo_competitive_or_stronger",
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, placebo: pd.DataFrame) -> None:
    proposed = summary[summary["policy"] == "proposed_high_confidence_true"].iloc[0]
    aa9 = summary[
        summary["policy"] == "AA9_gated_if_validation_safe_else_additive"
    ].iloc[0]
    best_placebo = summary[
        summary["policy"].isin(
            [
                "proposed_high_confidence_equal_placebo",
                "proposed_high_confidence_random_placebo",
                "proposed_high_confidence_shuffled_placebo",
            ]
        )
    ].sort_values("recommended_aprs", ascending=False).iloc[0]

    core_topk_improved = (
        int(proposed["main_top10_hits"]) > int(aa9["main_top10_hits"])
        or int(proposed["main_top25_hits"]) > int(aa9["main_top25_hits"])
        or int(proposed["main_top50_hits"]) > int(aa9["main_top50_hits"])
        or int(proposed["severe_top25_hits"]) > int(aa9["severe_top25_hits"])
    )
    success = (
        float(proposed["recommended_aprs"]) > AA9_REFERENCE_APRS
        and core_topk_improved
        and float(proposed["recommended_aprs"]) > float(best_placebo["recommended_aprs"])
    )

    lines = [
        "# Panel32 High-Confidence Alert Policy",
        "",
        "This report evaluates a fixed two-stage high-confidence alert policy under the main PortWatch + GDELT + WITS data scope. The policy reuses existing out-of-sample scores and keeps a fixed top-k alert budget.",
        "",
        "## Policy",
        "",
        "`proposed_high_confidence_true = 0.4 * rank(AA9 validation-safe score) + 0.4 * rank(GCL4 true WITS conversion-propensity gated score) + 0.2 * rank(positive operational shortfall)`.",
        "",
        "AA9's additive-vs-gated component is the existing validation-selected fold policy. The conversion-propensity score table is test-fold OOS only, so this script uses a fixed consensus formula rather than selecting or tuning its weights. Equal/random/shuffled placebo versions use the same formula with matched placebo network scores.",
        "",
        "## Main Result",
        "",
        "| Policy | PR-AUC | top-10 | top-25 | top-50 | severe top-25 | APRS |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| AA9 reference | {float(aa9['mean_main_pr_auc']):.4f} | {int(aa9['main_top10_hits'])} | {int(aa9['main_top25_hits'])} | {int(aa9['main_top50_hits'])} | {int(aa9['severe_top25_hits'])} | {AA9_REFERENCE_APRS:.4f} |",
        f"| Proposed high-confidence | {float(proposed['mean_main_pr_auc']):.4f} | {int(proposed['main_top10_hits'])} | {int(proposed['main_top25_hits'])} | {int(proposed['main_top50_hits'])} | {int(proposed['severe_top25_hits'])} | {float(proposed['recommended_aprs']):.4f} |",
        f"| Best matched placebo | {float(best_placebo['mean_main_pr_auc']):.4f} | {int(best_placebo['main_top10_hits'])} | {int(best_placebo['main_top25_hits'])} | {int(best_placebo['main_top50_hits'])} | {int(best_placebo['severe_top25_hits'])} | {float(best_placebo['recommended_aprs']):.4f} |",
        "",
        "Note: the AA9 APRS in this comparison is the original/reference APRS carried from `reports/main_paper_aprs.md`. The summary CSV also includes a within-candidate-pool `recommended_aprs` diagnostic for AA9; do not use that diagnostic value to replace the original AA9 baseline.",
        "",
        "## Placebo Checks",
        "",
        "| Placebo | PR-AUC delta | top-25 delta | severe top-25 delta | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in placebo.iterrows():
        lines.append(
            f"| {row['placebo_policy']} | {float(row['main_pr_auc_delta']):.4f} | {int(row['main_top25_hit_delta'])} | {int(row['severe_top25_hit_delta'])} | {row['audit_result']} |"
        )

    lines.extend(
        [
            "",
            "## Gate Judgment",
            "",
            f"- APRS above AA9 reference: {'yes' if float(proposed['recommended_aprs']) > AA9_REFERENCE_APRS else 'no'} (`{float(proposed['recommended_aprs']):.4f}` vs `{AA9_REFERENCE_APRS:.4f}`).",
            f"- Core top-k improvement over AA9: {'yes' if core_topk_improved else 'no'}.",
            f"- Placebo APRS not leading: {'yes' if float(proposed['recommended_aprs']) > float(best_placebo['recommended_aprs']) else 'no'}.",
            "- Fixed alert budget: yes, every policy is evaluated at top-10/top-25/top-50 per fold.",
            "- Validation-only / no test-label tuning: yes for AA9 fold selection; the proposed consensus formula is fixed before reading test labels, and matched placebos use the same weights.",
            "",
            f"Overall status: {'PASS' if success else 'FAIL'} according to the pre-specified gate.",
            "",
            "## Interpretation",
            "",
            "The proposed policy should be treated as a deployment-oriented alert allocation rule, not as evidence that WITS is causal or universally predictive. Its value comes from combining validation-guarded model selection, conversion-oriented event scores, operational vulnerability, and matched placebo network checks.",
            "",
            "Raw outputs:",
            "",
            f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
            f"- `{OUT_BY_FOLD.relative_to(ROOT)}`",
            f"- `{OUT_ALERTS.relative_to(ROOT)}`",
            f"- `{OUT_PLACEBO.relative_to(ROOT)}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    test = build_test_frame()
    by_fold, summary = build_metrics(test)
    alerts = build_alerts(test)
    placebo = build_placebo_checks(summary)

    OUT_BY_FOLD.parent.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(OUT_BY_FOLD, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    alerts.to_csv(OUT_ALERTS, index=False)
    placebo.to_csv(OUT_PLACEBO, index=False)
    write_report(summary, placebo)

    proposed = summary[summary["policy"] == "proposed_high_confidence_true"].iloc[0]
    print(
        "proposed_high_confidence_true:",
        f"PR-AUC={float(proposed['mean_main_pr_auc']):.4f}",
        f"top25={int(proposed['main_top25_hits'])}",
        f"severe_top25={int(proposed['severe_top25_hits'])}",
        f"APRS={float(proposed['recommended_aprs']):.4f}",
    )
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
