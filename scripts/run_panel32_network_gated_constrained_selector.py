from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    FOLDS,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)
from scripts.run_panel32_network_gated_conversion_main import (  # noqa: E402
    RANDOM_SEED,
    SEVERE_TARGET,
    fit_model,
    make_feature_groups as make_national_feature_groups,
    make_models,
    safe_ap,
    top_hits,
)
from scripts.run_panel32_network_gated_port_system_main import (  # noqa: E402
    load_dataset,
    make_feature_groups as make_port_system_feature_groups,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_network_gated_constrained_selector.md"
CANDIDATE_METRICS = TABLE_DIR / "panel32_network_gated_constrained_candidate_metrics.csv"
CANDIDATE_PREDICTIONS = TABLE_DIR / "panel32_network_gated_constrained_candidate_predictions.csv"
SELECTIONS = TABLE_DIR / "panel32_network_gated_constrained_selections.csv"
POLICY_PREDICTIONS = TABLE_DIR / "panel32_network_gated_constrained_policy_predictions.csv"
POLICY_SUMMARY = TABLE_DIR / "panel32_network_gated_constrained_policy_summary.csv"
POLICY_DELTAS = TABLE_DIR / "panel32_network_gated_constrained_policy_deltas.csv"
CLAIM_MATRIX = TABLE_DIR / "panel32_network_gated_constrained_claim_matrix.csv"


@dataclass(frozen=True)
class CandidateSpec:
    candidate: str
    feature_group: str
    family: str
    role: str
    additive_baseline: str | None = None
    operational_baseline: str | None = None


CANDIDATES = [
    CandidateSpec("NAT_OP", "NG0_portwatch_operational", "national", "operational"),
    CandidateSpec("NAT_GDELT", "NG1_portwatch_gdelt_additive", "national", "gdelt_additive"),
    CandidateSpec("NAT_WITS_ADD", "NG2_portwatch_gdelt_wits_additive", "national", "wits_additive"),
    CandidateSpec(
        "NAT_GATED_TRUE",
        "NG5_compact_network_gated_true",
        "national",
        "true_gated",
        additive_baseline="NAT_WITS_ADD",
        operational_baseline="NAT_OP",
    ),
    CandidateSpec("NAT_GATED_EQUAL", "NG6_equal_compact_gated_placebo", "national", "placebo"),
    CandidateSpec("NAT_GATED_RANDOM", "NG6_random_compact_gated_placebo", "national", "placebo"),
    CandidateSpec("NAT_GATED_SHUFFLED", "NG6_shuffled_compact_gated_placebo", "national", "placebo"),
    CandidateSpec("PS_OP", "PSNG0b_portwatch_operational_port_system", "port_system", "operational"),
    CandidateSpec("PS_GDELT", "PSNG1_portwatch_port_system_gdelt_additive", "port_system", "gdelt_additive"),
    CandidateSpec("PS_WITS_ADD", "PSNG2_portwatch_port_system_gdelt_wits_additive", "port_system", "wits_additive"),
    CandidateSpec(
        "PS_GATED_TRUE",
        "PSNG5_port_system_compact_gated_true",
        "port_system",
        "true_gated",
        additive_baseline="PS_WITS_ADD",
        operational_baseline="PS_OP",
    ),
    CandidateSpec("PS_GATED_EQUAL", "PSNG6_equal_port_system_compact_gated_placebo", "port_system", "placebo"),
    CandidateSpec("PS_GATED_RANDOM", "PSNG6_random_port_system_compact_gated_placebo", "port_system", "placebo"),
    CandidateSpec("PS_GATED_SHUFFLED", "PSNG6_shuffled_port_system_compact_gated_placebo", "port_system", "placebo"),
]


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def candidate_by_name() -> dict[str, CandidateSpec]:
    return {spec.candidate: spec for spec in CANDIDATES}


def score_frame(frame: pd.DataFrame, candidate: str, split: str) -> dict:
    out = {
        "fold": frame["fold"].iloc[0],
        "split": split,
        "candidate": candidate,
        "rows": len(frame),
        "positives": int(frame[TARGET].sum()),
        "severe_positives": int(frame[SEVERE_TARGET].sum()),
        "main_pr_auc": safe_ap(frame[TARGET], frame["predicted_probability"]),
        "severe_pr_auc": safe_ap(frame[SEVERE_TARGET], frame["predicted_probability"]),
    }
    for k in [10, 25, 50]:
        out[f"main_top{k}_hits"] = top_hits(frame, TARGET, k)
        out[f"severe_top{k}_hits"] = top_hits(frame, SEVERE_TARGET, k)
    return out


def make_combined_feature_groups(frame: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    groups = {}
    groups.update(make_national_feature_groups(frame, country_features))
    groups.update(make_port_system_feature_groups(frame, country_features))
    return groups


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_combined_feature_groups(train, country_features)
    model_name = "sklearn_gradient_boosting"
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in CANDIDATES:
        features = usable_features(train, groups[spec.feature_group])
        model, fit_mode = make_models()[model_name]
        fit_model(model, fit_mode, train[features], train[TARGET])
        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold, val_f1 = select_threshold(validation[TARGET].reset_index(drop=True), val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        test_scores = evaluate_predictions(test[TARGET].reset_index(drop=True), test_proba, threshold)

        for split, source, proba in [
            ("validation", validation, val_proba),
            ("test", test, test_proba),
        ]:
            pred = source[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
            pred["fold"] = fold.name
            pred["split"] = split
            pred["candidate"] = spec.candidate
            pred["feature_group"] = spec.feature_group
            pred["family"] = spec.family
            pred["role"] = spec.role
            pred["model"] = model_name
            pred["predicted_probability"] = proba
            prediction_frames.append(pred)
            metric_row = score_frame(pred, spec.candidate, split)
            metric_row.update(
                {
                    "feature_group": spec.feature_group,
                    "family": spec.family,
                    "role": spec.role,
                    "model": model_name,
                    "feature_count": len(features),
                    "selected_threshold": threshold,
                    "validation_f1": val_f1,
                }
            )
            if split == "test":
                metric_row.update(
                    {
                        "roc_auc": test_scores["roc_auc"],
                        "f1": test_scores["f1"],
                        "precision": test_scores["precision"],
                        "recall": test_scores["recall"],
                    }
                )
            metric_rows.append(metric_row)
    return metric_rows, prediction_frames


def top_hits_by_candidate(metrics: pd.DataFrame, fold: str, candidate: str, field: str) -> float:
    row = metrics.loc[
        metrics["fold"].eq(fold) & metrics["split"].eq("validation") & metrics["candidate"].eq(candidate)
    ]
    if row.empty:
        return np.nan
    return float(row[field].iloc[0])


def select_unconstrained(metrics: pd.DataFrame, fold: str) -> tuple[str, dict]:
    val = metrics.loc[metrics["fold"].eq(fold) & metrics["split"].eq("validation")].copy()
    row = val.sort_values(["main_pr_auc", "severe_pr_auc", "main_top25_hits"], ascending=False).iloc[0]
    return str(row["candidate"]), {"selection_reason": "highest validation PR-AUC, placebo allowed"}


def select_nonplacebo(metrics: pd.DataFrame, fold: str) -> tuple[str, dict]:
    val = metrics.loc[
        metrics["fold"].eq(fold) & metrics["split"].eq("validation") & ~metrics["role"].eq("placebo")
    ].copy()
    row = val.sort_values(["main_pr_auc", "severe_pr_auc", "main_top25_hits"], ascending=False).iloc[0]
    return str(row["candidate"]), {"selection_reason": "highest validation PR-AUC among non-placebo candidates"}


def max_family_placebo(metrics: pd.DataFrame, fold: str, family: str, field: str) -> float:
    vals = metrics.loc[
        metrics["fold"].eq(fold)
        & metrics["split"].eq("validation")
        & metrics["family"].eq(family)
        & metrics["role"].eq("placebo"),
        field,
    ]
    return float(vals.max()) if not vals.empty else -np.inf


def validation_value(metrics: pd.DataFrame, fold: str, candidate: str, field: str) -> float:
    row = metrics.loc[
        metrics["fold"].eq(fold) & metrics["split"].eq("validation") & metrics["candidate"].eq(candidate)
    ]
    return float(row[field].iloc[0])


def select_placebo_penalized(metrics: pd.DataFrame, fold: str) -> tuple[str, dict]:
    specs = candidate_by_name()
    val = metrics.loc[
        metrics["fold"].eq(fold) & metrics["split"].eq("validation") & ~metrics["role"].eq("placebo")
    ].copy()
    adjusted_rows = []
    for _, row in val.iterrows():
        spec = specs[str(row["candidate"])]
        penalty = 0.0
        max_placebo_ap = max_family_placebo(metrics, fold, spec.family, "main_pr_auc")
        max_placebo_severe = max_family_placebo(metrics, fold, spec.family, "severe_pr_auc")
        if spec.role == "true_gated":
            add_ap = validation_value(metrics, fold, spec.additive_baseline or spec.candidate, "main_pr_auc")
            add_severe = validation_value(metrics, fold, spec.additive_baseline or spec.candidate, "severe_pr_auc")
            penalty += 2.0 * max(0.0, max_placebo_ap - float(row["main_pr_auc"]))
            penalty += 1.0 * max(0.0, add_ap - float(row["main_pr_auc"]))
            penalty += 0.5 * max(0.0, max_placebo_severe - float(row["severe_pr_auc"]))
            penalty += 0.25 * max(0.0, add_severe - float(row["severe_pr_auc"]))
        elif spec.role in {"wits_additive", "gdelt_additive"}:
            penalty += 0.5 * max(0.0, max_placebo_ap - float(row["main_pr_auc"]))
        adjusted = float(row["main_pr_auc"]) + 0.25 * float(row["severe_pr_auc"]) - penalty
        adjusted_rows.append({**row.to_dict(), "adjusted_score": adjusted, "penalty": penalty})
    adjusted_df = pd.DataFrame(adjusted_rows)
    selected = adjusted_df.sort_values(["adjusted_score", "main_pr_auc", "main_top25_hits"], ascending=False).iloc[0]
    return str(selected["candidate"]), {
        "selection_reason": "max validation PR-AUC with additive/placebo penalties",
        "adjusted_score": float(selected["adjusted_score"]),
        "penalty": float(selected["penalty"]),
    }


def select_gated_if_valid_else_additive(metrics: pd.DataFrame, fold: str) -> tuple[str, dict]:
    specs = candidate_by_name()
    eligible = []
    for candidate in ["NAT_GATED_TRUE", "PS_GATED_TRUE"]:
        spec = specs[candidate]
        main_ap = validation_value(metrics, fold, candidate, "main_pr_auc")
        severe_ap = validation_value(metrics, fold, candidate, "severe_pr_auc")
        add_ap = validation_value(metrics, fold, spec.additive_baseline or candidate, "main_pr_auc")
        add_severe = validation_value(metrics, fold, spec.additive_baseline or candidate, "severe_pr_auc")
        placebo_ap = max_family_placebo(metrics, fold, spec.family, "main_pr_auc")
        placebo_severe = max_family_placebo(metrics, fold, spec.family, "severe_pr_auc")
        if main_ap >= add_ap and main_ap >= placebo_ap and severe_ap >= add_severe and severe_ap >= placebo_severe:
            eligible.append((candidate, main_ap, severe_ap))
    if eligible:
        eligible.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return eligible[0][0], {"selection_reason": "gated candidate passed additive and placebo validation guards"}
    candidates = ["NAT_WITS_ADD", "PS_WITS_ADD", "NAT_GDELT", "PS_GDELT", "NAT_OP", "PS_OP"]
    best = max(
        candidates,
        key=lambda c: (
            validation_value(metrics, fold, c, "main_pr_auc"),
            validation_value(metrics, fold, c, "severe_pr_auc"),
        ),
    )
    return best, {"selection_reason": "no gated candidate passed validation guards; fallback to best additive/operational"}


def select_topk_guarded(metrics: pd.DataFrame, fold: str) -> tuple[str, dict]:
    specs = candidate_by_name()
    eligible = []
    for spec in CANDIDATES:
        if spec.role == "placebo":
            continue
        main_top25 = validation_value(metrics, fold, spec.candidate, "main_top25_hits")
        severe_top25 = validation_value(metrics, fold, spec.candidate, "severe_top25_hits")
        main_ap = validation_value(metrics, fold, spec.candidate, "main_pr_auc")
        op = spec.operational_baseline or ("PS_OP" if spec.family == "port_system" else "NAT_OP")
        op_main_top25 = top_hits_by_candidate(metrics, fold, op, "main_top25_hits")
        op_severe_top25 = top_hits_by_candidate(metrics, fold, op, "severe_top25_hits")
        if main_top25 < op_main_top25 or severe_top25 < op_severe_top25:
            continue
        if spec.role == "true_gated":
            max_placebo_top25 = max_family_placebo(metrics, fold, spec.family, "main_top25_hits")
            if main_top25 < max_placebo_top25:
                continue
        eligible.append((spec.candidate, main_top25, severe_top25, main_ap))
    if eligible:
        eligible.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        return eligible[0][0], {"selection_reason": "best validation top-25 under operational/severe/placebo guards"}
    return "NAT_OP", {"selection_reason": "no candidate passed top-k/severe guards; fallback to national operational"}


SELECTORS = {
    "CAS0_unconstrained_best_val_pr_auc": select_unconstrained,
    "CAS1_nonplacebo_best_val_pr_auc": select_nonplacebo,
    "CAS2_placebo_penalized_pr_auc": select_placebo_penalized,
    "CAS3_gated_if_valid_else_additive": select_gated_if_valid_else_additive,
    "CAS4_top25_severe_guarded": select_topk_guarded,
}

REFERENCE_POLICIES = {
    "REF_NAT_OP": "NAT_OP",
    "REF_NAT_WITS_ADD": "NAT_WITS_ADD",
    "REF_NAT_GATED_TRUE": "NAT_GATED_TRUE",
    "REF_PS_WITS_ADD": "PS_WITS_ADD",
    "REF_PS_GATED_TRUE": "PS_GATED_TRUE",
    "REF_NAT_EQUAL_PLACEBO": "NAT_GATED_EQUAL",
    "REF_PS_EQUAL_PLACEBO": "PS_GATED_EQUAL",
}


def build_policy_predictions(candidate_predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_predictions = candidate_predictions.loc[candidate_predictions["split"].eq("test")].copy()
    selections = []
    policy_frames = []
    for fold in sorted(metrics["fold"].unique()):
        for policy, selector in SELECTORS.items():
            candidate, extra = selector(metrics, fold)
            frame = test_predictions.loc[
                test_predictions["fold"].eq(fold) & test_predictions["candidate"].eq(candidate)
            ].copy()
            frame["policy"] = policy
            frame["selected_candidate"] = candidate
            policy_frames.append(frame)
            selected_metric = metrics.loc[
                metrics["fold"].eq(fold) & metrics["split"].eq("validation") & metrics["candidate"].eq(candidate)
            ].iloc[0]
            selections.append(
                {
                    "fold": fold,
                    "policy": policy,
                    "selected_candidate": candidate,
                    "selected_role": selected_metric["role"],
                    "selected_family": selected_metric["family"],
                    "validation_main_pr_auc": selected_metric["main_pr_auc"],
                    "validation_severe_pr_auc": selected_metric["severe_pr_auc"],
                    "validation_main_top25_hits": selected_metric["main_top25_hits"],
                    "validation_severe_top25_hits": selected_metric["severe_top25_hits"],
                    **extra,
                }
            )
        for policy, candidate in REFERENCE_POLICIES.items():
            frame = test_predictions.loc[
                test_predictions["fold"].eq(fold) & test_predictions["candidate"].eq(candidate)
            ].copy()
            frame["policy"] = policy
            frame["selected_candidate"] = candidate
            policy_frames.append(frame)
            selected_metric = metrics.loc[
                metrics["fold"].eq(fold) & metrics["split"].eq("validation") & metrics["candidate"].eq(candidate)
            ].iloc[0]
            selections.append(
                {
                    "fold": fold,
                    "policy": policy,
                    "selected_candidate": candidate,
                    "selected_role": selected_metric["role"],
                    "selected_family": selected_metric["family"],
                    "validation_main_pr_auc": selected_metric["main_pr_auc"],
                    "validation_severe_pr_auc": selected_metric["severe_pr_auc"],
                    "validation_main_top25_hits": selected_metric["main_top25_hits"],
                    "validation_severe_top25_hits": selected_metric["severe_top25_hits"],
                    "selection_reason": "fixed reference policy",
                }
            )
    return pd.concat(policy_frames, ignore_index=True), pd.DataFrame(selections)


def summarize_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, frame in predictions.groupby("policy"):
        main_fold_ap = [safe_ap(f[TARGET], f["predicted_probability"]) for _, f in frame.groupby("fold")]
        severe_fold_ap = [safe_ap(f[SEVERE_TARGET], f["predicted_probability"]) for _, f in frame.groupby("fold")]
        row = {
            "policy": policy,
            "folds": frame["fold"].nunique(),
            "mean_main_pr_auc": float(pd.Series(main_fold_ap).mean()),
            "median_main_pr_auc": float(pd.Series(main_fold_ap).median()),
            "pooled_main_pr_auc": safe_ap(frame[TARGET], frame["predicted_probability"]),
            "mean_severe_pr_auc": float(pd.Series(severe_fold_ap).mean()),
            "median_severe_pr_auc": float(pd.Series(severe_fold_ap).median()),
            "pooled_severe_pr_auc": safe_ap(frame[SEVERE_TARGET], frame["predicted_probability"]),
        }
        for k in [10, 25, 50]:
            row[f"main_top{k}_hits"] = sum(top_hits(f, TARGET, k) for _, f in frame.groupby("fold"))
            row[f"severe_top{k}_hits"] = sum(
                top_hits(f, SEVERE_TARGET, k)
                for _, f in frame.groupby("fold")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)


def pooled_policy_delta(
    predictions: pd.DataFrame,
    focus: str,
    baseline: str,
    target: str,
    n_boot: int = 2000,
) -> tuple[float, float, float, float]:
    focus_frame = predictions.loc[predictions["policy"].eq(focus)]
    baseline_frame = predictions.loc[predictions["policy"].eq(baseline)]
    merged = focus_frame.merge(
        baseline_frame,
        on=["ISO3", "country", "week", "fold", TARGET, SEVERE_TARGET],
        suffixes=("_focus", "_baseline"),
    ).reset_index(drop=True)
    point = safe_ap(merged[target], merged["predicted_probability_focus"]) - safe_ap(
        merged[target], merged["predicted_probability_baseline"]
    )
    draws = []
    for seed in range(n_boot):
        sample = merged.sample(n=len(merged), replace=True, random_state=RANDOM_SEED + seed)
        if sample[target].nunique() < 2:
            continue
        draws.append(
            average_precision_score(sample[target], sample["predicted_probability_focus"])
            - average_precision_score(sample[target], sample["predicted_probability_baseline"])
        )
    series = pd.Series(draws)
    return point, float(series.quantile(0.025)), float(series.quantile(0.975)), float((series > 0).mean())


def policy_hit_delta(predictions: pd.DataFrame, focus: str, baseline: str, target: str, k: int) -> int:
    delta = 0
    for fold, frame in predictions.loc[predictions["policy"].isin([focus, baseline])].groupby("fold"):
        focus_top = frame.loc[frame["policy"].eq(focus)].sort_values("predicted_probability", ascending=False).head(k)
        base_top = frame.loc[frame["policy"].eq(baseline)].sort_values("predicted_probability", ascending=False).head(k)
        delta += int(focus_top[target].sum()) - int(base_top[target].sum())
    return delta


def make_policy_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    focus_policies = list(SELECTORS.keys())
    baselines = [
        "REF_NAT_OP",
        "REF_NAT_WITS_ADD",
        "REF_NAT_GATED_TRUE",
        "REF_PS_WITS_ADD",
        "REF_PS_GATED_TRUE",
        "REF_NAT_EQUAL_PLACEBO",
        "REF_PS_EQUAL_PLACEBO",
    ]
    rows = []
    for focus in focus_policies:
        for baseline in baselines:
            for label_name, target in [("main", TARGET), ("severe", SEVERE_TARGET)]:
                pr = pooled_policy_delta(predictions, focus, baseline, target)
                rows.append(
                    {
                        "contrast": f"{focus}_vs_{baseline}",
                        "focus_policy": focus,
                        "baseline_policy": baseline,
                        "label": label_name,
                        "pooled_pr_auc_delta": pr[0],
                        "ci_low": pr[1],
                        "ci_high": pr[2],
                        "p_gt_0": pr[3],
                        "top10_hit_delta": policy_hit_delta(predictions, focus, baseline, target, 10),
                        "top25_hit_delta": policy_hit_delta(predictions, focus, baseline, target, 25),
                        "top50_hit_delta": policy_hit_delta(predictions, focus, baseline, target, 50),
                    }
                )
    return pd.DataFrame(rows)


def make_claim_matrix(summary: pd.DataFrame, deltas: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    indexed = summary.set_index("policy")
    guarded = indexed.loc["CAS4_top25_severe_guarded"]
    penalized = indexed.loc["CAS2_placebo_penalized_pr_auc"]
    penalized_vs_nat = deltas.loc[
        deltas["contrast"].eq("CAS2_placebo_penalized_pr_auc_vs_REF_NAT_WITS_ADD") & deltas["label"].eq("main")
    ].iloc[0]
    guarded_vs_op = deltas.loc[
        deltas["contrast"].eq("CAS4_top25_severe_guarded_vs_REF_NAT_OP") & deltas["label"].eq("main")
    ].iloc[0]
    placebo_selected = selections.loc[
        selections["policy"].eq("CAS0_unconstrained_best_val_pr_auc") & selections["selected_role"].eq("placebo")
    ]
    penalized_true_gated = selections.loc[
        selections["policy"].eq("CAS2_placebo_penalized_pr_auc") & selections["selected_role"].eq("true_gated")
    ]
    gated_guard_true_gated = selections.loc[
        selections["policy"].eq("CAS3_gated_if_valid_else_additive") & selections["selected_role"].eq("true_gated")
    ]
    return pd.DataFrame(
        [
            {
                "claim": "Unconstrained validation selection can be fooled by placebo-like network scores.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence": f"unconstrained selector chose placebo in {len(placebo_selected)}/{selections['fold'].nunique()} folds",
                "paper_bucket": "negative/motivation",
                "status": "computed",
            },
            {
                "claim": "Placebo-penalized selection prevents gated promotion under current evidence.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence": f"CAS2 selected true gated in {len(penalized_true_gated)}/{selections['fold'].nunique()} folds; mean PR-AUC {penalized['mean_main_pr_auc']:.4f}; delta vs NAT WITS additive {penalized_vs_nat['pooled_pr_auc_delta']:.4f}",
                "paper_bucket": "supporting/negative guardrail",
                "status": "computed",
            },
            {
                "claim": "Strict gated eligibility rejects all current gated candidates.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence": f"CAS3 selected true gated in {len(gated_guard_true_gated)}/{selections['fold'].nunique()} folds; it fell back to operational/additive policies",
                "paper_bucket": "negative/method guardrail",
                "status": "computed",
            },
            {
                "claim": "Top-k/severe guardrails can protect deployment behavior but do not rescue gated conversion.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence": f"CAS4 mean PR-AUC {guarded['mean_main_pr_auc']:.4f}; top25 {int(guarded['main_top25_hits'])}; delta vs operational top25 {guarded_vs_op['top25_hit_delta']}",
                "paper_bucket": "supporting/deployment guardrail",
                "status": "computed",
            },
        ]
    )


def write_report(
    candidate_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    matrix: pd.DataFrame,
) -> None:
    content = f"""# Panel32 Network-Gated Constrained Alert Selector

## Purpose

This experiment tests a validation-safe constrained alert-allocation layer for the main-paper **Network-Gated Event Conversion Model**. It uses only PortWatch, GDELT, and WITS. The goal is not to add more features, but to prevent a deployment policy from promoting gated scores that behave like equal/random/shuffled WITS placebos.

## Candidate Pool

- National operational, GDELT additive, WITS additive, compact true gated, and compact WITS-placebo gated scores.
- PortWatch selected-port operational, GDELT additive, WITS additive, compact true gated, and compact WITS-placebo gated scores.
- Model family: sklearn Gradient Boosting.
- Selection uses validation years only; test years are held out.

## Policies

- `CAS0_unconstrained_best_val_pr_auc`: highest validation PR-AUC, placebo allowed. Diagnostic only.
- `CAS1_nonplacebo_best_val_pr_auc`: highest validation PR-AUC excluding placebos.
- `CAS2_placebo_penalized_pr_auc`: validation PR-AUC with explicit additive/placebo penalties for true gated candidates.
- `CAS3_gated_if_valid_else_additive`: true gated is eligible only if it beats additive and same-family placebos on validation main/severe PR-AUC; otherwise fallback to additive/operational.
- `CAS4_top25_severe_guarded`: validation top-25 allocation with operational, severe, and placebo guards.

## Policy Summary

{summary.to_markdown(index=False)}

## Fold Selections

{selections.to_markdown(index=False)}

## Key Deltas

{deltas.to_markdown(index=False)}

## Claim Matrix

{matrix.to_markdown(index=False)}

## Reading

A promotable constrained selector should improve over operational/additive references while not selecting placebo-like network scores. If the placebo-penalized policies mostly fall back to additive or operational references, that is still useful negative evidence: it means the current gated scores do not pass a validation-safe network-specific promotion rule.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    metric_rows = []
    prediction_frames = []
    for fold in FOLDS:
        rows, preds = run_fold(fold, df)
        metric_rows.extend(rows)
        prediction_frames.extend(preds)
        print(f"Finished {fold.name}: {len(rows)} metric rows", flush=True)
    candidate_metrics = pd.DataFrame(metric_rows)
    candidate_predictions = pd.concat(prediction_frames, ignore_index=True)
    policy_predictions, selections = build_policy_predictions(candidate_predictions, candidate_metrics)
    summary = summarize_policies(policy_predictions)
    deltas = make_policy_deltas(policy_predictions)
    matrix = make_claim_matrix(summary, deltas, selections)
    candidate_metrics.to_csv(CANDIDATE_METRICS, index=False)
    candidate_predictions.to_csv(CANDIDATE_PREDICTIONS, index=False)
    selections.to_csv(SELECTIONS, index=False)
    policy_predictions.to_csv(POLICY_PREDICTIONS, index=False)
    summary.to_csv(POLICY_SUMMARY, index=False)
    deltas.to_csv(POLICY_DELTAS, index=False)
    matrix.to_csv(CLAIM_MATRIX, index=False)
    write_report(candidate_metrics, selections, summary, deltas, matrix)
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(selections.to_string(index=False))


if __name__ == "__main__":
    run()
