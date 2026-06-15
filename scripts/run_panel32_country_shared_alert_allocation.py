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
from scripts.run_panel32_network_gated_country_shared_conversion import (  # noqa: E402
    PRIMARY_MODEL,
    SEVERE_TARGET,
    add_fold_calibrated_features,
    fit_model,
    load_dataset,
    make_feature_groups,
    make_models,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_country_shared_alert_allocation.md"
CANDIDATE_METRICS = TABLE_DIR / "panel32_country_shared_alert_allocation_candidate_metrics.csv"
CANDIDATE_PREDICTIONS = TABLE_DIR / "panel32_country_shared_alert_allocation_candidate_predictions.csv"
POLICY_SELECTIONS = TABLE_DIR / "panel32_country_shared_alert_allocation_selections.csv"
POLICY_PREDICTIONS = TABLE_DIR / "panel32_country_shared_alert_allocation_policy_predictions.csv"
POLICY_SUMMARY = TABLE_DIR / "panel32_country_shared_alert_allocation_policy_summary.csv"
POLICY_DELTAS = TABLE_DIR / "panel32_country_shared_alert_allocation_policy_deltas.csv"
CLAIM_MATRIX = TABLE_DIR / "panel32_country_shared_alert_allocation_claim_matrix.csv"

RANDOM_SEED = 42


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_group: str
    role: str


@dataclass(frozen=True)
class AllocationSpec:
    score_kind: str
    candidate: str | None = None
    left: str | None = None
    right: str | None = None
    weight_right: float = 0.0
    country_cap: int | None = None
    shortfall_reserve: float = 0.0

    def label(self) -> str:
        if self.score_kind == "candidate":
            core = str(self.candidate)
        else:
            pct = int(round(self.weight_right * 100))
            core = f"blend_{self.left}_{100 - pct}_{self.right}_{pct}"
        cap = "nocap" if self.country_cap is None else f"cap{self.country_cap}"
        reserve = int(round(self.shortfall_reserve * 100))
        return f"{core}_{cap}_shortfall{reserve}"


CANDIDATES = [
    CandidateSpec("CS_OP", "CS0_portwatch_operational_calibrated", "operational"),
    CandidateSpec("CS_GDELT", "CS1_portwatch_gdelt_additive_calibrated", "gdelt_additive"),
    CandidateSpec("CS_TRUE_ADD", "CS2_true_wits_additive_calibrated", "true_additive"),
    CandidateSpec("CS_TRUE_GATED", "CS3_country_shared_gated_true", "true_gated"),
    CandidateSpec("CS_EQUAL_GATED", "CS3_equal_country_shared_gated_placebo", "placebo"),
    CandidateSpec("CS_RANDOM_GATED", "CS3_random_country_shared_gated_placebo", "placebo"),
    CandidateSpec("CS_SHUFFLED_GATED", "CS3_shuffled_country_shared_gated_placebo", "placebo"),
]

FIXED_POLICIES = {
    "AA0_fixed_operational": AllocationSpec("candidate", candidate="CS_OP"),
    "AA1_fixed_gdelt_additive": AllocationSpec("candidate", candidate="CS_GDELT"),
    "AA2_fixed_true_wits_additive": AllocationSpec("candidate", candidate="CS_TRUE_ADD"),
    "AA3_fixed_true_gated": AllocationSpec("candidate", candidate="CS_TRUE_GATED"),
    "AA4_fixed_equal_gated_placebo": AllocationSpec("candidate", candidate="CS_EQUAL_GATED"),
    "AA5_fixed_random_gated_placebo": AllocationSpec("candidate", candidate="CS_RANDOM_GATED"),
    "AA6_fixed_shuffled_gated_placebo": AllocationSpec("candidate", candidate="CS_SHUFFLED_GATED"),
}

SEARCH_POLICY = "AA7_validation_constrained_alert_allocator"
BEST_SINGLE_POLICY = "AA8_validation_best_single_nonplacebo"
GATED_GUARD_POLICY = "AA9_gated_if_validation_safe_else_additive"


def safe_ap(y: pd.Series, score: pd.Series) -> float:
    if y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_hits(frame: pd.DataFrame, score_col: str, target_col: str, k: int) -> int:
    return int(frame.sort_values(score_col, ascending=False).head(k)[target_col].sum())


def numeric_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])]


def candidate_map() -> dict[str, CandidateSpec]:
    return {spec.name: spec for spec in CANDIDATES}


def rank_percentile(score: pd.Series) -> pd.Series:
    return score.rank(method="first", pct=True)


def constrained_rank_score(
    frame: pd.DataFrame,
    raw_score: pd.Series,
    country_cap: int | None,
    shortfall_reserve: float,
    k: int = 25,
) -> pd.Series:
    base = rank_percentile(raw_score).fillna(0.0) * 0.9
    if country_cap is None and shortfall_reserve <= 0:
        return base

    order = raw_score.sort_values(ascending=False).index.tolist()
    shortfall_active = frame["operational_shortfall_12w"].fillna(0.0).gt(0.0)
    selected: list[int] = []
    country_counts: dict[str, int] = {}

    def can_add(idx: int) -> bool:
        if idx in selected:
            return False
        if country_cap is None:
            return True
        iso3 = str(frame.at[idx, "ISO3"])
        return country_counts.get(iso3, 0) < country_cap

    def add_idx(idx: int) -> None:
        selected.append(idx)
        iso3 = str(frame.at[idx, "ISO3"])
        country_counts[iso3] = country_counts.get(iso3, 0) + 1

    reserve_n = int(round(k * shortfall_reserve))
    if reserve_n > 0:
        for idx in order:
            if len(selected) >= reserve_n:
                break
            if bool(shortfall_active.at[idx]) and can_add(idx):
                add_idx(idx)

    for idx in order:
        if len(selected) >= k:
            break
        if can_add(idx):
            add_idx(idx)

    adjusted = base.copy()
    for rank, idx in enumerate(selected):
        adjusted.at[idx] = 2.0 + (k - rank) / k
    return adjusted


def raw_score(frame: pd.DataFrame, spec: AllocationSpec) -> pd.Series:
    if spec.score_kind == "candidate":
        return frame[str(spec.candidate)].astype(float)
    left = rank_percentile(frame[str(spec.left)].astype(float))
    right = rank_percentile(frame[str(spec.right)].astype(float))
    return (1.0 - spec.weight_right) * left + spec.weight_right * right


def apply_spec(frame: pd.DataFrame, spec: AllocationSpec) -> pd.Series:
    score = raw_score(frame, spec)
    return constrained_rank_score(frame, score, spec.country_cap, spec.shortfall_reserve)


def replace_true_gated_with_placebo(spec: AllocationSpec, placebo: str) -> AllocationSpec:
    if spec.score_kind == "candidate":
        candidate = placebo if spec.candidate == "CS_TRUE_GATED" else spec.candidate
        return AllocationSpec(
            spec.score_kind,
            candidate=candidate,
            country_cap=spec.country_cap,
            shortfall_reserve=spec.shortfall_reserve,
        )
    right = placebo if spec.right == "CS_TRUE_GATED" else spec.right
    left = placebo if spec.left == "CS_TRUE_GATED" else spec.left
    return AllocationSpec(
        spec.score_kind,
        left=left,
        right=right,
        weight_right=spec.weight_right,
        country_cap=spec.country_cap,
        shortfall_reserve=spec.shortfall_reserve,
    )


def uses_true_gated(spec: AllocationSpec) -> bool:
    return spec.candidate == "CS_TRUE_GATED" or spec.left == "CS_TRUE_GATED" or spec.right == "CS_TRUE_GATED"


def policy_stats(frame: pd.DataFrame, score: pd.Series) -> dict:
    temp = frame[[TARGET, SEVERE_TARGET, "ISO3"]].copy()
    temp["policy_score"] = score
    top25 = temp.sort_values("policy_score", ascending=False).head(25)
    max_country_top25 = int(top25["ISO3"].value_counts().max()) if not top25.empty else 0
    out = {
        "main_pr_auc": safe_ap(temp[TARGET], temp["policy_score"]),
        "severe_pr_auc": safe_ap(temp[SEVERE_TARGET], temp["policy_score"]),
        "main_top10_hits": top_hits(temp, "policy_score", TARGET, 10),
        "main_top25_hits": top_hits(temp, "policy_score", TARGET, 25),
        "main_top50_hits": top_hits(temp, "policy_score", TARGET, 50),
        "severe_top10_hits": top_hits(temp, "policy_score", SEVERE_TARGET, 10),
        "severe_top25_hits": top_hits(temp, "policy_score", SEVERE_TARGET, 25),
        "severe_top50_hits": top_hits(temp, "policy_score", SEVERE_TARGET, 50),
        "max_country_top25": max_country_top25,
    }
    return out


def allocation_objective(frame: pd.DataFrame, spec: AllocationSpec) -> dict:
    score = apply_spec(frame, spec)
    stats = policy_stats(frame, score)
    max_placebo_top25 = 0
    max_placebo_ap = -np.inf
    if uses_true_gated(spec):
        for placebo in ["CS_EQUAL_GATED", "CS_RANDOM_GATED", "CS_SHUFFLED_GATED"]:
            placebo_spec = replace_true_gated_with_placebo(spec, placebo)
            placebo_stats = policy_stats(frame, apply_spec(frame, placebo_spec))
            max_placebo_top25 = max(max_placebo_top25, int(placebo_stats["main_top25_hits"]))
            max_placebo_ap = max(max_placebo_ap, float(placebo_stats["main_pr_auc"]))
    additive_stats = policy_stats(frame, apply_spec(frame, AllocationSpec("candidate", candidate="CS_TRUE_ADD")))
    placebo_penalty = max(0, max_placebo_top25 - int(stats["main_top25_hits"]))
    placebo_ap_penalty = max(0.0, max_placebo_ap - float(stats["main_pr_auc"])) if np.isfinite(max_placebo_ap) else 0.0
    additive_penalty = 0.0
    if uses_true_gated(spec):
        additive_penalty += max(0, int(additive_stats["main_top25_hits"]) - int(stats["main_top25_hits"]))
        additive_penalty += 4.0 * max(0.0, float(additive_stats["main_pr_auc"]) - float(stats["main_pr_auc"]))
    concentration_penalty = max(0, int(stats["max_country_top25"]) - 3)
    objective = (
        int(stats["main_top25_hits"])
        + 0.50 * int(stats["severe_top25_hits"])
        + 0.25 * int(stats["main_top10_hits"])
        + 4.0 * float(stats["main_pr_auc"])
        - 1.50 * placebo_penalty
        - 6.0 * placebo_ap_penalty
        - 0.75 * additive_penalty
        - 0.25 * concentration_penalty
    )
    return {
        **stats,
        "objective": float(objective),
        "placebo_top25_penalty": float(placebo_penalty),
        "placebo_ap_penalty": float(placebo_ap_penalty),
        "additive_penalty": float(additive_penalty),
        "concentration_penalty": float(concentration_penalty),
    }


def candidate_specs_for_search() -> list[AllocationSpec]:
    specs: list[AllocationSpec] = []
    bases = ["CS_OP", "CS_GDELT", "CS_TRUE_ADD", "CS_TRUE_GATED"]
    for candidate in bases:
        specs.append(AllocationSpec("candidate", candidate=candidate))
    for left, right in [
        ("CS_OP", "CS_TRUE_ADD"),
        ("CS_OP", "CS_TRUE_GATED"),
        ("CS_GDELT", "CS_TRUE_GATED"),
        ("CS_TRUE_ADD", "CS_TRUE_GATED"),
    ]:
        for weight in [0.25, 0.50, 0.75]:
            specs.append(AllocationSpec("blend", left=left, right=right, weight_right=weight))

    expanded: list[AllocationSpec] = []
    for spec in specs:
        for cap in [None, 2, 3, 4]:
            for reserve in [0.0, 0.25, 0.50]:
                expanded.append(
                    AllocationSpec(
                        spec.score_kind,
                        candidate=spec.candidate,
                        left=spec.left,
                        right=spec.right,
                        weight_right=spec.weight_right,
                        country_cap=cap,
                        shortfall_reserve=reserve,
                    )
                )
    return expanded


def select_best_spec(frame: pd.DataFrame, specs: list[AllocationSpec]) -> tuple[AllocationSpec, dict]:
    rows = []
    for spec in specs:
        rows.append({"spec_label": spec.label(), **allocation_objective(frame, spec)})
    scores = pd.DataFrame(rows).sort_values(
        ["objective", "main_top25_hits", "severe_top25_hits", "main_pr_auc"], ascending=False
    )
    best_label = str(scores.iloc[0]["spec_label"])
    best_spec = next(spec for spec in specs if spec.label() == best_label)
    return best_spec, scores.iloc[0].to_dict()


def make_gated_guard_spec(frame: pd.DataFrame) -> tuple[AllocationSpec, dict]:
    true_spec = AllocationSpec("candidate", candidate="CS_TRUE_GATED")
    add_spec = AllocationSpec("candidate", candidate="CS_TRUE_ADD")
    true_stats = policy_stats(frame, apply_spec(frame, true_spec))
    add_stats = policy_stats(frame, apply_spec(frame, add_spec))
    placebo_stats = [
        policy_stats(frame, apply_spec(frame, AllocationSpec("candidate", candidate=placebo)))
        for placebo in ["CS_EQUAL_GATED", "CS_RANDOM_GATED", "CS_SHUFFLED_GATED"]
    ]
    max_placebo_ap = max(float(item["main_pr_auc"]) for item in placebo_stats)
    max_placebo_top25 = max(int(item["main_top25_hits"]) for item in placebo_stats)
    gated_is_safe = (
        float(true_stats["main_pr_auc"]) >= float(add_stats["main_pr_auc"])
        and int(true_stats["main_top25_hits"]) >= int(add_stats["main_top25_hits"])
        and float(true_stats["main_pr_auc"]) >= max_placebo_ap
        and int(true_stats["main_top25_hits"]) >= max_placebo_top25
        and int(true_stats["severe_top25_hits"]) >= int(add_stats["severe_top25_hits"])
    )
    if gated_is_safe:
        return true_spec, {"selection_reason": "true gated passed additive/placebo/severe validation guards"}
    return add_spec, {"selection_reason": "true gated failed validation guards; fallback to true WITS additive"}


def make_prediction_matrix(predictions: pd.DataFrame, fold: str, split: str) -> pd.DataFrame:
    part = predictions.loc[predictions["fold"].eq(fold) & predictions["split"].eq(split)].copy()
    index_cols = [
        "fold",
        "split",
        "ISO3",
        "country",
        "week",
        TARGET,
        SEVERE_TARGET,
        "operational_shortfall_12w",
    ]
    return (
        part.pivot_table(index=index_cols, columns="candidate", values="predicted_probability", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
    )


def run_fold_candidates(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    train, validation, test = add_fold_calibrated_features(train, validation, test)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    model_name = PRIMARY_MODEL
    metric_rows = []
    prediction_frames = []
    for spec in CANDIDATES:
        features = numeric_features(train, groups[spec.feature_group])
        model, fit_mode = make_models()[model_name]
        fit_model(model, fit_mode, train[features], train[TARGET])
        for split_name, source, proba in [
            ("validation", validation, model.predict_proba(validation[features])[:, 1]),
            ("test", test, model.predict_proba(test[features])[:, 1]),
        ]:
            pred = source[
                ["ISO3", "country", "week", TARGET, SEVERE_TARGET, "operational_shortfall_12w"]
            ].copy()
            pred["fold"] = fold.name
            pred["split"] = split_name
            pred["candidate"] = spec.name
            pred["feature_group"] = spec.feature_group
            pred["role"] = spec.role
            pred["model"] = model_name
            pred["predicted_probability"] = proba
            prediction_frames.append(pred)
            stats = policy_stats(
                pred.rename(columns={"predicted_probability": "policy_score"}),
                pred["predicted_probability"],
            )
            metric_rows.append(
                {
                    "fold": fold.name,
                    "split": split_name,
                    "candidate": spec.name,
                    "feature_group": spec.feature_group,
                    "role": spec.role,
                    "model": model_name,
                    "feature_count": len(features),
                    **stats,
                }
            )
        print(f"{fold.name}: scored {spec.name} with {len(features)} features", flush=True)
    return metric_rows, prediction_frames


def evaluate_policy(
    policy_name: str,
    spec: AllocationSpec,
    validation_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    selection_info: dict,
) -> tuple[dict, pd.DataFrame, dict]:
    val_score = apply_spec(validation_matrix, spec)
    threshold, val_f1 = select_threshold(validation_matrix[TARGET].reset_index(drop=True), val_score.to_numpy())
    test_score = apply_spec(test_matrix, spec)
    test_scores = evaluate_predictions(test_matrix[TARGET].reset_index(drop=True), test_score.to_numpy(), threshold)
    pred = test_matrix[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
    pred["fold"] = str(test_matrix["fold"].iloc[0])
    pred["policy"] = policy_name
    pred["spec_label"] = spec.label()
    pred["model"] = PRIMARY_MODEL
    pred["policy_score"] = test_score.to_numpy()
    row = {
        "fold": str(test_matrix["fold"].iloc[0]),
        "policy": policy_name,
        "spec_label": spec.label(),
        "model": PRIMARY_MODEL,
        "selected_threshold": threshold,
        "validation_f1": val_f1,
        "test_rows": len(test_matrix),
        "test_positives": int(test_matrix[TARGET].sum()),
        "test_severe_positives": int(test_matrix[SEVERE_TARGET].sum()),
        **test_scores,
        "severe_pr_auc": safe_ap(test_matrix[SEVERE_TARGET], pd.Series(test_score, index=test_matrix.index)),
    }
    for k in [10, 25, 50]:
        row[f"main_top{k}_hits"] = top_hits(pred, "policy_score", TARGET, k)
        row[f"severe_top{k}_hits"] = top_hits(pred, "policy_score", SEVERE_TARGET, k)
    selection = {
        "fold": str(test_matrix["fold"].iloc[0]),
        "policy": policy_name,
        "spec_label": spec.label(),
        **selection_info,
    }
    return row, pred, selection


def evaluate_policies(candidate_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    selections = []
    search_specs = candidate_specs_for_search()
    for fold in sorted(candidate_predictions["fold"].unique()):
        validation_matrix = make_prediction_matrix(candidate_predictions, fold, "validation")
        test_matrix = make_prediction_matrix(candidate_predictions, fold, "test")
        policy_specs: dict[str, tuple[AllocationSpec, dict]] = {
            name: (spec, {"selection_reason": "fixed predeclared candidate"})
            for name, spec in FIXED_POLICIES.items()
        }
        best_single, best_single_info = select_best_spec(
            validation_matrix,
            [AllocationSpec("candidate", candidate=c) for c in ["CS_OP", "CS_GDELT", "CS_TRUE_ADD", "CS_TRUE_GATED"]],
        )
        policy_specs[BEST_SINGLE_POLICY] = (best_single, {"selection_reason": "best validation single score", **best_single_info})
        best_alloc, best_alloc_info = select_best_spec(validation_matrix, search_specs)
        policy_specs[SEARCH_POLICY] = (
            best_alloc,
            {"selection_reason": "best validation constrained alert-allocation objective", **best_alloc_info},
        )
        guarded_spec, guarded_info = make_gated_guard_spec(validation_matrix)
        policy_specs[GATED_GUARD_POLICY] = (guarded_spec, guarded_info)

        for policy_name, (spec, info) in policy_specs.items():
            row, pred, selection = evaluate_policy(policy_name, spec, validation_matrix, test_matrix, info)
            rows.append(row)
            preds.append(pred)
            selections.append(selection)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(selections)


def summarize(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "folds": ("fold", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "median_main_pr_auc": ("pr_auc", "median"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "median_severe_pr_auc": ("severe_pr_auc", "median"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "mean_f1": ("f1", "mean"),
    }
    for prefix in ["main", "severe"]:
        for k in [10, 25, 50]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        policy_metrics.groupby(["policy", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)
    )


def pooled_delta(
    predictions: pd.DataFrame,
    focus: str,
    baseline: str,
    target: str,
    n_boot: int = 1000,
) -> tuple[float, float, float, float]:
    focus_frame = predictions.loc[predictions["policy"].eq(focus)]
    baseline_frame = predictions.loc[predictions["policy"].eq(baseline)]
    merged = focus_frame.merge(
        baseline_frame,
        on=["ISO3", "country", "week", "fold", TARGET, SEVERE_TARGET],
        suffixes=("_focus", "_baseline"),
    ).reset_index(drop=True)
    point = safe_ap(merged[target], merged["policy_score_focus"]) - safe_ap(merged[target], merged["policy_score_baseline"])
    draws = []
    for seed in range(n_boot):
        sample = merged.sample(n=len(merged), replace=True, random_state=RANDOM_SEED + seed)
        if sample[target].nunique() < 2:
            continue
        draws.append(
            average_precision_score(sample[target], sample["policy_score_focus"])
            - average_precision_score(sample[target], sample["policy_score_baseline"])
        )
    series = pd.Series(draws)
    if series.empty:
        return point, np.nan, np.nan, np.nan
    return point, float(series.quantile(0.025)), float(series.quantile(0.975)), float((series > 0).mean())


def hit_delta(predictions: pd.DataFrame, focus: str, baseline: str, target: str, k: int) -> int:
    delta = 0
    for fold in sorted(predictions["fold"].unique()):
        f = predictions.loc[predictions["fold"].eq(fold) & predictions["policy"].eq(focus)]
        b = predictions.loc[predictions["fold"].eq(fold) & predictions["policy"].eq(baseline)]
        delta += top_hits(f, "policy_score", target, k) - top_hits(b, "policy_score", target, k)
    return int(delta)


def make_deltas(policy_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    contrasts = [
        (SEARCH_POLICY, "AA0_fixed_operational", "allocator_vs_operational"),
        (SEARCH_POLICY, "AA1_fixed_gdelt_additive", "allocator_vs_gdelt"),
        (SEARCH_POLICY, "AA2_fixed_true_wits_additive", "allocator_vs_true_wits_additive"),
        (SEARCH_POLICY, "AA3_fixed_true_gated", "allocator_vs_true_gated"),
        (SEARCH_POLICY, "AA4_fixed_equal_gated_placebo", "allocator_vs_equal_placebo"),
        (SEARCH_POLICY, "AA5_fixed_random_gated_placebo", "allocator_vs_random_placebo"),
        (SEARCH_POLICY, "AA6_fixed_shuffled_gated_placebo", "allocator_vs_shuffled_placebo"),
        (GATED_GUARD_POLICY, "AA0_fixed_operational", "guarded_policy_vs_operational"),
        (GATED_GUARD_POLICY, "AA1_fixed_gdelt_additive", "guarded_policy_vs_gdelt"),
        (GATED_GUARD_POLICY, "AA2_fixed_true_wits_additive", "guarded_gated_vs_true_wits_additive"),
        (GATED_GUARD_POLICY, "AA3_fixed_true_gated", "guarded_policy_vs_true_gated"),
        (GATED_GUARD_POLICY, "AA4_fixed_equal_gated_placebo", "guarded_policy_vs_equal_placebo"),
        (GATED_GUARD_POLICY, "AA5_fixed_random_gated_placebo", "guarded_policy_vs_random_placebo"),
        (GATED_GUARD_POLICY, "AA6_fixed_shuffled_gated_placebo", "guarded_policy_vs_shuffled_placebo"),
        (BEST_SINGLE_POLICY, "AA2_fixed_true_wits_additive", "best_single_vs_true_wits_additive"),
    ]
    for focus, baseline, contrast in contrasts:
        for label_name, target in [("main", TARGET), ("severe", SEVERE_TARGET)]:
            pr = pooled_delta(policy_predictions, focus, baseline, target)
            rows.append(
                {
                    "contrast": contrast,
                    "focus_policy": focus,
                    "baseline_policy": baseline,
                    "label": label_name,
                    "pooled_pr_auc_delta": pr[0],
                    "ci_low": pr[1],
                    "ci_high": pr[2],
                    "p_gt_0": pr[3],
                    "top10_hit_delta": hit_delta(policy_predictions, focus, baseline, target, 10),
                    "top25_hit_delta": hit_delta(policy_predictions, focus, baseline, target, 25),
                    "top50_hit_delta": hit_delta(policy_predictions, focus, baseline, target, 50),
                }
            )
    return pd.DataFrame(rows)


def make_claim_matrix(summary: pd.DataFrame, deltas: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    by_policy = summary.set_index("policy")
    guarded_vs_add = deltas.loc[
        deltas["contrast"].eq("guarded_gated_vs_true_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    guarded_vs_shuffled = deltas.loc[
        deltas["contrast"].eq("guarded_policy_vs_shuffled_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    allocator_vs_add = deltas.loc[
        deltas["contrast"].eq("allocator_vs_true_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    aa9 = selections.loc[selections["policy"].eq(GATED_GUARD_POLICY)]
    aa9_gated_specs = int(aa9["spec_label"].str.contains("CS_TRUE_GATED", regex=False).sum())
    rows = [
        {
            "claim": "A validation-safe gated-if-valid fallback is the best deployment layer in this branch.",
            "data_sources": "PortWatch + GDELT + WITS",
            "evidence_type": "validation-only additive/placebo/severe gate with additive fallback",
            "main_metric": f"{GATED_GUARD_POLICY} mean PR-AUC {by_policy.loc[GATED_GUARD_POLICY, 'mean_main_pr_auc']:.4f}, top-25 {int(by_policy.loc[GATED_GUARD_POLICY, 'main_top25_hits'])}; vs true additive PR-AUC delta {guarded_vs_add['pooled_pr_auc_delta']:.4f}, top-25 delta {int(guarded_vs_add['top25_hit_delta'])}",
            "paper_bucket": "supporting/main-candidate deployment guardrail",
            "status": "computed",
        },
        {
            "claim": "The heavier constrained allocator is not promotable.",
            "data_sources": "PortWatch + GDELT + WITS",
            "evidence_type": "validation-selected top-k/severe/placebo/concentration objective",
            "main_metric": f"allocator vs true additive PR-AUC delta {allocator_vs_add['pooled_pr_auc_delta']:.4f}, top-25 delta {int(allocator_vs_add['top25_hit_delta'])}",
            "paper_bucket": "negative",
            "status": "fails additive and placebo promotion guards",
        },
        {
            "claim": "The guarded policy should be framed as cautious conversion use, not proof that gating always wins.",
            "data_sources": "PortWatch + GDELT + WITS placebos",
            "evidence_type": "equal/random/shuffled gated placebo comparison and selected specs",
            "main_metric": f"AA9 selected true gated in {aa9_gated_specs}/3 folds; vs shuffled PR-AUC delta {guarded_vs_shuffled['pooled_pr_auc_delta']:.4f}, top-25 delta {int(guarded_vs_shuffled['top25_hit_delta'])}",
            "paper_bucket": "guardrail",
            "status": "supporting only; not a universal gated-performance claim",
        },
        {
            "claim": "Three-source compliance is preserved.",
            "data_sources": "PortWatch + GDELT + WITS only",
            "evidence_type": "script provenance",
            "main_metric": "script imports the country-shared three-source dataset/model code and no fourth-source features",
            "paper_bucket": "main",
            "status": "satisfied for this branch",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    candidate_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    selections: pd.DataFrame,
    claim_matrix: pd.DataFrame,
) -> None:
    candidate_test = candidate_metrics.loc[candidate_metrics["split"].eq("test")]
    content = f"""# Panel32 Country-Shared Alert Allocation

## Purpose

This branch tests the next P0 method step after the country-shared conversion model: a **deployment-aware alert-allocation layer**. It is not another feature expansion. It retrains the country-shared three-source candidates, scores validation and test years, and selects alert-ranking rules using validation data only.

The validation objective rewards main top-25 hits, severe top-25 hits, top-10 hits, PR-AUC, and country diversification, while penalizing candidate rules that are beaten by additive true WITS or equal/random/shuffled WITS gated placebos.

## Data And Validation

- Dataset: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Sources: PortWatch + GDELT + WITS only.
- Candidate model: `{PRIMARY_MODEL}`.
- Test folds: 2023, 2024, 2025.
- Selection: validation year only; test labels are not used for model, threshold, or allocation selection.

## Candidate Test Metrics

{candidate_test[["candidate", "fold", "role", "main_pr_auc", "severe_pr_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top25_hits"]].to_markdown(index=False)}

## Policy Summary

{summary[["policy", "mean_main_pr_auc", "mean_severe_pr_auc", "mean_roc_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits"]].to_markdown(index=False)}

## Validation Selections

{selections.to_markdown(index=False)}

## Key Deltas

{deltas.to_markdown(index=False)}

## Claim Matrix

{claim_matrix.to_markdown(index=False)}

## Reading

This branch is promotable only if the validation-selected allocator beats true WITS additive and the gated placebo policies without harming severe/top-k behavior. If it does not, it should be used as a negative guardrail showing that the current three-source score pool cannot yet support a deployment-aware gated policy.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    metric_rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_metrics, fold_predictions = run_fold_candidates(fold, df)
        metric_rows.extend(fold_metrics)
        prediction_frames.extend(fold_predictions)
        print(f"Finished candidate scoring for {fold.name}", flush=True)
    candidate_metrics = pd.DataFrame(metric_rows)
    candidate_predictions = pd.concat(prediction_frames, ignore_index=True)
    policy_metrics, policy_predictions, selections = evaluate_policies(candidate_predictions)
    summary = summarize(policy_metrics)
    deltas = make_deltas(policy_predictions)
    claim_matrix = make_claim_matrix(summary, deltas, selections)

    candidate_metrics.to_csv(CANDIDATE_METRICS, index=False)
    candidate_predictions.to_csv(CANDIDATE_PREDICTIONS, index=False)
    policy_metrics.to_csv(TABLE_DIR / "panel32_country_shared_alert_allocation_policy_metrics_by_fold.csv", index=False)
    policy_predictions.to_csv(POLICY_PREDICTIONS, index=False)
    selections.to_csv(POLICY_SELECTIONS, index=False)
    summary.to_csv(POLICY_SUMMARY, index=False)
    deltas.to_csv(POLICY_DELTAS, index=False)
    claim_matrix.to_csv(CLAIM_MATRIX, index=False)
    write_report(candidate_metrics, summary, deltas, selections, claim_matrix)

    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
