from __future__ import annotations

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

import scripts.run_panel32_locked_transfer_router as locked  # noqa: E402
import scripts.run_panel32_port_system_benchmark as psb  # noqa: E402
from scripts.run_panel32_portwatch_chokepoint_benchmark import fit_model, make_models  # noqa: E402
from scripts.run_panel_benchmark_models import TARGET, add_country_dummies, evaluate_predictions, select_threshold  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_port_system_loco_transfer.md"
RESULTS = TABLE_DIR / "panel32_port_system_loco_transfer_results.csv"
PREDICTIONS = TABLE_DIR / "panel32_port_system_loco_transfer_predictions.csv"
SUMMARY = TABLE_DIR / "panel32_port_system_loco_transfer_summary.csv"
SELECTIONS = TABLE_DIR / "panel32_port_system_loco_transfer_selections.csv"
DELTAS = TABLE_DIR / "panel32_port_system_loco_transfer_deltas.csv"
FAILURE_COUNTRIES = TABLE_DIR / "panel32_port_system_loco_failure_country_deltas.csv"

SEVERE_TARGET = psb.SEVERE_TARGET
RANDOM_SEED = 42
PRIORITY_FAILURE_ISO3 = ["POL", "EGY", "ITA", "GBR", "VNM", "CHN", "PHL"]


def split_loco(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return locked.split_loco(df, holdout)


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True, method="average").to_numpy()


def rank_blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    return left_weight * rank01(left) + (1.0 - left_weight) * rank01(right)


def top_hits(frame: pd.DataFrame, score_col: str, target_col: str, k: int) -> int:
    return int(frame.sort_values(score_col, ascending=False).head(k)[target_col].sum())


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def fit_predict_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_kind: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = usable_features(train, features)
    if model_kind == "gb_locked":
        model = locked.fit_locked_gb(train[features], train[TARGET])
    elif model_kind == "xgboost":
        model, fit_mode = make_models(train[TARGET])["advanced::xgboost"]
        fit_model(model, fit_mode, train[features], train[TARGET])
    else:
        raise ValueError(model_kind)
    return model.predict_proba(validation[features])[:, 1], model.predict_proba(test[features])[:, 1], len(features)


def build_candidate_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    country_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    locked_groups = locked.make_feature_groups(country_features)
    ps_groups = psb.make_feature_groups(train, country_features)
    candidate_specs = {
        "OP_GB": (locked_groups["OP"], "gb_locked"),
        "FULL_GB": (locked_groups["FULL"], "gb_locked"),
        "HAZ_GB": (locked_groups["HAZ"], "gb_locked"),
        "PS3_FULL_PORT_GB": (ps_groups["P32PS3_full_event_port_system"], "gb_locked"),
        "PS4_HAZ_PORT_GB": (ps_groups["P32PS4_full_event_hazard_port_system"], "gb_locked"),
        "PS6_PHYS_PORT_GB": (ps_groups["P32PS6_full_physical_port_system"], "gb_locked"),
    }
    if "advanced::xgboost" in make_models(train[TARGET]):
        candidate_specs.update(
            {
                "PS0_FULL_XGB": (ps_groups["P32PS0_full_event_network"], "xgboost"),
                "PS4_HAZ_PORT_XGB": (ps_groups["P32PS4_full_event_hazard_port_system"], "xgboost"),
            }
        )

    val_scores = validation[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
    test_scores = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
    feature_counts: dict[str, int] = {}
    for candidate, (features, model_kind) in candidate_specs.items():
        val_proba, test_proba, feature_count = fit_predict_candidate(train, validation, test, features, model_kind)
        val_scores[candidate] = val_proba
        test_scores[candidate] = test_proba
        feature_counts[candidate] = feature_count

    val_scores["BLEND_70HAZ30OP_GB"] = rank_blend(val_scores["HAZ_GB"].to_numpy(), val_scores["OP_GB"].to_numpy(), 0.70)
    test_scores["BLEND_70HAZ30OP_GB"] = rank_blend(
        test_scores["HAZ_GB"].to_numpy(), test_scores["OP_GB"].to_numpy(), 0.70
    )
    val_scores["BLEND_70PS4GB30OP_GB"] = rank_blend(
        val_scores["PS4_HAZ_PORT_GB"].to_numpy(), val_scores["OP_GB"].to_numpy(), 0.70
    )
    test_scores["BLEND_70PS4GB30OP_GB"] = rank_blend(
        test_scores["PS4_HAZ_PORT_GB"].to_numpy(), test_scores["OP_GB"].to_numpy(), 0.70
    )
    val_scores["BLEND_50PS4GB50HAZ_GB"] = rank_blend(
        val_scores["PS4_HAZ_PORT_GB"].to_numpy(), val_scores["HAZ_GB"].to_numpy(), 0.50
    )
    test_scores["BLEND_50PS4GB50HAZ_GB"] = rank_blend(
        test_scores["PS4_HAZ_PORT_GB"].to_numpy(), test_scores["HAZ_GB"].to_numpy(), 0.50
    )
    if "PS4_HAZ_PORT_XGB" in val_scores:
        val_scores["BLEND_70PS4XGB30OP_GB"] = rank_blend(
            val_scores["PS4_HAZ_PORT_XGB"].to_numpy(), val_scores["OP_GB"].to_numpy(), 0.70
        )
        test_scores["BLEND_70PS4XGB30OP_GB"] = rank_blend(
            test_scores["PS4_HAZ_PORT_XGB"].to_numpy(), test_scores["OP_GB"].to_numpy(), 0.70
        )
    for blend in [col for col in val_scores.columns if col.startswith("BLEND_")]:
        feature_counts[blend] = 0
    return val_scores, test_scores, feature_counts


def candidate_metrics(validation: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for score in score_cols:
        rows.append(
            {
                "candidate": score,
                "main_pr_auc": average_precision_score(validation[TARGET], validation[score]),
                "severe_pr_auc": average_precision_score(validation[SEVERE_TARGET], validation[score]),
                "main_top10": top_hits(validation, score, TARGET, 10),
                "main_top25": top_hits(validation, score, TARGET, 25),
                "severe_top10": top_hits(validation, score, SEVERE_TARGET, 10),
                "severe_top25": top_hits(validation, score, SEVERE_TARGET, 25),
            }
        )
    return pd.DataFrame(rows)


def choose_candidate(metrics: pd.DataFrame, objective: str) -> str:
    if objective == "main_pr_auc":
        return metrics.sort_values(["main_pr_auc", "main_top25"], ascending=False).iloc[0]["candidate"]
    if objective == "main_top25":
        return metrics.sort_values(["main_top25", "main_pr_auc"], ascending=False).iloc[0]["candidate"]
    if objective == "severe_pr_auc":
        return metrics.sort_values(["severe_pr_auc", "severe_top25", "main_pr_auc"], ascending=False).iloc[0][
            "candidate"
        ]
    if objective == "severe_top25":
        return metrics.sort_values(["severe_top25", "severe_pr_auc", "main_pr_auc"], ascending=False).iloc[0][
            "candidate"
        ]
    if objective == "rank_sum":
        frame = metrics.copy()
        frame["rank_sum"] = (
            frame["main_pr_auc"].rank(ascending=False, method="min")
            + frame["main_top25"].rank(ascending=False, method="min")
            + frame["severe_pr_auc"].rank(ascending=False, method="min")
            + frame["severe_top25"].rank(ascending=False, method="min")
        )
        return frame.sort_values(["rank_sum", "main_pr_auc"], ascending=[True, False]).iloc[0]["candidate"]
    raise ValueError(objective)


def policy_map(metrics: pd.DataFrame, score_cols: list[str]) -> dict[str, str]:
    fixed = {
        "PSL0_op_gb": "OP_GB",
        "PSL1_full_gb": "FULL_GB",
        "PSL2_haz_gb": "HAZ_GB",
        "PSL3_70haz30op_gb": "BLEND_70HAZ30OP_GB",
        "PSL4_ps4_gb": "PS4_HAZ_PORT_GB",
        "PSL5_70ps4gb30op": "BLEND_70PS4GB30OP_GB",
        "PSL6_50ps4gb50haz": "BLEND_50PS4GB50HAZ_GB",
    }
    if "PS4_HAZ_PORT_XGB" in score_cols:
        fixed.update(
            {
                "PSL7_ps4_xgb": "PS4_HAZ_PORT_XGB",
                "PSL8_ps0_xgb": "PS0_FULL_XGB",
                "PSL9_70ps4xgb30op": "BLEND_70PS4XGB30OP_GB",
            }
        )
    selected = {
        "PSLSEL_main_pr_auc": choose_candidate(metrics, "main_pr_auc"),
        "PSLSEL_main_top25": choose_candidate(metrics, "main_top25"),
        "PSLSEL_severe_pr_auc": choose_candidate(metrics, "severe_pr_auc"),
        "PSLSEL_severe_top25": choose_candidate(metrics, "severe_top25"),
        "PSLSEL_rank_sum": choose_candidate(metrics, "rank_sum"),
    }
    return {**fixed, **selected}


def score_holdout(df: pd.DataFrame, holdout: str) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_loco(df, holdout)
    if test.empty or validation.empty or train[TARGET].sum() < 5 or validation[TARGET].sum() < 5 or test[TARGET].sum() < 1:
        return [], []
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    val_scores, test_scores, feature_counts = build_candidate_scores(train, validation, test, country_features)
    score_cols = [col for col in val_scores.columns if col not in {"ISO3", "country", "week", TARGET, SEVERE_TARGET}]
    metrics = candidate_metrics(val_scores, score_cols)
    policies = policy_map(metrics, score_cols)

    rows = []
    predictions = []
    for policy, candidate in policies.items():
        threshold, val_f1 = select_threshold(val_scores[TARGET], val_scores[candidate].to_numpy())
        main_scores = evaluate_predictions(test_scores[TARGET].to_numpy(), test_scores[candidate].to_numpy(), threshold)
        pred = test_scores[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
        pred["holdout_iso3"] = holdout
        pred["policy"] = policy
        pred["selected_candidate"] = candidate
        pred["predicted_probability"] = test_scores[candidate].to_numpy()
        row = {
            "holdout_iso3": holdout,
            "holdout_country": test_scores["country"].iloc[0],
            "policy": policy,
            "selected_candidate": candidate,
            "feature_count": feature_counts.get(candidate, 0),
            "validation_candidate_main_pr_auc": float(metrics.loc[metrics["candidate"].eq(candidate), "main_pr_auc"].iloc[0]),
            "validation_candidate_severe_pr_auc": float(
                metrics.loc[metrics["candidate"].eq(candidate), "severe_pr_auc"].iloc[0]
            ),
            "validation_candidate_main_top25": int(metrics.loc[metrics["candidate"].eq(candidate), "main_top25"].iloc[0]),
            "validation_candidate_severe_top25": int(
                metrics.loc[metrics["candidate"].eq(candidate), "severe_top25"].iloc[0]
            ),
            "validation_f1": val_f1,
            "selected_threshold": threshold,
            "test_rows": len(test_scores),
            "test_positives": int(test_scores[TARGET].sum()),
            "test_severe_positives": int(test_scores[SEVERE_TARGET].sum()),
            **main_scores,
            "severe_pr_auc": average_precision_score(test_scores[SEVERE_TARGET], test_scores[candidate]),
        }
        for k in [5, 10, 25]:
            row[f"main_top{k}_hits"] = top_hits(pred, "predicted_probability", TARGET, k)
            row[f"severe_top{k}_hits"] = top_hits(pred, "predicted_probability", SEVERE_TARGET, k)
        rows.append(row)
        predictions.append(pred)
    return rows, predictions


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "holdouts": ("holdout_iso3", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "median_main_pr_auc": ("pr_auc", "median"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "median_severe_pr_auc": ("severe_pr_auc", "median"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "total_tp": ("tp", "sum"),
        "total_fp": ("fp", "sum"),
        "total_fn": ("fn", "sum"),
    }
    for prefix in ["main", "severe"]:
        for k in [5, 10, 25]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        results.groupby("policy", as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)
    )


def pooled_delta(
    predictions: pd.DataFrame, focus: str, baseline: str, target: str, n_boot: int = 2500
) -> tuple[float, float, float, float]:
    focus_frame = predictions.loc[predictions["policy"].eq(focus)]
    baseline_frame = predictions.loc[predictions["policy"].eq(baseline)]
    merged = focus_frame.merge(
        baseline_frame,
        on=["ISO3", "country", "week", "holdout_iso3", TARGET, SEVERE_TARGET],
        suffixes=("_focus", "_baseline"),
    ).reset_index(drop=True)
    point = average_precision_score(merged[target], merged["predicted_probability_focus"]) - average_precision_score(
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


def hit_delta(predictions: pd.DataFrame, focus: str, baseline: str, target: str, k: int) -> int:
    delta = 0
    subset = predictions.loc[predictions["policy"].isin([focus, baseline])]
    for _, frame in subset.groupby("holdout_iso3"):
        focus_top = frame.loc[frame["policy"].eq(focus)].sort_values("predicted_probability", ascending=False).head(k)
        base_top = frame.loc[frame["policy"].eq(baseline)].sort_values("predicted_probability", ascending=False).head(k)
        delta += int(focus_top[target].sum()) - int(base_top[target].sum())
    return delta


def make_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    contrasts = [
        ("PSL4_ps4_gb", "PSL2_haz_gb"),
        ("PSL4_ps4_gb", "PSL3_70haz30op_gb"),
        ("PSL4_ps4_gb", "PSL0_op_gb"),
        ("PSL5_70ps4gb30op", "PSL3_70haz30op_gb"),
        ("PSL6_50ps4gb50haz", "PSL2_haz_gb"),
        ("PSLSEL_main_pr_auc", "PSL2_haz_gb"),
        ("PSLSEL_rank_sum", "PSL3_70haz30op_gb"),
    ]
    if predictions["policy"].eq("PSL7_ps4_xgb").any():
        contrasts.extend(
            [
                ("PSL7_ps4_xgb", "PSL8_ps0_xgb"),
                ("PSL7_ps4_xgb", "PSL4_ps4_gb"),
                ("PSL9_70ps4xgb30op", "PSL3_70haz30op_gb"),
            ]
        )
    rows = []
    for focus, baseline in contrasts:
        if not predictions["policy"].eq(focus).any() or not predictions["policy"].eq(baseline).any():
            continue
        main = pooled_delta(predictions, focus, baseline, TARGET)
        severe = pooled_delta(predictions, focus, baseline, SEVERE_TARGET)
        rows.append(
            {
                "focus_policy": focus,
                "baseline_policy": baseline,
                "pooled_main_pr_auc_delta": main[0],
                "main_ci_low": main[1],
                "main_ci_high": main[2],
                "main_p_gt_0": main[3],
                "pooled_severe_pr_auc_delta": severe[0],
                "severe_ci_low": severe[1],
                "severe_ci_high": severe[2],
                "severe_p_gt_0": severe[3],
                "main_top10_hit_delta": hit_delta(predictions, focus, baseline, TARGET, 10),
                "main_top25_hit_delta": hit_delta(predictions, focus, baseline, TARGET, 25),
                "severe_top10_hit_delta": hit_delta(predictions, focus, baseline, SEVERE_TARGET, 10),
                "severe_top25_hit_delta": hit_delta(predictions, focus, baseline, SEVERE_TARGET, 25),
            }
        )
    return pd.DataFrame(rows)


def make_failure_country_deltas(results: pd.DataFrame) -> pd.DataFrame:
    key_policies = [
        "PSL0_op_gb",
        "PSL2_haz_gb",
        "PSL3_70haz30op_gb",
        "PSL4_ps4_gb",
        "PSL5_70ps4gb30op",
        "PSL7_ps4_xgb",
        "PSL9_70ps4xgb30op",
    ]
    frame = results.loc[results["holdout_iso3"].isin(PRIORITY_FAILURE_ISO3) & results["policy"].isin(key_policies)].copy()
    if frame.empty:
        return pd.DataFrame()
    metrics = ["pr_auc", "severe_pr_auc", "main_top10_hits", "main_top25_hits", "severe_top10_hits", "severe_top25_hits"]
    base_cols = ["holdout_iso3", "holdout_country", "test_positives", "test_severe_positives"]
    wide = None
    for policy in key_policies:
        part = frame.loc[frame["policy"].eq(policy), base_cols + metrics].copy()
        if part.empty:
            continue
        part = part.rename(columns={metric: f"{policy}_{metric}" for metric in metrics})
        wide = part if wide is None else wide.merge(part, on=base_cols, how="outer")
    if wide is None:
        return pd.DataFrame()
    for policy in key_policies:
        if f"{policy}_pr_auc" not in wide.columns:
            continue
        for baseline in ["PSL0_op_gb", "PSL2_haz_gb", "PSL3_70haz30op_gb"]:
            if policy == baseline or f"{baseline}_pr_auc" not in wide.columns:
                continue
            wide[f"{policy}_vs_{baseline}_pr_auc_delta"] = wide[f"{policy}_pr_auc"] - wide[f"{baseline}_pr_auc"]
            wide[f"{policy}_vs_{baseline}_top25_delta"] = (
                wide[f"{policy}_main_top25_hits"] - wide[f"{baseline}_main_top25_hits"]
            )
    return wide.sort_values("holdout_iso3")


def write_report(summary: pd.DataFrame, selections: pd.DataFrame, deltas: pd.DataFrame, failure: pd.DataFrame) -> None:
    selection_counts = selections.groupby(["policy", "selected_candidate"], as_index=False).size()
    failure_preview = failure if not failure.empty else pd.DataFrame()
    content = f"""# Panel32 Port-System LOCO Transfer

## Purpose

This experiment tests whether the expanded32 PortWatch selected-port system features help fully unseen-country transfer. Each country is held out completely; models train on other countries in 2021-2023, select thresholds or policies on other countries in 2024, and test on the held-out country in 2025.

## Candidate Family

- GB locked references: operational, full-event/network, full-event+hazard, and 70haz/30op rank blend.
- GB port-system candidates: full-event+port-system, full-event+hazard+port-system, physical/chokepoint+hazard+port-system, and fixed port-system blends.
- Optional XGBoost stress candidate: full-event+hazard+port-system and same-model full-event/network reference.

## Policy Summary

{summary.to_markdown(index=False)}

## Pooled PR-AUC And Top-K Deltas

{deltas.to_markdown(index=False)}

## Selection Counts

{selection_counts.to_markdown(index=False)}

## Priority Failure Countries

{failure_preview.to_markdown(index=False)}

## Reading

Promote port-system as a transfer component only if it improves over the locked hazard/router references on transfer PR-AUC or repairs top-k in priority failure countries without worsening severe behavior. If gains stay confined to temporal top-k, keep port-system as direct operational mechanism data and continue to waiting-time, AIS queue, berthing/service duration, closure/labor, route-duration, or schedule/blank-sailing sources.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = psb.load_or_build_dataset()
    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    if RESULTS.exists() and PREDICTIONS.exists():
        existing_results = pd.read_csv(RESULTS)
        existing_predictions = pd.read_csv(PREDICTIONS, parse_dates=["week"])
        rows.extend(existing_results.to_dict("records"))
        prediction_frames.append(existing_predictions)
        completed = set(existing_results["holdout_iso3"].dropna().unique())
        print(f"Resuming from checkpoint with {len(completed)} completed holdouts", flush=True)
    else:
        completed = set()
    for holdout in sorted(df["ISO3"].unique()):
        if holdout in completed:
            continue
        holdout_rows, holdout_predictions = score_holdout(df, holdout)
        rows.extend(holdout_rows)
        prediction_frames.extend(holdout_predictions)
        if rows and prediction_frames:
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS, index=False)
        print(f"Finished {holdout}: {len(holdout_rows)} policy rows", flush=True)

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = results[
        [
            "holdout_iso3",
            "policy",
            "selected_candidate",
            "validation_candidate_main_pr_auc",
            "validation_candidate_severe_pr_auc",
            "validation_candidate_main_top25",
            "validation_candidate_severe_top25",
        ]
    ].copy()
    summary = summarize(results)
    deltas = make_deltas(predictions)
    failure = make_failure_country_deltas(results)
    results.to_csv(RESULTS, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    selections.to_csv(SELECTIONS, index=False)
    summary.to_csv(SUMMARY, index=False)
    deltas.to_csv(DELTAS, index=False)
    failure.to_csv(FAILURE_COUNTRIES, index=False)
    write_report(summary, selections, deltas, failure)
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved failure country deltas: {FAILURE_COUNTRIES}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
