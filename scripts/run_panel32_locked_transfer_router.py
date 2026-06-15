from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_panel32_portwatch_chokepoint_benchmark as pwc  # noqa: E402
from scripts.run_panel32_hazard_memory_benchmark import HAZARD_FEATURES  # noqa: E402
from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    INTERACTION_FEATURES,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_NETWORK_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_locked_transfer_router.md"
RESULTS = TABLE_DIR / "panel32_locked_transfer_router_results.csv"
PREDICTIONS = TABLE_DIR / "panel32_locked_transfer_router_predictions.csv"
SUMMARY = TABLE_DIR / "panel32_locked_transfer_router_summary.csv"
SELECTIONS = TABLE_DIR / "panel32_locked_transfer_router_selections.csv"
DELTAS = TABLE_DIR / "panel32_locked_transfer_router_deltas.csv"

SEVERE_TARGET = pwc.SEVERE_TARGET
RANDOM_SEED = 42


def unique(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def split_loco(df: pd.DataFrame, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2021-01-01") & (df["week"] < "2024-01-01")].copy()
    validation = df[(df["ISO3"].ne(holdout)) & (df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["ISO3"].eq(holdout)) & (df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, validation, test


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    full_event = (
        OWN_NEWS_FEATURES
        + EXTERNAL_UNWEIGHTED_FEATURES
        + TOTAL_NETWORK_FEATURES
        + ME_NETWORK_FEATURES
        + INTERACTION_FEATURES
    )
    return {
        "OP": unique(base),
        "FULL": unique(base + full_event),
        "HAZ": unique(base + full_event + HAZARD_FEATURES),
    }


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def fit_locked_gb(train_x: pd.DataFrame, train_y: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=12,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )
    weights = compute_sample_weight(class_weight="balanced", y=train_y)
    model.fit(train_x, train_y, sample_weight=weights)
    return model


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True, method="average").to_numpy()


def rank_blend(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    return left_weight * rank01(left) + (1.0 - left_weight) * rank01(right)


def top_hits(frame: pd.DataFrame, score_col: str, target_col: str, k: int) -> int:
    return int(frame.sort_values(score_col, ascending=False).head(k)[target_col].sum())


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


def score_holdout(df: pd.DataFrame, holdout: str) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    train, validation, test = split_loco(df, holdout)
    if test.empty or validation.empty or train[TARGET].sum() < 5 or validation[TARGET].sum() < 5 or test[TARGET].sum() < 1:
        return [], [], []
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(country_features)
    val_scores = validation[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
    test_scores = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()

    for candidate, group_name in [("OP", "OP"), ("FULL", "FULL"), ("HAZ", "HAZ")]:
        features = usable_features(train, groups[group_name])
        model = fit_locked_gb(train[features], train[TARGET])
        val_scores[candidate] = model.predict_proba(validation[features])[:, 1]
        test_scores[candidate] = model.predict_proba(test[features])[:, 1]

    for name, weight in [("BLEND_80haz20op", 0.80), ("BLEND_70haz30op", 0.70), ("BLEND_60haz40op", 0.60)]:
        val_scores[name] = rank_blend(val_scores["HAZ"].to_numpy(), val_scores["OP"].to_numpy(), weight)
        test_scores[name] = rank_blend(test_scores["HAZ"].to_numpy(), test_scores["OP"].to_numpy(), weight)

    score_cols = ["OP", "FULL", "HAZ", "BLEND_80haz20op", "BLEND_70haz30op", "BLEND_60haz40op"]
    metrics = candidate_metrics(val_scores, score_cols)
    fixed_policies = {
        "LOCK0_op": "OP",
        "LOCK1_full": "FULL",
        "LOCK2_haz": "HAZ",
        "LOCK3_80haz20op": "BLEND_80haz20op",
        "LOCK4_70haz30op": "BLEND_70haz30op",
        "LOCK5_60haz40op": "BLEND_60haz40op",
    }
    selected_policies = {
        "LOCKSEL_main_pr_auc": choose_candidate(metrics, "main_pr_auc"),
        "LOCKSEL_main_top25": choose_candidate(metrics, "main_top25"),
        "LOCKSEL_severe_pr_auc": choose_candidate(metrics, "severe_pr_auc"),
        "LOCKSEL_severe_top25": choose_candidate(metrics, "severe_top25"),
        "LOCKSEL_rank_sum": choose_candidate(metrics, "rank_sum"),
    }
    policies = {**fixed_policies, **selected_policies}

    rows = []
    predictions = []
    selections = []
    for policy, candidate in policies.items():
        threshold, val_f1 = select_threshold(val_scores[TARGET], val_scores[candidate].to_numpy())
        main_scores = evaluate_predictions(test_scores[TARGET].to_numpy(), test_scores[candidate].to_numpy(), threshold)
        row = {
            "holdout_iso3": holdout,
            "holdout_country": test_scores["country"].iloc[0],
            "policy": policy,
            "selected_candidate": candidate,
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
        pred = test_scores[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
        pred["holdout_iso3"] = holdout
        pred["policy"] = policy
        pred["selected_candidate"] = candidate
        pred["predicted_probability"] = test_scores[candidate].to_numpy()
        for k in [5, 10, 25]:
            row[f"main_top{k}_hits"] = top_hits(pred, "predicted_probability", TARGET, k)
            row[f"severe_top{k}_hits"] = top_hits(pred, "predicted_probability", SEVERE_TARGET, k)
        rows.append(row)
        predictions.append(pred)
        selected = metrics.loc[metrics["candidate"].eq(candidate)].iloc[0].to_dict()
        selected.update({"holdout_iso3": holdout, "policy": policy, "selected_candidate": candidate})
        selections.append(selected)
    return rows, predictions, selections


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


def pooled_delta(predictions: pd.DataFrame, focus: str, baseline: str, target: str, n_boot: int = 2500) -> tuple[float, float, float, float]:
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
        sample = merged.sample(n=len(merged), replace=True, random_state=42 + seed)
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
        ("LOCK2_haz", "LOCK1_full"),
        ("LOCK2_haz", "LOCK0_op"),
        ("LOCK3_80haz20op", "LOCK2_haz"),
        ("LOCK4_70haz30op", "LOCK2_haz"),
        ("LOCK4_70haz30op", "LOCK0_op"),
        ("LOCKSEL_main_pr_auc", "LOCK2_haz"),
        ("LOCKSEL_rank_sum", "LOCK2_haz"),
    ]
    rows = []
    for focus, baseline in contrasts:
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


def write_report(summary: pd.DataFrame, selections: pd.DataFrame, deltas: pd.DataFrame) -> None:
    content = f"""# Panel32 Locked Transfer Router

## Purpose

This rerun turns the previous post-hoc transfer-router diagnostic into a locked expanded32 leave-one-country-out experiment. For each held-out country, models train on other countries in 2021-2023, validate on other countries in 2024, and test on the held-out country in 2025. Blend candidates are predeclared hazard/operational rank blends; validation-selected policies use validation data only.

## Policy Summary

{summary.to_markdown(index=False)}

## Pooled PR-AUC And Top-K Deltas

{deltas.to_markdown(index=False)}

## Selection Counts

{selections.groupby(["policy", "selected_candidate"], as_index=False).size().to_markdown(index=False)}

## Reading

A credible transfer router should either preserve the hazard PR-AUC lead while improving top-k, or clearly separate objective-specific deployment modes. This is a locked rerun over the predeclared hazard/operational candidate family; it still needs final paper framing against the broader panel14 and expanded32 evidence.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pwc.load_or_build_dataset()
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
        holdout_rows, holdout_predictions, holdout_selections = score_holdout(df, holdout)
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
    results.to_csv(RESULTS, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    selections.to_csv(SELECTIONS, index=False)
    summary.to_csv(SUMMARY, index=False)
    deltas.to_csv(DELTAS, index=False)
    write_report(summary, selections, deltas)
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
