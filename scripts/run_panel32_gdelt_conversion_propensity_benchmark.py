from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)
from scripts.run_panel32_network_gated_country_shared_conversion import (  # noqa: E402
    KIND_ORDER,
    NETWORK_BY_KIND,
    PRIMARY_MODEL,
    RANDOM_SEED,
    SEVERE_TARGET,
    add_fold_calibrated_features,
    available,
    calibration_cols,
    event_state_cols,
    fit_model,
    gated_cols,
    load_dataset,
    make_models,
    network_state_cols,
    numeric_features,
    safe_ap,
    top_hits,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_gdelt_conversion_propensity_benchmark.md"
METRICS = TABLE_DIR / "panel32_gdelt_conversion_propensity_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_gdelt_conversion_propensity_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_gdelt_conversion_propensity_predictions.csv"
DELTAS = TABLE_DIR / "panel32_gdelt_conversion_propensity_deltas.csv"
MATRIX = TABLE_DIR / "panel32_gdelt_conversion_propensity_claim_matrix.csv"

ALPHA = 60.0


def quantile_edges(values: pd.Series, n_bins: int = 3) -> np.ndarray | None:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.nunique() < 2:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(clean, quantiles))
    if len(edges) < 2:
        return None
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def add_bin_from_edges(frames: list[pd.DataFrame], col: str, out_col: str, edges: np.ndarray | None) -> None:
    for frame in frames:
        if edges is None or col not in frame.columns:
            frame[out_col] = 0
        else:
            frame[out_col] = (
                pd.cut(pd.to_numeric(frame[col], errors="coerce"), bins=edges, labels=False, include_lowest=True)
                .astype("float")
                .fillna(0)
                .astype(int)
            )


def add_smoothed_rate(
    train: pd.DataFrame,
    frames: list[pd.DataFrame],
    keys: list[str],
    out_col: str,
    alpha: float = ALPHA,
) -> None:
    global_rate = float(train[TARGET].mean())
    stats = train.groupby(keys, dropna=False, sort=False)[TARGET].agg(["sum", "count"]).reset_index()
    stats[out_col] = (stats["sum"] + alpha * global_rate) / (stats["count"] + alpha)
    rate_table = stats[keys + [out_col]]
    for frame in frames:
        mapped = frame[keys].merge(rate_table, how="left", on=keys)[out_col]
        frame[out_col] = mapped.fillna(global_rate).astype(float).to_numpy()
        frame[f"{out_col}_lift"] = frame[out_col] - global_rate


def add_fold_conversion_propensity_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [train.copy(), validation.copy(), test.copy()]
    train = frames[0]

    score_cols = ["cs_vulnerability_score", "cs_event_pressure_score"]
    for kind in KIND_ORDER:
        score_cols.append(f"cs_{kind}_network_pressure_score")

    for col in score_cols:
        add_bin_from_edges(frames, col, f"gcl_{col}_bin", quantile_edges(train[col], 3))

    for frame in frames:
        frame["gcl_shortfall_active"] = (frame["cs_vuln_shortfall"].fillna(0.0) > 0).astype(int)
        frame["gcl_event_tail"] = frame["cs_high_event_pressure"].astype(int)
        frame["gcl_vuln_tail"] = frame["cs_high_vulnerability"].astype(int)

    add_smoothed_rate(
        train,
        frames,
        ["gcl_cs_event_pressure_score_bin", "gcl_cs_vulnerability_score_bin"],
        "gcl_gdelt_event_vuln_rate",
    )
    add_smoothed_rate(
        train,
        frames,
        ["gcl_event_tail", "gcl_vuln_tail", "gcl_shortfall_active"],
        "gcl_gdelt_tail_state_rate",
    )
    add_smoothed_rate(
        train,
        frames,
        ["gcl_cs_event_pressure_score_bin", "gcl_shortfall_active"],
        "gcl_gdelt_event_shortfall_rate",
    )

    for frame in frames:
        rate_cols = [
            "gcl_gdelt_event_vuln_rate",
            "gcl_gdelt_tail_state_rate",
            "gcl_gdelt_event_shortfall_rate",
        ]
        lift_cols = [f"{col}_lift" for col in rate_cols]
        frame["gcl_gdelt_conversion_score"] = frame[rate_cols].mean(axis=1)
        frame["gcl_gdelt_conversion_lift"] = frame[lift_cols].mean(axis=1)
        frame["gcl_gdelt_event_x_conversion"] = (
            frame["cs_event_pressure_score"] * frame["gcl_gdelt_conversion_lift"]
        )
        frame["gcl_gdelt_vuln_event_x_conversion"] = (
            frame["cs_vulnerability_score"]
            * frame["cs_event_pressure_score"]
            * frame["gcl_gdelt_conversion_lift"]
        )

    for kind in KIND_ORDER:
        net_score = f"cs_{kind}_network_pressure_score"
        net_bin = f"gcl_{net_score}_bin"
        net_tail = f"gcl_{kind}_network_tail"
        high_col = f"cs_{kind}_high_network_pressure"
        for frame in frames:
            frame[net_tail] = frame[high_col].astype(int)

        add_smoothed_rate(
            train,
            frames,
            ["gcl_cs_event_pressure_score_bin", net_bin],
            f"gcl_{kind}_event_network_rate",
        )
        add_smoothed_rate(
            train,
            frames,
            ["gcl_cs_vulnerability_score_bin", net_bin],
            f"gcl_{kind}_vuln_network_rate",
        )
        add_smoothed_rate(
            train,
            frames,
            ["gcl_cs_event_pressure_score_bin", "gcl_cs_vulnerability_score_bin", net_bin],
            f"gcl_{kind}_event_vuln_network_rate",
        )
        add_smoothed_rate(
            train,
            frames,
            ["gcl_event_tail", "gcl_vuln_tail", net_tail],
            f"gcl_{kind}_tail_triad_rate",
        )

        for frame in frames:
            rate_cols = [
                f"gcl_{kind}_event_network_rate",
                f"gcl_{kind}_vuln_network_rate",
                f"gcl_{kind}_event_vuln_network_rate",
                f"gcl_{kind}_tail_triad_rate",
            ]
            lift_cols = [f"{col}_lift" for col in rate_cols]
            frame[f"gcl_{kind}_conversion_score"] = frame[rate_cols].mean(axis=1)
            frame[f"gcl_{kind}_conversion_lift"] = frame[lift_cols].mean(axis=1)
            frame[f"gcl_{kind}_event_network_x_conversion"] = (
                frame["cs_event_pressure_score"] * frame[net_score] * frame[f"gcl_{kind}_conversion_lift"]
            )
            frame[f"gcl_{kind}_vuln_event_network_x_conversion"] = (
                frame["cs_vulnerability_score"]
                * frame["cs_event_pressure_score"]
                * frame[net_score]
                * frame[f"gcl_{kind}_conversion_lift"]
            )

    for frame in frames:
        gcl_cols = [col for col in frame.columns if col.startswith("gcl_")]
        frame[gcl_cols] = frame[gcl_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frames[0], frames[1], frames[2]


def gdelt_propensity_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("gcl_gdelt_") and not col.endswith("_bin") and "tail" not in col
    ] + available(
        df,
        [
            "gcl_gdelt_tail_state_rate",
            "gcl_gdelt_tail_state_rate_lift",
            "gcl_gdelt_conversion_score",
            "gcl_gdelt_conversion_lift",
        ],
    )


def kind_propensity_cols(df: pd.DataFrame, kind: str) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith(f"gcl_{kind}_")
        and not col.endswith("_bin")
        and not col.endswith("_tail")
    ]


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base = available(df, OPERATIONAL_FEATURES) + country_features + calibration_cols(df)
    gdelt = available(df, OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES) + event_state_cols(df)
    gdelt_prop = gdelt_propensity_cols(df)
    true_network = available(df, NETWORK_BY_KIND["true"]) + network_state_cols(df, "true")
    groups = {
        "GCL0_portwatch_operational_calibrated": base,
        "GCL1_portwatch_gdelt_additive_calibrated": base + gdelt,
        "GCL2_true_wits_additive_calibrated": base + gdelt + true_network,
        "GCL3_gdelt_conversion_propensity": base + gdelt + gdelt_prop,
        "GCL4_true_wits_conversion_propensity_gated": (
            base + gdelt + gdelt_prop + true_network + kind_propensity_cols(df, "true") + gated_cols(df, "true")
        ),
    }
    for kind in ["equal", "random", "shuffled"]:
        network = available(df, NETWORK_BY_KIND[kind]) + network_state_cols(df, kind)
        groups[f"GCL2_{kind}_wits_additive_placebo"] = base + gdelt + network
        groups[f"GCL4_{kind}_conversion_propensity_gated_placebo"] = (
            base + gdelt + gdelt_prop + network + kind_propensity_cols(df, kind) + gated_cols(df, kind)
        )
    return {name: list(dict.fromkeys(features)) for name, features in groups.items()}


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    train, validation, test = add_fold_calibrated_features(train, validation, test)
    train, validation, test = add_fold_conversion_propensity_features(train, validation, test)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    rows = []
    predictions = []
    for group_name, raw_features in groups.items():
        features = numeric_features(train, raw_features)
        print(f"{fold.name}: fitting {group_name} with {len(features)} features", flush=True)
        for model_name, (model, fit_mode) in make_models().items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
            pred["fold"] = fold.name
            pred["feature_group"] = group_name
            pred["model"] = model_name
            pred["predicted_probability"] = test_proba
            row = {
                "fold": fold.name,
                "feature_group": group_name,
                "model": model_name,
                "feature_count": len(features),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positives": int(train[TARGET].sum()),
                "validation_positives": int(validation[TARGET].sum()),
                "test_positives": int(test[TARGET].sum()),
                "test_severe_positives": int(test[SEVERE_TARGET].sum()),
                "selected_threshold": threshold,
                "validation_f1": val_f1,
                **scores,
                "severe_pr_auc": safe_ap(test[SEVERE_TARGET], pd.Series(test_proba, index=test.index)),
            }
            for k in [10, 25, 50]:
                row[f"main_top{k}_hits"] = top_hits(pred, TARGET, k)
                row[f"severe_top{k}_hits"] = top_hits(pred, SEVERE_TARGET, k)
            rows.append(row)
            predictions.append(pred)
    return rows, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "folds": ("fold", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "median_main_pr_auc": ("pr_auc", "median"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "median_severe_pr_auc": ("severe_pr_auc", "median"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "mean_f1": ("f1", "mean"),
        "feature_count": ("feature_count", "max"),
    }
    for prefix in ["main", "severe"]:
        for k in [10, 25, 50]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        metrics.groupby(["feature_group", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)
    )


def pooled_delta(
    predictions: pd.DataFrame,
    focus: tuple[str, str],
    baseline: tuple[str, str],
    target: str,
    n_boot: int = 1000,
) -> tuple[float, float, float, float]:
    focus_group, focus_model = focus
    base_group, base_model = baseline
    focus_frame = predictions.loc[
        predictions["feature_group"].eq(focus_group) & predictions["model"].eq(focus_model)
    ]
    baseline_frame = predictions.loc[
        predictions["feature_group"].eq(base_group) & predictions["model"].eq(base_model)
    ]
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
            safe_ap(sample[target], sample["predicted_probability_focus"])
            - safe_ap(sample[target], sample["predicted_probability_baseline"])
        )
    series = pd.Series(draws)
    if series.empty:
        return point, np.nan, np.nan, np.nan
    return point, float(series.quantile(0.025)), float(series.quantile(0.975)), float((series > 0).mean())


def hit_delta(predictions: pd.DataFrame, focus: tuple[str, str], baseline: tuple[str, str], target: str, k: int) -> int:
    focus_group, focus_model = focus
    base_group, base_model = baseline
    focus_frame = predictions.loc[
        predictions["feature_group"].eq(focus_group) & predictions["model"].eq(focus_model)
    ]
    baseline_frame = predictions.loc[
        predictions["feature_group"].eq(base_group) & predictions["model"].eq(base_model)
    ]
    delta = 0
    for fold in sorted(set(focus_frame["fold"]) & set(baseline_frame["fold"])):
        f = focus_frame.loc[focus_frame["fold"].eq(fold)]
        b = baseline_frame.loc[baseline_frame["fold"].eq(fold)]
        delta += top_hits(f, target, k) - top_hits(b, target, k)
    return delta


def make_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model = PRIMARY_MODEL
    true_gated = ("GCL4_true_wits_conversion_propensity_gated", model)
    contrasts = [
        (true_gated, ("GCL0_portwatch_operational_calibrated", model), "true_cp_gated_vs_operational"),
        (true_gated, ("GCL1_portwatch_gdelt_additive_calibrated", model), "true_cp_gated_vs_gdelt_additive"),
        (true_gated, ("GCL2_true_wits_additive_calibrated", model), "true_cp_gated_vs_true_wits_additive"),
        (true_gated, ("GCL3_gdelt_conversion_propensity", model), "true_cp_gated_vs_gdelt_propensity"),
        (true_gated, ("GCL4_equal_conversion_propensity_gated_placebo", model), "true_cp_gated_vs_equal_placebo"),
        (true_gated, ("GCL4_random_conversion_propensity_gated_placebo", model), "true_cp_gated_vs_random_placebo"),
        (true_gated, ("GCL4_shuffled_conversion_propensity_gated_placebo", model), "true_cp_gated_vs_shuffled_placebo"),
        (
            ("GCL3_gdelt_conversion_propensity", model),
            ("GCL1_portwatch_gdelt_additive_calibrated", model),
            "gdelt_propensity_vs_gdelt_additive",
        ),
    ]
    for focus, baseline, contrast in contrasts:
        for label_name, target in [("main", TARGET), ("severe", SEVERE_TARGET)]:
            pr = pooled_delta(predictions, focus, baseline, target)
            rows.append(
                {
                    "contrast": contrast,
                    "focus_feature_group": focus[0],
                    "baseline_feature_group": baseline[0],
                    "model": model,
                    "label": label_name,
                    "pooled_pr_auc_delta": pr[0],
                    "ci_low": pr[1],
                    "ci_high": pr[2],
                    "p_gt_0": pr[3],
                    "top10_hit_delta": hit_delta(predictions, focus, baseline, target, 10),
                    "top25_hit_delta": hit_delta(predictions, focus, baseline, target, 25),
                    "top50_hit_delta": hit_delta(predictions, focus, baseline, target, 50),
                }
            )
    return pd.DataFrame(rows)


def make_claim_matrix(summary: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    primary = summary.loc[summary["model"].eq(PRIMARY_MODEL)].set_index("feature_group")
    add = deltas.loc[
        deltas["contrast"].eq("true_cp_gated_vs_true_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    gdelt = deltas.loc[
        deltas["contrast"].eq("gdelt_propensity_vs_gdelt_additive") & deltas["label"].eq("main")
    ].iloc[0]
    equal = deltas.loc[
        deltas["contrast"].eq("true_cp_gated_vs_equal_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    random = deltas.loc[
        deltas["contrast"].eq("true_cp_gated_vs_random_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    shuffled = deltas.loc[
        deltas["contrast"].eq("true_cp_gated_vs_shuffled_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "claim": "Fold-aware conversion propensity is a new three-source algorithmic representation, not paper framing.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence_type": "train-period smoothed conversion rates mapped to validation/test folds",
                "main_metric": f"{PRIMARY_MODEL} GCL4 true mean PR-AUC {primary.loc['GCL4_true_wits_conversion_propensity_gated','mean_main_pr_auc']:.4f}",
                "paper_bucket": "main candidate only if additive/placebo guards pass",
                "status": "computed",
            },
            {
                "claim": "GDELT conversion propensity should improve over raw GDELT additive controls.",
                "data_sources": "PortWatch + GDELT",
                "evidence_type": "pooled temporal PR-AUC delta",
                "main_metric": f"delta {gdelt['pooled_pr_auc_delta']:.4f}, CI [{gdelt['ci_low']:.4f},{gdelt['ci_high']:.4f}], top25 {gdelt['top25_hit_delta']:+.0f}",
                "paper_bucket": "Gate 1 evidence if positive",
                "status": "positive only if stable and not top-k weaker",
            },
            {
                "claim": "True WITS conversion propensity gated model should beat true WITS additive.",
                "data_sources": "PortWatch + GDELT + WITS",
                "evidence_type": "pooled temporal PR-AUC delta and top-k guard",
                "main_metric": f"delta {add['pooled_pr_auc_delta']:.4f}, CI [{add['ci_low']:.4f},{add['ci_high']:.4f}], top25 {add['top25_hit_delta']:+.0f}",
                "paper_bucket": "Gate 2 evidence if positive",
                "status": "promotion only if additive and placebo guards pass",
            },
            {
                "claim": "True WITS conversion propensity should separate from matched network placebos.",
                "data_sources": "PortWatch + GDELT + WITS with equal/random/shuffled placebos",
                "evidence_type": "placebo-controlled pooled temporal PR-AUC",
                "main_metric": f"vs equal {equal['pooled_pr_auc_delta']:.4f}; vs random {random['pooled_pr_auc_delta']:.4f}; vs shuffled {shuffled['pooled_pr_auc_delta']:.4f}",
                "paper_bucket": "placebo guard",
                "status": "promotion blocked if any matched placebo is competitive",
            },
            {
                "claim": "This branch stays inside the main-paper source boundary.",
                "data_sources": "PortWatch + GDELT + WITS only",
                "evidence_type": "script provenance",
                "main_metric": "reads the expanded32 PortWatch/GDELT/WITS processed panel and no fourth-source cache",
                "paper_bucket": "main/supplemental depending on results",
                "status": "satisfied",
            },
        ]
    )


def write_report(df: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, matrix: pd.DataFrame) -> None:
    primary = summary.loc[summary["model"].eq(PRIMARY_MODEL)]
    content = f"""# Panel32 Fold-Aware GDELT Conversion-Propensity Benchmark

## Purpose

This branch responds to the active research directive by testing a new conversion representation rather than reframing weak WITS-gated evidence. It does **not** create manual labels. It creates fold-aware, train-period smoothed conversion-propensity features that ask whether event pressure converts under operational vulnerability and WITS exposure states.

## Dataset and Validation

- Dataset: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Sources used: PortWatch, GDELT, WITS only.
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}
- Validation: rolling temporal folds; conversion-propensity rates are learned from training rows only and mapped to validation/test rows.

## Model Ladder

1. `GCL0_portwatch_operational_calibrated`: operational baseline plus train-period calibration.
2. `GCL1_portwatch_gdelt_additive_calibrated`: GCL0 plus raw/calibrated GDELT event pressure.
3. `GCL2_true_wits_additive_calibrated`: GCL1 plus true WITS exposure.
4. `GCL3_gdelt_conversion_propensity`: GCL1 plus fold-aware GDELT conversion-propensity features.
5. `GCL4_true_wits_conversion_propensity_gated`: GCL2 plus fold-aware conversion-propensity and calibrated gated interactions.
6. Matched `GCL4_equal/random/shuffled` WITS placebo gated variants.

## Primary Model Main Table

Primary model: `{PRIMARY_MODEL}`.

{primary[["feature_group", "mean_main_pr_auc", "mean_severe_pr_auc", "mean_roc_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "feature_count"]].to_markdown(index=False)}

## Key Primary-Model Deltas

{deltas.to_markdown(index=False)}

## Claim-Evidence Matrix

{matrix.to_markdown(index=False)}

## Reading

Promote this branch only if the true conversion-propensity gated model improves over true WITS additive and separates from equal/random/shuffled conversion-propensity gated placebos under temporal validation. If not, record it as a negative or supporting diagnostic and move to the next research-engineering branch.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(fold, df)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)
        print(f"Finished {fold.name}: {len(fold_rows)} rows", flush=True)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(metrics)
    deltas = make_deltas(predictions)
    matrix = make_claim_matrix(summary, deltas)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    deltas.to_csv(DELTAS, index=False)
    matrix.to_csv(MATRIX, index=False)
    write_report(df, summary, deltas, matrix)
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved matrix: {MATRIX}")
    print(f"Saved report: {REPORT}")
    print(summary.loc[summary["model"].eq(PRIMARY_MODEL)].to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
