from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_EQUAL_PLACEBO_FEATURES,
    TOTAL_NETWORK_FEATURES,
    TOTAL_RANDOM_PLACEBO_FEATURES,
    TOTAL_SHUFFLED_PLACEBO_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_network_gated_conversion_main.md"
METRICS = TABLE_DIR / "panel32_network_gated_conversion_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_network_gated_conversion_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_network_gated_conversion_predictions.csv"
DELTAS = TABLE_DIR / "panel32_network_gated_conversion_key_deltas.csv"
MATRIX = TABLE_DIR / "panel32_network_gated_claim_evidence_matrix.csv"

SEVERE_TARGET = "abnormal_next_week_container_2p0sigma"
RANDOM_SEED = 42

ME_PLACEBO_BY_KIND = {
    "equal": ["me_equal_strict_very_negative_exposure", "me_network_strict_article_count"],
    "random": ["me_random_strict_very_negative_exposure", "me_network_strict_article_count"],
    "shuffled": ["me_shuffled_strict_very_negative_exposure", "me_network_strict_article_count"],
}

TOTAL_PLACEBO_BY_KIND = {
    "equal": TOTAL_EQUAL_PLACEBO_FEATURES,
    "random": TOTAL_RANDOM_PLACEBO_FEATURES,
    "shuffled": TOTAL_SHUFFLED_PLACEBO_FEATURES,
}

NETWORK_ROOTS = {
    "true": [
        "network_very_negative_exposure",
        "network_trade_transport_exposure",
        "network_risk_theme_exposure",
        "me_network_strict_very_negative_exposure",
    ],
    "equal": [
        "equal_very_negative_exposure",
        "equal_trade_transport_exposure",
        "equal_risk_theme_exposure",
        "me_equal_strict_very_negative_exposure",
    ],
    "random": [
        "random_very_negative_exposure",
        "random_trade_transport_exposure",
        "random_risk_theme_exposure",
        "me_random_strict_very_negative_exposure",
    ],
    "shuffled": [
        "shuffled_very_negative_exposure",
        "shuffled_trade_transport_exposure",
        "shuffled_risk_theme_exposure",
        "me_shuffled_strict_very_negative_exposure",
    ],
}

EVENT_ROOTS = [
    "external_very_negative_article_share",
    "external_trade_transport_count",
    "external_risk_theme_count",
    "external_me_strict_very_negative_exposure",
]

VULNERABILITY_ROOTS = [
    "ng_vulnerability_shortfall",
    "ng_vulnerability_negative_trend",
]


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def safe_ap(y: pd.Series, score: pd.Series) -> float:
    if y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_hits(frame: pd.DataFrame, target_col: str, k: int) -> int:
    return int(frame.sort_values("predicted_probability", ascending=False).head(k)[target_col].sum())


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)
    df[SEVERE_TARGET] = (
        df["next_week_container"] < (df["rolling_mean_12w"] - 2.0 * df["rolling_std_12w"].replace(0, np.nan))
    ).astype(int)
    return add_network_gated_features(df)


def add_network_gated_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ng_vulnerability_shortfall"] = out["operational_shortfall_12w"].clip(lower=0).fillna(0.0)
    out["ng_vulnerability_negative_trend"] = out["negative_trend_4w"].clip(lower=0).fillna(0.0)
    for col in ["external_trade_transport_count", "external_risk_theme_count"]:
        if col in out.columns:
            out[f"ng_log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    event_cols = []
    for col in EVENT_ROOTS:
        if col in out.columns:
            if col in {"external_trade_transport_count", "external_risk_theme_count"}:
                event_cols.append(f"ng_log1p_{col}")
            else:
                event_cols.append(col)

    for network_kind, network_cols in NETWORK_ROOTS.items():
        available_network_cols = available(out, network_cols)
        gate_cols = {}
        for vuln in available(out, VULNERABILITY_ROOTS):
            for event in event_cols:
                gate_cols[f"ng_{network_kind}_{vuln}_x_{event}"] = out[vuln] * out[event]
            for event in event_cols:
                for network in available_network_cols:
                    gate_cols[f"ng_{network_kind}_{event}_x_{network}"] = out[event] * out[network]
            for event in event_cols:
                for network in available_network_cols:
                    gate_cols[f"ng_{network_kind}_{vuln}_x_{event}_x_{network}"] = out[vuln] * out[event] * out[network]
        if gate_cols:
            out = pd.concat([out, pd.DataFrame(gate_cols, index=out.index)], axis=1)
    compact_cols = {}
    for network_kind, network_cols in NETWORK_ROOTS.items():
        v_short = out["ng_vulnerability_shortfall"]
        v_trend = out["ng_vulnerability_negative_trend"]
        event_vneg = out["external_very_negative_article_share"]
        event_me = out["external_me_strict_very_negative_exposure"]
        network_vneg = out[network_cols[0]]
        network_me = out[network_cols[-1]]
        compact_cols[f"ngc_{network_kind}_shortfall_x_external_vneg"] = v_short * event_vneg
        compact_cols[f"ngc_{network_kind}_external_vneg_x_network_vneg"] = event_vneg * network_vneg
        compact_cols[f"ngc_{network_kind}_shortfall_x_network_vneg"] = v_short * network_vneg
        compact_cols[f"ngc_{network_kind}_shortfall_x_external_vneg_x_network_vneg"] = (
            v_short * event_vneg * network_vneg
        )
        compact_cols[f"ngc_{network_kind}_trend_x_external_vneg_x_network_vneg"] = v_trend * event_vneg * network_vneg
        compact_cols[f"ngc_{network_kind}_shortfall_x_external_me_x_network_me"] = v_short * event_me * network_me
    out = pd.concat([out, pd.DataFrame(compact_cols, index=out.index)], axis=1)
    gate_cols = [col for col in out.columns if col.startswith("ng_") or col.startswith("ngc_")]
    out[gate_cols] = out[gate_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def gated_cols(df: pd.DataFrame, network_kind: str) -> list[str]:
    return [col for col in df.columns if col.startswith(f"ng_{network_kind}_")]


def compact_gated_cols(df: pd.DataFrame, network_kind: str) -> list[str]:
    return [col for col in df.columns if col.startswith(f"ngc_{network_kind}_")]


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base = available(df, OPERATIONAL_FEATURES) + country_features
    gdelt = available(df, OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES)
    true_network = available(df, TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES)
    groups = {
        "NG0_portwatch_operational": base,
        "NG1_portwatch_gdelt_additive": base + gdelt,
        "NG2_portwatch_gdelt_wits_additive": base + gdelt + true_network,
        "NG3_network_gated_conversion_true": base + gdelt + true_network + gated_cols(df, "true"),
        "NG5_compact_network_gated_true": base + gdelt + true_network + compact_gated_cols(df, "true"),
    }
    for kind in ["equal", "random", "shuffled"]:
        placebo_network = available(df, TOTAL_PLACEBO_BY_KIND[kind] + ME_PLACEBO_BY_KIND[kind])
        groups[f"NG4_{kind}_wits_gated_placebo"] = base + gdelt + placebo_network + gated_cols(df, kind)
        groups[f"NG6_{kind}_compact_gated_placebo"] = (
            base + gdelt + placebo_network + compact_gated_cols(df, kind)
        )
    return {name: list(dict.fromkeys(features)) for name, features in groups.items()}


def make_models() -> dict[str, tuple[object, str]]:
    return {
        "logistic": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
                ]
            ),
            "plain",
        ),
        "sklearn_gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=12,
                subsample=0.85,
                random_state=RANDOM_SEED,
            ),
            "sample_weight",
        ),
    }


def fit_model(model, fit_mode: str, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if fit_mode == "sample_weight":
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
        model.fit(train_x, train_y, sample_weight=weights)
    else:
        model.fit(train_x, train_y)


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    rows = []
    predictions = []
    for group_name, raw_features in groups.items():
        features = [f for f in raw_features if f in train.columns and pd.api.types.is_numeric_dtype(train[f])]
        for model_name, (model, fit_mode) in make_models().items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            main_scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
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
                **main_scores,
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


def pooled_delta(predictions: pd.DataFrame, focus: tuple[str, str], baseline: tuple[str, str], target: str, n_boot: int = 2500) -> tuple[float, float, float, float]:
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
            average_precision_score(sample[target], sample["predicted_probability_focus"])
            - average_precision_score(sample[target], sample["predicted_probability_baseline"])
        )
    series = pd.Series(draws)
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
    model = "sklearn_gradient_boosting"
    true_gated = ("NG3_network_gated_conversion_true", model)
    compact_true = ("NG5_compact_network_gated_true", model)
    contrasts = [
        (true_gated, ("NG0_portwatch_operational", model), "gated_true_vs_operational"),
        (true_gated, ("NG1_portwatch_gdelt_additive", model), "gated_true_vs_gdelt_additive"),
        (true_gated, ("NG2_portwatch_gdelt_wits_additive", model), "gated_true_vs_wits_additive"),
        (true_gated, ("NG4_equal_wits_gated_placebo", model), "gated_true_vs_equal_placebo"),
        (true_gated, ("NG4_random_wits_gated_placebo", model), "gated_true_vs_random_placebo"),
        (true_gated, ("NG4_shuffled_wits_gated_placebo", model), "gated_true_vs_shuffled_placebo"),
        (compact_true, ("NG0_portwatch_operational", model), "compact_gated_true_vs_operational"),
        (compact_true, ("NG1_portwatch_gdelt_additive", model), "compact_gated_true_vs_gdelt_additive"),
        (compact_true, ("NG2_portwatch_gdelt_wits_additive", model), "compact_gated_true_vs_wits_additive"),
        (compact_true, ("NG6_equal_compact_gated_placebo", model), "compact_gated_true_vs_equal_placebo"),
        (compact_true, ("NG6_random_compact_gated_placebo", model), "compact_gated_true_vs_random_placebo"),
        (compact_true, ("NG6_shuffled_compact_gated_placebo", model), "compact_gated_true_vs_shuffled_placebo"),
        (("NG2_portwatch_gdelt_wits_additive", model), ("NG1_portwatch_gdelt_additive", model), "wits_additive_vs_gdelt_additive"),
    ]
    rows = []
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
    gb = summary.loc[summary["model"].eq("sklearn_gradient_boosting")].set_index("feature_group")
    true_vs_add = deltas.loc[
        deltas["contrast"].eq("gated_true_vs_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    compact_vs_add = deltas.loc[
        deltas["contrast"].eq("compact_gated_true_vs_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    true_vs_equal = deltas.loc[
        deltas["contrast"].eq("gated_true_vs_equal_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    true_vs_random = deltas.loc[
        deltas["contrast"].eq("gated_true_vs_random_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    rows = [
        {
            "claim": "Event signal alone is noisy but useful only conditionally.",
            "data_sources": "PortWatch + GDELT",
            "model_family": "NG1 additive GDELT vs NG0 operational",
            "evidence_type": "temporal PR-AUC/top-k",
            "main_metric": f"GB NG1 mean PR-AUC {gb.loc['NG1_portwatch_gdelt_additive','mean_main_pr_auc']:.4f} vs NG0 {gb.loc['NG0_portwatch_operational','mean_main_pr_auc']:.4f}",
            "paper_bucket": "main/supporting",
            "status": "usable if framed as additive baseline, not final model",
        },
        {
            "claim": "WITS network exposure adds structural gating beyond unweighted GDELT.",
            "data_sources": "PortWatch + GDELT + WITS",
            "model_family": "NG2 additive WITS, NG3 wide gated, NG5 compact gated",
            "evidence_type": "temporal PR-AUC/placebo",
            "main_metric": f"NG3 vs NG2 pooled PR-AUC delta {true_vs_add['pooled_pr_auc_delta']:.4f}; NG5 vs NG2 {compact_vs_add['pooled_pr_auc_delta']:.4f}; NG3 vs equal {true_vs_equal['pooled_pr_auc_delta']:.4f}; NG3 vs random {true_vs_random['pooled_pr_auc_delta']:.4f}",
            "paper_bucket": "main if positive vs additive and placebos; otherwise supporting/negative",
            "status": "computed in this run",
        },
        {
            "claim": "Network-gated conversion is the proposed model class.",
            "data_sources": "PortWatch + GDELT + WITS",
            "model_family": "OperationalVulnerability x EventPressure x NetworkExposure",
            "evidence_type": "temporal and severe guardrails",
            "main_metric": f"NG3 wide mean PR-AUC {gb.loc['NG3_network_gated_conversion_true','mean_main_pr_auc']:.4f}; NG5 compact mean PR-AUC {gb.loc['NG5_compact_network_gated_true','mean_main_pr_auc']:.4f}",
            "paper_bucket": "main model table",
            "status": "requires LOCO follow-up before strong deployment claim",
        },
        {
            "claim": "Fourth data sources are not part of the main model.",
            "data_sources": "PortWatch + GDELT + WITS only",
            "model_family": "all NG groups",
            "evidence_type": "data provenance",
            "main_metric": "script reads multicountry32_container_event_network_benchmark.csv and uses no NASA/UNCTAD/BTS/Linerlytica/WorldBank/SeaIntel/USDA/ARIC features",
            "paper_bucket": "main",
            "status": "satisfied for this script",
        },
    ]
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, matrix: pd.DataFrame) -> None:
    gb = summary.loc[summary["model"].eq("sklearn_gradient_boosting")]
    content = f"""# Panel32 Network-Gated Event Conversion Main Benchmark

## Purpose

This is a main-paper consolidation benchmark for the proposed **Network-Gated Event Conversion Model for Port Disruption Early Warning**. It intentionally uses only three data sources: PortWatch operational activity/labels, GDELT own and partner event pressure, and WITS true/equal/random/shuffled trade-network exposure.

## Dataset

- File: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}

## Required Model Ladder

1. PortWatch operational only: `NG0_portwatch_operational`
2. Operational + GDELT additive: `NG1_portwatch_gdelt_additive`
3. Operational + GDELT + WITS additive: `NG2_portwatch_gdelt_wits_additive`
4. Proposed true network-gated conversion: wide `NG3_network_gated_conversion_true` and compact `NG5_compact_network_gated_true`
5. Same conversion model with equal/random/shuffled WITS placebos: `NG4_*_wits_gated_placebo` and `NG6_*_compact_gated_placebo`

## Gradient Boosting Main Table

{gb[["feature_group", "mean_main_pr_auc", "mean_severe_pr_auc", "mean_roc_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "feature_count"]].to_markdown(index=False)}

## Key Gradient-Boosting Deltas

{deltas.to_markdown(index=False)}

## Claim-Evidence Matrix

{matrix.to_markdown(index=False)}

## Reading

This table tests the paper's proposed mechanism directly: events are allowed to convert into abnormal-port risk through operational vulnerability and WITS network exposure. Promotion depends on whether `NG3_network_gated_conversion_true` improves over additive GDELT/WITS baselines and separates from equal/random/shuffled WITS gated placebos. If placebo-gated variants match or beat the true WITS model, the paper should frame network gating as attribution/structure and not as a universal predictive win.
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
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
