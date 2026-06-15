from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
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
REPORT = PROJECT_ROOT / "reports" / "panel32_network_gated_country_shared_conversion.md"
METRICS = TABLE_DIR / "panel32_network_gated_country_shared_conversion_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_network_gated_country_shared_conversion_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_network_gated_country_shared_conversion_predictions.csv"
DELTAS = TABLE_DIR / "panel32_network_gated_country_shared_conversion_deltas.csv"
MATRIX = TABLE_DIR / "panel32_network_gated_country_shared_conversion_claim_matrix.csv"

SEVERE_TARGET = "abnormal_next_week_container_2p0sigma"
RANDOM_SEED = 42
PRIMARY_MODEL = "sklearn_gradient_boosting"

KIND_ORDER = ["true", "equal", "random", "shuffled"]

NETWORK_BY_KIND = {
    "true": TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES,
    "equal": TOTAL_EQUAL_PLACEBO_FEATURES
    + ["me_equal_strict_very_negative_exposure", "me_network_strict_article_count"],
    "random": TOTAL_RANDOM_PLACEBO_FEATURES
    + ["me_random_strict_very_negative_exposure", "me_network_strict_article_count"],
    "shuffled": TOTAL_SHUFFLED_PLACEBO_FEATURES
    + ["me_shuffled_strict_very_negative_exposure", "me_network_strict_article_count"],
}

NETWORK_ROOTS = {
    "true": [
        "network_negative_exposure",
        "network_very_negative_exposure",
        "network_trade_transport_exposure",
        "network_risk_theme_exposure",
        "me_network_strict_very_negative_exposure",
        "me_network_strict_article_count",
    ],
    "equal": [
        "equal_negative_exposure",
        "equal_very_negative_exposure",
        "equal_trade_transport_exposure",
        "equal_risk_theme_exposure",
        "me_equal_strict_very_negative_exposure",
        "me_network_strict_article_count",
    ],
    "random": [
        "random_negative_exposure",
        "random_very_negative_exposure",
        "random_trade_transport_exposure",
        "random_risk_theme_exposure",
        "me_random_strict_very_negative_exposure",
        "me_network_strict_article_count",
    ],
    "shuffled": [
        "shuffled_negative_exposure",
        "shuffled_very_negative_exposure",
        "shuffled_trade_transport_exposure",
        "shuffled_risk_theme_exposure",
        "me_shuffled_strict_very_negative_exposure",
        "me_network_strict_article_count",
    ],
}

VULNERABILITY_COMPONENTS = [
    "cs_vuln_shortfall",
    "cs_vuln_negative_trend",
    "cs_vuln_shortfall_ratio",
    "cs_vuln_volatility_ratio",
    "cs_vuln_recovery_gap",
]

EVENT_COMPONENTS = [
    "cs_log_own_article_count",
    "cs_own_negative_share",
    "cs_own_very_negative_share",
    "cs_log_own_trade_transport",
    "cs_log_own_risk_theme",
    "cs_log_external_article_count",
    "cs_external_negative_share",
    "cs_external_very_negative_share",
    "cs_log_external_trade_transport",
    "cs_log_external_risk_theme",
    "cs_external_me_very_negative",
    "cs_log_external_me_article_count",
]


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def numeric_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])]


def safe_ap(y: pd.Series, score: pd.Series) -> float:
    if y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_hits(frame: pd.DataFrame, target_col: str, k: int) -> int:
    return int(frame.sort_values("predicted_probability", ascending=False).head(k)[target_col].sum())


def add_static_country_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cs_vuln_shortfall"] = out["operational_shortfall_12w"].clip(lower=0).fillna(0.0)
    out["cs_vuln_negative_trend"] = out["negative_trend_4w"].clip(lower=0).fillna(0.0)
    out["cs_vuln_shortfall_active"] = (out["cs_vuln_shortfall"] > 0).astype(float)
    out["cs_vuln_shortfall_ratio"] = out["cs_vuln_shortfall"] / (out["rolling_mean_container_12w"].abs() + 1.0)
    out["cs_vuln_volatility_ratio"] = out["rolling_std_container_12w"] / (
        out["rolling_mean_container_12w"].abs() + 1.0
    )
    out["cs_vuln_recovery_gap"] = (
        out["rolling_mean_container_12w"] - out["rolling_mean_container_4w"]
    ) / (out["rolling_mean_container_12w"].abs() + 1.0)
    out["cs_vuln_recovery_gap"] = out["cs_vuln_recovery_gap"].clip(lower=0).fillna(0.0)

    out["cs_log_own_article_count"] = np.log1p(out["article_count"].clip(lower=0).fillna(0.0))
    out["cs_own_negative_share"] = out["negative_article_share"].fillna(0.0)
    out["cs_own_very_negative_share"] = out["very_negative_article_share"].fillna(0.0)
    out["cs_log_own_trade_transport"] = np.log1p(out["trade_transport_count"].clip(lower=0).fillna(0.0))
    out["cs_log_own_risk_theme"] = np.log1p(out["risk_theme_count"].clip(lower=0).fillna(0.0))
    out["cs_log_external_article_count"] = np.log1p(out["external_article_count"].clip(lower=0).fillna(0.0))
    out["cs_external_negative_share"] = out["external_negative_article_share"].fillna(0.0)
    out["cs_external_very_negative_share"] = out["external_very_negative_article_share"].fillna(0.0)
    out["cs_log_external_trade_transport"] = np.log1p(
        out["external_trade_transport_count"].clip(lower=0).fillna(0.0)
    )
    out["cs_log_external_risk_theme"] = np.log1p(out["external_risk_theme_count"].clip(lower=0).fillna(0.0))
    out["cs_external_me_very_negative"] = out["external_me_strict_very_negative_exposure"].fillna(0.0)
    out["cs_log_external_me_article_count"] = np.log1p(
        out["external_me_strict_article_count"].clip(lower=0).fillna(0.0)
    )

    cs_cols = [col for col in out.columns if col.startswith("cs_")]
    out[cs_cols] = out[cs_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)
    df[SEVERE_TARGET] = (
        df["next_week_container"] < (df["rolling_mean_12w"] - 2.0 * df["rolling_std_12w"].replace(0, np.nan))
    ).astype(int)
    return add_static_country_shared_features(df)


def add_country_priors(train: pd.DataFrame, frames: list[pd.DataFrame], target_col: str, out_col: str) -> None:
    alpha = 26.0
    global_rate = float(train[target_col].mean())
    stats = train.groupby("ISO3", sort=False)[target_col].agg(["sum", "count"])
    priors = (stats["sum"] + alpha * global_rate) / (stats["count"] + alpha)
    for frame in frames:
        frame[out_col] = frame["ISO3"].map(priors).fillna(global_rate).astype(float)


def zscore_from_train(train: pd.DataFrame, frames: list[pd.DataFrame], col: str, out_col: str) -> None:
    values = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = float(values.mean()) if values.notna().any() else 0.0
    std = float(values.std(ddof=0)) if values.notna().any() else 0.0
    if std <= 1e-9:
        std = 1.0
    for frame in frames:
        frame[out_col] = (
            (pd.to_numeric(frame[col], errors="coerce").fillna(mean) - mean) / std
        ).clip(-4.0, 4.0)


def score_from_z(frame: pd.DataFrame, cols: list[str], out_col: str) -> None:
    usable = available(frame, cols)
    if usable:
        frame[out_col] = frame[usable].mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        frame[out_col] = 0.0


def add_fold_calibrated_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [train.copy(), validation.copy(), test.copy()]
    train = frames[0]

    add_country_priors(train, frames, TARGET, "cs_country_main_prior")
    add_country_priors(train, frames, SEVERE_TARGET, "cs_country_severe_prior")

    z_vuln_cols = []
    z_event_cols = []
    for col in available(train, VULNERABILITY_COMPONENTS):
        out_col = f"csz_vuln_{col.removeprefix('cs_vuln_')}"
        zscore_from_train(train, frames, col, out_col)
        z_vuln_cols.append(out_col)
    for col in available(train, EVENT_COMPONENTS):
        out_col = f"csz_event_{col.removeprefix('cs_')}"
        zscore_from_train(train, frames, col, out_col)
        z_event_cols.append(out_col)

    for frame in frames:
        score_from_z(frame, z_vuln_cols, "cs_vulnerability_score")
        score_from_z(frame, z_event_cols, "cs_event_pressure_score")

    vuln_q75 = float(frames[0]["cs_vulnerability_score"].quantile(0.75))
    event_q75 = float(frames[0]["cs_event_pressure_score"].quantile(0.75))
    for frame in frames:
        frame["cs_high_vulnerability"] = (frame["cs_vulnerability_score"] >= vuln_q75).astype(float)
        frame["cs_high_event_pressure"] = (frame["cs_event_pressure_score"] >= event_q75).astype(float)
        frame["cs_vuln_prior_pressure"] = frame["cs_vulnerability_score"] * frame["cs_country_main_prior"]
        frame["cs_event_prior_pressure"] = frame["cs_event_pressure_score"] * frame["cs_country_main_prior"]

    for kind in KIND_ORDER:
        z_network_cols = []
        for col in available(train, NETWORK_ROOTS[kind]):
            out_col = f"csz_{kind}_{col}"
            zscore_from_train(train, frames, col, out_col)
            z_network_cols.append(out_col)
        score_col = f"cs_{kind}_network_pressure_score"
        high_col = f"cs_{kind}_high_network_pressure"
        network_q75 = 0.0
        for frame in frames:
            score_from_z(frame, z_network_cols, score_col)
        if score_col in frames[0].columns:
            network_q75 = float(frames[0][score_col].quantile(0.75))
        for frame in frames:
            vuln = frame["cs_vulnerability_score"]
            event = frame["cs_event_pressure_score"]
            network = frame[score_col]
            high_vuln = frame["cs_high_vulnerability"]
            high_event = frame["cs_high_event_pressure"]
            high_network = (network >= network_q75).astype(float)
            active_shortfall = frame["cs_vuln_shortfall_active"]
            main_prior = frame["cs_country_main_prior"]
            severe_prior = frame["cs_country_severe_prior"]
            frame[high_col] = high_network
            frame[f"csg_{kind}_vuln_x_event"] = vuln * event
            frame[f"csg_{kind}_event_x_network"] = event * network
            frame[f"csg_{kind}_vuln_x_network"] = vuln * network
            frame[f"csg_{kind}_vuln_x_event_x_network"] = vuln * event * network
            frame[f"csg_{kind}_shortfall_x_event_x_network"] = active_shortfall * event * network
            frame[f"csg_{kind}_tail_event_x_tail_network"] = high_event * high_network * event * network
            frame[f"csg_{kind}_tail_vuln_event_network"] = high_vuln * event * network
            frame[f"csg_{kind}_country_prior_x_triple"] = main_prior * vuln * event * network
            frame[f"csg_{kind}_severe_prior_x_triple"] = severe_prior * vuln * event * network

    for frame in frames:
        cs_cols = [col for col in frame.columns if col.startswith("cs") or col.startswith("csg_")]
        frame[cs_cols] = frame[cs_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frames[0], frames[1], frames[2]


def calibration_cols(df: pd.DataFrame) -> list[str]:
    explicit = [
        "cs_country_main_prior",
        "cs_country_severe_prior",
        "cs_vulnerability_score",
        "cs_high_vulnerability",
        "cs_vuln_prior_pressure",
    ]
    return available(df, explicit)


def event_state_cols(df: pd.DataFrame) -> list[str]:
    explicit = [
        "cs_event_pressure_score",
        "cs_high_event_pressure",
        "cs_event_prior_pressure",
    ]
    return available(df, explicit) + [col for col in df.columns if col.startswith("csz_event_")]


def network_state_cols(df: pd.DataFrame, kind: str) -> list[str]:
    explicit = [
        f"cs_{kind}_network_pressure_score",
        f"cs_{kind}_high_network_pressure",
    ]
    z_cols = [col for col in df.columns if col.startswith(f"csz_{kind}_")]
    return available(df, explicit) + z_cols


def gated_cols(df: pd.DataFrame, kind: str) -> list[str]:
    return [col for col in df.columns if col.startswith(f"csg_{kind}_")]


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base = available(df, OPERATIONAL_FEATURES) + country_features + calibration_cols(df)
    gdelt = available(df, OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES) + event_state_cols(df)
    true_network = available(df, NETWORK_BY_KIND["true"]) + network_state_cols(df, "true")
    groups = {
        "CS0_portwatch_operational_calibrated": base,
        "CS1_portwatch_gdelt_additive_calibrated": base + gdelt,
        "CS2_true_wits_additive_calibrated": base + gdelt + true_network,
        "CS3_country_shared_gated_true": base + gdelt + true_network + gated_cols(df, "true"),
    }
    for kind in ["equal", "random", "shuffled"]:
        network = available(df, NETWORK_BY_KIND[kind]) + network_state_cols(df, kind)
        groups[f"CS2_{kind}_wits_additive_placebo"] = base + gdelt + network
        groups[f"CS3_{kind}_country_shared_gated_placebo"] = base + gdelt + network + gated_cols(df, kind)
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
                n_estimators=260,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=14,
                subsample=0.85,
                random_state=RANDOM_SEED,
            ),
            "sample_weight",
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=18,
                l2_regularization=0.02,
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
    train, validation, test = add_fold_calibrated_features(train, validation, test)
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
            average_precision_score(sample[target], sample["predicted_probability_focus"])
            - average_precision_score(sample[target], sample["predicted_probability_baseline"])
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
    true_gated = ("CS3_country_shared_gated_true", model)
    contrasts = [
        (true_gated, ("CS0_portwatch_operational_calibrated", model), "true_gated_vs_operational"),
        (true_gated, ("CS1_portwatch_gdelt_additive_calibrated", model), "true_gated_vs_gdelt"),
        (true_gated, ("CS2_true_wits_additive_calibrated", model), "true_gated_vs_true_wits_additive"),
        (true_gated, ("CS3_equal_country_shared_gated_placebo", model), "true_gated_vs_equal_placebo"),
        (true_gated, ("CS3_random_country_shared_gated_placebo", model), "true_gated_vs_random_placebo"),
        (true_gated, ("CS3_shuffled_country_shared_gated_placebo", model), "true_gated_vs_shuffled_placebo"),
        (
            ("CS2_true_wits_additive_calibrated", model),
            ("CS1_portwatch_gdelt_additive_calibrated", model),
            "true_wits_additive_vs_gdelt",
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
    true_vs_add = deltas.loc[
        deltas["contrast"].eq("true_gated_vs_true_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    true_vs_equal = deltas.loc[
        deltas["contrast"].eq("true_gated_vs_equal_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    true_vs_random = deltas.loc[
        deltas["contrast"].eq("true_gated_vs_random_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    true_vs_shuffled = deltas.loc[
        deltas["contrast"].eq("true_gated_vs_shuffled_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    rows = [
        {
            "claim": "Country-shared calibration is a new algorithmic branch, not post-hoc framing.",
            "data_sources": "PortWatch + GDELT + WITS",
            "evidence_type": "fold-safe train-period country priors and global z-score scales",
            "main_metric": f"{PRIMARY_MODEL} true gated mean PR-AUC {primary.loc['CS3_country_shared_gated_true','mean_main_pr_auc']:.4f}",
            "paper_bucket": "main candidate if placebo guard passes; otherwise supporting/negative",
            "status": "computed",
        },
        {
            "claim": "True WITS gated conversion should beat additive true WITS.",
            "data_sources": "PortWatch + GDELT + WITS",
            "evidence_type": "pooled temporal PR-AUC delta",
            "main_metric": f"delta {true_vs_add['pooled_pr_auc_delta']:.4f}, CI [{true_vs_add['ci_low']:.4f},{true_vs_add['ci_high']:.4f}], p>0 {true_vs_add['p_gt_0']:.3f}",
            "paper_bucket": "gate evidence",
            "status": "positive only if CI mostly above zero and top-k not weaker",
        },
        {
            "claim": "True WITS gated conversion should separate from placebo networks.",
            "data_sources": "PortWatch + GDELT + WITS with equal/random/shuffled WITS placebos",
            "evidence_type": "placebo-controlled pooled temporal PR-AUC",
            "main_metric": f"vs equal {true_vs_equal['pooled_pr_auc_delta']:.4f}; vs random {true_vs_random['pooled_pr_auc_delta']:.4f}; vs shuffled {true_vs_shuffled['pooled_pr_auc_delta']:.4f}",
            "paper_bucket": "gate evidence",
            "status": "promotion allowed only if true gated is not placebo-competitive",
        },
        {
            "claim": "Fourth-source expansion remains outside this main test.",
            "data_sources": "PortWatch + GDELT + WITS only",
            "evidence_type": "script provenance",
            "main_metric": "script reads multicountry32_container_event_network_benchmark.csv and no NASA/UNCTAD/BTS/LPI/AIS/commercial sources",
            "paper_bucket": "main",
            "status": "satisfied for this branch",
        },
    ]
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, matrix: pd.DataFrame) -> None:
    primary = summary.loc[summary["model"].eq(PRIMARY_MODEL)]
    content = f"""# Panel32 Country-Shared Network-Gated Conversion Benchmark

## Purpose

This branch responds to the active research directive: weak gated-conversion evidence should trigger a new algorithmic test, not only defensive paper framing. The model keeps the main-paper three-source scope and tests whether a country-shared, train-calibrated conversion representation improves next-week abnormal container activity prediction.

## Dataset and Validation

- Dataset: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Sources used: PortWatch operational labels/features, GDELT own/external event pressure, WITS true/equal/random/shuffled exposure.
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}
- Validation: rolling temporal folds; thresholds selected on validation years, never on test years.

## Model Ladder

1. `CS0_portwatch_operational_calibrated`: operational PortWatch + country dummies + train-period country risk priors.
2. `CS1_portwatch_gdelt_additive_calibrated`: CS0 + raw and calibrated GDELT event pressure.
3. `CS2_true_wits_additive_calibrated`: CS1 + true WITS exposure and calibrated network pressure.
4. `CS3_country_shared_gated_true`: CS2 + calibrated vulnerability x event x true-network conversion interactions.
5. `CS2_*` and `CS3_*` equal/random/shuffled placebo versions.

## Primary Model Main Table

Primary model: `{PRIMARY_MODEL}`.

{primary[["feature_group", "mean_main_pr_auc", "mean_severe_pr_auc", "mean_roc_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "feature_count"]].to_markdown(index=False)}

## Key Primary-Model Deltas

{deltas.to_markdown(index=False)}

## Claim-Evidence Matrix

{matrix.to_markdown(index=False)}

## Reading

This is a main-candidate algorithm branch only if the true WITS gated model improves over the additive true-WITS baseline and separates from equal/random/shuffled gated placebos. If it fails either condition, the result should be recorded as a negative guardrail and used to guide the next research-engineering branch, such as validation-safe alert allocation or a reproducible new data acquisition branch.
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
