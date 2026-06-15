from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fetch_portwatch_chokepoint_panel as pwc_source  # noqa: E402
from scripts.run_panel32_hazard_memory_benchmark import (  # noqa: E402
    HAZARD_FEATURES,
    add_hazard_features,
)
from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    INTERACTION_FEATURES,
    ME_NETWORK_FEATURES,
    ME_PLACEBO_FEATURES,
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


BASE_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
CHOKEPOINT_WEEKLY = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_weekly.csv"
ROUTE_WEEKLY = PROJECT_ROOT / "data" / "interim" / "portwatch_chokepoint_route_exposure_panel32_weekly.csv"
ROUTE_WEIGHTS_OUT = PROJECT_ROOT / "reports" / "tables" / "panel32_portwatch_chokepoint_route_weights.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_portwatch_chokepoint_benchmark.csv"

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_portwatch_chokepoint_benchmark.md"
METRICS = TABLE_DIR / "panel32_portwatch_chokepoint_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_portwatch_chokepoint_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_portwatch_chokepoint_predictions.csv"
ALERTS = TABLE_DIR / "panel32_portwatch_chokepoint_alert_budget.csv"
KEY = TABLE_DIR / "panel32_portwatch_chokepoint_key_contrasts.csv"

SEVERE_TARGET = "abnormal_next_week_container_2p0sigma"


# Transparent physical-route exposure priors for the 18 countries added in expanded32.
# These are scenario/exposure assumptions over official PortWatch chokepoints, not
# estimated trade-lane shares or causal route weights.
EXPANDED_ROUTE_WEIGHTS = {
    "BEL": {"chokepoint1": 0.28, "chokepoint4": 0.17, "chokepoint8": 0.16, "chokepoint9": 0.16, "chokepoint10": 0.06, "chokepoint7": 0.11, "chokepoint2": 0.06},
    "BRA": {"chokepoint2": 0.24, "chokepoint7": 0.22, "chokepoint8": 0.15, "chokepoint1": 0.12, "chokepoint4": 0.08, "chokepoint21": 0.08, "chokepoint5": 0.06},
    "CAN": {"chokepoint2": 0.22, "chokepoint26": 0.14, "chokepoint9": 0.14, "chokepoint8": 0.12, "chokepoint1": 0.12, "chokepoint4": 0.08, "chokepoint22": 0.08},
    "CHL": {"chokepoint21": 0.24, "chokepoint2": 0.24, "chokepoint7": 0.15, "chokepoint5": 0.12, "chokepoint19": 0.08, "chokepoint1": 0.08, "chokepoint4": 0.06},
    "EGY": {"chokepoint1": 0.42, "chokepoint4": 0.20, "chokepoint6": 0.12, "chokepoint8": 0.10, "chokepoint3": 0.06, "chokepoint7": 0.06},
    "ESP": {"chokepoint8": 0.28, "chokepoint1": 0.24, "chokepoint4": 0.15, "chokepoint7": 0.12, "chokepoint9": 0.08, "chokepoint2": 0.06, "chokepoint3": 0.04},
    "FRA": {"chokepoint1": 0.27, "chokepoint8": 0.18, "chokepoint9": 0.13, "chokepoint4": 0.16, "chokepoint7": 0.11, "chokepoint10": 0.06, "chokepoint2": 0.05},
    "GBR": {"chokepoint9": 0.24, "chokepoint8": 0.18, "chokepoint1": 0.22, "chokepoint4": 0.14, "chokepoint7": 0.10, "chokepoint2": 0.06, "chokepoint10": 0.04},
    "IND": {"chokepoint5": 0.24, "chokepoint6": 0.20, "chokepoint1": 0.18, "chokepoint4": 0.15, "chokepoint7": 0.10, "chokepoint15": 0.06, "chokepoint14": 0.04},
    "ITA": {"chokepoint1": 0.34, "chokepoint4": 0.20, "chokepoint8": 0.12, "chokepoint3": 0.10, "chokepoint6": 0.08, "chokepoint7": 0.08, "chokepoint5": 0.04},
    "MEX": {"chokepoint2": 0.28, "chokepoint22": 0.16, "chokepoint23": 0.10, "chokepoint24": 0.08, "chokepoint1": 0.12, "chokepoint8": 0.08, "chokepoint7": 0.06},
    "PAK": {"chokepoint6": 0.30, "chokepoint1": 0.20, "chokepoint4": 0.18, "chokepoint5": 0.12, "chokepoint7": 0.10, "chokepoint15": 0.04},
    "PAN": {"chokepoint2": 0.45, "chokepoint22": 0.12, "chokepoint23": 0.08, "chokepoint24": 0.06, "chokepoint1": 0.10, "chokepoint8": 0.06, "chokepoint7": 0.05},
    "PHL": {"chokepoint14": 0.22, "chokepoint11": 0.16, "chokepoint5": 0.16, "chokepoint25": 0.12, "chokepoint27": 0.10, "chokepoint1": 0.08, "chokepoint6": 0.06},
    "POL": {"chokepoint10": 0.22, "chokepoint9": 0.20, "chokepoint1": 0.22, "chokepoint4": 0.14, "chokepoint8": 0.10, "chokepoint7": 0.06, "chokepoint3": 0.03},
    "SWE": {"chokepoint10": 0.28, "chokepoint9": 0.20, "chokepoint1": 0.20, "chokepoint4": 0.12, "chokepoint8": 0.08, "chokepoint7": 0.06, "chokepoint26": 0.03},
    "TUR": {"chokepoint3": 0.28, "chokepoint1": 0.22, "chokepoint4": 0.14, "chokepoint6": 0.12, "chokepoint8": 0.08, "chokepoint28": 0.06, "chokepoint7": 0.05},
    "ZAF": {"chokepoint7": 0.48, "chokepoint1": 0.14, "chokepoint4": 0.12, "chokepoint8": 0.08, "chokepoint5": 0.08, "chokepoint6": 0.05},
}


def unique(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def prefixed_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def build_route_panel32(countries: list[str]) -> pd.DataFrame:
    if not CHOKEPOINT_WEEKLY.exists():
        raise FileNotFoundError(
            f"Missing {CHOKEPOINT_WEEKLY}. Run scripts/fetch_portwatch_chokepoint_panel.py first."
        )
    weekly = pd.read_csv(CHOKEPOINT_WEEKLY, parse_dates=["week"])
    pwc_source.ROUTE_WEIGHTS.update(EXPANDED_ROUTE_WEIGHTS)
    weights = pwc_source.route_weight_frame(countries, weekly["portid"].drop_duplicates().tolist())
    ROUTE_WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(ROUTE_WEIGHTS_OUT, index=False)

    frames = []
    for prefix, weight_col in [
        ("pwc_route", "route_weight"),
        ("pwc_equal", "route_equal_weight"),
        ("pwc_random", "route_random_weight"),
    ]:
        frames.append(pwc_source.weighted_route_features(weekly, weights, prefix, weight_col))
    route = frames[0]
    for frame in frames[1:]:
        route = route.merge(frame, on=["ISO3", "week"], how="outer")

    global_cols = [
        "cpw_n_container_neg_z52",
        "cpw_n_total_neg_z52",
        "cpw_capacity_container_neg_z52",
        "cpw_capacity_neg_z52",
        "cpw_active_event_count",
        "cpw_red_event_count",
        "cpw_max_alert_score",
        "cpw_active_event_days",
        "cpw_active_red",
    ]
    global_week = (
        weekly.groupby("week", as_index=False)[global_cols]
        .agg(
            {
                "cpw_n_container_neg_z52": "mean",
                "cpw_n_total_neg_z52": "mean",
                "cpw_capacity_container_neg_z52": "mean",
                "cpw_capacity_neg_z52": "mean",
                "cpw_active_event_count": "sum",
                "cpw_red_event_count": "sum",
                "cpw_max_alert_score": "max",
                "cpw_active_event_days": "sum",
                "cpw_active_red": "max",
            }
        )
        .rename(columns={col: f"pwc_global_{col.replace('cpw_', '')}" for col in global_cols})
    )
    route = route.merge(global_week, on="week", how="left")
    pwc_cols = [col for col in route.columns if col.startswith("pwc_")]
    route[pwc_cols] = route[pwc_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    route = route.sort_values(["ISO3", "week"]).reset_index(drop=True)
    ROUTE_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    route.to_csv(ROUTE_WEEKLY, index=False)
    return route


def add_portwatch_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    roots = [
        "active_event_count",
        "red_event_count",
        "max_alert_score",
        "active_event_days",
        "active_red",
        "active_any_disruption",
        "n_container_neg_z52",
        "n_total_neg_z52",
        "capacity_container_neg_z52",
        "capacity_neg_z52",
        "container_utilization_pos_z52",
        "total_utilization_pos_z52",
        "n_container_change_4w",
        "n_total_change_4w",
    ]
    for root in roots:
        route_col = f"pwc_route_{root}"
        equal_col = f"pwc_equal_{root}"
        random_col = f"pwc_random_{root}"
        if route_col in out.columns and equal_col in out.columns:
            out[f"pwc_route_minus_equal_{root}"] = out[route_col] - out[equal_col]
        if route_col in out.columns and random_col in out.columns:
            out[f"pwc_route_minus_random_{root}"] = out[route_col] - out[random_col]

    out["pwc_route_physical_stress"] = (
        out.get("pwc_route_n_container_neg_z52", 0.0)
        + out.get("pwc_route_n_total_neg_z52", 0.0)
        + out.get("pwc_route_capacity_container_neg_z52", 0.0)
        + out.get("pwc_route_red_event_count", 0.0)
        + out.get("pwc_route_active_red", 0.0)
    )
    out["pwc_equal_physical_stress"] = (
        out.get("pwc_equal_n_container_neg_z52", 0.0)
        + out.get("pwc_equal_n_total_neg_z52", 0.0)
        + out.get("pwc_equal_capacity_container_neg_z52", 0.0)
        + out.get("pwc_equal_red_event_count", 0.0)
        + out.get("pwc_equal_active_red", 0.0)
    )
    out["pwc_random_physical_stress"] = (
        out.get("pwc_random_n_container_neg_z52", 0.0)
        + out.get("pwc_random_n_total_neg_z52", 0.0)
        + out.get("pwc_random_capacity_container_neg_z52", 0.0)
        + out.get("pwc_random_red_event_count", 0.0)
        + out.get("pwc_random_active_red", 0.0)
    )
    out["pwc_route_physical_stress_x_shortfall"] = out["pwc_route_physical_stress"] * out[
        "operational_shortfall_12w"
    ].clip(lower=0.0)
    out["pwc_route_physical_stress_x_network"] = out["pwc_route_physical_stress"] * out[
        "network_very_negative_exposure"
    ]
    out["pwc_route_red_x_me_network"] = out.get("pwc_route_red_event_count", 0.0) * out[
        "me_network_strict_very_negative_exposure"
    ]
    out["pwc_route_stress_x_hazard"] = out["pwc_route_physical_stress"] * out["haz_current_low_depth"]
    pwc_cols = [col for col in out.columns if col.startswith("pwc_")]
    out[pwc_cols] = out[pwc_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.sort_values(["week", "ISO3"]).reset_index(drop=True)


def load_or_build_dataset() -> pd.DataFrame:
    if OUTPUT_DATASET.exists():
        return pd.read_csv(OUTPUT_DATASET, parse_dates=["week"])
    base_df = pd.read_csv(BASE_DATASET, parse_dates=["week"])
    base_df = add_hazard_features(base_df)
    countries = sorted(base_df["ISO3"].unique())
    route = build_route_panel32(countries)
    out = base_df.merge(route, on=["ISO3", "week"], how="left")
    pwc_cols = [col for col in out.columns if col.startswith("pwc_")]
    out[pwc_cols] = out.groupby("ISO3", sort=False)[pwc_cols].transform(lambda g: g.ffill().bfill()).fillna(0.0)
    out[SEVERE_TARGET] = (
        out["next_week_container"]
        < (out["rolling_mean_12w"] - 2.0 * out["rolling_std_12w"].replace(0, np.nan))
    ).astype(int)
    out = add_portwatch_features(out)
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DATASET, index=False)
    return out


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base_features = OPERATIONAL_FEATURES + country_features
    raw = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES
    m7 = OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES + TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES + INTERACTION_FEATURES
    all_placebos = TOTAL_EQUAL_PLACEBO_FEATURES + TOTAL_SHUFFLED_PLACEBO_FEATURES + TOTAL_RANDOM_PLACEBO_FEATURES
    route = prefixed_cols(df, "pwc_route_")
    equal = prefixed_cols(df, "pwc_equal_")
    random = prefixed_cols(df, "pwc_random_")
    route_delta = prefixed_cols(df, "pwc_route_minus_")
    global_cp = prefixed_cols(df, "pwc_global_")
    route_interactions = [
        "pwc_route_physical_stress_x_shortfall",
        "pwc_route_physical_stress_x_network",
        "pwc_route_red_x_me_network",
        "pwc_route_stress_x_hazard",
    ]
    return {
        "P32PWC0_operational": unique(base_features),
        "P32PWC1_full_event_network": unique(base_features + m7),
        "P32PWC2_global_chokepoint": unique(base_features + raw + global_cp),
        "P32PWC3_route_official": unique(base_features + raw + route + route_delta),
        "P32PWC4_route_equal_placebo": unique(base_features + raw + equal),
        "P32PWC5_route_random_placebo": unique(base_features + raw + random),
        "P32PWC6_event_route_network": unique(base_features + m7 + route + route_delta),
        "P32PWC7_event_route_hazard": unique(base_features + m7 + route + route_delta + HAZARD_FEATURES),
        "P32PWC8_event_route_interactions": unique(
            base_features + m7 + route + route_delta + HAZARD_FEATURES + route_interactions
        ),
        "P32PWC9_placebo_hazard": unique(base_features + all_placebos + ME_PLACEBO_FEATURES + equal + random + HAZARD_FEATURES),
    }


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in frame.columns and pd.api.types.is_numeric_dtype(frame[feature])]


def make_models(train_y: pd.Series) -> dict[str, tuple[object, str]]:
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    scale_pos_weight = neg / pos
    models: dict[str, tuple[object, str]] = {
        "fast::random_forest": (
            RandomForestClassifier(
                n_estimators=220,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "plain",
        ),
        "advanced::extra_trees": (
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "plain",
        ),
        "advanced::sklearn_gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=14,
                subsample=0.85,
                random_state=42,
            ),
            "sample_weight",
        ),
    }
    try:
        xgboost = __import__("xgboost")
        models["advanced::xgboost"] = (
            xgboost.XGBClassifier(
                n_estimators=220,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                reg_alpha=0.05,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
            "plain",
        )
    except Exception:
        pass
    return models


def fit_model(model, fit_mode: str, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if fit_mode == "plain":
        model.fit(train_x, train_y)
    else:
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
        model.fit(train_x, train_y, sample_weight=weights)


def topk_hits(frame: pd.DataFrame, target_col: str, k: int) -> int:
    return int(frame.sort_values("predicted_probability", ascending=False).head(k)[target_col].sum())


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    rows = []
    predictions = []
    for group_name, raw_features in groups.items():
        features = usable_features(train, raw_features)
        for model_name, (model, fit_mode) in make_models(train[TARGET]).items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            main_scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            severe_pr_auc = average_precision_score(test[SEVERE_TARGET], test_proba)
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
                "severe_pr_auc": severe_pr_auc,
            }
            for k in [10, 25, 50, 100]:
                row[f"main_top{k}_hits"] = topk_hits(pred, TARGET, k)
                row[f"severe_top{k}_hits"] = topk_hits(pred, SEVERE_TARGET, k)
            rows.append(row)
            predictions.append(pred)
    return rows, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "folds": ("fold", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "mean_f1": ("f1", "mean"),
    }
    for prefix in ["main", "severe"]:
        for k in [10, 25, 50, 100]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        metrics.groupby(["feature_group", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits", "mean_severe_pr_auc"], ascending=False)
    )


def alert_budget(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, model, fold), frame in predictions.groupby(["feature_group", "model", "fold"], sort=False):
        for k in [10, 25, 50, 100]:
            top = frame.sort_values("predicted_probability", ascending=False).head(k)
            rows.append(
                {
                    "feature_group": group,
                    "model": model,
                    "fold": fold,
                    "top_k_per_fold": k,
                    "main_hits": int(top[TARGET].sum()),
                    "severe_hits": int(top[SEVERE_TARGET].sum()),
                    "alerts": len(top),
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby(["feature_group", "model", "top_k_per_fold"], as_index=False)
        .agg(main_hits=("main_hits", "sum"), severe_hits=("severe_hits", "sum"), alerts=("alerts", "sum"))
        .assign(
            main_precision=lambda d: d["main_hits"] / d["alerts"],
            severe_precision=lambda d: d["severe_hits"] / d["alerts"],
        )
        .sort_values(["top_k_per_fold", "main_hits", "severe_hits"], ascending=[True, False, False])
    )


def key_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("advanced::sklearn_gradient_boosting", "P32PWC1_full_event_network"),
        ("advanced::sklearn_gradient_boosting", "P32PWC6_event_route_network"),
        ("advanced::sklearn_gradient_boosting", "P32PWC7_event_route_hazard"),
        ("advanced::sklearn_gradient_boosting", "P32PWC8_event_route_interactions"),
        ("advanced::sklearn_gradient_boosting", "P32PWC0_operational"),
        ("advanced::extra_trees", "P32PWC3_route_official"),
        ("advanced::extra_trees", "P32PWC4_route_equal_placebo"),
        ("advanced::extra_trees", "P32PWC5_route_random_placebo"),
        ("advanced::xgboost", "P32PWC6_event_route_network"),
        ("fast::random_forest", "P32PWC3_route_official"),
        ("fast::random_forest", "P32PWC4_route_equal_placebo"),
    ]
    rows = []
    for model, group in specs:
        match = summary.loc[summary["model"].eq(model) & summary["feature_group"].eq(group)]
        if not match.empty:
            rows.append(match.iloc[0].to_dict())
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, alerts: pd.DataFrame, key: pd.DataFrame) -> None:
    top = summary.head(25)
    top25 = alerts.loc[alerts["top_k_per_fold"].eq(25)].head(20)
    route_countries = sorted(df["ISO3"].unique())
    content = f"""# Panel32 PortWatch Official Chokepoint Benchmark

## Purpose

This experiment ports official PortWatch chokepoint exposure to the expanded32 panel. It uses daily official chokepoint vessel/capacity observations and disruption events, mapped to countries with transparent route-exposure priors. Equal and random route-weight placebos are included over the same official chokepoint signals.

## Dataset

- File: `data/processed/multicountry32_portwatch_chokepoint_benchmark.csv`
- Route exposure cache: `data/interim/portwatch_chokepoint_route_exposure_panel32_weekly.csv`
- Route weights: `reports/tables/panel32_portwatch_chokepoint_route_weights.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}
- Country set: `{", ".join(route_countries)}`

## Evaluation

- Temporal rolling-origin validation with test years 2023, 2024, and 2025.
- Thresholds are selected on the immediately preceding validation year only.
- Main ranking metric: PR-AUC; alert-budget top-k hits are secondary.
- Model families: Logistic/RF plus advanced sklearn GB, ExtraTrees, and optional XGBoost/LightGBM/CatBoost.
- Placebos: equal and fixed random route weights over the same official chokepoint series.

## Top Mean PR-AUC Results

{top[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "severe_top50_hits"]].to_markdown(index=False)}

## Top-25 Alert Leaders

{top25[["feature_group", "model", "main_hits", "severe_hits", "alerts", "main_precision", "severe_precision"]].to_markdown(index=False)}

## Key Contrasts

{key[["feature_group", "model", "mean_main_pr_auc", "mean_severe_pr_auc", "main_top10_hits", "main_top25_hits", "severe_top10_hits", "severe_top25_hits"]].to_markdown(index=False)}

## Reading

This branch should be promoted only if official route-weighted chokepoint features improve over the expanded32 fixed GB full-event/network reference (`0.2095` main PR-AUC, main top-25 `33`, severe top-25 `23`) and separate from equal/random route placebos. If route placebos remain competitive, use the result as physical-exposure/mechanism evidence rather than a predictive network-weighting claim.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_dataset()
    rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(fold, df)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(metrics)
    alerts = alert_budget(predictions)
    key = key_contrasts(summary)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    alerts.to_csv(ALERTS, index=False)
    key.to_csv(KEY, index=False)
    write_report(df, summary, alerts, key)
    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved route exposure: {ROUTE_WEEKLY}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
