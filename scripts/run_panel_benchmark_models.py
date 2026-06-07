from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_benchmark_results.md"
TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


FOLDS = [
    TemporalFold("test_2023", "2021-01-01", "2022-01-01", "2022-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
    TemporalFold("test_2024", "2021-01-01", "2023-01-01", "2023-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
    TemporalFold("test_2025", "2021-01-01", "2024-01-01", "2024-01-01", "2025-01-01", "2025-01-01", "2026-01-01"),
]

OPERATIONAL_FEATURES = [
    "portcalls_container",
    "lag_container_1w",
    "lag_container_2w",
    "lag_container_4w",
    "rolling_mean_container_4w",
    "rolling_std_container_4w",
    "rolling_mean_container_8w",
    "rolling_std_container_8w",
    "rolling_mean_container_12w",
    "rolling_std_container_12w",
    "rolling_change_container_4w",
    "operational_shortfall_12w",
    "negative_trend_4w",
    "month",
    "quarter",
]

OWN_NEWS_FEATURES = [
    "article_count",
    "avg_tone",
    "negative_article_share",
    "very_negative_article_share",
    "trade_transport_count",
    "risk_theme_count",
]

EXTERNAL_UNWEIGHTED_FEATURES = [
    "external_article_count",
    "external_avg_tone",
    "external_negative_article_share",
    "external_very_negative_article_share",
    "external_trade_transport_count",
    "external_risk_theme_count",
]

TOTAL_NETWORK_FEATURES = [
    "network_partner_article_count",
    "network_negative_exposure",
    "network_very_negative_exposure",
    "network_trade_transport_exposure",
    "network_risk_theme_exposure",
]

TOTAL_EQUAL_PLACEBO_FEATURES = [
    "network_partner_article_count",
    "equal_negative_exposure",
    "equal_very_negative_exposure",
    "equal_trade_transport_exposure",
    "equal_risk_theme_exposure",
]

TOTAL_SHUFFLED_PLACEBO_FEATURES = [
    "network_partner_article_count",
    "shuffled_negative_exposure",
    "shuffled_very_negative_exposure",
    "shuffled_trade_transport_exposure",
    "shuffled_risk_theme_exposure",
]

TOTAL_RANDOM_PLACEBO_FEATURES = [
    "network_partner_article_count",
    "random_negative_exposure",
    "random_very_negative_exposure",
    "random_trade_transport_exposure",
    "random_risk_theme_exposure",
]

ME_NETWORK_FEATURES = [
    "me_network_strict_very_negative_exposure",
    "me_network_strict_article_count",
]

ME_PLACEBO_FEATURES = [
    "me_equal_strict_very_negative_exposure",
    "me_shuffled_strict_very_negative_exposure",
    "me_random_strict_very_negative_exposure",
    "me_network_strict_article_count",
]

INTERACTION_FEATURES = [
    "network_minus_equal_very_negative",
    "network_to_equal_very_negative_ratio",
    "shortfall_x_network_very_negative",
    "trend_x_network_very_negative",
    "shortfall_x_external_very_negative",
    "trend_x_external_very_negative",
    "me_network_minus_equal_strict",
    "shortfall_x_me_network_strict",
    "trend_x_me_network_strict",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["week"])
    return df.sort_values(["week", "ISO3"]).reset_index(drop=True)


def split_fold(df: pd.DataFrame, fold: TemporalFold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= fold.train_start) & (df["week"] < fold.train_end)].copy()
    validation = df[(df["week"] >= fold.validation_start) & (df["week"] < fold.validation_end)].copy()
    test = df[(df["week"] >= fold.test_start) & (df["week"] < fold.test_end)].copy()
    return train, validation, test


def add_country_dummies(*frames: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    countries = sorted(frames[0]["ISO3"].unique())
    out_frames = []
    for frame in frames:
        out = frame.copy()
        for country in countries:
            out[f"country_{country}"] = (out["ISO3"] == country).astype(int)
        out_frames.append(out)
    return out_frames, [f"country_{country}" for country in countries]


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    return {
        "M1_operational": base,
        "M2_own_country_news": base + OWN_NEWS_FEATURES,
        "M3_external_unweighted_events": base + EXTERNAL_UNWEIGHTED_FEATURES,
        "M4_total_import_network": base + TOTAL_NETWORK_FEATURES,
        "M5_me_strict_network": base + ME_NETWORK_FEATURES,
        "M6a_total_equal_placebo": base + TOTAL_EQUAL_PLACEBO_FEATURES,
        "M6b_total_shuffled_placebo": base + TOTAL_SHUFFLED_PLACEBO_FEATURES,
        "M6c_total_random_placebo": base + TOTAL_RANDOM_PLACEBO_FEATURES,
        "M6d_me_placebo_bundle": base + ME_PLACEBO_FEATURES,
        "M7_full_event_network": base
        + OWN_NEWS_FEATURES
        + EXTERNAL_UNWEIGHTED_FEATURES
        + TOTAL_NETWORK_FEATURES
        + ME_NETWORK_FEATURES
        + INTERACTION_FEATURES,
    }


def make_models() -> dict[str, object]:
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def select_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.02, 0.80, 79)
    scores = [
        f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return np.nan
    top_k = min(k, len(y_true))
    order = np.argsort(-proba)[:top_k]
    return float(y_true[order].mean())


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_at_10": precision_at_k(y_true, proba, 10),
        "precision_at_25": precision_at_k(y_true, proba, 25),
        "precision_at_50": precision_at_k(y_true, proba, 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_fold(fold: TemporalFold, df: pd.DataFrame) -> tuple[list[dict], list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    metric_rows = []
    prediction_rows = []
    interpretation_frames = []

    for group_name, features in feature_groups.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise KeyError(f"{group_name} missing features: {missing}")

        for model_name, model in make_models().items():
            model.fit(train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            y_test = test[TARGET].to_numpy()
            scores = evaluate_predictions(y_test, test_proba, threshold)

            metric_rows.append(
                {
                    "fold": fold.name,
                    "feature_group": group_name,
                    "model": model_name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "train_positives": int(train[TARGET].sum()),
                    "validation_positives": int(validation[TARGET].sum()),
                    "test_positives": int(test[TARGET].sum()),
                    "selected_threshold": threshold,
                    "validation_f1": val_f1,
                    **scores,
                }
            )

            pred_frame = test[["week", "ISO3", "country", TARGET]].copy()
            pred_frame["fold"] = fold.name
            pred_frame["feature_group"] = group_name
            pred_frame["model"] = model_name
            pred_frame["predicted_probability"] = test_proba
            pred_frame["selected_threshold"] = threshold
            pred_frame["prediction"] = (test_proba >= threshold).astype(int)
            prediction_rows.extend(pred_frame.to_dict("records"))
            interpretation_frames.append(extract_interpretation(fold.name, group_name, model_name, model, features))

    return metric_rows, prediction_rows, interpretation_frames


def extract_interpretation(
    fold_name: str,
    group_name: str,
    model_name: str,
    model,
    features: list[str],
) -> pd.DataFrame:
    if model_name == "logistic":
        values = model.named_steps["model"].coef_[0]
        value_type = "coefficient"
    else:
        values = model.feature_importances_
        value_type = "feature_importance"

    return pd.DataFrame(
        {
            "fold": fold_name,
            "feature_group": group_name,
            "model": model_name,
            "feature": features,
            "value_type": value_type,
            "value": values,
        }
    ).sort_values("value", key=lambda s: s.abs(), ascending=False)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["feature_group", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_precision_at_25=("precision_at_25", "mean"),
            total_tp=("tp", "sum"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            total_tn=("tn", "sum"),
        )
        .sort_values(["mean_pr_auc", "mean_roc_auc"], ascending=False)
    )


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, df: pd.DataFrame) -> None:
    top = summary.head(12)
    rf = summary[summary["model"].eq("random_forest")].copy()
    comparison = rf[
        rf["feature_group"].isin(
            [
                "M1_operational",
                "M2_own_country_news",
                "M3_external_unweighted_events",
                "M4_total_import_network",
                "M5_me_strict_network",
                "M6a_total_equal_placebo",
                "M6b_total_shuffled_placebo",
                "M6c_total_random_placebo",
                "M7_full_event_network",
            ]
        )
    ].sort_values("mean_pr_auc", ascending=False)

    content = f"""# Panel Benchmark Results

## Dataset

- File: `data/processed/multicountry_container_event_network_benchmark.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}

## Evaluation

- Temporal rolling-origin validation.
- Test folds: 2023, 2024, and 2025.
- Thresholds are selected on the immediately preceding validation year only.
- Main ranking metric: PR-AUC, because abnormal weeks are rare.
- Models: balanced Logistic Regression and Random Forest.

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Random Forest Benchmark Comparison

{comparison[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## Interpretation

This report should be read as the formal panel benchmark, not as a causal estimate. The network layer is an exposure-mapping mechanism: it asks whether external event pressure becomes more useful when routed through observed trade dependencies. If network-weighted variants outperform unweighted and placebo variants across folds, that supports predictive value. If they do not, the defensible contribution is still a reproducible event-network benchmark and evidence about when simple network weighting is redundant.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    metric_rows = []
    prediction_rows = []
    interpretation_frames = []

    for fold in FOLDS:
        fold_metrics, fold_predictions, fold_interpretation = run_fold(fold, df)
        metric_rows.extend(fold_metrics)
        prediction_rows.extend(fold_predictions)
        interpretation_frames.extend(fold_interpretation)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    interpretation = pd.concat(interpretation_frames, ignore_index=True)
    summary = summarize_metrics(metrics)

    metrics.to_csv(TABLE_DIR / "panel_benchmark_metrics_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_benchmark_summary.csv", index=False)
    predictions.to_csv(TABLE_DIR / "panel_benchmark_predictions.csv", index=False)
    interpretation.to_csv(TABLE_DIR / "panel_benchmark_model_interpretation.csv", index=False)
    write_report(metrics, summary, df)

    print(f"Saved fold metrics: {TABLE_DIR / 'panel_benchmark_metrics_by_fold.csv'}")
    print(f"Saved summary: {TABLE_DIR / 'panel_benchmark_summary.csv'}")
    print(f"Saved predictions: {TABLE_DIR / 'panel_benchmark_predictions.csv'}")
    print(f"Saved interpretation: {TABLE_DIR / 'panel_benchmark_model_interpretation.csv'}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
