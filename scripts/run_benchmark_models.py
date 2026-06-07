from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "processed" / "japan_container_event_network_benchmark.csv"
)
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
RESULTS_MD = PROJECT_ROOT / "reports" / "benchmark_results.md"

TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42

FEATURE_GROUPS = {
    "M1_operational": [
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
        "month",
        "quarter",
    ],
    "M2_simple_news": [
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
        "month",
        "quarter",
        "news_article_count",
        "news_avg_tone",
        "unweighted_negative_exposure",
        "unweighted_very_negative_exposure",
        "news_trade_transport_count",
        "news_risk_theme_count",
    ],
    "M3_unweighted_me_event": [
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
        "month",
        "quarter",
        "me_strict_unweighted_exposure",
        "me_strict_article_count",
    ],
    "M4_total_import_network": [
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
        "month",
        "quarter",
        "total_network_negative_exposure",
        "total_network_very_negative_exposure",
        "total_network_risk_theme_exposure",
        "total_network_trade_transport_exposure",
    ],
    "M5_me_network": [
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
        "month",
        "quarter",
        "me_strict_network_exposure",
        "me_strict_article_count",
    ],
    "M6_equal_placebo": [
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
        "month",
        "quarter",
        "me_strict_equal_exposure",
        "me_strict_article_count",
    ],
    "M6_shuffled_placebo": [
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
        "month",
        "quarter",
        "me_strict_shuffled_exposure",
        "me_strict_article_count",
    ],
    "M6_random_placebo": [
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
        "month",
        "quarter",
        "me_strict_random_exposure",
        "me_strict_article_count",
    ],
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values("week").reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= "2021-01-01") & (df["week"] < "2024-01-01")].copy()
    validation = df[(df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, validation, test


def make_models() -> dict[str, object]:
    return {
        "logistic": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def select_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        scores.append(f1_score(y_true, pred, zero_division=0))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def score_with_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric_fn,
    n_boot: int = 500,
) -> tuple[float, float, float]:
    point = metric_fn(y_true, proba)
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], proba[idx]))
    if not values:
        return float(point), np.nan, np.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return float(point), float(low), float(high)


def evaluate_model(
    model_name: str,
    model,
    group_name: str,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame]:
    model.fit(train[features], train[TARGET])

    val_proba = model.predict_proba(validation[features])[:, 1]
    threshold, val_f1 = select_threshold(validation[TARGET], val_proba)

    test_proba = model.predict_proba(test[features])[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    y_test = test[TARGET].to_numpy()
    roc, roc_low, roc_high = score_with_ci(y_test, test_proba, roc_auc_score)
    pr, pr_low, pr_high = score_with_ci(y_test, test_proba, average_precision_score)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    metrics = {
        "feature_group": group_name,
        "model": model_name,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_positives": int(train[TARGET].sum()),
        "validation_positives": int(validation[TARGET].sum()),
        "test_positives": int(test[TARGET].sum()),
        "selected_threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "roc_auc": roc,
        "roc_auc_ci_low": roc_low,
        "roc_auc_ci_high": roc_high,
        "pr_auc": pr,
        "pr_auc_ci_low": pr_low,
        "pr_auc_ci_high": pr_high,
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
    }
    confusion = {
        "feature_group": group_name,
        "model": model_name,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    interpretation = extract_interpretation(model_name, model, group_name, features)
    return metrics, confusion, interpretation


def extract_interpretation(
    model_name: str,
    model,
    group_name: str,
    features: list[str],
) -> pd.DataFrame:
    if model_name == "logistic":
        values = model.named_steps["model"].coef_[0]
        kind = "coefficient"
    elif model_name == "random_forest":
        values = model.feature_importances_
        kind = "feature_importance"
    else:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "feature_group": group_name,
            "model": model_name,
            "feature": features,
            "value_type": kind,
            "value": values,
        }
    ).sort_values("value", key=lambda s: s.abs(), ascending=False)


def write_report(metrics: pd.DataFrame, path: Path) -> None:
    best = metrics.sort_values("pr_auc", ascending=False).head(10)
    content = f"""# Benchmark Results

## Dataset Split

- Train: 2021-2023
- Validation: 2024
- Test: 2025
- Threshold selection: validation F1 only
- Test metrics: reported once using the validation-selected threshold

## Status

These are benchmark script outputs from the currently cached 2021-2025 GDELT window. They should be interpreted with care because the test split contains only five positive labels.

## Top Test PR-AUC Results

{best[["feature_group", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "selected_threshold"]].to_markdown(index=False)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    df = load_dataset(Path(args.dataset))
    train, validation, test = temporal_split(df)

    metrics_rows = []
    confusion_rows = []
    interpretation_frames = []

    for group_name, features in FEATURE_GROUPS.items():
        for model_name, model in make_models().items():
            metrics, confusion, interpretation = evaluate_model(
                model_name,
                model,
                group_name,
                features,
                train,
                validation,
                test,
            )
            metrics_rows.append(metrics)
            confusion_rows.append(confusion)
            interpretation_frames.append(interpretation)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    confusion_df = pd.DataFrame(confusion_rows)
    interpretation_df = pd.concat(interpretation_frames, ignore_index=True)

    metrics_df.to_csv(TABLE_DIR / "benchmark_metrics.csv", index=False)
    confusion_df.to_csv(TABLE_DIR / "confusion_matrices.csv", index=False)
    interpretation_df.to_csv(TABLE_DIR / "model_interpretation.csv", index=False)
    write_report(metrics_df, RESULTS_MD)

    print(f"Saved metrics: {TABLE_DIR / 'benchmark_metrics.csv'}")
    print(f"Saved confusion matrices: {TABLE_DIR / 'confusion_matrices.csv'}")
    print(f"Saved interpretation: {TABLE_DIR / 'model_interpretation.csv'}")
    print(f"Saved report: {RESULTS_MD}")
    print(metrics_df.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
