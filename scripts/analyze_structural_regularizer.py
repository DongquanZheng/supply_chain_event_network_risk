from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark_models import (  # noqa: E402
    DEFAULT_DATASET,
    FEATURE_GROUPS,
    TARGET,
    load_dataset,
    make_models,
    select_threshold,
    temporal_split,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REGULARIZER_TABLE = TABLE_DIR / "structural_regularizer_diagnostics.csv"
FALSE_ALERT_TABLE = TABLE_DIR / "m2_suppressed_false_alerts.csv"
ALERT_BUDGET_TABLE = TABLE_DIR / "alert_budget_diagnostics.csv"
REGULARIZER_MD = PROJECT_ROOT / "reports" / "structural_regularizer_diagnostics.md"

GROUPS = [
    "M2_simple_news",
    "M3_unweighted_me_event",
    "M5_me_network",
    "M6_equal_placebo",
    "M6_shuffled_placebo",
    "M6_random_placebo",
]


def threshold_for_min_recall(
    y_true: pd.Series,
    proba: np.ndarray,
    min_recall: float = 0.8,
) -> float:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), proba])
    candidates = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        alerts = int(pred.sum())
        if recall >= min_recall:
            candidates.append((threshold, precision, f1, alerts))

    if not candidates:
        return float(0.5)

    # Keep recall at the requested floor, then prefer fewer alerts.
    candidates = sorted(candidates, key=lambda x: (-x[1], -x[2], x[3], -x[0]))
    return float(candidates[0][0])


def evaluate_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "alerts": int(pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
        "mean_probability": float(np.mean(proba)),
        "mean_probability_on_negatives": float(np.mean(proba[y_true == 0])),
        "mean_probability_on_positives": float(np.mean(proba[y_true == 1])),
        "mean_probability_on_false_positives": (
            float(np.mean(proba[(y_true == 0) & (pred == 1)]))
            if fp
            else np.nan
        ),
    }


def collect_group_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation, test = temporal_split(df)
    metric_rows = []
    prediction_frames = []

    for group in GROUPS:
        features = FEATURE_GROUPS[group]
        model = make_models()["random_forest"]
        model.fit(train[features], train[TARGET])

        val_proba = model.predict_proba(validation[features])[:, 1]
        f1_threshold, _ = select_threshold(validation[TARGET], val_proba)
        fixed_recall_threshold = threshold_for_min_recall(
            validation[TARGET],
            val_proba,
            min_recall=0.8,
        )

        test_proba = model.predict_proba(test[features])[:, 1]
        y_test = test[TARGET].to_numpy()

        for protocol, threshold in [
            ("validation_f1", f1_threshold),
            ("validation_min_recall_0.8", fixed_recall_threshold),
        ]:
            row = evaluate_at_threshold(y_test, test_proba, threshold)
            row.update({"feature_group": group, "protocol": protocol})
            metric_rows.append(row)

        predictions = test[
            ["week", "portcalls_container", "next_week_container", TARGET]
        ].copy()
        predictions["feature_group"] = group
        predictions["probability"] = test_proba
        predictions["f1_threshold"] = f1_threshold
        predictions["fixed_recall_threshold"] = fixed_recall_threshold
        predictions["f1_prediction"] = (test_proba >= f1_threshold).astype(int)
        predictions["fixed_recall_prediction"] = (
            test_proba >= fixed_recall_threshold
        ).astype(int)
        prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def suppressed_false_alerts(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    m2 = predictions[predictions["feature_group"].eq("M2_simple_news")]
    m5 = predictions[predictions["feature_group"].eq("M5_me_network")]

    paired = m2.merge(
        m5,
        on=["week", "portcalls_container", "next_week_container", TARGET],
        suffixes=("_m2", "_m5"),
    )
    suppressed = paired[
        paired[TARGET].eq(0)
        & paired["f1_prediction_m2"].eq(1)
        & paired["f1_prediction_m5"].eq(0)
    ].copy()

    feature_cols = [
        "week",
        "news_article_count",
        "news_avg_tone",
        "unweighted_negative_exposure",
        "unweighted_very_negative_exposure",
        "me_strict_unweighted_exposure",
        "me_strict_network_exposure",
        "me_strict_shuffled_exposure",
        "me_strict_random_exposure",
        "me_strict_article_count",
    ]
    suppressed = suppressed.merge(df[feature_cols], on="week", how="left")
    return suppressed.sort_values("week")


def alert_budget_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in predictions.groupby("feature_group"):
        y_true = part[TARGET].to_numpy()
        proba = part["probability"].to_numpy()

        for alert_budget in [10, 13, 16, 20]:
            pred = np.zeros(len(part), dtype=int)
            pred[np.argsort(-proba)[:alert_budget]] = 1
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "feature_group": group,
                    "diagnostic": f"top_{alert_budget}_alerts",
                    "threshold": np.nan,
                    "alerts": int(pred.sum()),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": precision_score(y_true, pred, zero_division=0),
                    "recall": recall_score(y_true, pred, zero_division=0),
                    "f1": f1_score(y_true, pred, zero_division=0),
                }
            )

        oracle_candidates = []
        for threshold in np.unique(proba):
            pred = (proba >= threshold).astype(int)
            recall = recall_score(y_true, pred, zero_division=0)
            if recall >= 0.8:
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                oracle_candidates.append(
                    {
                        "feature_group": group,
                        "diagnostic": "test_oracle_min_fp_at_recall_0.8",
                        "threshold": float(threshold),
                        "alerts": int(pred.sum()),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "precision": precision_score(y_true, pred, zero_division=0),
                        "recall": recall,
                        "f1": f1_score(y_true, pred, zero_division=0),
                    }
                )
        if oracle_candidates:
            rows.append(
                sorted(
                    oracle_candidates,
                    key=lambda x: (x["fp"], -x["tp"], x["alerts"]),
                )[0]
            )

    return pd.DataFrame(rows)


def write_report(
    metrics: pd.DataFrame,
    suppressed: pd.DataFrame,
    alert_budget: pd.DataFrame,
    path: Path,
) -> None:
    f1_metrics = metrics[metrics["protocol"].eq("validation_f1")].copy()
    fixed_metrics = metrics[metrics["protocol"].eq("validation_min_recall_0.8")].copy()
    oracle = alert_budget[
        alert_budget["diagnostic"].eq("test_oracle_min_fp_at_recall_0.8")
    ].copy()
    top13 = alert_budget[alert_budget["diagnostic"].eq("top_13_alerts")].copy()

    def row(group: str, source: pd.DataFrame) -> pd.Series:
        return source[source["feature_group"].eq(group)].iloc[0]

    m2 = row("M2_simple_news", f1_metrics)
    m5 = row("M5_me_network", f1_metrics)
    m6_random = row("M6_random_placebo", f1_metrics)

    false_positive_reduction = m2["fp"] - m5["fp"]
    alert_reduction = m2["alerts"] - m5["alerts"]

    content = f"""# Structural Regularizer Diagnostics

## Question

Does the network layer behave like a structural relevance filter that suppresses NLP-driven false alerts, rather than merely adding another predictive feature?

## Main F1-Threshold Result

At validation-selected F1 thresholds:

- `M2_simple_news` alerts: {int(m2["alerts"])}; true positives: {int(m2["tp"])}; false positives: {int(m2["fp"])}
- `M5_me_network` alerts: {int(m5["alerts"])}; true positives: {int(m5["tp"])}; false positives: {int(m5["fp"])}
- M5 reduces false positives by {int(false_positive_reduction)} and total alerts by {int(alert_reduction)}, but misses one positive that M2 captures.

This supports a selective-filter interpretation, not a blanket predictive-improvement claim.

## Placebo Challenge

The random placebo network has {int(m6_random["alerts"])} alerts, {int(m6_random["tp"])} true positives, and {int(m6_random["fp"])} false positives. Because the placebo is close to the true network in this test split, the current evidence is not strong enough to claim that the true trade network uniquely causes the false-positive reduction.

## Fixed-Recall Diagnostic

The table below selects thresholds on validation data to target at least 0.8 recall and then evaluates them on the 2025 test split.

{fixed_metrics[["feature_group", "alerts", "tp", "fp", "fn", "precision", "recall", "f1", "threshold"]].to_markdown(index=False)}

## Alert-Budget Diagnostic

The table below compares models when each is allowed exactly 13 test alerts. This is a diagnostic only; it uses a fixed alert budget rather than a validation-selected threshold.

{top13[["feature_group", "alerts", "tp", "fp", "fn", "precision", "recall", "f1"]].to_markdown(index=False)}

## Test-Oracle Threshold Diagnostic

The table below asks, after seeing the test labels, what is the smallest number of false positives each score can achieve while preserving at least 0.8 test recall. This is not a valid final benchmark metric, but it is useful for detecting whether M5's false-positive reduction is merely a threshold artifact.

{oracle[["feature_group", "alerts", "tp", "fp", "fn", "precision", "recall", "f1", "threshold"]].to_markdown(index=False)}

## Calibration and Over-Alerting

{f1_metrics[["feature_group", "alerts", "tp", "fp", "fn", "precision", "recall", "f1", "brier", "mean_probability_on_false_positives"]].to_markdown(index=False)}

## M2 False Alerts Suppressed By M5

- Count: {len(suppressed)}

{suppressed[["week", "portcalls_container", "next_week_container", "probability_m2", "probability_m5", "unweighted_very_negative_exposure", "me_strict_network_exposure", "me_strict_shuffled_exposure", "me_strict_random_exposure"]].to_markdown(index=False)}

## Current Conclusion

The evidence supports only a weak structural-regularizer claim. M5 suppresses some M2 false alerts under validation-selected F1 thresholds, but alert-budget and test-oracle diagnostics show that M2 can also be made selective. Therefore, the current benchmark does not yet prove that the true trade network removes spurious NLP overfitting. The next methodological need is stronger validation of network-specific filtering: rolling-origin stability, negative-control outcomes, and stricter partner/channel event definitions.
"""
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    metrics, predictions = collect_group_predictions(df)
    suppressed = suppressed_false_alerts(predictions, df)
    alert_budget = alert_budget_diagnostics(predictions)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REGULARIZER_TABLE, index=False)
    suppressed.to_csv(FALSE_ALERT_TABLE, index=False)
    alert_budget.to_csv(ALERT_BUDGET_TABLE, index=False)
    write_report(metrics, suppressed, alert_budget, REGULARIZER_MD)

    print(f"Saved diagnostics: {REGULARIZER_TABLE}")
    print(f"Saved suppressed false alerts: {FALSE_ALERT_TABLE}")
    print(f"Saved alert-budget diagnostics: {ALERT_BUDGET_TABLE}")
    print(f"Saved report: {REGULARIZER_MD}")
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
