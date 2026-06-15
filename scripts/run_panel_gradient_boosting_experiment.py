from __future__ import annotations

from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    FOLDS,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    load_dataset,
    make_feature_groups,
    select_threshold,
    split_fold,
    summarize_metrics,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_gradient_boosting_experiment.md"
RANDOM_SEED = 42


def make_models() -> dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            early_stopping=False,
            random_state=RANDOM_SEED,
        ),
    }


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    metric_rows = []
    prediction_rows = []

    sample_weight = compute_sample_weight(class_weight="balanced", y=train[TARGET])

    for group_name, features in feature_groups.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise KeyError(f"{group_name} missing features: {missing}")

        for model_name, model in make_models().items():
            model.fit(train[features], train[TARGET], sample_weight=sample_weight)
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

    return metric_rows, prediction_rows


def build_event_layer_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = metrics.loc[
        metrics["feature_group"].eq("M1_operational"),
        ["model", "fold", "pr_auc", "roc_auc", "f1"],
    ].rename(
        columns={
            "pr_auc": "m1_pr_auc",
            "roc_auc": "m1_roc_auc",
            "f1": "m1_f1",
        }
    )
    for group in [
        "M2_own_country_news",
        "M3_external_unweighted_events",
        "M4_total_import_network",
        "M5_me_strict_network",
    ]:
        comparison = metrics.loc[
            metrics["feature_group"].eq(group),
            ["model", "fold", "pr_auc", "roc_auc", "f1"],
        ].merge(baselines, on=["model", "fold"], how="inner")
        comparison["feature_group"] = group
        comparison["pr_auc_delta_vs_m1"] = comparison["pr_auc"] - comparison["m1_pr_auc"]
        comparison["roc_auc_delta_vs_m1"] = comparison["roc_auc"] - comparison["m1_roc_auc"]
        comparison["f1_delta_vs_m1"] = comparison["f1"] - comparison["m1_f1"]
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True)


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, df: pd.DataFrame) -> None:
    top = summary.head(15)
    model_sections = []
    for model_name in sorted(summary["model"].unique()):
        model_summary = summary.loc[summary["model"].eq(model_name)].sort_values("mean_pr_auc", ascending=False)
        model_sections.append(
            f"""## {model_name}

{model_summary[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}
"""
        )

    delta_summary = (
        deltas.groupby(["model", "feature_group"], as_index=False)
        .agg(
            mean_pr_auc_delta_vs_m1=("pr_auc_delta_vs_m1", "mean"),
            folds_beating_m1=("pr_auc_delta_vs_m1", lambda s: int((s > 0).sum())),
            mean_roc_auc_delta_vs_m1=("roc_auc_delta_vs_m1", "mean"),
        )
        .sort_values(["model", "mean_pr_auc_delta_vs_m1"], ascending=[True, False])
    )

    content = f"""# Panel Gradient Boosting Experiment

## Purpose

This exploratory experiment adds dependency-safe sklearn gradient boosting model families to the existing panel benchmark without changing the dataset, feature groups, temporal folds, or validation-selected threshold protocol.

## Dataset And Protocol

- Dataset rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Test folds: 2023, 2024, and 2025 rolling-origin folds
- Main metric: PR-AUC
- Sample weighting: balanced training sample weights
- No threshold tuning on test labels

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

{"".join(model_sections)}
## Event And Network Layer Deltas Versus M1

{delta_summary.to_markdown(index=False)}

## Reading

This is an exploratory model-family breadth check. A publishable upgrade would require the gradient boosting results to be integrated into the main benchmark report only after the ranking pattern is judged stable and useful. If M2 or M3 does not beat M1 consistently, Gate 1 remains unresolved even if network variants occasionally improve average PR-AUC.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    metric_rows = []
    prediction_rows = []
    for fold in FOLDS:
        fold_metrics, fold_predictions = run_fold(fold, df)
        metric_rows.extend(fold_metrics)
        prediction_rows.extend(fold_predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_metrics(metrics)
    deltas = build_event_layer_deltas(metrics)

    metrics.to_csv(TABLE_DIR / "panel_gradient_boosting_metrics_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_gradient_boosting_summary.csv", index=False)
    predictions.to_csv(TABLE_DIR / "panel_gradient_boosting_predictions.csv", index=False)
    deltas.to_csv(TABLE_DIR / "panel_gradient_boosting_event_layer_deltas.csv", index=False)
    write_report(metrics, summary, deltas, df)

    print(f"Saved fold metrics: {TABLE_DIR / 'panel_gradient_boosting_metrics_by_fold.csv'}")
    print(f"Saved summary: {TABLE_DIR / 'panel_gradient_boosting_summary.csv'}")
    print(f"Saved predictions: {TABLE_DIR / 'panel_gradient_boosting_predictions.csv'}")
    print(f"Saved deltas: {TABLE_DIR / 'panel_gradient_boosting_event_layer_deltas.csv'}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
