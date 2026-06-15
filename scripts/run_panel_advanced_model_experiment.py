from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score
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
REPORT = PROJECT_ROOT / "reports" / "panel_advanced_model_experiment.md"
METRICS = TABLE_DIR / "panel_advanced_model_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel_advanced_model_summary.csv"
PREDICTIONS = TABLE_DIR / "panel_advanced_model_predictions.csv"
DELTAS = TABLE_DIR / "panel_advanced_model_bootstrap_deltas.csv"
KEY_DELTAS = TABLE_DIR / "panel_advanced_model_key_deltas.csv"
BASELINE_GROUP = "M1_operational"
RANDOM_SEED = 42


def optional_imports() -> dict[str, object]:
    modules = {}
    for name in ["xgboost", "lightgbm", "catboost"]:
        try:
            modules[name] = __import__(name)
        except Exception:
            continue
    return modules


def make_models(train_y: pd.Series) -> dict[str, tuple[object, str]]:
    modules = optional_imports()
    pos = max(int(train_y.sum()), 1)
    neg = max(int(len(train_y) - train_y.sum()), 1)
    scale_pos_weight = neg / pos

    models: dict[str, tuple[object, str]] = {
        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
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

    if "xgboost" in modules:
        models["xgboost"] = (
            modules["xgboost"].XGBClassifier(
                n_estimators=260,
                learning_rate=0.035,
                max_depth=3,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                reg_alpha=0.05,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                tree_method="hist",
            ),
            "plain",
        )

    if "lightgbm" in modules:
        models["lightgbm"] = (
            modules["lightgbm"].LGBMClassifier(
                n_estimators=260,
                learning_rate=0.035,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            "plain",
        )

    if "catboost" in modules:
        models["catboost"] = (
            modules["catboost"].CatBoostClassifier(
                iterations=260,
                learning_rate=0.035,
                depth=4,
                l2_leaf_reg=5.0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="Balanced",
                random_seed=RANDOM_SEED,
                verbose=False,
                allow_writing_files=False,
            ),
            "plain",
        )

    return models


def fit_model(model, fit_mode: str, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if fit_mode == "sample_weight":
        weights = compute_sample_weight(class_weight="balanced", y=train_y)
        model.fit(train_x, train_y, sample_weight=weights)
    else:
        model.fit(train_x, train_y)


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    prediction_frames = []
    for group_name, features in feature_groups.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise KeyError(f"{group_name} missing features: {missing}")

        for model_name, (model, fit_mode) in make_models(train[TARGET]).items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)

            rows.append(
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

            predictions = test[["ISO3", "country", "week", TARGET]].copy()
            predictions["fold"] = fold.name
            predictions["feature_group"] = group_name
            predictions["model"] = model_name
            predictions["predicted_probability"] = test_proba
            predictions["selected_threshold"] = threshold
            predictions["validation_f1"] = val_f1
            prediction_frames.append(predictions)

    return rows, prediction_frames


def paired_bootstrap_delta(
    focus: pd.DataFrame,
    baseline: pd.DataFrame,
    metric_fn,
    n_boot: int = 1500,
) -> tuple[float, float, float, float]:
    merged = focus.merge(
        baseline,
        on=["fold", "week", "ISO3", TARGET],
        suffixes=("_focus", "_baseline"),
    )
    y = merged[TARGET].to_numpy()
    focus_score = merged["predicted_probability_focus"].to_numpy()
    base_score = merged["predicted_probability_baseline"].to_numpy()
    point = metric_fn(y, focus_score) - metric_fn(y, base_score)

    rng = np.random.default_rng(RANDOM_SEED)
    deltas = []
    n = len(merged)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(metric_fn(y[idx], focus_score[idx]) - metric_fn(y[idx], base_score[idx]))

    delta_array = np.array(deltas)
    low, high = np.percentile(delta_array, [2.5, 97.5])
    return float(point), float(low), float(high), float(np.mean(delta_array > 0))


def bootstrap_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model_predictions in predictions.groupby("model"):
        baseline = model_predictions.loc[model_predictions["feature_group"].eq(BASELINE_GROUP)].copy()
        for group in sorted(model_predictions["feature_group"].unique()):
            if group == BASELINE_GROUP:
                continue
            focus = model_predictions.loc[model_predictions["feature_group"].eq(group)].copy()
            pr_point, pr_low, pr_high, pr_p = paired_bootstrap_delta(
                focus, baseline, average_precision_score
            )
            roc_point, roc_low, roc_high, roc_p = paired_bootstrap_delta(focus, baseline, roc_auc_score)
            rows.append(
                {
                    "model": model_name,
                    "feature_group": group,
                    "baseline_group": BASELINE_GROUP,
                    "pooled_pr_auc_delta": pr_point,
                    "pr_delta_ci_low": pr_low,
                    "pr_delta_ci_high": pr_high,
                    "pr_delta_bootstrap_p_gt_0": pr_p,
                    "pooled_roc_auc_delta": roc_point,
                    "roc_delta_ci_low": roc_low,
                    "roc_delta_ci_high": roc_high,
                    "roc_delta_bootstrap_p_gt_0": roc_p,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "pooled_pr_auc_delta"], ascending=[True, False])


def key_bootstrap_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("extra_trees", "M4_total_import_network", "M1_operational"),
        ("extra_trees", "M4_total_import_network", "M6a_total_equal_placebo"),
        ("extra_trees", "M4_total_import_network", "M6b_total_shuffled_placebo"),
        ("extra_trees", "M4_total_import_network", "M6c_total_random_placebo"),
        ("extra_trees", "M5_me_strict_network", "M1_operational"),
        ("lightgbm", "M5_me_strict_network", "M1_operational"),
    ]
    rows = []
    for model_name, focus_group, baseline_group in specs:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)].copy()
        if model_predictions.empty:
            continue
        focus = model_predictions.loc[model_predictions["feature_group"].eq(focus_group)].copy()
        baseline = model_predictions.loc[model_predictions["feature_group"].eq(baseline_group)].copy()
        if focus.empty or baseline.empty:
            continue
        for metric_name, metric_fn in [
            ("pr_auc", average_precision_score),
            ("roc_auc", roc_auc_score),
        ]:
            point, low, high, p_gt_0 = paired_bootstrap_delta(
                focus, baseline, metric_fn, n_boot=5000
            )
            rows.append(
                {
                    "model": model_name,
                    "focus_group": focus_group,
                    "baseline_group": baseline_group,
                    "metric": metric_name,
                    "bootstrap_draws": 5000,
                    "point_delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "p_gt_0": p_gt_0,
                }
            )
    return pd.DataFrame(rows)


def event_network_delta_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[
        metrics["feature_group"].eq(BASELINE_GROUP),
        ["model", "fold", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": "baseline_pr_auc", "roc_auc": "baseline_roc_auc"})
    deltas = metrics.merge(baseline, on=["model", "fold"], how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq(BASELINE_GROUP)].copy()
    deltas["pr_auc_delta_vs_m1"] = deltas["pr_auc"] - deltas["baseline_pr_auc"]
    deltas["roc_auc_delta_vs_m1"] = deltas["roc_auc"] - deltas["baseline_roc_auc"]
    return (
        deltas.groupby(["model", "feature_group"], as_index=False)
        .agg(
            mean_pr_auc_delta_vs_m1=("pr_auc_delta_vs_m1", "mean"),
            folds_beating_m1=("pr_auc_delta_vs_m1", lambda s: int((s > 0).sum())),
            mean_roc_auc_delta_vs_m1=("roc_auc_delta_vs_m1", "mean"),
        )
        .sort_values(["model", "mean_pr_auc_delta_vs_m1"], ascending=[True, False])
    )


def write_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    key_deltas: pd.DataFrame,
) -> None:
    top = summary.head(25)
    model_sections = []
    for model_name in sorted(summary["model"].unique()):
        view = summary.loc[summary["model"].eq(model_name)].sort_values("mean_pr_auc", ascending=False)
        model_sections.append(
            f"""## {model_name}

{view[["feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}
"""
        )
    delta_summary = event_network_delta_summary(metrics)
    best_deltas = deltas.groupby("model", group_keys=False).head(5)

    content = f"""# Panel Advanced Model Experiment

## Purpose

This active research expansion tests stronger algorithm families beyond Logistic Regression, Random Forest, and HistGradientBoosting. It keeps the locked 11-country dataset, M1-M7 feature groups, rolling-origin temporal folds, validation-selected thresholds, and PR-AUC priority unchanged.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_event_network_benchmark.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Test folds: 2023, 2024, 2025 rolling-origin
- Models attempted: ExtraTrees, sklearn GradientBoosting, XGBoost, LightGBM, CatBoost if installed
- Main metric: PR-AUC
- Thresholds: validation-selected only

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

{"".join(model_sections)}
## Mean Fold Deltas Versus M1

{delta_summary.to_markdown(index=False)}

## Best Pooled Bootstrap Deltas Versus M1

{best_deltas.to_markdown(index=False)}

## Key 5000-Draw Checks

{key_deltas.to_markdown(index=False)}

## Reading

This experiment is a research expansion, not a replacement for the locked benchmark. A substantive algorithmic breakthrough would require an advanced model to improve event/network groups over its own operational baseline with positive paired PR-AUC intervals, preferably without being matched by placebo variants.
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

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    deltas = bootstrap_deltas(predictions)
    key_deltas = key_bootstrap_deltas(predictions)

    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    deltas.to_csv(DELTAS, index=False)
    key_deltas.to_csv(KEY_DELTAS, index=False)
    write_report(df, metrics, summary, deltas, key_deltas)

    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved key deltas: {KEY_DELTAS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(25).to_string(index=False))
    print(deltas.groupby("model", group_keys=False).head(5).to_string(index=False))


if __name__ == "__main__":
    run()
