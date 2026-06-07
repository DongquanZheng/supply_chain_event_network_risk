from __future__ import annotations

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

from scripts.run_nlp_taxonomy_experiment import (
    CURRENT_NEWS,
    DEFAULT_BENCHMARK,
    DEFAULT_DATASET,
    DEFAULT_ME_WEIGHTS,
    DEFAULT_TAXONOMY,
    OPERATIONAL,
    TARGET,
    build_dataset,
    temporal_split,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "network_feature_options.md"
RANDOM_SEED = 42


TAX_BUNDLE = [
    "taxonomy_candidate_article_count",
    "taxonomy_candidate_negative_severity",
    "tax_maritime_unweighted_score",
    "tax_machinery_electronics_unweighted_score",
    "tax_energy_unweighted_score",
    "tax_trade_policy_unweighted_score",
    "tax_broad_supply_unweighted_score",
]


def add_network_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9

    out["operational_shortfall_12w"] = (
        out["rolling_mean_container_12w"] - out["portcalls_container"]
    ) / out["rolling_std_container_12w"].replace(0, np.nan)
    out["negative_trend_4w"] = (
        -out["rolling_change_container_4w"] / out["rolling_std_container_12w"].replace(0, np.nan)
    )

    out["me_network_minus_unweighted"] = (
        out["tax_machinery_electronics_me_network_score"]
        - out["tax_machinery_electronics_unweighted_score"]
    )
    out["me_network_minus_equal"] = (
        out["tax_machinery_electronics_me_network_score"]
        - out["tax_machinery_electronics_me_equal_score"]
    )
    out["me_network_to_unweighted_ratio"] = (
        out["tax_machinery_electronics_me_network_score"]
        / (out["tax_machinery_electronics_unweighted_score"] + eps)
    )
    out["me_network_to_equal_ratio"] = (
        out["tax_machinery_electronics_me_network_score"]
        / (out["tax_machinery_electronics_me_equal_score"] + eps)
    )

    event_cols = [
        "unweighted_very_negative_exposure",
        "tax_machinery_electronics_unweighted_score",
        "tax_machinery_electronics_me_network_score",
        "tax_broad_supply_unweighted_score",
    ]
    for col in event_cols:
        out[f"shortfall_x_{col}"] = out["operational_shortfall_12w"] * out[col]
        out[f"trend_x_{col}"] = out["negative_trend_4w"] * out[col]

    out["news_x_me_network_ratio"] = (
        out["unweighted_very_negative_exposure"] * out["me_network_to_equal_ratio"]
    )
    out["taxonomy_x_me_network_ratio"] = (
        out["tax_machinery_electronics_unweighted_score"] * out["me_network_to_equal_ratio"]
    )

    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_models() -> dict[str, object]:
    return {
        "logistic": Pipeline(
            [
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
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def select_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def top_k_score(y_true: pd.Series, proba: np.ndarray, k: int = 10) -> tuple[int, float]:
    ranked = pd.DataFrame({"y": y_true.to_numpy(), "proba": proba}).sort_values(
        "proba", ascending=False
    )
    tp = int(ranked.head(k)["y"].sum())
    return tp, tp / k


def feature_groups() -> dict[str, list[str]]:
    network_structure = [
        "tax_machinery_electronics_me_network_score",
        "me_network_minus_unweighted",
        "me_network_minus_equal",
        "me_network_to_unweighted_ratio",
        "me_network_to_equal_ratio",
    ]
    interaction = [
        "operational_shortfall_12w",
        "negative_trend_4w",
        "shortfall_x_unweighted_very_negative_exposure",
        "shortfall_x_tax_machinery_electronics_unweighted_score",
        "shortfall_x_tax_machinery_electronics_me_network_score",
        "shortfall_x_tax_broad_supply_unweighted_score",
        "trend_x_unweighted_very_negative_exposure",
        "trend_x_tax_machinery_electronics_unweighted_score",
        "trend_x_tax_machinery_electronics_me_network_score",
        "trend_x_tax_broad_supply_unweighted_score",
        "news_x_me_network_ratio",
        "taxonomy_x_me_network_ratio",
    ]
    return {
        "M2_current_simple_news": OPERATIONAL + CURRENT_NEWS,
        "N7_news_plus_taxonomy_bundle": OPERATIONAL + CURRENT_NEWS + TAX_BUNDLE,
        "C1_news_taxonomy_plus_network_structure": OPERATIONAL
        + CURRENT_NEWS
        + TAX_BUNDLE
        + network_structure,
        "C2_news_taxonomy_plus_interactions": OPERATIONAL
        + CURRENT_NEWS
        + TAX_BUNDLE
        + interaction,
        "C3_news_taxonomy_network_interactions": OPERATIONAL
        + CURRENT_NEWS
        + TAX_BUNDLE
        + network_structure
        + interaction,
        "C4_operational_network_interactions_only": OPERATIONAL
        + network_structure
        + interaction,
    }


def evaluate_group(name: str, features: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for model_name, model in make_models().items():
        model.fit(train[features], train[TARGET])
        val_proba = model.predict_proba(val[features])[:, 1]
        threshold = select_threshold(val[TARGET], val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test[TARGET], test_pred, labels=[0, 1]).ravel()
        tp_at_10, precision_at_10 = top_k_score(test[TARGET], test_proba, k=10)
        rows.append(
            {
                "feature_group": name,
                "model": model_name,
                "roc_auc": roc_auc_score(test[TARGET], test_proba),
                "pr_auc": average_precision_score(test[TARGET], test_proba),
                "selected_threshold": threshold,
                "precision": precision_score(test[TARGET], test_pred, zero_division=0),
                "recall": recall_score(test[TARGET], test_pred, zero_division=0),
                "f1": f1_score(test[TARGET], test_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tp_at_10": tp_at_10,
                "precision_at_10": precision_at_10,
            }
        )
    return rows


def write_report(metrics: pd.DataFrame, dataset: pd.DataFrame) -> None:
    best = metrics.sort_values(["pr_auc", "f1"], ascending=False)
    rf = metrics.loc[metrics["model"].eq("random_forest")].sort_values(
        ["pr_auc", "f1"], ascending=False
    )
    content = f"""# Network Feature Options Diagnostic

## Purpose

This diagnostic tests whether increasing network complexity inside the Japan-centered benchmark helps once NLP taxonomy features are available.

## Data

- Dataset: `data/processed/japan_container_nlp_taxonomy_benchmark_2023_2025.csv`
- Rows: {len(dataset)}
- Positive labels: {int(dataset[TARGET].sum())}
- Split: train 2023, validation 2024, test 2025

## Tested Network Extensions

- Network structure ratios: true machinery/electronics network exposure minus/relative to unweighted and equal-weight exposure.
- Operational vulnerability interactions: current operational shortfall and recent negative trend multiplied by event/network signals.
- Combined news + taxonomy + network interaction feature sets.

## Results

{best[["feature_group", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "fp", "tp", "tp_at_10"]].to_markdown(index=False)}

## Random Forest Only

{rf[["feature_group", "roc_auc", "pr_auc", "precision", "recall", "f1", "fp", "tp", "tp_at_10"]].to_markdown(index=False)}

## Interpretation

If interaction groups outperform the news + taxonomy baseline, network should be developed as a structural conditioning layer. If they do not, the Japan-only benchmark is likely too narrow for network structure to show predictive value, and the next serious network direction should be a multi-country panel rather than a more complex Japan-only model.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    dataset = build_dataset(DEFAULT_BENCHMARK, DEFAULT_TAXONOMY, DEFAULT_ME_WEIGHTS, DEFAULT_DATASET)
    dataset = add_network_diagnostics(dataset)
    train, val, test = temporal_split(dataset)

    rows = []
    for name, features in feature_groups().items():
        rows.extend(evaluate_group(name, features, train, val, test))

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "f1"], ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLE_DIR / "network_feature_options.csv"
    metrics.to_csv(output, index=False)
    write_report(metrics, dataset)

    print(f"Saved: {output}")
    print(f"Report: {REPORT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run()
