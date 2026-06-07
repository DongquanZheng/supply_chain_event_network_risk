from __future__ import annotations

import argparse
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

from src.config import GDELT_TO_ISO3, JAPAN


DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "data" / "processed" / "japan_container_event_network_benchmark.csv"
)
DEFAULT_TAXONOMY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_nlp_taxonomy_partner_week_2023-01-01_2025-12-31.csv"
)
DEFAULT_ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "japan_me_dependency_weights_2023.csv"
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "japan_container_nlp_taxonomy_benchmark_2023_2025.csv"
)
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
RESULTS_MD = PROJECT_ROOT / "reports" / "nlp_taxonomy_experiment.md"

TARGET = "abnormal_next_week_container"
RANDOM_SEED = 42

OPERATIONAL = [
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
]

CURRENT_NEWS = [
    "news_article_count",
    "news_avg_tone",
    "unweighted_negative_exposure",
    "unweighted_very_negative_exposure",
    "news_trade_transport_count",
    "news_risk_theme_count",
]

TAXONOMY_SCORE_COLUMNS = [
    "nlp_maritime_disruption_prob_negative_score",
    "nlp_machinery_electronics_disruption_prob_negative_score",
    "nlp_energy_disruption_prob_negative_score",
    "nlp_trade_policy_disruption_prob_negative_score",
    "nlp_broad_supply_disruption_prob_negative_score",
]


def load_base(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week"]).sort_values("week").reset_index(drop=True)


def build_taxonomy_exposures(taxonomy_path: Path, me_weights_path: Path) -> pd.DataFrame:
    taxonomy = pd.read_csv(taxonomy_path, parse_dates=["event_week"])
    taxonomy["partner_iso3"] = taxonomy["code"].map(GDELT_TO_ISO3)
    taxonomy = taxonomy.loc[
        taxonomy["partner_iso3"].notna() & taxonomy["partner_iso3"].ne(JAPAN.iso3)
    ].copy()

    me_weights = pd.read_csv(me_weights_path)
    weight_table = (
        me_weights[["partner_iso3", "me_dependency_share"]]
        .drop_duplicates()
        .sort_values("partner_iso3")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(weight_table)
    weight_table["me_equal_weight"] = 1 / n
    weight_table["me_shuffled_weight"] = rng.permutation(
        weight_table["me_dependency_share"].to_numpy()
    )
    random_raw = rng.random(n)
    weight_table["me_random_weight"] = random_raw / random_raw.sum()

    taxonomy = taxonomy.merge(weight_table, on="partner_iso3", how="inner")

    for col in TAXONOMY_SCORE_COLUMNS:
        short = col.replace("nlp_", "").replace("_disruption_prob_negative_score", "")
        taxonomy[f"tax_{short}_unweighted_score"] = taxonomy[col]
        taxonomy[f"tax_{short}_me_network_score"] = taxonomy[col] * taxonomy["me_dependency_share"]
        taxonomy[f"tax_{short}_me_equal_score"] = taxonomy[col] * taxonomy["me_equal_weight"]
        taxonomy[f"tax_{short}_me_shuffled_score"] = taxonomy[col] * taxonomy["me_shuffled_weight"]
        taxonomy[f"tax_{short}_me_random_score"] = taxonomy[col] * taxonomy["me_random_weight"]

    unweighted_aggs = {
        f"tax_{col.replace('nlp_', '').replace('_disruption_prob_negative_score', '')}_unweighted_score": (
            f"tax_{col.replace('nlp_', '').replace('_disruption_prob_negative_score', '')}_unweighted_score",
            "mean",
        )
        for col in TAXONOMY_SCORE_COLUMNS
    }

    weighted_aggs = {}
    for col in TAXONOMY_SCORE_COLUMNS:
        short = col.replace("nlp_", "").replace("_disruption_prob_negative_score", "")
        for kind in ["me_network", "me_equal", "me_shuffled", "me_random"]:
            weighted_aggs[f"tax_{short}_{kind}_score"] = (
                f"tax_{short}_{kind}_score",
                "sum",
            )

    return (
        taxonomy.groupby("event_week", as_index=False)
        .agg(
            taxonomy_candidate_article_count=("article_count", "sum"),
            taxonomy_candidate_negative_severity=("nlp_candidate_negative_severity", "mean"),
            **unweighted_aggs,
            **weighted_aggs,
        )
        .sort_values("event_week")
    )


def build_dataset(base_path: Path, taxonomy_path: Path, me_weights_path: Path, output: Path) -> pd.DataFrame:
    base = load_base(base_path)
    exposures = build_taxonomy_exposures(taxonomy_path, me_weights_path)

    dataset = (
        base.merge(exposures, left_on="week", right_on="event_week", how="inner")
        .drop(columns=["event_week"])
        .sort_values("week")
        .reset_index(drop=True)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    return dataset


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["week"] >= "2023-01-01") & (df["week"] < "2024-01-01")].copy()
    validation = df[(df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df[(df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    return train, validation, test


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


def evaluate_group(
    group_name: str,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict]:
    rows = []
    for model_name, model in make_models().items():
        model.fit(train[features], train[TARGET])
        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold = select_threshold(validation[TARGET], val_proba)

        test_proba = model.predict_proba(test[features])[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            test[TARGET], test_pred, labels=[0, 1]
        ).ravel()

        rows.append(
            {
                "feature_group": group_name,
                "model": model_name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positives": int(train[TARGET].sum()),
                "validation_positives": int(validation[TARGET].sum()),
                "test_positives": int(test[TARGET].sum()),
                "selected_threshold": threshold,
                "roc_auc": roc_auc_score(test[TARGET], test_proba),
                "pr_auc": average_precision_score(test[TARGET], test_proba),
                "precision": precision_score(test[TARGET], test_pred, zero_division=0),
                "recall": recall_score(test[TARGET], test_pred, zero_division=0),
                "f1": f1_score(test[TARGET], test_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return rows


def feature_groups() -> dict[str, list[str]]:
    tax_bundle = [
        "taxonomy_candidate_article_count",
        "taxonomy_candidate_negative_severity",
        "tax_maritime_unweighted_score",
        "tax_machinery_electronics_unweighted_score",
        "tax_energy_unweighted_score",
        "tax_trade_policy_unweighted_score",
        "tax_broad_supply_unweighted_score",
    ]
    return {
        "M1_operational_2023_2025": OPERATIONAL,
        "M2_current_simple_news": OPERATIONAL + CURRENT_NEWS,
        "M3_current_me_strict_unweighted": OPERATIONAL
        + ["me_strict_unweighted_exposure", "me_strict_article_count"],
        "N1_taxonomy_unweighted_bundle": OPERATIONAL + tax_bundle,
        "N2_taxonomy_me_unweighted": OPERATIONAL
        + [
            "tax_machinery_electronics_unweighted_score",
            "taxonomy_candidate_article_count",
        ],
        "N3_taxonomy_me_network": OPERATIONAL
        + [
            "tax_machinery_electronics_me_network_score",
            "taxonomy_candidate_article_count",
        ],
        "N4_taxonomy_me_equal_placebo": OPERATIONAL
        + [
            "tax_machinery_electronics_me_equal_score",
            "taxonomy_candidate_article_count",
        ],
        "N5_taxonomy_me_shuffled_placebo": OPERATIONAL
        + [
            "tax_machinery_electronics_me_shuffled_score",
            "taxonomy_candidate_article_count",
        ],
        "N6_taxonomy_me_random_placebo": OPERATIONAL
        + [
            "tax_machinery_electronics_me_random_score",
            "taxonomy_candidate_article_count",
        ],
        "N7_current_news_plus_taxonomy_bundle": OPERATIONAL + CURRENT_NEWS + tax_bundle,
        "N8_current_news_plus_taxonomy_network": OPERATIONAL
        + CURRENT_NEWS
        + [
            "tax_machinery_electronics_me_network_score",
            "taxonomy_candidate_article_count",
        ],
    }


def write_report(metrics: pd.DataFrame, dataset: pd.DataFrame, path: Path) -> None:
    best = metrics.sort_values(["pr_auc", "roc_auc"], ascending=False).head(12)
    rf = metrics.loc[metrics["model"].eq("random_forest")].sort_values("pr_auc", ascending=False)
    content = f"""# NLP Taxonomy Experiment

## Purpose

This experiment tests whether a lightweight NLP taxonomy layer improves the event side of the benchmark before adding more network complexity.

## Data

- Dataset: `data/processed/japan_container_nlp_taxonomy_benchmark_2023_2025.csv`
- Rows: {len(dataset)}
- Week range: {dataset["week"].min().date()} to {dataset["week"].max().date()}
- Positive labels: {int(dataset[TARGET].sum())}
- Split: train 2023, validation 2024, test 2025

## Method

The taxonomy features come from weakly supervised document classification over URL slugs plus GKG themes/names/organizations. The current GDELT GKG public table does not provide full article bodies, so this is a reproducible event-taxonomy proxy rather than full-text NLP.

## Top Results

{best[["feature_group", "model", "roc_auc", "pr_auc", "precision", "recall", "f1", "fp", "tp"]].to_markdown(index=False)}

## Random Forest Ranking

{rf[["feature_group", "roc_auc", "pr_auc", "precision", "recall", "f1", "fp", "tp"]].to_markdown(index=False)}

## Interpretation

This table should be read as an NLP-upgrade diagnostic, not as the final paper benchmark. If taxonomy features beat current simple GKG controls, the next paper version should treat NLP taxonomy as the main event-signal layer. If taxonomy features do not beat simple controls, the next step is better text access or a stricter event-definition audit rather than adding network complexity.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    dataset = build_dataset(
        Path(args.benchmark),
        Path(args.taxonomy),
        Path(args.me_weights),
        Path(args.output_dataset),
    )
    train, validation, test = temporal_split(dataset)

    rows = []
    for name, features in feature_groups().items():
        rows.extend(evaluate_group(name, features, train, validation, test))

    metrics = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLE_DIR / "nlp_taxonomy_experiment_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    write_report(metrics, dataset, RESULTS_MD)

    print(f"Saved dataset: {args.output_dataset}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {RESULTS_MD}")
    print(metrics.head(12).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--me-weights", default=str(DEFAULT_ME_WEIGHTS))
    parser.add_argument("--output-dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
