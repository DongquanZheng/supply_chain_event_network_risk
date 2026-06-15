from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    OPERATIONAL_FEATURES,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    load_dataset,
    precision_at_k,
    select_threshold,
    summarize_metrics,
)
from src.config import GDELT_TO_ISO3, ISO3_TO_GDELT  # noqa: E402


TAXONOMY_PATH = PROJECT_ROOT / "data" / "interim" / "gkg_nlp_taxonomy_partner_week_2023-01-01_2025-12-31.csv"
TOTAL_WEIGHTS_PATH = PROJECT_ROOT / "data" / "interim" / "panel_total_dependency_weights_2023.csv"
ME_WEIGHTS_PATH = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_nlp_taxonomy_benchmark_2023_2025.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_nlp_taxonomy_benchmark.md"
RANDOM_SEED = 42

TAXONOMY_SCORE_COLUMNS = [
    "nlp_maritime_disruption_prob_negative_score",
    "nlp_machinery_electronics_disruption_prob_negative_score",
    "nlp_energy_disruption_prob_negative_score",
    "nlp_trade_policy_disruption_prob_negative_score",
    "nlp_broad_supply_disruption_prob_negative_score",
]

CURRENT_OWN_NEWS_FEATURES = [
    "article_count",
    "avg_tone",
    "negative_article_share",
    "very_negative_article_share",
    "trade_transport_count",
    "risk_theme_count",
]

CURRENT_EXTERNAL_FEATURES = [
    "external_article_count",
    "external_avg_tone",
    "external_negative_article_share",
    "external_very_negative_article_share",
    "external_trade_transport_count",
    "external_risk_theme_count",
]

CURRENT_ME_NETWORK_FEATURES = [
    "me_network_strict_very_negative_exposure",
    "me_network_strict_article_count",
]


def short_name(column: str) -> str:
    return column.replace("nlp_", "").replace("_disruption_prob_negative_score", "")


def add_placebo_weights(weights: pd.DataFrame, weight_col: str, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + (17 if prefix == "me" else 0))
    frames = []
    for target, group in weights.groupby("ISO3", sort=True):
        out = group.copy().sort_values("partner_iso3").reset_index(drop=True)
        n = len(out)
        out[f"{prefix}_equal_weight"] = 1 / n
        out[f"{prefix}_shuffled_weight"] = rng.permutation(out[weight_col].to_numpy())
        raw_random = rng.random(n)
        out[f"{prefix}_random_weight"] = raw_random / raw_random.sum()
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    total = weights.sum()
    if total == 0:
        return 0.0
    return float((values * weights).sum() / total)


def build_taxonomy_exposures(countries: list[str]) -> pd.DataFrame:
    taxonomy = pd.read_csv(TAXONOMY_PATH, parse_dates=["event_week"])
    taxonomy["partner_iso3"] = taxonomy["code"].map(GDELT_TO_ISO3)
    taxonomy = taxonomy.loc[taxonomy["partner_iso3"].isin(countries)].copy()

    total_weights = pd.read_csv(TOTAL_WEIGHTS_PATH)
    total_weights = total_weights.loc[
        total_weights["ISO3"].isin(countries) & total_weights["partner_iso3"].isin(countries)
    ].copy()
    total_weights = add_placebo_weights(total_weights, "import_dependency_share", "total")

    me_weights = pd.read_csv(ME_WEIGHTS_PATH)
    me_weights = me_weights.loc[
        me_weights["ISO3"].isin(countries) & me_weights["partner_iso3"].isin(countries)
    ].copy()
    me_weights = add_placebo_weights(me_weights, "import_dependency_share", "me")

    rows = []
    for week, week_events in taxonomy.groupby("event_week"):
        for target in countries:
            own = week_events.loc[week_events["partner_iso3"].eq(target)]
            external = week_events.loc[week_events["partner_iso3"].ne(target)]
            row = {
                "week": week,
                "ISO3": target,
                "taxonomy_own_article_count": float(own["article_count"].sum()),
                "taxonomy_external_article_count": float(external["article_count"].sum()),
            }

            for col in TAXONOMY_SCORE_COLUMNS:
                name = short_name(col)
                row[f"taxonomy_own_{name}_score"] = float(own[col].iloc[0]) if len(own) else 0.0
                row[f"taxonomy_external_{name}_score"] = weighted_average(
                    external[col], external["article_count"]
                )

            total_joined = total_weights.loc[total_weights["ISO3"].eq(target)].merge(
                week_events,
                on="partner_iso3",
                how="inner",
            )
            me_joined = me_weights.loc[me_weights["ISO3"].eq(target)].merge(
                week_events,
                on="partner_iso3",
                how="inner",
            )

            for col in TAXONOMY_SCORE_COLUMNS:
                name = short_name(col)
                for prefix, joined, weight_specs in [
                    (
                        "taxonomy_total",
                        total_joined,
                        {
                            "network": "import_dependency_share",
                            "equal": "total_equal_weight",
                            "shuffled": "total_shuffled_weight",
                            "random": "total_random_weight",
                        },
                    ),
                    (
                        "taxonomy_me",
                        me_joined,
                        {
                            "network": "import_dependency_share",
                            "equal": "me_equal_weight",
                            "shuffled": "me_shuffled_weight",
                            "random": "me_random_weight",
                        },
                    ),
                ]:
                    for variant, weight_col in weight_specs.items():
                        row[f"{prefix}_{variant}_{name}_score"] = float(
                            (joined[col] * joined[weight_col]).sum()
                        )

            rows.append(row)

    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    base = load_dataset()
    countries = sorted(base["ISO3"].unique())
    taxonomy_exposures = build_taxonomy_exposures(countries)
    dataset = (
        base.loc[(base["week"] >= "2023-01-01") & (base["week"] < "2026-01-01")]
        .merge(taxonomy_exposures, on=["week", "ISO3"], how="inner")
        .sort_values(["week", "ISO3"])
        .reset_index(drop=True)
    )
    dataset["gdelt_code"] = dataset["ISO3"].map(ISO3_TO_GDELT)
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_DATASET, index=False)
    return dataset


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            early_stopping=False,
            random_state=RANDOM_SEED,
        ),
    }


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    own_taxonomy = [
        "taxonomy_own_article_count",
        *[f"taxonomy_own_{short_name(col)}_score" for col in TAXONOMY_SCORE_COLUMNS],
    ]
    external_taxonomy = [
        "taxonomy_external_article_count",
        *[f"taxonomy_external_{short_name(col)}_score" for col in TAXONOMY_SCORE_COLUMNS],
    ]
    total_network_taxonomy = [
        *[f"taxonomy_total_network_{short_name(col)}_score" for col in TAXONOMY_SCORE_COLUMNS],
    ]
    me_network_taxonomy = [
        "taxonomy_me_network_machinery_electronics_score",
    ]
    me_placebo_taxonomy = [
        "taxonomy_me_equal_machinery_electronics_score",
        "taxonomy_me_shuffled_machinery_electronics_score",
        "taxonomy_me_random_machinery_electronics_score",
    ]

    return {
        "C1_operational_2023_2025": base,
        "C2_current_own_news": base + CURRENT_OWN_NEWS_FEATURES,
        "C3_current_external_events": base + CURRENT_EXTERNAL_FEATURES,
        "C5_current_me_strict_network": base + CURRENT_ME_NETWORK_FEATURES,
        "T2_taxonomy_own_events": base + own_taxonomy,
        "T3_taxonomy_external_events": base + external_taxonomy,
        "T4_taxonomy_total_network": base + total_network_taxonomy,
        "T5_taxonomy_me_network": base + me_network_taxonomy,
        "T6_taxonomy_me_placebo_bundle": base + me_placebo_taxonomy,
        "T7_taxonomy_full_event_network": base
        + own_taxonomy
        + external_taxonomy
        + total_network_taxonomy
        + me_network_taxonomy,
        "T8_current_external_plus_taxonomy": base + CURRENT_EXTERNAL_FEATURES + external_taxonomy,
    }


def fit_model(model_name: str, model, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if model_name == "hist_gradient_boosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)
        model.fit(train_x, train_y, sample_weight=sample_weight)
    else:
        model.fit(train_x, train_y)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = build_dataset()
    train, validation, test = split_dataset(df)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    metric_rows = []
    prediction_rows = []
    for group_name, features in feature_groups.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise KeyError(f"{group_name} missing features: {missing}")
        for model_name, model in make_models().items():
            fit_model(model_name, model, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)

            metric_rows.append(
                {
                    "fold": "test_2025",
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
                    "precision_at_10": precision_at_k(test[TARGET].to_numpy(), test_proba, 10),
                    "precision_at_25": precision_at_k(test[TARGET].to_numpy(), test_proba, 25),
                    "precision_at_50": precision_at_k(test[TARGET].to_numpy(), test_proba, 50),
                }
            )

            pred_frame = test[["week", "ISO3", "country", TARGET]].copy()
            pred_frame["feature_group"] = group_name
            pred_frame["model"] = model_name
            pred_frame["predicted_probability"] = test_proba
            pred_frame["selected_threshold"] = threshold
            pred_frame["prediction"] = (test_proba >= threshold).astype(int)
            prediction_rows.extend(pred_frame.to_dict("records"))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    return df, metrics, predictions


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary.head(18)
    by_model = []
    for model_name in sorted(summary["model"].unique()):
        model_summary = summary.loc[summary["model"].eq(model_name)].sort_values("mean_pr_auc", ascending=False)
        by_model.append(
            f"""## {model_name}

{model_summary[["feature_group", "mean_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}
"""
        )

    m1 = metrics.loc[
        metrics["feature_group"].eq("C1_operational_2023_2025"),
        ["model", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": "m1_pr_auc", "roc_auc": "m1_roc_auc"})
    deltas = metrics.merge(m1, on="model", how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq("C1_operational_2023_2025")].copy()
    deltas["pr_auc_delta_vs_m1"] = deltas["pr_auc"] - deltas["m1_pr_auc"]
    deltas["roc_auc_delta_vs_m1"] = deltas["roc_auc"] - deltas["m1_roc_auc"]
    delta_table = deltas.sort_values(["model", "pr_auc_delta_vs_m1"], ascending=[True, False])

    content = f"""# Panel NLP Taxonomy Benchmark

## Purpose

This exploratory diagnostic tests whether a weakly supervised GDELT event-taxonomy layer improves the event side of the 11-country panel benchmark before adding more network complexity.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_nlp_taxonomy_benchmark_2023_2025.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Split: train 2023, validation 2024, test 2025
- Thresholds: selected on 2024 validation only
- Main metric: PR-AUC

## Important Caveat

The taxonomy layer is weakly supervised from URL slugs plus GKG themes, names, and organizations. It is more semantic than raw GDELT tone/theme counts, but it is not manually validated full-text NLP.

## Top Mean Results

{top[["feature_group", "model", "mean_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

{"".join(by_model)}
## PR-AUC Deltas Versus Operational Baseline

{delta_table[["model", "feature_group", "pr_auc", "m1_pr_auc", "pr_auc_delta_vs_m1", "roc_auc_delta_vs_m1"]].to_markdown(index=False)}

## Reading

This is an exploratory event-layer upgrade check. If taxonomy feature groups consistently beat the current raw GDELT controls and the operational baseline, this can become a candidate replacement event layer for the formal benchmark. If gains are model- or split-specific, it should be used as a diagnostic direction rather than a paper claim.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, metrics, predictions = evaluate()
    summary = summarize_metrics(metrics)

    metrics.to_csv(TABLE_DIR / "panel_nlp_taxonomy_metrics_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_nlp_taxonomy_summary.csv", index=False)
    predictions.to_csv(TABLE_DIR / "panel_nlp_taxonomy_predictions.csv", index=False)
    write_report(df, metrics, summary)

    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved metrics: {TABLE_DIR / 'panel_nlp_taxonomy_metrics_by_fold.csv'}")
    print(f"Saved summary: {TABLE_DIR / 'panel_nlp_taxonomy_summary.csv'}")
    print(f"Saved predictions: {TABLE_DIR / 'panel_nlp_taxonomy_predictions.csv'}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    run()
