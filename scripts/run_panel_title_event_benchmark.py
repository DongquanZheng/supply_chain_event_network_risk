from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse
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
    select_threshold,
    summarize_metrics,
)
from src.config import GDELT_TO_ISO3  # noqa: E402


DOCS = PROJECT_ROOT / "data" / "interim" / "gkg_nlp_candidate_docs_2023-01-01_2025-12-31.csv"
TOTAL_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_total_dependency_weights_2023.csv"
ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_title_event_benchmark_2023_2025.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_title_event_benchmark.md"
RANDOM_SEED = 42


PATTERNS = {
    "trade_policy": re.compile(
        r"\b(?:tariff|tariffs|sanction|sanctions|export control|export ban|import ban|"
        r"customs|trade war|trade restriction|trade restrictions|embargo|export|exports|import|imports)\b",
        re.I,
    ),
    "maritime_logistics": re.compile(
        r"\b(?:port|ports|shipping|freight|cargo|container|vessel|maritime|logistics|"
        r"suez|panama|red sea|red-sea|tanker|shipwreck|ship|ships)\b",
        re.I,
    ),
    "energy_transport": re.compile(
        r"\b(?:oil|gas|fuel|lng|crude|petroleum|refinery|tanker)\b",
        re.I,
    ),
    "manufacturing_electronics": re.compile(
        r"\b(?:semiconductor|semiconductors|chip|chips|electronics|machinery|factory|"
        r"factories|manufactur|industrial equipment|rare earth|antimony|drones)\b",
        re.I,
    ),
    "weather_disruption": re.compile(
        r"\b(?:flood|floods|flooding|typhoon|storm|earthquake|tsunami|landslide|wildfire)\b",
        re.I,
    ),
}


def slug_text(url: str) -> str:
    parsed = urlparse(str(url))
    text = " ".join([parsed.netloc, parsed.path, parsed.query])
    text = unquote(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_title_events(countries: list[str]) -> pd.DataFrame:
    docs = pd.read_csv(
        DOCS,
        usecols=["event_week", "code", "DocumentIdentifier", "tone"],
        parse_dates=["event_week"],
    )
    docs["partner_iso3"] = docs["code"].map(GDELT_TO_ISO3)
    docs = docs.loc[docs["partner_iso3"].isin(countries)].copy()
    docs["title_text"] = docs["DocumentIdentifier"].fillna("").map(slug_text)
    docs["negative_severity"] = (-docs["tone"]).clip(lower=0, upper=10) / 10

    for name, pattern in PATTERNS.items():
        docs[f"title_{name}_flag"] = docs["title_text"].str.contains(pattern).astype(int)
        docs[f"title_{name}_negative_score"] = docs[f"title_{name}_flag"] * docs["negative_severity"]

    flag_cols = [f"title_{name}_flag" for name in PATTERNS]
    docs["title_any_direct_flag"] = docs[flag_cols].max(axis=1)
    docs["title_any_direct_negative_score"] = docs["title_any_direct_flag"] * docs["negative_severity"]

    named_aggs = {
        "title_doc_count": ("title_text", "size"),
        "title_direct_doc_count": ("title_any_direct_flag", "sum"),
        "title_direct_negative_score": ("title_any_direct_negative_score", "mean"),
        "title_avg_tone": ("tone", "mean"),
    }
    for name in PATTERNS:
        named_aggs[f"title_{name}_count"] = (f"title_{name}_flag", "sum")
        named_aggs[f"title_{name}_negative_score"] = (f"title_{name}_negative_score", "mean")

    return (
        docs.groupby(["event_week", "partner_iso3"], as_index=False)
        .agg(**named_aggs)
        .sort_values(["event_week", "partner_iso3"])
    )


def weighted_exposures(events: pd.DataFrame, weights: pd.DataFrame, prefix: str) -> pd.DataFrame:
    joined = weights.merge(events, on="partner_iso3", how="inner")
    score_cols = [col for col in events.columns if col.endswith("_negative_score")]
    count_cols = [col for col in events.columns if col.endswith("_count") and col != "title_doc_count"]

    for col in score_cols + count_cols:
        joined[f"{prefix}_{col}"] = joined["import_dependency_share"] * joined[col]

    agg = {
        f"{prefix}_{col}": (f"{prefix}_{col}", "sum")
        for col in score_cols + count_cols
    }
    agg[f"{prefix}_title_doc_count"] = ("title_doc_count", "sum")
    return joined.groupby(["event_week", "ISO3"], as_index=False).agg(**agg)


def external_unweighted(events: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    rows = []
    score_cols = [col for col in events.columns if col.endswith("_negative_score")]
    count_cols = [col for col in events.columns if col.endswith("_count")]
    for week, week_events in events.groupby("event_week"):
        for target in countries:
            external = week_events.loc[week_events["partner_iso3"].ne(target)]
            row = {"event_week": week, "ISO3": target}
            for col in score_cols:
                row[f"external_{col}"] = float(external[col].mean()) if len(external) else 0.0
            for col in count_cols:
                row[f"external_{col}"] = float(external[col].sum()) if len(external) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def own_events(events: pd.DataFrame) -> pd.DataFrame:
    rename = {
        col: f"own_{col}"
        for col in events.columns
        if col not in {"event_week", "partner_iso3"}
    }
    return events.rename(columns={"partner_iso3": "ISO3", **rename})


def build_dataset() -> pd.DataFrame:
    base = load_dataset()
    base = base.loc[(base["week"] >= "2023-01-01") & (base["week"] < "2026-01-01")].copy()
    countries = sorted(base["ISO3"].unique())
    events = load_title_events(countries)

    total_weights = pd.read_csv(TOTAL_WEIGHTS)
    total_weights = total_weights.loc[
        total_weights["ISO3"].isin(countries) & total_weights["partner_iso3"].isin(countries)
    ].copy()
    me_weights = pd.read_csv(ME_WEIGHTS)
    me_weights = me_weights.loc[
        me_weights["ISO3"].isin(countries) & me_weights["partner_iso3"].isin(countries)
    ].copy()

    dataset = (
        base.merge(own_events(events), left_on=["week", "ISO3"], right_on=["event_week", "ISO3"], how="inner")
        .drop(columns=["event_week"])
        .merge(external_unweighted(events, countries), left_on=["week", "ISO3"], right_on=["event_week", "ISO3"], how="inner")
        .drop(columns=["event_week"])
        .merge(weighted_exposures(events, total_weights, "total_network"), left_on=["week", "ISO3"], right_on=["event_week", "ISO3"], how="inner")
        .drop(columns=["event_week"])
        .merge(weighted_exposures(events, me_weights, "me_network"), left_on=["week", "ISO3"], right_on=["event_week", "ISO3"], how="inner")
        .drop(columns=["event_week"])
        .sort_values(["week", "ISO3"])
        .reset_index(drop=True)
    )
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
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
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


def title_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_title_direct_doc_count",
        f"{prefix}_title_direct_negative_score",
        f"{prefix}_title_trade_policy_count",
        f"{prefix}_title_trade_policy_negative_score",
        f"{prefix}_title_maritime_logistics_count",
        f"{prefix}_title_maritime_logistics_negative_score",
        f"{prefix}_title_energy_transport_count",
        f"{prefix}_title_energy_transport_negative_score",
        f"{prefix}_title_manufacturing_electronics_count",
        f"{prefix}_title_manufacturing_electronics_negative_score",
        f"{prefix}_title_weather_disruption_count",
        f"{prefix}_title_weather_disruption_negative_score",
    ]


def make_feature_groups(country_features: list[str]) -> dict[str, list[str]]:
    base = OPERATIONAL_FEATURES + country_features
    return {
        "S1_operational_2023_2025": base,
        "S2_title_own_events": base + title_columns("own"),
        "S3_title_external_events": base + title_columns("external"),
        "S4_title_total_network": base + title_columns("total_network"),
        "S5_title_me_network": base + title_columns("me_network"),
        "S6_title_full_event_network": base
        + title_columns("own")
        + title_columns("external")
        + title_columns("total_network")
        + title_columns("me_network"),
    }


def fit_model(model_name: str, model, train_x: pd.DataFrame, train_y: pd.Series) -> None:
    if model_name == "hist_gradient_boosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)
        model.fit(train_x, train_y, sample_weight=sample_weight)
    else:
        model.fit(train_x, train_y)


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    train, validation, test = split_dataset(df)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    feature_groups = make_feature_groups(country_features)

    rows = []
    for group_name, features in feature_groups.items():
        for model_name, model in make_models().items():
            fit_model(model_name, model, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            rows.append(
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
                }
            )

    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(TABLE_DIR / "panel_title_event_metrics_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_title_event_summary.csv", index=False)
    write_report(df, metrics, summary)

    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved metrics: {TABLE_DIR / 'panel_title_event_metrics_by_fold.csv'}")
    print(f"Saved summary: {TABLE_DIR / 'panel_title_event_summary.csv'}")
    print(f"Saved report: {REPORT}")
    print(summary.head(18).to_string(index=False))


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary.head(18)
    m1 = metrics.loc[
        metrics["feature_group"].eq("S1_operational_2023_2025"),
        ["model", "pr_auc", "roc_auc"],
    ].rename(columns={"pr_auc": "m1_pr_auc", "roc_auc": "m1_roc_auc"})
    deltas = metrics.merge(m1, on="model", how="left")
    deltas = deltas.loc[~deltas["feature_group"].eq("S1_operational_2023_2025")].copy()
    deltas["pr_auc_delta_vs_m1"] = deltas["pr_auc"] - deltas["m1_pr_auc"]
    deltas["roc_auc_delta_vs_m1"] = deltas["roc_auc"] - deltas["m1_roc_auc"]
    delta_table = deltas.sort_values(["model", "pr_auc_delta_vs_m1"], ascending=[True, False])

    content = f"""# Panel Title-Level Event Benchmark

## Purpose

This exploratory experiment tests a stricter event layer derived only from GDELT candidate-document URL/title slugs. It uses direct title-level supply-chain terms instead of broad GKG themes, motivated by the high-risk-week sanity check.

## Dataset And Protocol

- Dataset: `data/processed/multicountry_container_title_event_benchmark_2023_2025.csv`
- Rows: {len(df)}
- Countries: {df["ISO3"].nunique()}
- Positive labels: {int(df[TARGET].sum())}
- Positive rate: {df[TARGET].mean():.3f}
- Split: train 2023, validation 2024, test 2025
- Thresholds: selected on 2024 validation only
- Main metric: PR-AUC

## Top Results

{top[["feature_group", "model", "mean_pr_auc", "mean_roc_auc", "mean_f1", "mean_precision", "mean_recall", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## PR-AUC Deltas Versus Operational Baseline

{delta_table[["model", "feature_group", "pr_auc", "m1_pr_auc", "pr_auc_delta_vs_m1", "roc_auc_delta_vs_m1"]].to_markdown(index=False)}

## Reading

This is an exploratory Gate 1 diagnostic. If title-level event groups beat the operational baseline, they provide a cleaner event-layer candidate than broad raw GDELT controls. If they do not, the project should keep the title-level audit as interpretability evidence rather than replacing the locked benchmark features.
"""
    REPORT.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    run()
