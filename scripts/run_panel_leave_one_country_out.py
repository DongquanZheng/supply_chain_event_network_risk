from __future__ import annotations

from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (
    EXTERNAL_UNWEIGHTED_FEATURES,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    RANDOM_SEED,
    TARGET,
    TOTAL_NETWORK_FEATURES,
    select_threshold,
)


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_leave_one_country_out.md"


FEATURE_GROUPS = {
    "M1_operational": OPERATIONAL_FEATURES,
    "M3_external_unweighted_events": OPERATIONAL_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES,
    "M4_total_import_network": OPERATIONAL_FEATURES + TOTAL_NETWORK_FEATURES,
    "M5_me_strict_network": OPERATIONAL_FEATURES + ME_NETWORK_FEATURES,
}


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"])


def add_train_country_dummies(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    countries = sorted(train["ISO3"].unique())
    frames = []
    for frame in [train, validation, test]:
        out = frame.copy()
        for country in countries:
            out[f"country_{country}"] = (out["ISO3"] == country).astype(int)
        frames.append(out)
    return frames[0], frames[1], frames[2], [f"country_{country}" for country in countries]


def run_holdout(df: pd.DataFrame, holdout_iso3: str) -> list[dict]:
    train = df[
        (df["ISO3"].ne(holdout_iso3))
        & (df["week"] >= "2021-01-01")
        & (df["week"] < "2024-01-01")
    ].copy()
    validation = df[
        (df["ISO3"].ne(holdout_iso3))
        & (df["week"] >= "2024-01-01")
        & (df["week"] < "2025-01-01")
    ].copy()
    test = df[
        (df["ISO3"].eq(holdout_iso3))
        & (df["week"] >= "2025-01-01")
        & (df["week"] < "2026-01-01")
    ].copy()
    train, validation, test, country_features = add_train_country_dummies(train, validation, test)
    rows = []

    for group_name, base_features in FEATURE_GROUPS.items():
        features = base_features + country_features
        model = make_model()
        model.fit(train[features], train[TARGET])
        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        pred = (test_proba >= threshold).astype(int)
        y = test[TARGET].to_numpy()

        rows.append(
            {
                "holdout_iso3": holdout_iso3,
                "holdout_country": test["country"].iloc[0],
                "feature_group": group_name,
                "test_rows": len(test),
                "test_positives": int(y.sum()),
                "roc_auc": roc_auc_score(y, test_proba),
                "pr_auc": average_precision_score(y, test_proba),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "validation_f1": val_f1,
                "selected_threshold": threshold,
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("feature_group", as_index=False)
        .agg(
            holdout_countries=("holdout_iso3", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values("mean_pr_auc", ascending=False)
    )


def write_report(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    pivot = results.pivot(index="holdout_iso3", columns="feature_group", values="pr_auc").reset_index()
    if {"M5_me_strict_network", "M1_operational", "M3_external_unweighted_events"}.issubset(pivot.columns):
        pivot["m5_minus_m1"] = pivot["M5_me_strict_network"] - pivot["M1_operational"]
        pivot["m5_minus_m3"] = pivot["M5_me_strict_network"] - pivot["M3_external_unweighted_events"]

    content = f"""# Leave-One-Country-Out Panel Generalization

## Question

Does event-network exposure help when the model must generalize to a country that was never observed during training?

For each holdout country, the model trains on 2021-2023 data from the other 10 countries, selects a threshold on 2024 data from the other 10 countries, and tests on the holdout country's 2025 observations.

## Summary

{summary.to_markdown(index=False)}

## Holdout PR-AUC Table

{pivot.to_markdown(index=False)}

## Fold-Level Results

{results.to_markdown(index=False)}

## Interpretation

This is a stricter domain-generalization check than the standard panel benchmark. If M5 improves here, network exposure is not merely fitting country fixed effects; it carries information that transfers across countries.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    df = load_dataset()
    rows = []
    for holdout_iso3 in sorted(df["ISO3"].unique()):
        rows.extend(run_holdout(df, holdout_iso3))
    results = pd.DataFrame(rows)
    summary = summarize(results)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_DIR / "panel_leave_one_country_out_results.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_leave_one_country_out_summary.csv", index=False)
    write_report(results, summary)
    print(summary.to_string(index=False))
    print(f"Saved report: {REPORT}")


if __name__ == "__main__":
    run()
