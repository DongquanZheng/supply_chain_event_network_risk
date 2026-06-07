from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3
from scripts.run_panel_benchmark_models import (
    FOLDS,
    OPERATIONAL_FEATURES,
    RANDOM_SEED,
    TARGET,
    add_country_dummies,
    select_threshold,
    split_fold,
)


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
ME_EVENTS = PROJECT_ROOT / "data" / "interim" / "gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv"
WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_counterfactual_network_swap.md"
SIGNAL = "machinery_electronics_disruption_very_negative_share"


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(DATASET, parse_dates=["week"])
    events = pd.read_csv(ME_EVENTS, parse_dates=["event_week"])
    events["partner_iso3"] = events["code"].map(GDELT_TO_ISO3)
    weights = pd.read_csv(WEIGHTS)
    return dataset, events.dropna(subset=["partner_iso3"]).copy(), weights


def donor_swapped_exposure(events: pd.DataFrame, weights: pd.DataFrame, donor_iso3: str) -> pd.DataFrame:
    donor_weights = weights[weights["ISO3"].eq(donor_iso3)][["partner_iso3", "import_dependency_share"]].copy()
    targets = sorted(set(GDELT_TO_ISO3.values()))
    rows = []

    for target_iso3 in targets:
        usable_weights = donor_weights[donor_weights["partner_iso3"].ne(target_iso3)].copy()
        usable_weights["counterfactual_weight"] = (
            usable_weights["import_dependency_share"] / usable_weights["import_dependency_share"].sum()
        )
        joined = events.merge(usable_weights[["partner_iso3", "counterfactual_weight"]], on="partner_iso3", how="inner")
        weekly = (
            joined.assign(contribution=joined["counterfactual_weight"] * joined[SIGNAL])
            .groupby("event_week", as_index=False)
            .agg(counterfactual_me_exposure=("contribution", "sum"))
        )
        weekly["ISO3"] = target_iso3
        rows.append(weekly)

    return pd.concat(rows, ignore_index=True)


def evaluate_feature(dataset: pd.DataFrame, feature: str, variant: str) -> list[dict]:
    rows = []
    for fold in FOLDS:
        train, validation, test = split_fold(dataset, fold)
        [train, validation, test], country_features = add_country_dummies(train, validation, test)
        features = OPERATIONAL_FEATURES + country_features + [feature, "me_network_strict_article_count"]

        model = make_model()
        model.fit(train[features], train[TARGET])
        val_proba = model.predict_proba(validation[features])[:, 1]
        threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
        test_proba = model.predict_proba(test[features])[:, 1]
        pred = (test_proba >= threshold).astype(int)
        y = test[TARGET].to_numpy()

        rows.append(
            {
                "variant": variant,
                "fold": fold.name,
                "rows": len(test),
                "positives": int(y.sum()),
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
        results.groupby("variant", as_index=False)
        .agg(
            folds=("fold", "nunique"),
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
    true_row = summary[summary["variant"].eq("true_target_specific_me_network")]
    true_pr = float(true_row["mean_pr_auc"].iloc[0])
    swapped = summary[summary["variant"].str.startswith("donor_")].copy()
    better_swaps = int((swapped["mean_pr_auc"] > true_pr).sum())

    content = f"""# Counterfactual Network Swap Check

## Question

Does target-specific machinery/electronics dependency structure matter, or could any country's network weights work equally well?

For each donor country, this check replaces every target country's machinery/electronics network exposure with exposure computed from the donor country's WITS machinery/electronics dependency vector, renormalized after dropping the target country itself from the partner set.

## Summary

{summary.to_markdown(index=False)}

## Fold-Level Results

{results.to_markdown(index=False)}

## Interpretation

The true target-specific network has mean PR-AUC `{true_pr:.3f}`. `{better_swaps}` donor-swapped variants have higher mean PR-AUC than the true target-specific network. If zero or very few donor swaps beat the true network, this supports target-specific dependency structure. If many swaps beat it, the current network signal is not yet structurally specific enough.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    dataset, events, weights = load_inputs()
    rows = evaluate_feature(dataset, "me_network_strict_very_negative_exposure", "true_target_specific_me_network")

    for donor_iso3 in sorted(set(GDELT_TO_ISO3.values())):
        swapped = donor_swapped_exposure(events, weights, donor_iso3)
        swapped_dataset = (
            dataset.drop(columns=["counterfactual_me_exposure"], errors="ignore")
            .merge(
                swapped,
                left_on=["week", "ISO3"],
                right_on=["event_week", "ISO3"],
                how="inner",
            )
            .drop(columns=["event_week"])
            .sort_values(["week", "ISO3"])
            .reset_index(drop=True)
        )
        rows.extend(evaluate_feature(swapped_dataset, "counterfactual_me_exposure", f"donor_{donor_iso3}"))

    results = pd.DataFrame(rows)
    summary = summarize(results)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_DIR / "panel_counterfactual_network_swap_by_fold.csv", index=False)
    summary.to_csv(TABLE_DIR / "panel_counterfactual_network_swap_summary.csv", index=False)
    write_report(results, summary)
    print(summary.to_string(index=False))
    print(f"Saved report: {REPORT}")


if __name__ == "__main__":
    run()
