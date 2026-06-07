from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3
from src.wits import build_partner_dependency_weights, fetch_partner_trade_by_product


PREDICTIONS = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_predictions.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel_network_mechanism_checks.md"
WEIGHTS_CACHE = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
TARGET = "abnormal_next_week_container"
ME_PRODUCT_CODE = "84-85_MachElec"


def build_or_load_me_weights() -> pd.DataFrame:
    if WEIGHTS_CACHE.exists():
        return pd.read_csv(WEIGHTS_CACHE)

    countries = sorted(set(GDELT_TO_ISO3.values()))
    frames = []
    for target_iso3 in countries:
        partners = [iso3 for iso3 in countries if iso3 != target_iso3]
        trade = fetch_partner_trade_by_product(target_iso3, year=2023, product=ME_PRODUCT_CODE)
        weights = build_partner_dependency_weights(trade, partner_iso3=partners)
        weights["ISO3"] = target_iso3
        frames.append(
            weights[
                ["ISO3", "partner_iso3", "value_thousand_usd", "import_dependency_share"]
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WEIGHTS_CACHE, index=False)
    return out


def concentration_metrics(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso3, group in weights.groupby("ISO3"):
        shares = group["import_dependency_share"].to_numpy()
        entropy = -float(np.sum(shares * np.log(shares + 1e-12)))
        rows.append(
            {
                "ISO3": iso3,
                "me_hhi": float(np.sum(shares**2)),
                "me_top1_share": float(np.max(shares)),
                "me_top3_share": float(np.sort(shares)[-3:].sum()),
                "me_entropy": entropy,
                "me_effective_partners": float(np.exp(entropy)),
            }
        )
    return pd.DataFrame(rows)


def score_by_country(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions[predictions["model"].eq("random_forest")].copy()
    groups = [
        "M1_operational",
        "M3_external_unweighted_events",
        "M4_total_import_network",
        "M5_me_strict_network",
        "M6b_total_shuffled_placebo",
    ]
    predictions = predictions[predictions["feature_group"].isin(groups)]

    rows = []
    for (iso3, feature_group), group in predictions.groupby(["ISO3", "feature_group"]):
        y = group[TARGET].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        proba = group["predicted_probability"].to_numpy()
        rows.append(
            {
                "ISO3": iso3,
                "country": group["country"].iloc[0],
                "feature_group": feature_group,
                "rows": len(group),
                "positives": int(y.sum()),
                "pr_auc": average_precision_score(y, proba),
                "roc_auc": roc_auc_score(y, proba),
            }
        )
    scores = pd.DataFrame(rows)
    wide = scores.pivot(index=["ISO3", "country"], columns="feature_group", values="pr_auc").reset_index()
    wide.columns.name = None
    wide["m5_minus_m1_pr_auc"] = wide["M5_me_strict_network"] - wide["M1_operational"]
    wide["m5_minus_m3_pr_auc"] = wide["M5_me_strict_network"] - wide["M3_external_unweighted_events"]
    wide["m5_minus_m4_pr_auc"] = wide["M5_me_strict_network"] - wide["M4_total_import_network"]
    wide["m5_minus_shuffled_pr_auc"] = wide["M5_me_strict_network"] - wide["M6b_total_shuffled_placebo"]
    return wide


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    concentration_cols = ["me_hhi", "me_top1_share", "me_top3_share", "me_entropy", "me_effective_partners"]
    delta_cols = [
        "m5_minus_m1_pr_auc",
        "m5_minus_m3_pr_auc",
        "m5_minus_m4_pr_auc",
        "m5_minus_shuffled_pr_auc",
    ]
    rows = []
    for concentration in concentration_cols:
        for delta in delta_cols:
            rows.append(
                {
                    "concentration_metric": concentration,
                    "delta_metric": delta,
                    "pearson_corr": df[concentration].corr(df[delta], method="pearson"),
                    "spearman_corr": df[concentration].corr(df[delta], method="spearman"),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_corr", ascending=False)


def write_report(country_results: pd.DataFrame, correlations: pd.DataFrame) -> None:
    top_corr = correlations.head(12)
    content = f"""# Panel Network Mechanism Checks

## Question

Does machinery/electronics network exposure help more for countries whose machinery/electronics import dependency is structurally concentrated?

This test is exploratory because there are only 11 countries, but it is important for mechanism building. A positive relationship would support the idea that network value is conditional on dependency structure rather than being a generic feature-engineering trick.

## Country-Level Results

{country_results.to_markdown(index=False)}

## Concentration vs Network-Gain Correlations

{top_corr.to_markdown(index=False)}

## Interpretation Rule

If `me_hhi`, `me_top1_share`, or `me_top3_share` correlate positively with `m5_minus_*` deltas, network gains are larger for more concentrated dependency structures. If entropy/effective partners correlate positively instead, the evidence points toward network exposure helping more in diversified systems. If all correlations are weak, the current panel does not yet explain where network weighting helps.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    weights = build_or_load_me_weights()
    concentration = concentration_metrics(weights)
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["week"])
    country_scores = score_by_country(predictions)
    country_results = country_scores.merge(concentration, on="ISO3", how="left")
    correlations = correlation_table(country_results)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    country_results.to_csv(TABLE_DIR / "panel_network_mechanism_country_results.csv", index=False)
    correlations.to_csv(TABLE_DIR / "panel_network_mechanism_correlations.csv", index=False)
    write_report(country_results, correlations)

    print(country_results.sort_values("m5_minus_m3_pr_auc", ascending=False).to_string(index=False))
    print("\nTop correlations:")
    print(correlations.head(12).to_string(index=False))
    print(f"Saved report: {REPORT}")


if __name__ == "__main__":
    run()
