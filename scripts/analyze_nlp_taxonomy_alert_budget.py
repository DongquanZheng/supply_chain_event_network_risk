from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_nlp_taxonomy_experiment import (
    DEFAULT_BENCHMARK,
    DEFAULT_DATASET,
    DEFAULT_ME_WEIGHTS,
    DEFAULT_TAXONOMY,
    TARGET,
    build_dataset,
    feature_groups,
    make_models,
    temporal_split,
)


TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "nlp_taxonomy_alert_budget.md"


def top_k_metrics(y_true: pd.Series, proba, k: int) -> dict:
    ranked = (
        pd.DataFrame({"y": y_true.to_numpy(), "proba": proba})
        .sort_values("proba", ascending=False)
        .head(k)
    )
    tp = int(ranked["y"].sum())
    return {
        "alert_budget": k,
        "tp": tp,
        "fp": int(k - tp),
        "precision_at_k": tp / k,
        "recall_at_k": tp / int(y_true.sum()),
    }


def run() -> None:
    dataset = build_dataset(
        DEFAULT_BENCHMARK,
        DEFAULT_TAXONOMY,
        DEFAULT_ME_WEIGHTS,
        DEFAULT_DATASET,
    )
    train, validation, test = temporal_split(dataset)

    model = make_models()["random_forest"]
    rows = []
    prediction_frames = []

    for group_name, features in feature_groups().items():
        fitted = model.__class__(**model.get_params())
        fitted.fit(train[features], train[TARGET])
        proba = fitted.predict_proba(test[features])[:, 1]

        for k in [5, 10, 15, 20]:
            row = top_k_metrics(test[TARGET], proba, k)
            row["feature_group"] = group_name
            rows.append(row)

        frame = test[["week", TARGET]].copy()
        frame["feature_group"] = group_name
        frame["risk_score"] = proba
        frame["rank"] = frame["risk_score"].rank(method="first", ascending=False).astype(int)
        prediction_frames.append(frame)

    budget = pd.DataFrame(rows).sort_values(["alert_budget", "tp", "precision_at_k"], ascending=[True, False, False])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    positives = predictions.loc[predictions[TARGET].eq(1)].sort_values(["feature_group", "rank"])

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    budget_path = TABLE_DIR / "nlp_taxonomy_alert_budget.csv"
    positives_path = TABLE_DIR / "nlp_taxonomy_positive_week_ranks.csv"
    budget.to_csv(budget_path, index=False)
    positives.to_csv(positives_path, index=False)

    content = f"""# NLP Taxonomy Alert-Budget Diagnostic

## Purpose

This diagnostic compares Random Forest risk rankings under fixed alert budgets. It avoids interpreting validation-selected thresholds as evidence of mechanism.

## Fixed-Budget Results

{budget.to_markdown(index=False)}

## Positive-Week Ranks

{positives[["feature_group", "week", "risk_score", "rank"]].to_markdown(index=False)}

## Interpretation

If a taxonomy or network model captures more true positives at the same alert budget, it is improving risk ranking under limited analyst attention. If it performs similarly to operational or placebo models, then the current taxonomy/network layer is more useful for explanation and auditing than for ranking improvement.
"""
    REPORT.write_text(content, encoding="utf-8")

    print(f"Saved: {budget_path}")
    print(f"Saved: {positives_path}")
    print(f"Report: {REPORT}")
    print(budget.to_string(index=False))


if __name__ == "__main__":
    run()
