from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_benchmark_models import (  # noqa: E402
    DEFAULT_DATASET,
    FEATURE_GROUPS,
    TABLE_DIR,
    evaluate_model,
    load_dataset,
    make_models,
    temporal_split,
)


PLACEBO_TABLE = TABLE_DIR / "placebo_results.csv"
PLACEBO_MD = PROJECT_ROOT / "reports" / "placebo_checks.md"

PLACEBO_GROUPS = {
    "M3_unweighted_me_event": FEATURE_GROUPS["M3_unweighted_me_event"],
    "M5_me_network": FEATURE_GROUPS["M5_me_network"],
    "M6_equal_placebo": FEATURE_GROUPS["M6_equal_placebo"],
    "M6_shuffled_placebo": FEATURE_GROUPS["M6_shuffled_placebo"],
    "M6_random_placebo": FEATURE_GROUPS["M6_random_placebo"],
}


def write_report(results: pd.DataFrame, path: Path) -> None:
    rf = results[results["model"].eq("random_forest")].copy()
    rf = rf.sort_values("pr_auc", ascending=False)

    content = f"""# Placebo Checks

## Purpose

These checks test whether machinery/electronics network exposure adds value beyond simpler or intentionally weakened exposure mappings.

## Placebo Definitions

- `M3_unweighted_me_event`: partner event signal aggregated without trade weights.
- `M5_me_network`: machinery/electronics dependency-weighted exposure.
- `M6_equal_placebo`: equal partner weights.
- `M6_shuffled_placebo`: real weights assigned to the wrong partners.
- `M6_random_placebo`: deterministic random weights with fixed seed.

## Evaluation Protocol

- Train: 2021-2023
- Validation: 2024
- Test: 2025
- Threshold: selected on validation F1 only
- Main metric: test PR-AUC

## Random Forest Placebo Summary

{rf[["feature_group", "roc_auc", "pr_auc", "precision", "recall", "f1", "selected_threshold"]].to_markdown(index=False)}

## Interpretation Guardrail

If placebo variants perform similarly to the true network, the project should not claim that trade-network weighting independently improves prediction. In that case, the network contribution should be framed as exposure attribution and a transparent benchmark component, not as proven predictive superiority.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    df = load_dataset(Path(args.dataset))
    train, validation, test = temporal_split(df)

    rows = []
    for group_name, features in PLACEBO_GROUPS.items():
        for model_name, model in make_models().items():
            metrics, _, _ = evaluate_model(
                model_name,
                model,
                group_name,
                features,
                train,
                validation,
                test,
            )
            rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PLACEBO_TABLE, index=False)
    write_report(results, PLACEBO_MD)

    print(f"Saved placebo results: {PLACEBO_TABLE}")
    print(f"Saved placebo report: {PLACEBO_MD}")
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
