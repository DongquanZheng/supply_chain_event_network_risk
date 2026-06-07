# Placebo Checks

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

| feature_group          |   roc_auc |   pr_auc |   precision |   recall |       f1 |   selected_threshold |
|:-----------------------|----------:|---------:|------------:|---------:|---------:|---------------------:|
| M3_unweighted_me_event |  0.888889 | 0.547619 |    0.208333 |      1   | 0.344828 |                 0.32 |
| M6_equal_placebo       |  0.888889 | 0.547619 |    0.208333 |      1   | 0.344828 |                 0.32 |
| M6_random_placebo      |  0.884444 | 0.545455 |    0.25     |      0.8 | 0.380952 |                 0.54 |
| M5_me_network          |  0.884444 | 0.525359 |    0.307692 |      0.8 | 0.444444 |                 0.59 |
| M6_shuffled_placebo    |  0.884444 | 0.43     |    0.217391 |      1   | 0.357143 |                 0.32 |

## Interpretation Guardrail

If placebo variants perform similarly to the true network, the project should not claim that trade-network weighting independently improves prediction. In that case, the network contribution should be framed as exposure attribution and a transparent benchmark component, not as proven predictive superiority.
