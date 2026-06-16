# Public Result Snapshot

This folder provides a small, sanitized snapshot of the current main-paper evidence so that a GitHub reader can inspect the core empirical results without local `reports/`, `paper/`, or private project notes.

The full generated reports, manuscript drafts, cached data, and agent handoff documents remain local-only by design. The tables here are lightweight exports of paper-facing result tables generated from the PortWatch/GDELT/WITS pipeline.

## Scope

- Panel: expanded32 country-week panel.
- Rows: 8,384 country-weeks.
- Positive labels: 892 next-week abnormal container-activity weeks.
- Positive rate: 0.106.
- Week range: 2020-12-28 to 2025-12-29.
- Main data sources: PortWatch-style operational activity, GDELT GKG event signals, and WITS import-dependency weights.

## Files

- `tables/data_task_summary.csv`: dataset and prediction-task summary.
- `tables/main_model_ladder.csv`: temporal and LOCO model-ladder results.
- `tables/network_audit.csv`: true WITS versus additive/placebo audit contrasts.
- `tables/deployment_subgroups.csv`: target sensitivity and operational-shortfall subgroup checks.
- `tables/high_confidence_alert_policy.csv`: known-country temporal deployment policy and matched placebo comparison.

## Reading Guardrails

The public snapshot supports a reliability-aware framing, not a universal network-win claim.

- GDELT/event features provide noisy but sometimes useful external signals.
- WITS is used as exposure mapping, attribution, and placebo audit structure.
- True WITS does not consistently dominate equal/random/shuffled placebos.
- The high-confidence policy improves known-country temporal PR-AUC/top-k/APRS in the current run, but it is not a LOCO transfer claim.
- No causal propagation claim is made.
