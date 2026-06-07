# Reproducibility

## Environment

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

The benchmark scripts are written to run without notebook state.

## Main Pipeline

```bash
python scripts/build_panel_benchmark_dataset.py
python scripts/run_panel_benchmark_models.py
python scripts/analyze_panel_benchmark_deltas.py
python scripts/make_panel_benchmark_figures.py
python -m unittest discover -s tests
```

## Determinism

- Random Forest models use fixed `random_state=42`.
- Random placebo network weights use fixed seeds in the dataset builder.
- Thresholds are selected on validation folds only.
- Tests verify chronological splits, key schema columns, leakage guardrails, and cached random placebo determinism.

## Data Caching

Modeling scripts read cached GDELT outputs from `data/interim/`. They do not query BigQuery directly.

If GDELT features need to be refreshed, use the BigQuery scripts only with:

- partition filters,
- dry-run cost checks,
- deterministic output filenames,
- no credentials committed to the repository.

## Current Benchmark Outputs

- `data/processed/multicountry_container_event_network_benchmark.csv`
- `reports/tables/panel_benchmark_metrics_by_fold.csv`
- `reports/tables/panel_benchmark_summary.csv`
- `reports/tables/panel_benchmark_m5_deltas.csv`
- `reports/panel_benchmark_results.md`
- `reports/panel_benchmark_delta_checks.md`
- `reports/panel_benchmark_paper_conclusion.md`

## Known Limits

The current conclusion depends on country-level PortWatch targets, static 2023 WITS weights, and GDELT proxy event features. These are appropriate for a benchmark paper but not sufficient for causal claims about supply-chain propagation.
