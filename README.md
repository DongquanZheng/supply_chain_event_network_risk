# Event-Informed Port Disruption Prediction with Network-Weighted Trade Exposure

This repository builds a reproducible applied data science benchmark for next-week abnormal container port activity prediction.

The core question is:

> Do external event signals improve port disruption prediction beyond operational baselines, and does trade-network-weighted exposure add value over simpler unweighted event controls?

The project combines three layers:

- **ML operational baseline**: lagged PortWatch container activity, rolling volatility, trend, seasonality, and country fixed effects.
- **NLP/event layer**: GDELT GKG country-week event signals, including tone, negative article shares, and disruption-oriented event counts.
- **Network layer**: WITS bilateral import dependency weights that map external event pressure into target-country exposure.

The network layer is treated as an exposure-mapping and interpretation mechanism, not as a causal effect.

## Current Benchmark Status

The main benchmark is now an 11-country panel covering the currently mapped GDELT/ISO3 countries:

Japan, China, Korea, United States, Australia, United Arab Emirates, Saudi Arabia, Vietnam, Thailand, Indonesia, and Germany.

The processed panel dataset contains:

- 2,860 country-week rows
- 314 positive next-week abnormal container labels
- 2020-12-28 to 2025-12-29 weekly coverage
- no missing values in the processed benchmark table

The current formal result is directional:

- The strongest average Random Forest PR-AUC across 2023, 2024, and 2025 rolling-origin test folds comes from `M5_me_strict_network`.
- The gain over simpler and placebo alternatives is modest, and paired bootstrap intervals cross zero.
- The defensible claim is that commodity-specific network exposure is a useful benchmark feature and attribution layer, not that it conclusively dominates every alternative.

See:

- `reports/panel_benchmark_results.md`
- `reports/panel_benchmark_delta_checks.md`
- `reports/panel_benchmark_paper_conclusion.md`
- `docs/DECISIONS_AND_RESULTS.md`

## Data Sources

- **PortWatch / IMF-style ArcGIS service**: country-level daily port activity, aggregated to weekly container port calls.
- **WITS / World Bank Trade Stats API**: 2023 bilateral import dependency weights.
- **GDELT GKG via BigQuery**: cached country-week event features.

BigQuery queries must use partition filters and dry-run cost estimation before execution. Cached GDELT outputs are stored under `data/interim/` and are not intended to be committed if large.

## Public Release Policy

This repository is script-first. Exploratory notebooks, private proposal drafts, credentials, raw/interim/processed data caches, and row-level prediction dumps are excluded from version control.

The public repository keeps reproducible code, documentation, aggregate benchmark outputs, and publication-style figures. See `docs/PUBLIC_RELEASE_CHECKLIST.md` for the release boundary.

## Repository Structure

```text
data/
  interim/       cached API and BigQuery-derived intermediate data
  processed/     model-ready benchmark datasets
docs/            decisions, methodology, benchmark plan
paper/           paper outline and cautious claim variants
reports/
  figures/       generated paper-style figures
  tables/        generated metrics, predictions, diagnostics
scripts/         reproducible data and modeling scripts
src/             reusable data-source and exposure helpers
tests/           fast offline integrity tests
```

## Reproducible Pipeline

Run from the project root.

```bash
python scripts/build_panel_benchmark_dataset.py
python scripts/run_panel_benchmark_models.py
python scripts/analyze_panel_benchmark_deltas.py
python scripts/make_panel_benchmark_figures.py
python -m unittest discover -s tests
```

Primary outputs:

- `data/processed/multicountry_container_event_network_benchmark.csv`
- `reports/panel_benchmark_dataset_summary.md`
- `reports/panel_benchmark_results.md`
- `reports/panel_benchmark_delta_checks.md`
- `reports/panel_benchmark_paper_conclusion.md`
- `reports/tables/panel_benchmark_summary.csv`
- `reports/tables/panel_benchmark_metrics_by_fold.csv`
- `reports/tables/panel_benchmark_m5_deltas.csv`
- `reports/figures/fig_panel_model_comparison_pr_auc.png`
- `reports/figures/fig_panel_pr_auc_by_fold.png`
- `reports/figures/fig_panel_target_exposure_timeseries.png`
- `reports/figures/fig_panel_m5_delta_bootstrap.png`

## Benchmark Models

- `M1_operational`: operational baseline only.
- `M2_own_country_news`: operational baseline plus own-country event controls.
- `M3_external_unweighted_events`: operational baseline plus unweighted external partner-event controls.
- `M4_total_import_network`: operational baseline plus total-import network-weighted exposure.
- `M5_me_strict_network`: operational baseline plus machinery/electronics strict network exposure.
- `M6`: placebo alternatives using equal, shuffled, and random network weights.
- `M7_full_event_network`: supplementary combined feature model.

Evaluation uses temporal rolling-origin validation only. Thresholds are selected on validation years, not test years. PR-AUC is the primary metric because abnormal port-activity labels are rare.

## Limitations

- The network weights are static 2023 WITS import dependencies.
- The current PortWatch target is country-level, not port-level.
- GDELT GKG features are event proxies, not manually validated full-text NLP labels.
- Current network gains are directional and should be reported with placebo and bootstrap caveats.

## Citation Placeholder

Citation details will be added after the benchmark paper draft is prepared.
