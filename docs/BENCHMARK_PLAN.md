# Benchmark Plan

This document locks the scope for turning the current PortWatch + WITS + GDELT GKG project into a reproducible applied benchmark.

## Paper Positioning

Working title:

**Event-Informed Port Disruption Prediction with Network-Weighted Trade Exposure**

Primary benchmark question:

Do external event signals improve next-week abnormal port activity prediction beyond strong operational baselines, and does trade-network-weighted exposure provide additional value over simpler unweighted news/event controls?

The network layer is an exposure-mapping mechanism, not a causal effect.

## Locked MVP Scope

- Primary empirical benchmark: 11-country panel using the currently mapped GDELT/ISO3 countries.
- Japan-centered benchmark: retained as a detailed case study and development MVP.
- Primary operational target: next-week abnormal container port activity from weekly PortWatch country-level `portcalls_container`.
- Primary network/event channel: machinery/electronics dependency exposure.
- Primary external event source: GDELT GKG through BigQuery, cached before modeling.
- Primary trade dependency source: WITS 2023 import weights for each target country in the panel.
- Secondary checks: aggregate port calls and other vessel-type targets only as supplementary robustness, not the main benchmark.

## Current Repository Audit

### Existing Modules

- `src/config.py`: target-country config and GDELT-to-ISO3 mapping.
- `src/portwatch.py`: PortWatch country-level daily fetch and aggregate weekly operational base for total port calls.
- `src/wits.py`: WITS total-import partner fetch and dependency-weight construction.
- `src/gkg_bigquery.py`: partition-filtered GDELT GKG query builder and BigQuery dry-run/run helpers.
- `src/network_exposure.py`: ISO3 attachment, total-import network exposure aggregation, and merge helper.

### Existing Cached Data

- `data/interim/japan_portwatch_weekly_operational_model_base.csv`
- `data/interim/gkg_partner_event_features_2024-10-01_2025-03-31.csv`
- `data/interim/gkg_partner_channel_specific_event_features_2024-10-01_2025-03-31.csv`
- `data/interim/gkg_partner_channel_specific_strict_event_features_2024-10-01_2025-03-31.csv`
- `data/processed/japan_channel_specific_strict_exposure_vessel_labels_2024q4_2025q1.csv`
- `data/processed/japan_strict_channel_pairing_diagnostic_2024q4_2025q1.csv`
- `data/processed/japan_partner_contribution_2024q4_2025q1.csv`

### Existing Validated Notebook Results

- Japan total port-call operational baseline.
- Japan container operational baseline.
- GDELT GKG partner-week event features for a six-month window.
- Total-import exposure redundancy diagnostic.
- Commodity-specific dependency diagnostics.
- Strict machinery/electronics-to-container channel diagnostic.

### Gaps

- No formal benchmark dataset builder script.
- No reproducible M1-M6 evaluation script.
- No placebo-check script.
- No benchmark reports or paper figures.
- No tests for leakage, temporal splits, deterministic placebo weights, or dataset schema.
- Machinery/electronics WITS product-group weights and strict channel-specific GKG logic are not yet modularized in `src/`.

## Benchmark Model Sequence

- M1: operational baseline only.
- M2: operational baseline + simple news controls.
- M3: operational baseline + unweighted partner event signal.
- M4: operational baseline + total-import network-weighted exposure.
- M5: operational baseline + machinery/electronics network-weighted strict exposure.
- M6: operational baseline + placebo network exposure.

Evaluation must use temporal validation only. Thresholds must be selected on validation data, never on test data.

## Phase TODO

### Phase 1 - Audit and Scope Lock

- [x] Inspect repository structure.
- [x] Read current project documentation.
- [x] Identify existing validated modules, notebook-only logic, and cached data.
- [x] Write this benchmark plan.

### Phase 2 - Benchmark Dataset

- [x] Create `scripts/build_benchmark_dataset.py`.
- [x] Build Japan weekly container operational target and features.
- [x] Load or build WITS total-import and machinery/electronics dependency weights.
- [x] Load cached GDELT GKG partner-week features.
- [x] Compute unweighted, total-import, machinery/electronics, equal-weight, shuffled-weight, and random-weight exposures.
- [x] Save `data/processed/japan_container_event_network_benchmark.csv`.
- [x] Write `reports/benchmark_dataset_summary.md`.

### Phase 3 - Benchmark Evaluation

- [x] Create `scripts/run_benchmark_models.py`.
- [x] Implement temporal train/validation/test evaluation.
- [x] Evaluate Logistic Regression and Random Forest for M1-M6.
- [x] Select F1 threshold on validation data only.
- [x] Save metrics, confusion matrices, and interpretation tables under `reports/tables/`.
- [x] Write `reports/benchmark_results.md`.

### Phase 4 - Placebo and Robustness

- [x] Create `scripts/run_placebo_checks.py`.
- [x] Evaluate equal, shuffled, and random network exposures.
- [ ] Add optional future-event placebo only if leakage-safe.
- [x] Save `reports/tables/placebo_results.csv`.
- [x] Write `reports/placebo_checks.md`.

### Phase 5 - Figures

- [x] Create `scripts/make_benchmark_figures.py`.
- [x] Generate target time series, model comparison, exposure time series, partner contribution, and exposure-correlation figures.
- [x] Save figures under `reports/figures/`.

### Phase 5b - Test-Week Diagnostics

- [x] Create `scripts/analyze_benchmark_diagnostics.py`.
- [x] Save selected test predictions under `reports/tables/test_predictions_selected.csv`.
- [x] Write `reports/benchmark_diagnostics.md`.

### Phase 6 - Paper Scaffold

- [x] Create `paper/outline.md`.
- [x] Create `paper/contribution_claims.md` with strong, mixed, and null-result claim variants.

### Phase 7 - Reproducibility and Tests

- [ ] Update `README.md`.
- [ ] Add `docs/REPRODUCIBILITY.md`.
- [ ] Update `docs/DATA_SOURCES.md`.
- [x] Add fast tests for panel schema, temporal splits, target leakage, deterministic placebo outputs, and key columns.

## Immediate Next Step

The next step is paper preparation and robustness tightening. The current panel result should be framed as directional evidence: machinery/electronics strict network exposure is the strongest average Random Forest PR-AUC feature group in the multi-country panel, but paired bootstrap intervals still cross zero. The benchmark can support a cautious applied-data-science contribution, not a causal claim.

## Panel Benchmark Addendum

- `scripts/build_panel_benchmark_dataset.py` builds `data/processed/multicountry_container_event_network_benchmark.csv`.
- `scripts/run_panel_benchmark_models.py` runs rolling-origin M1-M7 panel evaluation.
- `scripts/analyze_panel_benchmark_deltas.py` runs paired bootstrap delta checks for the main M5 network model.
- `scripts/make_panel_benchmark_figures.py` creates panel benchmark figures.
- Results are saved under `reports/panel_benchmark_results.md`, `reports/panel_benchmark_delta_checks.md`, and `reports/panel_benchmark_paper_conclusion.md`.
