# Supply Chain Event Network Risk

Code for a reliability-aware ML-NLP-Network pipeline for port disruption early warning.

The project predicts next-week abnormal container port activity from three public-data layers:

- **PortWatch-style operational data**: port activity, lagged shortfall, and operational vulnerability features.
- **GDELT event signals**: own-country and partner-country external event features.
- **WITS trade network exposure**: bilateral import-dependency weights used for network exposure, attribution, and placebo audits.

The network layer is an exposure-mapping and audit layer. It should not be interpreted as causal propagation evidence.

## Repository Scope

This public repository is code-first. It intentionally excludes generated data, reports, paper drafts, private notes, credentials, and local agent handoff documents.

```text
data/
  raw/           placeholder only
  interim/       placeholder only; cached inputs stay local
  processed/     placeholder only; generated datasets stay local
notebooks/
  reproduce_panel_benchmark.ipynb
scripts/         data collection, feature construction, benchmarks, audits
src/             reusable data and feature-construction helpers
tests/           offline integrity tests
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Some optional model scripts use packages such as XGBoost, LightGBM, or CatBoost when available. The core sklearn workflows remain dependency-safe.

## Local Inputs

Large and source-restricted files are not committed. The main workflows expect locally generated inputs under `data/interim/` and write outputs under `data/processed/`, `reports/`, and `outputs/`.

Typical local inputs include:

```text
data/interim/gkg_partner_event_features_2021-01-01_2025-12-31.csv
data/interim/gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv
```

GDELT GKG feature caches should be generated with partition-filtered BigQuery queries and dry-run cost checks before execution.

## Main Workflow

Run commands from the repository root. Exact availability depends on which local data caches have already been generated.

```bash
# Build panel data
python scripts/build_expanded32_panel_benchmark_dataset.py

# Baseline and model-family benchmarks
python scripts/run_panel32_benchmark_models.py
python scripts/run_panel32_advanced_model_experiment.py

# Network exposure, placebo, and guarded integration checks
python scripts/run_panel32_network_gated_conversion_main.py
python scripts/run_panel32_gdelt_conversion_propensity_benchmark.py
python scripts/run_panel32_country_shared_alert_allocation.py

# Reliability-aware deployment policy
python scripts/run_panel32_high_confidence_alert_policy.py

# Paper-facing tables and audit summaries generated locally
python scripts/make_main_paper_consolidated_tables.py
python scripts/make_main_paper_aprs_tables.py
python scripts/make_gdelt_conversion_audit.py
```

Generated outputs are local-only and ignored by Git.

## Additional Data Utilities

The repository also includes scripts for reproducibly fetching or building supporting inputs:

- PortWatch panel variants: `scripts/fetch_portwatch_*.py`
- WITS dependency weights: `scripts/fetch_panel32_dependency_weights.py`
- GDELT event caches: `scripts/fetch_expanded_gdelt_*.py`
- Scope and feasibility checks: `scripts/make_*feasibility*.py`

## Evaluation Principles

The benchmark uses temporal validation only. Random train/test splits are not used for final claims.

Key evaluation views include:

- PR-AUC for rare positive abnormal-activity weeks.
- Top-k alert precision under fixed alert budgets.
- Severe-event guardrails.
- True WITS versus equal/random/shuffled placebo network audits.
- Leave-one-country-out transfer diagnostics.
- Subgroup checks for operational vulnerability and event conversion.
- APRS, a reporting metric that combines predictive performance, audit reliability, and guardrail robustness.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check schema expectations, temporal split ordering, leakage guardrails, and deterministic placebo behavior.

## Data Sources

- PortWatch-style port activity services.
- GDELT GKG via BigQuery.
- World Bank WITS Trade Stats API.

Users are responsible for complying with each source's access terms and for keeping credentials outside version control.
