# Supply Chain Event Network Risk

Reproducible Python code for building and evaluating event-informed port disruption prediction features.

The repository combines:

- PortWatch-style weekly country-level container port activity.
- WITS bilateral import dependency weights.
- Cached GDELT GKG country-week event features.
- Temporal benchmark models for next-week abnormal container activity.

The network layer is implemented as an exposure-mapping feature construction step. It should not be interpreted as a causal effect estimate.

## Repository Scope

This public repository is code-first. It includes only the files needed to reproduce the benchmark pipeline:

```text
data/
  raw/           placeholder only
  interim/       placeholder only; cached input files are local
  processed/     placeholder only; generated datasets are local
notebooks/
  reproduce_panel_benchmark.ipynb
scripts/         data, modeling, robustness, and figure-generation scripts
src/             reusable data and feature-construction helpers
tests/           fast offline integrity tests
requirements.txt
```

Private research notes, paper drafts, generated reports, large data files, notebooks with exploratory outputs, and credentials are intentionally excluded from version control.

## Setup

```bash
pip install -r requirements.txt
```

## Required Local Inputs

The main panel pipeline expects cached GDELT feature files under `data/interim/`:

```text
data/interim/gkg_partner_event_features_2021-01-01_2025-12-31.csv
data/interim/gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv
```

These files should be generated from partition-filtered GDELT GKG BigQuery queries and kept local. They are not committed to GitHub.

## Reproduce

Run the full workflow from the project root:

```bash
python scripts/build_panel_benchmark_dataset.py
python scripts/run_panel_benchmark_models.py
python scripts/make_panel_benchmark_figures.py
python scripts/make_supply_chain_network_overview.py
python -m unittest discover -s tests
```

Or run:

```text
notebooks/reproduce_panel_benchmark.ipynb
```

Generated datasets, reports, tables, and figures are written locally under `data/processed/` and `reports/`.

## Data Sources

- PortWatch-style ArcGIS service for country-level port activity.
- World Bank WITS Trade Stats API for bilateral import dependency weights.
- GDELT GKG via BigQuery for event signals.

BigQuery queries should always use partition filters and dry-run cost checks before execution.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check schema expectations, temporal split ordering, leakage guardrails, and deterministic placebo behavior.
