# Data Sources

## PortWatch Country-Level Port Activity

Used for weekly operational targets and baseline features.

- Unit: country-week.
- Main target: next-week abnormal `portcalls_container`.
- Construction: daily country-level records are aggregated into complete Monday-start weeks.
- Abnormal label: next-week container port calls below the current rolling 12-week mean minus 1.5 rolling standard deviations.

## WITS Trade Stats API

Used for trade dependency weights.

- Total-import network: 2023 bilateral import value by target country and partner country.
- Machinery/electronics network: WITS product group `84-85_MachElec`.
- Weight definition: partner import share normalized within the currently mapped partner set.

The weights are exposure mappings, not causal propagation estimates.

## GDELT GKG BigQuery

Used for external event signals.

- General GKG partner-week features: tone, negative article share, very-negative article share, transport/trade counts, broad risk-theme counts.
- Machinery/electronics strict event features: channel-specific very-negative share and article count.
- All GDELT-derived outputs must be cached before modeling.

BigQuery guardrails:

- Always use partition filters.
- Always dry-run and record estimated bytes before execution.
- Never run unbounded scans.
- Do not commit credentials or local ADC files.

## Current Cached Inputs

- `data/interim/gkg_partner_event_features_2021-01-01_2025-12-31.csv`
- `data/interim/gkg_partner_me_strict_event_features_2021-01-01_2025-12-31.csv`
- `data/interim/gkg_nlp_taxonomy_partner_week_2023-01-01_2025-12-31.csv`

Large cached files are kept local under `data/interim/`.
