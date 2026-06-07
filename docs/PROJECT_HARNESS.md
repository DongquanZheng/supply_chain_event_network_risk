# Project Harness

This document defines the non-negotiable research and engineering rules for the project. It should be updated only when the research design deliberately changes.

## Research Positioning

This project is not a generic network visualization exercise. The network layer is used to define exposure, dependency, and possible spillover structure for event signals.

The central empirical question is:

> Do network-aligned event signals provide incremental predictive or explanatory value for supply chain disruption outcomes beyond strong operational baselines and simple news controls?

## Core Comparison Design

The project should compare models in an incremental sequence:

1. naive or seasonal baseline;
2. strong operational ML baseline;
3. operational baseline plus news volume and tone controls;
4. operational baseline plus unweighted NLP event signals;
5. operational baseline plus spatially aligned event signals;
6. operational baseline plus network-weighted event exposure;
7. placebo variants using shuffled dates, shuffled locations, or shuffled network weights.

The project must not claim that NLP or network structure is useful merely because adding extra variables improves a weak model.

## Baseline Standard

The operational baseline must be strong enough to be a fair comparison.

Required baseline features, where available:

- lagged operational activity;
- rolling mean;
- rolling volatility;
- recent change or trend;
- seasonality variables;
- recent abnormality history;
- target-specific fixed effects if multiple ports or countries are modeled.

At least one transparent model and one stronger non-linear model should be evaluated where data size allows.

## Temporal Validation

Random train-test splitting is not acceptable for final claims.

Final evaluation should use temporal validation, such as:

- train on earlier years and test on later years;
- walk-forward validation;
- no same-period or future information leakage.

All event signals must be constructed using information available before the predicted outcome period.

## NLP Signal Rules

NLP features must be compared against simpler controls:

- article count;
- source count;
- average tone or sentiment;
- keyword/rule-based event count;
- shuffled or irrelevant-region event signals.

The project should not claim that semantic NLP adds value unless it outperforms these simpler controls in out-of-sample evaluation.

## Network Rules

Trade or maritime connectivity is treated as dependency or exposure structure, not as risk itself.

Acceptable interpretations:

- trade dependency;
- network exposure;
- structural importance;
- potential propagation pathway;
- connectivity context.

Avoid unsupported interpretations:

- trade volume directly equals supply chain risk;
- centrality directly proves disruption vulnerability;
- network association alone proves causality.

## Causal Language Policy

This project may use causal-inspired terms such as exposure mapping, direct exposure, spillover exposure, and placebo checks.

It must not claim causal effects unless a valid causal identification strategy is explicitly established.

Preferred language:

> network-defined exposure provides incremental predictive and explanatory value.

Avoid:

> network exposure causally causes disruption.

## Placebo and Robustness Checks

At minimum, the project should include one or more of:

- shuffled network weights;
- shuffled event dates;
- irrelevant-region event exposure;
- future-event placebo;
- comparison between true network and random network exposure.

Network contribution is credible only if true network exposure performs better than placebo exposure under the same modeling protocol.

## Documentation Rules

Whenever the project makes a confirmed design decision, empirical finding, limitation, or change in scope, it must be recorded in `docs/DECISIONS_AND_RESULTS.md`.

Notebook outputs should not become the only source of truth.

## Engineering Rules

- Keep data acquisition, feature construction, modeling, and evaluation modular.
- Start in notebooks for supervision and interpretation.
- Move stable logic into `src/` or `scripts/` after it has been validated.
- Do not commit large raw data files.
- Keep source URLs, API parameters, and download dates documented.
- Prefer reproducible public data sources.

