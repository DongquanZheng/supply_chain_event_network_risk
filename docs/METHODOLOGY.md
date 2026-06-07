# Methodology Notes

This document will grow into the formal methodology description after notebook experiments validate the design.

## Initial Framework

The planned empirical framework has three layers.

### 1. Operational Baseline

Use historical operational indicators to predict future abnormal activity or disruption-related outcomes.

Candidate feature groups:

- lagged activity;
- rolling activity level;
- rolling volatility;
- recent trend;
- seasonality;
- target-specific historical abnormality.

### 2. Event Signal Layer

Use public news or event data to construct weekly event-risk signals.

Candidate feature groups:

- news volume;
- average tone or sentiment;
- rule-based disruption keyword scores;
- NLP-derived event risk probabilities;
- spatially aligned country, region, or port event signals.

### 3. Network Exposure Layer

Use trade or maritime network structure to convert event signals into exposure measures.

Candidate exposure formulas:

```text
local_exposure[j,t] = event_signal[j,t]

network_exposure[j,t] =
    sum_i dependency_weight[i,j] * event_signal[i,t]

second_order_exposure[j,t] =
    sum_k sum_i dependency_weight[i,k] * dependency_weight[k,j] * event_signal[i,t]
```

These are exposure measures, not causal effects by themselves.

## Planned Evaluation

The project should evaluate whether each additional layer improves out-of-sample prediction and interpretability under temporal validation.

Expected model sequence:

```text
operational baseline
operational + simple news controls
operational + NLP event signal
operational + spatial event signal
operational + network-weighted event exposure
operational + placebo network exposure
```

