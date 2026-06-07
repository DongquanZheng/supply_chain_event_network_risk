# Paper Outline

Working title:

**Event-Informed Port Disruption Prediction with Network-Weighted Trade Exposure**

## 1. Introduction

- Motivation: port activity disruptions are shaped by operational dynamics and external events.
- Gap: event-informed disruption prediction often lacks transparent comparison against strong operational baselines and network/placebo controls.
- Research question: do external event signals improve next-week abnormal port activity prediction, and does trade-network-weighted exposure add value beyond unweighted event controls?
- Contribution summary: reproducible benchmark, event-network feature pipeline, temporal evaluation, placebo checks, and mixed evidence on network-weighted exposure.

## 2. Related Work

- Port disruption and maritime risk prediction.
- Event-informed supply-chain risk analytics.
- Trade and supply-chain network exposure.
- Benchmarking and temporal validation for rare disruption events.

## 3. Data

- PortWatch country-level weekly container port calls for Japan.
- WITS bilateral import dependency weights.
- GDELT GKG partner-country event features.
- Cached data windows and reproducibility constraints.

## 4. Benchmark Design

- Primary target: next-week abnormal Japan container port activity.
- Target construction: rolling 12-week historical threshold.
- Temporal split: train 2021-2023, validation 2024, test 2025.
- Rare-event setting and PR-AUC as central metric.

## 5. Event and Network Exposure Features

- M1 operational features.
- M2 simple news/event controls.
- M3 unweighted machinery/electronics event signal.
- M4 total-import network-weighted exposure.
- M5 machinery/electronics network-weighted exposure.
- M6 placebo exposure variants.

## 6. Experimental Setup

- Logistic Regression with class balancing.
- Random Forest with class balancing.
- Threshold selection on validation F1 only.
- Test metrics: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Bootstrap confidence intervals where feasible.

## 7. Results

- M1-M6 benchmark table.
- Random Forest versus Logistic Regression comparison.
- Test-set PR-AUC and ROC-AUC.
- Thresholded alerting tradeoff.

## 8. Placebo and Robustness Checks

- Equal weights.
- Shuffled weights.
- Random weights with fixed seed.
- Interpretation of placebo competitiveness.

## 9. Discussion and Limitations

- Current results do not support universal network-weighting superiority.
- Network-weighted machinery/electronics exposure acts as a stricter alert filter in the current benchmark.
- The network layer provides interpretable exposure attribution even when PR-AUC does not improve.
- Limitations: Japan-centered design, small test positive count, static trade weights, country-level rather than port-level network structure.

## 10. Conclusion

- External event signals improve over operational-only baselines in the current benchmark.
- Simple news controls currently provide the strongest PR-AUC.
- Network-weighted exposure should be treated as a relevance-filtering and attribution mechanism, with predictive value tested rather than assumed.
