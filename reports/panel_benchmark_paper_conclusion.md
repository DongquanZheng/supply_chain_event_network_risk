# Paper-Facing Panel Benchmark Conclusion

## What We Can Claim Now

The strongest current result comes from the multi-country panel benchmark, not from the Japan-only benchmark.

Across 11 countries and three rolling-origin test years, the Random Forest model using machinery/electronics strict network exposure achieves the highest mean PR-AUC among the tested feature groups:

- `M5_me_strict_network`: mean PR-AUC `0.223`
- `M4_total_import_network`: mean PR-AUC `0.212`
- `M1_operational`: mean PR-AUC `0.210`
- `M2_own_country_news`: mean PR-AUC `0.205`
- `M3_external_unweighted_events`: mean PR-AUC `0.205`

This supports a benchmark-level claim that commodity-specific trade-network exposure can add useful ranking information in a cross-country setting.

## What We Cannot Claim Yet

The result is not a decisive network victory. Paired bootstrap intervals for M5 PR-AUC deltas cross zero, and the advantage over total-import network exposure is negligible. Therefore, the paper should not claim that network weighting universally or statistically dominates simpler alternatives.

## Best Current Framing

This project should be positioned as a reproducible benchmark for event-informed port disruption prediction with network-weighted exposure.

The most defensible mechanism is:

1. Operational ML captures endogenous port-activity vulnerability.
2. NLP/GDELT event features measure external event pressure.
3. Trade networks convert general event pressure into country-specific exposure.
4. Commodity-specific network exposure is more coherent than a Japan-only static network or unweighted external news pressure.

## Stronger Mechanism After Additional Diagnostics

The additional mechanism checks suggest that the network layer should not be treated as one monolithic feature family.

- **Total-import network exposure** is more useful for leave-one-country-out generalization. When the model trains on 10 countries and tests on a completely unseen country, `M4_total_import_network` has the highest mean PR-AUC (`0.278`), above external unweighted events (`0.274`) and the operational baseline (`0.262`).
- **Machinery/electronics strict network exposure** is stronger in the standard multi-country panel benchmark, where it achieves the best Random Forest mean PR-AUC (`0.223`).
- **Exact target-specific machinery/electronics weights** are not yet uniquely validated: three donor-network swaps slightly outperform the true target-specific network, so this should be framed as commodity-channel exposure rather than precise causal dependency mapping.

This yields a sharper contribution:

> Network granularity matters. Broad trade networks can support cross-country transfer, while commodity-specific networks provide interpretable channel exposure in panel prediction.

## Suggested Abstract-Level Claim

We construct a reproducible PortWatch-WITS-GDELT benchmark for next-week abnormal container port activity prediction. In a multi-country panel, commodity-specific machinery/electronics network exposure achieves the strongest average PR-AUC among tested Random Forest feature groups. In leave-one-country-out evaluation, broader total-import network exposure provides stronger cross-country generalization than commodity-specific exposure. These results suggest that network granularity is a key design choice: broad networks support transfer, while commodity-specific networks support channel-level exposure attribution. Placebo, counterfactual network-swap, and bootstrap diagnostics show that the gains are directional rather than conclusive, motivating future work on dynamic and more granular event-network mappings.

## Main Limitation

The current network layer uses static 2023 WITS dependency weights and country-level PortWatch activity. Future work should test longer windows, dynamic trade weights, finer port-level targets, and stronger event taxonomy features before making causal claims about supply-chain disruption propagation.
