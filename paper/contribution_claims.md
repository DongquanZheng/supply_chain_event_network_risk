# Contribution Claim Variants

This file records cautious claim variants for the paper. The final manuscript should choose the version supported by the locked benchmark results.

## Strong Result Version

Commodity-specific trade-network exposure improves event-informed port disruption prediction beyond operational baselines and unweighted event controls. The result suggests that supply-chain network structure can act as a relevance-mapping mechanism for external event signals.

Use only if:

- M5 consistently outperforms M1-M4 and M6 on PR-AUC across validation designs.
- Placebo variants are clearly weaker.
- Confidence intervals do not make the result ambiguous.

## Mixed Result Version

External event signals improve next-week abnormal container activity prediction beyond operational baselines, but network weighting does not universally improve ranking performance. Machinery/electronics network exposure changes the precision-recall tradeoff and provides interpretable dependency-channel attribution, suggesting that network structure is useful as a relevance-filtering mechanism even when predictive gains are mixed.

Use if current results remain stable:

- M2 has the best PR-AUC.
- M5 improves or changes thresholded alert selectivity.
- Placebo variants remain competitive.
- Partner/channel attribution remains interpretable.

## Null Result Version

The benchmark shows that simple event/news controls can outperform naive trade-network weighting for Japan container disruption prediction. This negative result highlights the importance of strong operational baselines, placebo networks, and channel-specific validation before claiming network-informed predictive value in event-driven supply-chain risk analytics.

Use if:

- M5 performs no better than M1-M4 or placebo variants.
- Case studies do not provide useful exposure attribution.
- Network features remain redundant or unstable.

## Current Preferred Claim

The current benchmark supports a cautious mixed/null-borderline version. Simple news controls currently win test PR-AUC, and stricter diagnostics show that machinery/electronics network exposure does not yet prove unique structural regularization over unweighted or placebo event signals. The strongest current contribution is a transparent benchmark showing how to test event-informed prediction, network-weighted exposure, placebo networks, and false-alert filtering claims without overclaiming network superiority.
