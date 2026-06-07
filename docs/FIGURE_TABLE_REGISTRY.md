# Figure and Table Registry

This registry records what each generated figure or table is for. It is meant to prevent later confusion when writing the paper, README, slides, or rebuttal material.

## Recommended Main Paper Figures

| File | Role | What it shows | Suggested placement |
| --- | --- | --- | --- |
| `reports/figures/fig_pipeline_overview.png` | Pipeline overview | How PortWatch, WITS, GDELT GKG, network exposure, and temporal ML fit together. | Data / benchmark design |
| `reports/figures/fig_all_country_supply_chain_network.png` | Network visualization | Import-dependency network with 2025 event-exposure scores. Node size is dependency importance; node color is event exposure; edges show selected total-import and machinery/electronics dependency links. | Event and network exposure features |
| `reports/figures/fig_panel_target_exposure_timeseries.png` | Data behavior | Panel-level weekly container activity, event-network exposure, and positive-label rate over time. | Data / exploratory analysis |
| `reports/figures/fig_panel_model_comparison_pr_auc.png` | Core benchmark result | Random Forest mean PR-AUC across rolling test folds for M1-M6 style feature groups. | Results |
| `reports/figures/fig_panel_model_comparison_auc.png` | Core benchmark result | PR-AUC and ROC-AUC side by side, showing why PR-AUC should be emphasized under class imbalance. | Results |
| `reports/figures/fig_panel_pr_auc_by_fold.png` | Temporal stability | PR-AUC across rolling-origin test years for key model groups. | Results / robustness |
| `reports/figures/fig_panel_m5_delta_bootstrap.png` | Incremental value check | Paired bootstrap PR-AUC deltas comparing M5 against baselines and placebos. | Robustness |
| `reports/figures/fig_panel_exposure_correlation.png` | Feature redundancy diagnostic | Correlations among unweighted, network-weighted, equal-weight, and shuffled exposure features. | Discussion / robustness |

## Recommended Appendix or Supplementary Figures

| File | Role | What it shows | Suggested placement |
| --- | --- | --- | --- |
| `reports/figures/fig_panel_country_network_gain.png` | Heterogeneity diagnostic | Country-level M5 minus M3 PR-AUC gain, showing where machinery/electronics network weighting helps or hurts. | Appendix |
| `reports/figures/fig_panel_leave_one_country_out.png` | Generalization diagnostic | Leave-one-country-out PR-AUC by feature group; useful for cross-country transfer claims. | Appendix / robustness |
| `reports/figures/fig_panel_counterfactual_network_swap.png` | Falsification diagnostic | Performance under target-specific and donor-country network weights; tests whether exact target-specific weights are uniquely supported. | Appendix / robustness |
| `reports/figures/fig_all_country_supply_chain_network_linear_original.png` | Visual comparison backup | Earlier network figure with linear node sizing before publication-style refinements. | Internal reference only |

## Japan-Centered MVP Figures

These are useful for explaining the original Japan-centered prototype, but the main paper should prioritize the multi-country panel figures unless the paper explicitly includes a Japan case study.

| File | Role | What it shows |
| --- | --- | --- |
| `reports/figures/fig_target_timeseries.png` | Japan target visualization | Japan weekly container calls and positive next-week abnormal labels. |
| `reports/figures/fig_model_comparison_auc.png` | Japan benchmark result | Japan-centered model comparison for PR-AUC and ROC-AUC. |
| `reports/figures/fig_network_exposure_timeseries.png` | Japan exposure time series | Japan event exposure features over time. |
| `reports/figures/fig_partner_contributions.png` | Attribution case study | Partner contributions to Japan machinery/electronics network exposure in selected positive weeks. |
| `reports/figures/fig_exposure_correlation.png` | Japan feature redundancy | Correlation among Japan exposure features. |

## Recommended Main Paper Tables

| File | Role | What it shows | Suggested placement |
| --- | --- | --- | --- |
| `reports/tables/panel_benchmark_summary.csv` | Main benchmark table | Mean ROC-AUC, PR-AUC, F1, precision, recall, and aggregate confusion counts by feature group and model. | Results |
| `reports/tables/panel_benchmark_metrics_by_fold.csv` | Fold-level benchmark table | Per-fold temporal validation metrics. | Results / appendix |
| `reports/tables/panel_benchmark_m5_deltas.csv` | Incremental value table | Paired bootstrap PR-AUC deltas for M5 against baselines and placebos. | Robustness |
| `reports/tables/all_country_network_overview_edges.csv` | Network figure support table | Edge list used in the all-country network figure, including dependency layer and import share. | Data appendix |

## Recommended Appendix or Diagnostic Tables

| File | Role | What it shows |
| --- | --- | --- |
| `reports/tables/panel_leave_one_country_out_summary.csv` | Generalization summary | Mean leave-one-country-out performance by feature group. |
| `reports/tables/panel_leave_one_country_out_results.csv` | Generalization detail | Per-holdout-country results. |
| `reports/tables/panel_counterfactual_network_swap_summary.csv` | Counterfactual summary | Average performance under donor-country network swaps. |
| `reports/tables/panel_counterfactual_network_swap_by_fold.csv` | Counterfactual detail | Donor-network swap results by fold. |
| `reports/tables/panel_network_mechanism_country_results.csv` | Country heterogeneity | Country-level PR-AUC and dependency concentration diagnostics. |
| `reports/tables/panel_network_mechanism_correlations.csv` | Mechanism diagnostic | Correlations between dependency concentration measures and network gains. |
| `reports/tables/panel_benchmark_model_interpretation.csv` | Model interpretation | Coefficients or feature importance summaries. |
| `reports/tables/panel_benchmark_predictions.csv` | Prediction audit | Out-of-sample predictions by model, fold, country, and week. Useful for case studies. |

## Current Caption for Network Figure

Figure X. Import-dependency network with 2025 event-exposure scores. Nodes represent countries, with node size proportional to each country’s summed dependency importance using a linear scale. Directed edges point from partner countries to exposed target countries, and edge width reflects the 2023 WITS import-dependency share. Orange edges indicate the top four machinery/electronics dependency links per target country, while blue edges indicate the top three total-import dependency links per target country. Node color represents the average machinery/electronics-related event-exposure score in 2025. Node positions are determined by a force-directed layout for visualization only and should not be interpreted as geographic or metric distances. Only the strongest dependency links are shown for readability.

