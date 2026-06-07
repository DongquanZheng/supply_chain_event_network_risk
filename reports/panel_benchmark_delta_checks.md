# Panel Benchmark Delta Checks

## Purpose

This diagnostic compares `M5_me_strict_network` against operational, unweighted event, total-import network, and placebo alternatives using pooled out-of-sample predictions from the 2023, 2024, and 2025 rolling-origin test folds.

The bootstrap is paired at the country-week prediction level. It is a stability diagnostic, not a formal causal test.

## Random Forest M5 Delta Results

| focus_group          | baseline_group                |   pooled_pr_auc_delta |   pr_delta_ci_low |   pr_delta_ci_high |   pr_delta_bootstrap_p_gt_0 |   pooled_roc_auc_delta |   roc_delta_ci_low |   roc_delta_ci_high |   roc_delta_bootstrap_p_gt_0 |
|:---------------------|:------------------------------|----------------------:|------------------:|-------------------:|----------------------------:|-----------------------:|-------------------:|--------------------:|-----------------------------:|
| M5_me_strict_network | M6b_total_shuffled_placebo    |           0.0220349   |      -0.000512228 |          0.0476229 |                      0.9725 |             0.00533195 |         -0.0191663 |           0.030797  |                       0.67   |
| M5_me_strict_network | M6a_total_equal_placebo       |           0.0135809   |      -0.00550602  |          0.0340349 |                      0.9165 |             0.00397276 |         -0.016285  |           0.0246622 |                       0.6735 |
| M5_me_strict_network | M6c_total_random_placebo      |           0.00971143  |      -0.0135135   |          0.0337346 |                      0.7765 |            -0.00391528 |         -0.0284837 |           0.0204136 |                       0.3885 |
| M5_me_strict_network | M3_external_unweighted_events |           0.00944589  |      -0.0107813   |          0.0285394 |                      0.823  |             0.00325936 |         -0.0168762 |           0.0240476 |                       0.6325 |
| M5_me_strict_network | M2_own_country_news           |           0.00890548  |      -0.0166445   |          0.0383538 |                      0.7535 |             0.0108093  |         -0.0210257 |           0.041015  |                       0.7595 |
| M5_me_strict_network | M1_operational                |           0.00273389  |      -0.0192922   |          0.0239731 |                      0.592  |             0.00165673 |         -0.021921  |           0.024662  |                       0.5635 |
| M5_me_strict_network | M4_total_import_network       |           0.000145349 |      -0.0206995   |          0.0199296 |                      0.464  |            -0.0111981  |         -0.034196  |           0.0107854 |                       0.166  |

## Reading

Positive PR-AUC deltas mean the machinery/electronics strict network model ranks abnormal country-weeks better than the comparison model on pooled test predictions. If intervals cross zero, the result should be framed as directional rather than conclusive.
