# Panel Benchmark Results

## Dataset

- File: `data/processed/multicountry_container_event_network_benchmark.csv`
- Countries: 11
- Rows: 2860
- Positive labels: 314
- Positive rate: 0.110
- Week range: 2020-12-28 to 2025-12-29

## Evaluation

- Temporal rolling-origin validation.
- Test folds: 2023, 2024, and 2025.
- Thresholds are selected on the immediately preceding validation year only.
- Main ranking metric: PR-AUC, because abnormal weeks are rare.
- Models: balanced Logistic Regression and Random Forest.

## Top Mean Results

| feature_group                 | model         |   mean_pr_auc |   std_pr_auc |   mean_roc_auc |   mean_f1 |   mean_precision |   mean_recall |   total_tp |   total_fp |   total_fn |
|:------------------------------|:--------------|--------------:|-------------:|---------------:|----------:|-----------------:|--------------:|-----------:|-----------:|-----------:|
| M5_me_strict_network          | random_forest |      0.222682 |    0.0738766 |       0.664684 |  0.258277 |         0.24122  |      0.547826 |        106 |        539 |         90 |
| M6d_me_placebo_bundle         | logistic      |      0.219189 |    0.0623443 |       0.669323 |  0.19524  |         0.225906 |      0.405475 |         77 |        424 |        119 |
| M4_total_import_network       | logistic      |      0.2187   |    0.0648356 |       0.661536 |  0.265968 |         0.200774 |      0.47841  |         94 |        437 |        102 |
| M6a_total_equal_placebo       | logistic      |      0.217917 |    0.0563883 |       0.663952 |  0.254933 |         0.229404 |      0.357167 |         69 |        304 |        127 |
| M6d_me_placebo_bundle         | random_forest |      0.216581 |    0.0820903 |       0.653616 |  0.170841 |         0.169821 |      0.276105 |         55 |        257 |        141 |
| M6c_total_random_placebo      | logistic      |      0.216544 |    0.0614066 |       0.662301 |  0.279685 |         0.236816 |      0.375514 |         74 |        265 |        122 |
| M7_full_event_network         | logistic      |      0.2145   |    0.0534231 |       0.652286 |  0.241087 |         0.187806 |      0.467191 |         91 |        475 |        105 |
| M2_own_country_news           | logistic      |      0.213929 |    0.049893  |       0.66477  |  0.239528 |         0.236282 |      0.274282 |         53 |        193 |        143 |
| M5_me_strict_network          | logistic      |      0.212864 |    0.0555689 |       0.664464 |  0.220456 |         0.203291 |      0.483763 |         91 |        505 |        105 |
| M1_operational                | logistic      |      0.212571 |    0.0683865 |       0.670464 |  0.282754 |         0.198714 |      0.541443 |        106 |        482 |         90 |
| M4_total_import_network       | random_forest |      0.212431 |    0.0662021 |       0.667492 |  0.210568 |         0.192311 |      0.357521 |         70 |        281 |        126 |
| M3_external_unweighted_events | logistic      |      0.211631 |    0.063995  |       0.666801 |  0.271066 |         0.20144  |      0.425389 |         85 |        339 |        111 |

## Random Forest Benchmark Comparison

| feature_group                 |   mean_pr_auc |   std_pr_auc |   mean_roc_auc |   mean_f1 |   mean_precision_at_25 |   total_tp |   total_fp |   total_fn |
|:------------------------------|--------------:|-------------:|---------------:|----------:|-----------------------:|-----------:|-----------:|-----------:|
| M5_me_strict_network          |      0.222682 |    0.0738766 |       0.664684 |  0.258277 |               0.293333 |        106 |        539 |         90 |
| M4_total_import_network       |      0.212431 |    0.0662021 |       0.667492 |  0.210568 |               0.293333 |         70 |        281 |        126 |
| M6c_total_random_placebo      |      0.210736 |    0.0671403 |       0.662086 |  0.27666  |               0.293333 |         99 |        430 |         97 |
| M1_operational                |      0.209636 |    0.0620284 |       0.654868 |  0.210813 |               0.253333 |         58 |        218 |        138 |
| M6a_total_equal_placebo       |      0.209545 |    0.0772069 |       0.660517 |  0.233841 |               0.253333 |         74 |        313 |        122 |
| M2_own_country_news           |      0.205166 |    0.0796078 |       0.643613 |  0.229798 |               0.266667 |         88 |        448 |        108 |
| M3_external_unweighted_events |      0.204995 |    0.0712705 |       0.658205 |  0.24748  |               0.24     |        102 |        483 |         94 |
| M7_full_event_network         |      0.204844 |    0.0617454 |       0.661198 |  0.228922 |               0.24     |         71 |        294 |        125 |
| M6b_total_shuffled_placebo    |      0.197326 |    0.0602486 |       0.65215  |  0.254917 |               0.213333 |         98 |        435 |         98 |

## Interpretation

This report should be read as the formal panel benchmark, not as a causal estimate. The network layer is an exposure-mapping mechanism: it asks whether external event pressure becomes more useful when routed through observed trade dependencies. If network-weighted variants outperform unweighted and placebo variants across folds, that supports predictive value. If they do not, the defensible contribution is still a reproducible event-network benchmark and evidence about when simple network weighting is redundant.
