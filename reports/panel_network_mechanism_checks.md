# Panel Network Mechanism Checks

## Question

Does machinery/electronics network exposure help more for countries whose machinery/electronics import dependency is structurally concentrated?

This test is exploratory because there are only 11 countries, but it is important for mechanism building. A positive relationship would support the idea that network value is conditional on dependency structure rather than being a generic feature-engineering trick.

## Country-Level Results

| ISO3   | country              |   M1_operational |   M3_external_unweighted_events |   M4_total_import_network |   M5_me_strict_network |   M6b_total_shuffled_placebo |   m5_minus_m1_pr_auc |   m5_minus_m3_pr_auc |   m5_minus_m4_pr_auc |   m5_minus_shuffled_pr_auc |   me_hhi |   me_top1_share |   me_top3_share |   me_entropy |   me_effective_partners |
|:-------|:---------------------|-----------------:|--------------------------------:|--------------------------:|-----------------------:|-----------------------------:|---------------------:|---------------------:|---------------------:|---------------------------:|---------:|----------------:|----------------:|-------------:|------------------------:|
| ARE    | United Arab Emirates |         0.250895 |                        0.228996 |                  0.241048 |               0.256521 |                     0.309749 |           0.00562541 |           0.0275252  |           0.0154723  |                -0.053228   | 0.429389 |        0.631016 |        0.860975 |      1.2803  |                 3.59773 |
| AUS    | Australia            |         0.212889 |                        0.166084 |                  0.17331  |               0.182903 |                     0.148676 |          -0.0299854  |           0.0168196  |           0.00959278 |                 0.0342271  | 0.351097 |        0.5436   |        0.827353 |      1.42493 |                 4.15756 |
| CHN    | China                |         0.265088 |                        0.286481 |                  0.272409 |               0.310315 |                     0.249539 |           0.0452273  |           0.0238347  |           0.0379062  |                 0.060776   | 0.209399 |        0.308484 |        0.727904 |      1.68746 |                 5.40571 |
| DEU    | Germany              |         0.320006 |                        0.314252 |                  0.377047 |               0.351308 |                     0.265361 |           0.0313019  |           0.0370564  |          -0.0257392  |                 0.0859476  | 0.451291 |        0.644854 |        0.883477 |      1.18053 |                 3.2561  |
| IDN    | Indonesia            |         0.31321  |                        0.391176 |                  0.279143 |               0.281996 |                     0.406319 |          -0.0312141  |          -0.109181   |           0.00285288 |                -0.124323   | 0.429288 |        0.634974 |        0.826511 |      1.30442 |                 3.68555 |
| JPN    | Japan                |         0.227273 |                        0.238735 |                  0.237301 |               0.219021 |                     0.250213 |          -0.00825168 |          -0.0197135  |          -0.0182795  |                -0.0311919  | 0.41211  |        0.616854 |        0.817296 |      1.30282 |                 3.67967 |
| KOR    | Korea                |         0.174538 |                        0.164512 |                  0.191118 |               0.1676   |                     0.159448 |          -0.00693799 |           0.00308796 |          -0.0235181  |                 0.00815239 | 0.336127 |        0.531194 |        0.805982 |      1.40262 |                 4.06584 |
| SAU    | Saudi Arabia         |         0.208439 |                        0.188182 |                  0.206934 |               0.180152 |                     0.277932 |          -0.0282875  |          -0.00803036 |          -0.0267822  |                -0.0977808  | 0.349445 |        0.547821 |        0.820207 |      1.46159 |                 4.3128  |
| THA    | Thailand             |         0.119601 |                        0.127242 |                  0.132223 |               0.144294 |                     0.135609 |           0.0246924  |           0.0170514  |           0.012071   |                 0.0086852  | 0.357313 |        0.550476 |        0.832481 |      1.38711 |                 4.00327 |
| USA    | United States        |         0.339963 |                        0.308681 |                  0.404889 |               0.315645 |                     0.357293 |          -0.0243181  |           0.00696407 |          -0.089244   |                -0.0416474  | 0.272691 |        0.465567 |        0.716899 |      1.59846 |                 4.94542 |
| VNM    | Vietnam              |         0.235428 |                        0.2407   |                  0.225869 |               0.243943 |                     0.190806 |           0.00851449 |           0.00324272 |           0.0180737  |                 0.0531365  | 0.35771  |        0.497498 |        0.909313 |      1.26744 |                 3.55175 |

## Concentration vs Network-Gain Correlations

| concentration_metric   | delta_metric             |   pearson_corr |   spearman_corr |
|:-----------------------|:-------------------------|---------------:|----------------:|
| me_top3_share          | m5_minus_m4_pr_auc       |      0.35391   |       0.354545  |
| me_top3_share          | m5_minus_m3_pr_auc       |      0.0215112 |       0.336364  |
| me_top3_share          | m5_minus_shuffled_pr_auc |      0.162821  |       0.272727  |
| me_top3_share          | m5_minus_m1_pr_auc       |      0.0864474 |       0.236364  |
| me_hhi                 | m5_minus_m3_pr_auc       |     -0.269665  |       0.118182  |
| me_hhi                 | m5_minus_m4_pr_auc       |      0.0285119 |       0.0636364 |
| me_hhi                 | m5_minus_m1_pr_auc       |     -0.17697   |       0.0181818 |
| me_top1_share          | m5_minus_m3_pr_auc       |     -0.297522  |       0.0181818 |
| me_hhi                 | m5_minus_shuffled_pr_auc |     -0.233414  |      -0.0636364 |
| me_entropy             | m5_minus_m3_pr_auc       |      0.141956  |      -0.127273  |
| me_effective_partners  | m5_minus_m3_pr_auc       |      0.162503  |      -0.127273  |
| me_top1_share          | m5_minus_m1_pr_auc       |     -0.338056  |      -0.136364  |

## Interpretation Rule

If `me_hhi`, `me_top1_share`, or `me_top3_share` correlate positively with `m5_minus_*` deltas, network gains are larger for more concentrated dependency structures. If entropy/effective partners correlate positively instead, the evidence points toward network exposure helping more in diversified systems. If all correlations are weak, the current panel does not yet explain where network weighting helps.
