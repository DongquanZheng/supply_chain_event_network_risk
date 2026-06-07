# Leave-One-Country-Out Panel Generalization

## Question

Does event-network exposure help when the model must generalize to a country that was never observed during training?

For each holdout country, the model trains on 2021-2023 data from the other 10 countries, selects a threshold on 2024 data from the other 10 countries, and tests on the holdout country's 2025 observations.

## Summary

| feature_group                 |   holdout_countries |   mean_pr_auc |   std_pr_auc |   mean_roc_auc |   mean_f1 |   mean_precision |   mean_recall |
|:------------------------------|--------------------:|--------------:|-------------:|---------------:|----------:|-----------------:|--------------:|
| M4_total_import_network       |                  11 |      0.278373 |    0.105622  |       0.649071 |  0.23228  |         0.173819 |      0.4671   |
| M3_external_unweighted_events |                  11 |      0.273592 |    0.115311  |       0.643554 |  0.240643 |         0.169705 |      0.490801 |
| M1_operational                |                  11 |      0.262069 |    0.117809  |       0.627748 |  0.21215  |         0.151383 |      0.436039 |
| M5_me_strict_network          |                  11 |      0.248857 |    0.0964355 |       0.630551 |  0.204004 |         0.140838 |      0.464069 |

## Holdout PR-AUC Table

| holdout_iso3   |   M1_operational |   M3_external_unweighted_events |   M4_total_import_network |   M5_me_strict_network |   m5_minus_m1 |   m5_minus_m3 |
|:---------------|-----------------:|--------------------------------:|--------------------------:|-----------------------:|--------------:|--------------:|
| ARE            |        0.258317  |                       0.297121  |                  0.310495 |              0.330013  |    0.0716967  |   0.0328923   |
| AUS            |        0.0849928 |                       0.0935736 |                  0.109524 |              0.0783109 |   -0.00668192 |  -0.0152627   |
| CHN            |        0.263681  |                       0.284636  |                  0.28312  |              0.217467  |   -0.046214   |  -0.0671684   |
| DEU            |        0.334871  |                       0.267747  |                  0.343637 |              0.336529  |    0.0016586  |   0.0687828   |
| IDN            |        0.530804  |                       0.523922  |                  0.458036 |              0.322023  |   -0.208781   |  -0.2019      |
| JPN            |        0.200137  |                       0.29208   |                  0.227143 |              0.205591  |    0.00545455 |  -0.0864891   |
| KOR            |        0.182778  |                       0.165149  |                  0.162828 |              0.163492  |   -0.0192857  |  -0.00165733  |
| SAU            |        0.148821  |                       0.160234  |                  0.165558 |              0.159625  |    0.010804   |  -0.000608854 |
| THA            |        0.3434    |                       0.380421  |                  0.391282 |              0.387755  |    0.0443556  |   0.00733365  |
| USA            |        0.255721  |                       0.252105  |                  0.341035 |              0.323671  |    0.0679505  |   0.0715662   |
| VNM            |        0.279233  |                       0.29252   |                  0.269444 |              0.212951  |   -0.0662818  |  -0.079569    |

## Fold-Level Results

| holdout_iso3   | holdout_country      | feature_group                 |   test_rows |   test_positives |   roc_auc |    pr_auc |   precision |   recall |       f1 |   validation_f1 |   selected_threshold |
|:---------------|:---------------------|:------------------------------|------------:|-----------------:|----------:|----------:|------------:|---------:|---------:|----------------:|---------------------:|
| ARE            | United Arab Emirates | M1_operational                |          50 |                8 |  0.604167 | 0.258317  |   0.214286  | 0.75     | 0.333333 |        0.237113 |                 0.42 |
| ARE            | United Arab Emirates | M3_external_unweighted_events |          50 |                8 |  0.699405 | 0.297121  |   0.222222  | 0.75     | 0.342857 |        0.244681 |                 0.42 |
| ARE            | United Arab Emirates | M4_total_import_network       |          50 |                8 |  0.6875   | 0.310495  |   0.173913  | 1        | 0.296296 |        0.268551 |                 0.34 |
| ARE            | United Arab Emirates | M5_me_strict_network          |          50 |                8 |  0.741071 | 0.330013  |   0.173913  | 1        | 0.296296 |        0.237762 |                 0.32 |
| AUS            | Australia            | M1_operational                |          50 |                3 |  0.361702 | 0.0849928 |   0.0769231 | 0.333333 | 0.125    |        0.265193 |                 0.42 |
| AUS            | Australia            | M3_external_unweighted_events |          50 |                3 |  0.382979 | 0.0935736 |   0.0625    | 0.333333 | 0.105263 |        0.251046 |                 0.37 |
| AUS            | Australia            | M4_total_import_network       |          50 |                3 |  0.460993 | 0.109524  |   0.0714286 | 0.333333 | 0.117647 |        0.265957 |                 0.4  |
| AUS            | Australia            | M5_me_strict_network          |          50 |                3 |  0.411348 | 0.0783109 |   0.0666667 | 0.333333 | 0.111111 |        0.230453 |                 0.36 |
| CHN            | China                | M1_operational                |          50 |                7 |  0.611296 | 0.263681  |   0         | 0        | 0        |        0.234234 |                 0.52 |
| CHN            | China                | M3_external_unweighted_events |          50 |                7 |  0.671096 | 0.284636  |   0.222222  | 0.285714 | 0.25     |        0.211838 |                 0.31 |
| CHN            | China                | M4_total_import_network       |          50 |                7 |  0.667774 | 0.28312   |   0         | 0        | 0        |        0.245161 |                 0.44 |
| CHN            | China                | M5_me_strict_network          |          50 |                7 |  0.644518 | 0.217467  |   0         | 0        | 0        |        0.239521 |                 0.43 |
| DEU            | Germany              | M1_operational                |          50 |               10 |  0.59     | 0.334871  |   0.166667  | 0.1      | 0.125    |        0.257669 |                 0.43 |
| DEU            | Germany              | M3_external_unweighted_events |          50 |               10 |  0.58     | 0.267747  |   0.166667  | 0.2      | 0.181818 |        0.247788 |                 0.37 |
| DEU            | Germany              | M4_total_import_network       |          50 |               10 |  0.6275   | 0.343637  |   0.111111  | 0.1      | 0.105263 |        0.257028 |                 0.36 |
| DEU            | Germany              | M5_me_strict_network          |          50 |               10 |  0.58     | 0.336529  |   0.181818  | 0.2      | 0.190476 |        0.255102 |                 0.39 |
| IDN            | Indonesia            | M1_operational                |          50 |                8 |  0.824405 | 0.530804  |   0.225806  | 0.875    | 0.358974 |        0.254098 |                 0.36 |
| IDN            | Indonesia            | M3_external_unweighted_events |          50 |                8 |  0.755952 | 0.523922  |   0.259259  | 0.875    | 0.4      |        0.230769 |                 0.4  |
| IDN            | Indonesia            | M4_total_import_network       |          50 |                8 |  0.72619  | 0.458036  |   0.194444  | 0.875    | 0.318182 |        0.249135 |                 0.33 |
| IDN            | Indonesia            | M5_me_strict_network          |          50 |                8 |  0.71131  | 0.322023  |   0.26087   | 0.75     | 0.387097 |        0.255556 |                 0.42 |
| JPN            | Japan                | M1_operational                |          50 |                5 |  0.706667 | 0.200137  |   0.166667  | 0.2      | 0.181818 |        0.265625 |                 0.49 |
| JPN            | Japan                | M3_external_unweighted_events |          50 |                5 |  0.768889 | 0.29208   |   0.166667  | 0.2      | 0.181818 |        0.242105 |                 0.41 |
| JPN            | Japan                | M4_total_import_network       |          50 |                5 |  0.76     | 0.227143  |   0.166667  | 0.2      | 0.181818 |        0.256881 |                 0.38 |
| JPN            | Japan                | M5_me_strict_network          |          50 |                5 |  0.706667 | 0.205591  |   0.166667  | 0.2      | 0.181818 |        0.247059 |                 0.42 |
| KOR            | Korea                | M1_operational                |          50 |                4 |  0.657609 | 0.182778  |   0.0833333 | 0.5      | 0.142857 |        0.229412 |                 0.27 |
| KOR            | Korea                | M3_external_unweighted_events |          50 |                4 |  0.652174 | 0.165149  |   0.130435  | 0.75     | 0.222222 |        0.217391 |                 0.31 |
| KOR            | Korea                | M4_total_import_network       |          50 |                4 |  0.641304 | 0.162828  |   0.125     | 0.5      | 0.2      |        0.235294 |                 0.37 |
| KOR            | Korea                | M5_me_strict_network          |          50 |                4 |  0.668478 | 0.163492  |   0.125     | 0.75     | 0.214286 |        0.224242 |                 0.28 |
| SAU            | Saudi Arabia         | M1_operational                |          50 |                8 |  0.4375   | 0.148821  |   0         | 0        | 0        |        0.237548 |                 0.35 |
| SAU            | Saudi Arabia         | M3_external_unweighted_events |          50 |                8 |  0.470238 | 0.160234  |   0         | 0        | 0        |        0.25     |                 0.36 |
| SAU            | Saudi Arabia         | M4_total_import_network       |          50 |                8 |  0.470238 | 0.165558  |   0.125     | 0.125    | 0.125    |        0.250951 |                 0.36 |
| SAU            | Saudi Arabia         | M5_me_strict_network          |          50 |                8 |  0.473214 | 0.159625  |   0         | 0        | 0        |        0.245989 |                 0.41 |
| THA            | Thailand             | M1_operational                |          50 |                7 |  0.66113  | 0.3434    |   0.235294  | 0.571429 | 0.333333 |        0.253275 |                 0.36 |
| THA            | Thailand             | M3_external_unweighted_events |          50 |                7 |  0.657807 | 0.380421  |   0.222222  | 0.571429 | 0.32     |        0.243728 |                 0.33 |
| THA            | Thailand             | M4_total_import_network       |          50 |                7 |  0.667774 | 0.391282  |   0.5       | 0.571429 | 0.533333 |        0.282609 |                 0.4  |
| THA            | Thailand             | M5_me_strict_network          |          50 |                7 |  0.664452 | 0.387755  |   0.2       | 0.571429 | 0.296296 |        0.25498  |                 0.33 |
| USA            | United States        | M1_operational                |          50 |                5 |  0.666667 | 0.255721  |   0.210526  | 0.8      | 0.333333 |        0.240602 |                 0.35 |
| USA            | United States        | M3_external_unweighted_events |          50 |                5 |  0.648889 | 0.252105  |   0.176471  | 0.6      | 0.272727 |        0.250923 |                 0.35 |
| USA            | United States        | M4_total_import_network       |          50 |                5 |  0.657778 | 0.341035  |   0.166667  | 0.6      | 0.26087  |        0.247706 |                 0.38 |
| USA            | United States        | M5_me_strict_network          |          50 |                5 |  0.626667 | 0.323671  |   0.16      | 0.8      | 0.266667 |        0.236162 |                 0.33 |
| VNM            | Vietnam              | M1_operational                |          50 |                6 |  0.784091 | 0.279233  |   0.285714  | 0.666667 | 0.4      |        0.223256 |                 0.38 |
| VNM            | Vietnam              | M3_external_unweighted_events |          50 |                6 |  0.791667 | 0.29252   |   0.238095  | 0.833333 | 0.37037  |        0.236559 |                 0.34 |
| VNM            | Vietnam              | M4_total_import_network       |          50 |                6 |  0.772727 | 0.269444  |   0.277778  | 0.833333 | 0.416667 |        0.269058 |                 0.35 |
| VNM            | Vietnam              | M5_me_strict_network          |          50 |                6 |  0.708333 | 0.212951  |   0.214286  | 0.5      | 0.3      |        0.224299 |                 0.37 |

## Interpretation

This is a stricter domain-generalization check than the standard panel benchmark. If M5 improves here, network exposure is not merely fitting country fixed effects; it carries information that transfers across countries.
