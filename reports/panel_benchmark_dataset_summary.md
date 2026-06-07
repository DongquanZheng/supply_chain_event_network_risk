# Panel Benchmark Dataset Summary

## Dataset

- File: `data/processed/multicountry_container_event_network_benchmark.csv`
- Countries: 11
- Rows: 2860
- Positive labels: 314
- Positive rate: 0.110
- Week range: 2020-12-28 to 2025-12-29
- Target: `abnormal_next_week_container`, defined as next-week container port calls below the current rolling 12-week mean minus 1.5 standard deviations.

## Feature Blocks

- Operational container time-series features
- Own-country GDELT event controls
- External unweighted partner-event controls
- Total-import network-weighted partner-event exposure
- Equal, shuffled, and random-weight placebo exposure
- Machinery/electronics strict event-network exposure
- Operational vulnerability and event-network interaction features

## Country Summary

| ISO3   | country              |   rows |   positives |   positive_rate |
|:-------|:---------------------|-------:|------------:|----------------:|
| ARE    | United Arab Emirates |    260 |          26 |       0.1       |
| AUS    | Australia            |    260 |          22 |       0.0846154 |
| CHN    | China                |    260 |          30 |       0.115385  |
| DEU    | Germany              |    260 |          35 |       0.134615  |
| IDN    | Indonesia            |    260 |          26 |       0.1       |
| JPN    | Japan                |    260 |          26 |       0.1       |
| KOR    | Korea                |    260 |          33 |       0.126923  |
| SAU    | Saudi Arabia         |    260 |          27 |       0.103846  |
| THA    | Thailand             |    260 |          32 |       0.123077  |
| USA    | United States        |    260 |          24 |       0.0923077 |
| VNM    | Vietnam              |    260 |          33 |       0.126923  |

## Year Summary

|   year |   rows |   positives |   positive_rate |
|-------:|-------:|------------:|----------------:|
|   2020 |     11 |           3 |       0.272727  |
|   2021 |    572 |          68 |       0.118881  |
|   2022 |    572 |          47 |       0.0821678 |
|   2023 |    572 |          70 |       0.122378  |
|   2024 |    583 |          55 |       0.0943396 |
|   2025 |    550 |          71 |       0.129091  |

## Missingness

No missing values.
