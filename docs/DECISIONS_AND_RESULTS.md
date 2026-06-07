# Decisions and Results Log

This document records confirmed project decisions, empirical results, and limitations. It is the project memory and should be updated whenever a conclusion becomes stable.

## 2026-05-20

### Confirmed Direction

The third project will integrate ML, NLP, and network structure rather than only visualizing a trade or maritime network.

### Confirmed Research Logic

The network layer will be used as an exposure mapping layer. Its role is to weight and contextualize external event signals according to trade or maritime dependency structure.

### Confirmed Methodological Guardrail

The project must test incremental value beyond a strong operational baseline. It must also compare NLP and network-weighted NLP signals against simpler controls such as news volume, tone, keyword signals, and placebo network/event variants.

### Open Implementation Choice

The final empirical setting is not yet fixed. Candidate settings include:

- Japan-centered country-level trade dependency network;
- East Asia maritime/trade exposure network;
- multi-port PortWatch operational outcome with network-weighted event exposure.

The next notebook step will inspect feasible operational outcome and network data options before fixing the empirical scope.

### Repository Setup

The project repository has been initialized with a GitHub-ready structure, a research harness, a methodology note, and a data-source tracking document. The local PhD research proposal draft is intentionally excluded from version control through `.gitignore`.

## 2026-05-21

### WITS Feasibility Result

WITS Trade Stats can be accessed from the notebook environment for Japan 2023 import-by-partner data. The raw query returned 224 partner rows. After removing World Bank regional and aggregate partners through the World Bank country metadata filter, 198 country/economy partner rows remained.

### Confirmed Network Interpretation

For the first network layer, bilateral import data will be interpreted as trade dependency or exposure structure. For Japan 2023, `import_dependency_share` is defined as each partner's import value divided by Japan's total country-only imports in the filtered WITS data.

### Initial Japan Import Dependency Pattern

The top country/economy partners in the filtered Japan 2023 import data are China, United States, Australia, United Arab Emirates, Saudi Arabia, Korea, Viet Nam, Thailand, Indonesia, and Germany. This pattern is plausible and supports using WITS as a candidate source for country-level trade dependency networks.

### PortWatch Endpoint Issue

The previously used ArcGIS endpoint for `Daily_Trade_Data` returned HTTP 200 with an ArcGIS error payload: `Item does not exist or is inaccessible`. This means the old PortWatch query URL is not currently usable as an operational outcome source. A current, reproducible PortWatch access path must be found before fixing PortWatch as the ML outcome data source.

### PortWatch Country-Level Alternative

The PortWatch daily country/economy aggregation endpoint `Daily_Trade_Data_REG` is accessible and returns Japan records with fields including `ISO3`, `country`, `date`, `portcalls`, vessel-type port calls, import shipment estimates, export shipment estimates, and shipment totals. This may be a better operational outcome layer than single-port data because it aligns more naturally with country-level WITS trade dependency networks.

### PortWatch Country Coverage Check

The `Daily_Trade_Data_REG` endpoint returned 2,692 daily rows for each tested economy: Japan, China, Korea, Singapore, Viet Nam, Malaysia, Thailand, Indonesia, United States, Netherlands, and Germany. This supports using country-level port-system activity as the operational outcome layer for the MVP.

### Candidate Operational Target

Japan daily country-level PortWatch records were aggregated to 383 complete weekly observations. A preliminary next-week abnormal activity label was constructed using weekly `portcalls` and a rolling 12-week historical threshold: next-week portcalls below `rolling_mean_12w - 1.5 * rolling_std_12w`. This candidate target produced 370 usable modeling rows and 46 positive cases, with a positive rate of 12.4%. The target is plausible but not yet final; threshold sensitivity should be checked before locking it.

### Preliminary Operational Baseline

Using Japan country-level weekly PortWatch features and a temporal split with training through 2023 and testing from 2024 onward, a balanced Logistic Regression operational baseline achieved ROC-AUC 0.740 and PR-AUC 0.313 on the future test period. At a default 0.5 threshold, recall was high at 0.769 while precision was low at 0.204. This is a preliminary transparent baseline, not the final strong operational baseline.

### Strong Baseline Candidate

A Random Forest operational baseline achieved higher test-period ranking metrics than Logistic Regression on the same temporal split: ROC-AUC 0.773 and PR-AUC 0.369. At the default threshold it was conservative, but a diagnostic threshold sweep showed a best-F1 value of 0.400 versus 0.333 for Logistic Regression. Threshold selection was performed on the test set only for diagnosis and is not a final evaluation protocol.

### Interim Dataset Saved

The Japan weekly PortWatch operational modeling base was saved to `data/interim/japan_portwatch_weekly_operational_model_base.csv` with 362 rows and 23 columns.

### GDELT DOC Access Note

GDELT DOC API returned article metadata including title, seen date, domain, language, source country, and URL for broad maritime/logistics queries. A later stricter query attempt hit HTTP 429 rate limiting. This confirms that GDELT DOC is usable but must be accessed with throttling, caching, and small query batches. The project should prefer article titles and metadata as lightweight reproducible NLP inputs rather than relying on large-scale full-text crawling.

### Event Data Direction Update

The preferred event-data route is now GDELT GKG through Google BigQuery rather than frequent GDELT DOC API calls. GKG is better aligned with the project because it supports scalable event/news aggregation using themes, locations, tone, dates, and URLs. The NLP/event layer will focus on transparent metadata-derived event features rather than full-text article modeling in the MVP.

### BigQuery Setup Confirmed

A dedicated Google Cloud project `supply-chain-network-risk` was created and configured for BigQuery access. The notebook successfully queried metadata from `gdelt-bq.gdeltv2.__TABLES__`, returning 62 tables. This confirms that BigQuery-based GDELT access is operational in the local research environment.

### BigQuery Cost Guardrail

An initial GKG sample query filtered by the integer `DATE` field would have scanned about 2.96 TB and was blocked by the `maximum_bytes_billed` limit. This confirms that all GKG queries must use dry runs and proper partition filters before execution. Integer date filtering alone is not sufficient as a cost-control strategy.

### GKG Location Parsing Feasibility

GKG `V2Locations` can be parsed with a regular expression to extract two-letter GDELT/FIPS-style country codes. A one-day test for major Japan trade partners returned records for US, China, Germany, Australia, Korea, Viet Nam, Indonesia, Japan, Saudi Arabia, Thailand, and the United Arab Emirates. This confirms that GKG location codes can serve as the bridge from event records to WITS trade partners, provided that a country-code mapping is maintained.

### First GKG Partner Event Feature Sample

A one-day GKG query for 2025-01-01 produced partner-country event features for 11 Japan-relevant GDELT country codes. The output included article count, average tone, negative article share, very negative article share, trade/transport count, and risk-theme count. This confirms the feasibility of constructing country-period event pressure features from GKG for use in network-weighted exposure.

### First Network-Weighted Event Exposure Calculation

The first network-weighted event exposure calculation was completed for Japan using a one-day GKG sample and WITS 2023 import dependency weights for mapped major partners. Partner-level GKG features were weighted by import dependency share and aggregated into Japan-level network exposure measures, including negative exposure, very negative exposure, risk-theme exposure, and trade/transport exposure. This confirms that the core ML + event + network data linkage is technically feasible.

### First Full-Week GKG Feature Cache

A full weekly GKG partner-event feature query for 2024-12-30 to 2025-01-05 was executed and saved to `data/interim/gkg_partner_event_features_2024-12-30_2025-01-05.csv`. The query returned 11 partner-country rows for the Japan-relevant mapped partners.

### Module Refactor

The notebook-validated pipeline has been modularized into `src/config.py`, `src/portwatch.py`, `src/wits.py`, `src/gkg_bigquery.py`, and `src/network_exposure.py`. The modules support target-country configuration, PortWatch operational base construction, WITS dependency weights, partition-safe GKG BigQuery queries, and network-weighted event exposure aggregation. A light import check passed.

### Eight-Week GKG Event Cache

An eight-week GKG partner-event feature window for 2024-12-02 to 2025-01-26 was executed and cached at `data/interim/gkg_partner_event_features_2024-12-02_2025-01-26.csv`. The query returned 88 rows, corresponding to 8 weeks and 11 mapped partner-country codes.

### Eight-Week Mini-Window Observation

In the 2024-12-02 to 2025-01-26 mini-window, two of eight weeks had positive `abnormal_next_week` labels. Mean network-weighted very negative exposure was higher in positive-label weeks than in negative-label weeks, while partner article count, broad risk-theme exposure, and trade/transport exposure were lower in positive-label weeks. This is only a sanity-check observation from a very small sample, not a statistical result. It suggests that focused adverse-tone exposure may be more informative than raw news volume or broad theme counts, and should be tested over a longer window with placebo comparisons.

### Planned Visualization Output

The final project should include an interactive week-level network exposure visualization. The user should be able to select a week and inspect the target country, partner countries, trade dependency edges, partner event pressure, weighted exposure contributions, and the target country's next-week abnormal activity label. This is planned as an explanatory output after the modeling pipeline is stabilized.

### Six-Month GKG Event Cache

A six-month GKG partner-event feature window for 2024-10-01 to 2025-03-31 was executed and cached at `data/interim/gkg_partner_event_features_2024-10-01_2025-03-31.csv`. The query returned 297 rows, corresponding to 27 weeks and 11 mapped partner-country codes. This window is large enough for preliminary M1-M5 feature comparison, but not sufficient for final model validation.

### Partner Contribution Diagnostic

Partner-level decomposition of the six-month Japan network exposure showed that network weighting changes the ranking of event signals. China and the United States frequently dominate Japan's weighted very-negative exposure because their event-tone signals are combined with high import-dependency weights. Around the positive-label weeks of 2024-12-23 and 2024-12-30, China, the United States, and Korea were among the leading contributors. This supports the interpretation of the network layer as an exposure-mapping mechanism rather than a visualization-only add-on. However, it does not yet prove predictive superiority over unweighted or placebo-weighted event features.

### Partner Contribution Visualization

A stacked bar visualization of the 2024-12-02 to 2025-01-20 window showed that Japan's network-weighted very-negative exposure increased around the two positive-label weeks. The contribution pattern was concentrated in major dependency partners, especially China, the United States, and Korea, with additional contributions from Australia, Saudi Arabia, the United Arab Emirates, and other mapped partners. This figure is suitable as an explanatory README or methodology output because it makes the exposure-mapping logic visible.

### Research Direction Correction

The project should not treat a visually interpretable network exposure as sufficient evidence of network value. The deeper research question is whether network structure defines a theoretically meaningful and empirically testable exposure mapping for external event signals. The current Japan-only total-import weighting is a first MVP, but it may be too coarse to demonstrate predictive gains. Next-stage analysis should test alternative network definitions, including commodity-specific dependency networks, second-order propagation exposure, partner centrality, panel-country designs, and placebo/falsification networks. The project claim should remain empirical: network-informed exposure may improve relevance and interpretability, but predictive improvement must be tested rather than assumed.

### Total-Import Network Redundancy Diagnostic

For the six-month Japan window, the true total-import-weighted very-negative exposure was highly correlated with the unweighted very-negative exposure (`r = 0.971`). The true total-import-weighted negative exposure was also highly correlated with unweighted negative exposure (`r = 0.937`). This indicates that the current total-import network weighting produces limited independent variation relative to unweighted event signals. The lack of predictive improvement from this network definition is therefore not surprising. The next methodological step should be commodity-specific or sector-specific dependency networks, where partner weights can differ more meaningfully across risk channels.

### WITS Product Code Correction

The WITS Trade Stats SDMX endpoint does not accept raw HS chapter codes such as `27`, `85`, or `87` in the `product` path. These requests return `Invalid_Product`. For the current API route, commodity-specific dependency prototypes should use WITS product group codes such as `27-27_Fuels`, `84-85_MachElec`, and `86-89_Transport`.

### Commodity-Specific Network Feasibility

Commodity-specific WITS product group networks for Japan show materially different dependency structures from the total-import network. In the fuels network, Australia, the United Arab Emirates, and Saudi Arabia dominate the mapped partner set, while China has a very small share. In the machinery/electronics network, China dominates with a mapped-partner share above 0.61. In the transport-equipment network, China, Germany, the United States, and Thailand are the leading partners. This confirms that commodity-specific dependency networks provide meaningful variation and are a stronger research direction than total-import weighting alone.

### Commodity-Specific Exposure Diagnostic

In the six-month Japan window, commodity-specific very-negative exposure features showed stronger exploratory separation of positive-label weeks than the aggregate total-import exposure. Machinery/electronics and transport-equipment very-negative exposure both achieved perfect single-feature ranking on this small window (`ROC-AUC = 1.00`, `PR-AUC = 1.00`), while unweighted very-negative exposure and total-import-weighted very-negative exposure achieved `ROC-AUC = 0.98` and `PR-AUC = 0.833`. Fuels very-negative exposure was weaker (`ROC-AUC = 0.86`, `PR-AUC = 0.375`). This result is not final evidence because the window contains only 27 weeks and 2 positive labels, but it supports the hypothesis that network value depends on using economically relevant dependency channels rather than aggregate trade weights.

### Commodity Exposure Redundancy Diagnostic

Commodity-specific exposure features are not equally redundant with unweighted event signals. Total-import-weighted very-negative exposure was highly correlated with unweighted very-negative exposure (`r = 0.971`). Fuels and machinery/electronics very-negative exposure showed lower correlations with unweighted exposure (`r = 0.807` and `r = 0.843`), indicating more distinct variation. Transport-equipment exposure remained highly correlated with total-import and unweighted exposure (`r = 0.956` with total-import exposure and `r = 0.934` with unweighted exposure). This suggests that commodity-specific networks can introduce additional structure, but some channels may still overlap strongly with aggregate event pressure.

### Channel-Level Exposure Visualization

The channel-level time-series plot for Japan shows that very-negative exposure rises around the two positive-label weeks, but channels differ in timing and magnitude. Fuels exposure has a sharp pre-window spike, while machinery/electronics and transport-equipment exposures remain elevated through the positive-label window. This supports lead-lag testing by channel rather than treating all network exposures as contemporaneous predictors.

### Channel Lead-Lag Diagnostic

Lead-lag diagnostics suggest different timing structures across commodity-specific exposure channels. Fuels very-negative exposure performed strongest at a one-week lag, while machinery/electronics and transport-equipment very-negative exposure performed strongest contemporaneously in the current event week. This is only exploratory because the six-month window contains two positive labels, but it supports the hypothesis that different supply-chain dependency channels may transmit event signals over different lead times.

### Vessel-Type Outcome Feasibility

PortWatch country-level vessel-type weekly series are available for Japan, including container, tanker, dry bulk, roro, and general cargo port calls. Using the same rolling 12-week threshold rule, the resulting next-week abnormal labels are not too sparse: the vessel-type targets have between 38 and 50 positive labels across 372 usable rows. This supports testing channel-aligned outcomes, such as fuel exposure against tanker activity and machinery/electronics exposure against container activity.

### Channel-Aligned Outcome Diagnostic

Within the six-month GKG window, preliminary channel-aligned tests were directionally consistent: machinery/electronics very-negative exposure separated the single container positive label, fuels exposure showed strong separation for dry-bulk positives, and transport-equipment exposure showed positive separation for roro and general-cargo positives. These findings are exploratory because the aligned window contains only 1 to 3 positive labels per vessel-type target, but they support testing commodity-specific event-network exposure against vessel-type operational outcomes rather than only aggregate port calls.

### Channel Pairing Placebo Diagnostic

A cross-channel placebo diagnostic showed that expected commodity-to-vessel pairings do not consistently outperform mismatched pairings in the current six-month window. For example, transport-equipment exposure also ranked the single container positive label highly, and fuels exposure performed strongly for roro and general-cargo labels. This suggests that the current event signal is still too generic: commodity-specific trade weights are being multiplied by general country-level very-negative news, not by commodity-specific event signals. The next research step should therefore construct channel-specific NLP/GKG features before treating channel alignment as evidence of mechanism.

### Channel-Specific GKG Feature Cache

Channel-specific GKG event features were queried and cached for the 2024-10-01 to 2025-03-31 window. The query returned 297 partner-week rows across 27 weeks and 11 mapped partner-country codes. For each country-week, it computed fuel-related, machinery/electronics-related, and transport-equipment-related very-negative article shares and article counts using GKG themes plus URL/text keyword matching. This provides a more theoretically aligned event layer for commodity-specific network exposure.

### Channel-Specific Pairing Diagnostic

After combining channel-specific NLP signals with commodity-specific dependency weights, expected channel pairings only partially outperformed mismatched pairings. Machinery/electronics exposure ranked highest for the container target, which is theoretically consistent. However, transport-equipment exposure also performed strongly for several non-transport targets, while fuel-specific exposure was weak for tanker and dry-bulk targets in this six-month window. This suggests that the current keyword/theme channel definitions are still imperfect: transport-related terms may capture broad logistical disruption rather than transport-equipment supply chains, and fuel-related terms may capture energy-market news that does not align cleanly with Japanese tanker activity. Further NLP feature refinement is needed before making mechanism claims.

### Channel NLP Audit

A partner-contribution audit of channel-specific exposure around the positive-label window showed both plausible and problematic patterns. Fuels exposure is often driven by expected energy partners such as the United Arab Emirates, Saudi Arabia, and Australia, and transport-equipment exposure is often driven by Germany, the United States, China, and Thailand. However, Korea appears as an unusually large contributor in some fuel and transport-equipment weeks because its channel-specific event signal is very high despite smaller commodity dependency weights. This suggests that the current keyword/theme rules may still capture broad national crisis or logistics news rather than strictly commodity-relevant supply-chain events. The next NLP iteration should use stricter channel definitions and separate broad transport/logistics disruption from transport-equipment/vehicle supply-chain disruption.

### Strict Channel-Specific GKG Feature Cache

A stricter channel-specific GKG feature query was executed for the 2024-10-01 to 2025-03-31 window. The strict version requires both commodity-relevance terms and disruption/risk terms before an article contributes to a channel-specific signal. The returned partner-week table is not too sparse: channel disruption article counts remain available across the mapped country-week rows. This stricter event layer should be evaluated against the previous broad channel features using expected-pair and placebo-pair diagnostics.

### Strict Channel Pairing Result

The stricter channel-specific event layer improved the container pairing: machinery/electronics exposure ranked highest for the container target, while mismatched fuel and transport-equipment exposures weakened. However, the expected pairings for fuels-to-tanker/dry-bulk and transport-equipment-to-roro/general-cargo did not consistently outperform mismatched pairings. This suggests that the machinery/electronics-to-container channel is currently the most coherent channel-aligned signal in the MVP, while other channels require better event definitions, alternative operational outcomes, or longer validation windows.

### Container Operational Baseline

A container-specific operational modeling base was constructed from Japan weekly PortWatch container port calls. The resulting target has 371 rows, 36 positive next-week abnormal labels, and a positive rate of 9.7%. With a temporal split that trains before 2024 and tests from 2024 onward, a balanced Logistic Regression baseline achieved `ROC-AUC = 0.772` and `PR-AUC = 0.261`; a Random Forest baseline achieved `ROC-AUC = 0.756` and `PR-AUC = 0.215`, with higher default-threshold F1. This provides a clean M1 operational benchmark for testing machinery/electronics event-network exposure.

## 2026-06-06

### IEEE-Style Benchmark Scope Locked

The project has been reframed as a reproducible applied benchmark rather than an open-ended feature exploration. The main benchmark question is whether external event signals improve next-week abnormal container port activity prediction beyond a strong operational baseline, and whether trade-network-weighted exposure adds value over unweighted event controls. The primary empirical scope is Japan-centered, with weekly PortWatch container port calls as the outcome and machinery/electronics dependency exposure as the main network-event channel.

### Extended Benchmark Dataset Generated

The processed benchmark dataset `data/processed/japan_container_event_network_benchmark.csv` was generated using the cached 2021-2025 GDELT window. It contains 260 weekly observations from 2020-12-28 to 2025-12-29, with 26 positive next-week abnormal container labels and no missing feature values. The temporal benchmark split is: train 2021-2023, validation 2024, and test 2025. Positive labels are available in all three splits, but the test split still contains only five positives, so final claims must remain cautious.

### Initial M1-M6 Benchmark Result

The first reproducible M1-M6 benchmark run used Logistic Regression and Random Forest models with validation-selected F1 thresholds and test-only final reporting. On the 2025 test split, the best PR-AUC was achieved by `M2_simple_news` with Random Forest (`ROC-AUC = 0.911`, `PR-AUC = 0.684`, `F1 = 0.400`). `M5_me_network` with Random Forest achieved lower PR-AUC (`0.525`) but higher thresholded F1 (`0.444`) because it produced fewer false positives while still capturing four of five positives. `M4_total_import_network` performed between simple news and machinery/electronics network in PR-AUC (`0.570`), while placebo variants were competitive enough to require formal placebo and robustness checks.

### Benchmark Interpretation Guardrail

The current benchmark does not support claiming that network weighting generally improves prediction. A defensible interpretation is that event features improve over the operational baseline in this test split, while network weighting changes the precision-recall tradeoff and provides interpretable exposure attribution. The next required step is Phase 4 placebo and robustness testing before making any paper-level claim about incremental network value.

### Placebo Check Result

The first formal placebo check compared machinery/electronics network exposure with unweighted, equal-weight, shuffled-weight, and random-weight alternatives under the same temporal validation protocol. For Random Forest models, unweighted and equal-weight machinery/electronics event exposure achieved `PR-AUC = 0.548`, random-weight placebo achieved `PR-AUC = 0.545`, and true machinery/electronics network exposure achieved `PR-AUC = 0.525`. The true network exposure had the highest thresholded F1 among these variants (`F1 = 0.444`) because it reduced false positives, but it did not win on PR-AUC. This is a mixed result: network weighting appears to alter the alerting tradeoff and improve interpretability, but current evidence does not show robust ranking superiority over simpler event aggregation.

### Test-Week Diagnostic Result

A week-level diagnostic of the 2025 test split showed why the machinery/electronics network model has higher F1 but lower PR-AUC than the simple-news model. The `M2_simple_news` Random Forest captured all five positive test weeks but produced 15 false positives. The `M5_me_network` Random Forest captured four of five positive weeks and reduced false positives to nine. Six false alerts raised by M2 were suppressed by M5, while M5 missed one positive week that M2 captured. This supports a more precise interpretation: the current network layer is not an indispensable ranking signal, but it behaves as a stricter relevance filter that changes the alerting tradeoff and offers dependency-channel attribution.

### PhD RP Implication

For the PhD research proposal, the strongest defensible framing is not that network weighting automatically improves prediction. A stronger and more realistic claim is that network structure can define which external event signals are supply-chain-relevant for a target country or sector. In the current MVP, the ML model tests operational abnormality, the NLP layer detects external event pressure, and the network layer filters and attributes event relevance through dependency channels. Future PhD work should investigate when this relevance-filtering mechanism improves prediction, when it only improves interpretability, and when simple event controls are sufficient.

### Structural Regularizer Diagnostic

A stricter diagnostic tested whether machinery/electronics network exposure can be interpreted as a structural regularizer that removes spurious NLP-driven over-alerting. Under validation-selected F1 thresholds, `M5_me_network` reduced the number of 2025 test alerts from 20 to 13 relative to `M2_simple_news`, reduced false positives from 15 to 9, and retained four of five true positives. However, alert-budget and test-oracle diagnostics weakened the causal interpretation. When each model was allowed exactly 13 test alerts, M2, M3, M5, and placebo variants all captured four positives with nine false positives. A test-oracle threshold diagnostic showed that M2 could achieve 0.8 recall with only three false positives, while M5 required seven false positives. Therefore, the current benchmark does not prove that true trade-network weighting uniquely removes NLP overfitting. The defensible conclusion is that network-as-regularizer is a promising hypothesis, but it requires rolling-origin stability checks, stronger placebo designs, and negative-control outcomes before it can be treated as an empirical finding.

### Rolling-Origin Stability Result

A rolling-origin check was run for three temporal test years: 2023, 2024, and 2025. Random Forest models were trained and thresholded using only prior years. Simple news controls (`M2_simple_news`) had the highest average PR-AUC across rolling windows (`0.557`), followed by total-import network exposure (`0.498`), random placebo (`0.479`), unweighted machinery/electronics event exposure (`0.468`), and machinery/electronics network exposure (`0.463`). This confirms that machinery/electronics network weighting does not provide stable predictive superiority in the current benchmark. The article should therefore not be positioned as a network-performance win. A more distinctive and defensible mechanism is to treat the network layer as a structural plausibility audit: it tests whether event-informed predictions are explainable through supply-chain dependency channels, rather than assuming every predictive NLP signal is supply-chain-relevant.

### NLP Taxonomy Prototype

A 2023-2025 GDELT GKG candidate-document cache was queried for Japan-relevant partner countries. The query used partition filters, a dry-run estimate of `765.23 GB`, and deterministic country-week sampling. It returned `205,920` candidate document rows. Because the GKG table does not provide article bodies, the NLP prototype builds document text from URL slugs, GKG themes, names, and organizations. A weakly supervised taxonomy classifier was then trained on 2023 documents using lightweight hashed text features and logistic SGD classifiers, producing partner-week probabilities for maritime disruption, machinery/electronics disruption, energy disruption, trade-policy disruption, and broad supply disruption.

This is a meaningful upgrade over raw GKG tone/theme counts, but it remains a reproducible event-taxonomy proxy rather than manually validated full-text NLP. The machinery/electronics weak-label rate is low (`0.8%`), which is useful for specificity but means the signal should be interpreted carefully.

### NLP Taxonomy Benchmark Diagnostic

The taxonomy features were merged into a 2023-2025 Japan container benchmark and evaluated with train 2023, validation 2024, and test 2025. The current simple-news Random Forest model still achieved the highest PR-AUC (`0.664`). The standalone taxonomy machinery/electronics network model achieved lower PR-AUC (`0.460`) but better thresholded alert behavior than simple news: under the validation-selected threshold, it captured all five 2025 positives with 14 false positives, compared with 39 false positives for the simple-news model.

The strongest taxonomy result came from combining current news controls with the taxonomy bundle. This model did not beat simple news on PR-AUC (`0.446` vs. `0.664`), but under the validation-selected threshold it captured all five positive test weeks and reduced false positives from 39 to 8, improving F1 from `0.204` to `0.556`. A fixed alert-budget diagnostic weakened the ranking interpretation: at 5, 10, 15, and 20 alerts, taxonomy models did not capture more positives than the simpler baselines. Therefore, the current evidence supports taxonomy as a calibration/false-alert-control and semantic filtering layer, not yet as a superior risk-ranking layer.

### Updated Mechanism Framing

The most distinctive current mechanism is not "network improves prediction." A sharper and more defensible framing is: operational ML detects abnormality patterns, NLP taxonomy converts noisy global event data into interpretable event categories, and network exposure provides structural relevance and attribution checks for those event categories. In the current Japan-container MVP, NLP taxonomy appears more promising than adding network complexity. Network weighting should remain in the benchmark as a falsifiable structural layer, but the next research iteration should prioritize better event taxonomy, stricter text evidence, and calibration/alert-quality evaluation.

### Japan-Only Network Complexity Diagnostic

A diagnostic tested whether a more complex Japan-only network layer improves the taxonomy benchmark. The tested features included true machinery/electronics network exposure relative to unweighted and equal-weight exposure, network-minus-placebo gaps, operational shortfall, recent negative trend, and interactions between operational vulnerability and event/network signals. These variants did not outperform the simple-news model on PR-AUC, and they did not outperform the `current news + taxonomy bundle` model on thresholded false-positive control. The best PR-AUC remained `M2_current_simple_news` with Random Forest (`0.664`), while `current news + taxonomy bundle` remained the strongest alert-quality result (`TP = 5`, `FP = 8`, `F1 = 0.556`). Interaction-heavy network variants increased model complexity without producing a clear Japan-only performance gain.

This suggests that simply adding more Japan-only network transformations is not the right next step. The static country-level import network has limited time variation, so its predictive contribution is naturally constrained in a single-target weekly time series.

### Multi-Country Panel Feasibility

A PortWatch feasibility diagnostic was run for the 11 currently mapped GDELT/ISO3 countries: Japan, China, Korea, United States, Australia, United Arab Emirates, Saudi Arabia, Vietnam, Thailand, Indonesia, and Germany. The panel contains `4,103` weekly country observations from 2019-03-25 to 2026-05-11, with `452` positive next-week abnormal container labels and an overall positive rate of `11.0%`. Each country has 373 usable weekly rows and between 34 and 48 positive labels.

This materially changes the network strategy. A multi-country panel gives enough cross-sectional variation for network exposure to matter empirically: the same global event environment can be mapped through different country-specific dependency structures and compared against different port outcomes. The recommended network direction is therefore not a more complex Japan-only model, but a panel benchmark where network structure explains heterogeneous event exposure across countries.

### Panel M1/M2/Network Sanity Benchmark

A first multi-country panel benchmark was run for the 11 mapped countries using weekly container port-call targets and cached 2021-2025 GDELT country-week features. The split was train 2021-2023, validation 2024, and test 2025. The panel benchmark has `1,716` train rows, `583` validation rows, and `550` test rows, with `71` positive labels in the 2025 test split.

The initial country-event check showed that own-country GDELT event controls did not clearly improve over the operational panel baseline. External partner-event controls improved recall but produced many false positives. The important result emerged after adding target-specific total-import network exposure over the mapped partner countries. For Random Forest models, the best PR-AUC came from event/network feature groups:

- `P7_operational_plus_all_events_and_network`: `PR-AUC = 0.301`
- `P5_operational_plus_total_import_network`: `PR-AUC = 0.301`
- `P6_operational_plus_external_and_network`: `PR-AUC = 0.283`
- `P2_operational_plus_country_events`: `PR-AUC = 0.278`
- `P5b_operational_plus_equal_weight_placebo`: `PR-AUC = 0.274`
- `P1_operational_panel`: `PR-AUC = 0.264`
- `P3_operational_plus_external_events`: `PR-AUC = 0.264`

This is the first result suggesting that network weighting may matter empirically when evaluated in the correct data structure. The improvement is not yet a final claim: equal-weight placebo still has competitive ROC-AUC and F1, and thresholded accuracy is not the main rare-event metric. However, true total-import network exposure has better PR-AUC than operational, own-country events, external unweighted events, and equal-weight placebo in this panel sanity check. This supports moving the benchmark from Japan-only to a multi-country panel for the formal network evaluation.

### Formal Multi-Country Panel Benchmark

A formal multi-country panel benchmark was implemented using `data/processed/multicountry_container_event_network_benchmark.csv`. The dataset contains `2,860` country-week observations from `2020-12-28` to `2025-12-29` across 11 countries, with `314` positive next-week abnormal container labels and an overall positive rate of `11.0%`. The benchmark uses rolling-origin temporal validation with test years 2023, 2024, and 2025. Thresholds are selected only on the immediately preceding validation year, and PR-AUC is treated as the main ranking metric because positives are rare.

The benchmark compares operational baselines, own-country news controls, external unweighted event pressure, total-import network exposure, machinery/electronics strict network exposure, and placebo alternatives. For Random Forest models, the best average PR-AUC across the three test folds is achieved by `M5_me_strict_network`:

- `M5_me_strict_network`: mean PR-AUC `0.223`, mean ROC-AUC `0.665`
- `M4_total_import_network`: mean PR-AUC `0.212`, mean ROC-AUC `0.667`
- `M6c_total_random_placebo`: mean PR-AUC `0.211`, mean ROC-AUC `0.662`
- `M1_operational`: mean PR-AUC `0.210`, mean ROC-AUC `0.655`
- `M6a_total_equal_placebo`: mean PR-AUC `0.210`, mean ROC-AUC `0.661`
- `M2_own_country_news`: mean PR-AUC `0.205`, mean ROC-AUC `0.644`
- `M3_external_unweighted_events`: mean PR-AUC `0.205`, mean ROC-AUC `0.658`
- `M6b_total_shuffled_placebo`: mean PR-AUC `0.197`, mean ROC-AUC `0.652`

This is the strongest evidence so far that the network layer can matter when evaluated as a cross-country exposure-mapping mechanism rather than as a Japan-only time-series add-on. The key empirical shift is that the same external event environment is routed through different country-specific trade dependencies and tested against different national container port outcomes.

### Panel Delta and Placebo Interpretation

A paired bootstrap delta diagnostic compared `M5_me_strict_network` against baselines using pooled out-of-sample predictions from the 2023, 2024, and 2025 rolling-origin test folds. The machinery/electronics strict network model has positive pooled PR-AUC deltas against shuffled, equal-weight, random-weight, unweighted external event, own-country news, and operational baselines. The largest delta is against the shuffled total-network placebo (`+0.022` PR-AUC; bootstrap probability delta > 0 is `0.973`). However, the 95% bootstrap intervals cross zero for all comparisons, and the delta against total-import network exposure is essentially zero (`+0.0001` PR-AUC).

The paper-level conclusion should therefore be directional rather than overclaimed. The current evidence supports the statement that commodity-specific network exposure is the best-performing panel feature group in the benchmark and provides a theoretically coherent exposure map. It does not yet support a strong statistical claim that machinery/electronics network weighting decisively dominates every simpler event or placebo alternative.

### Current Paper-Level Claim

The defensible contribution is now stronger than the Japan-only result. The project can be framed as a reproducible multi-country benchmark showing that:

1. operational history remains a strong baseline for next-week abnormal container activity;
2. external event signals need structural filtering because unweighted event pressure can be noisy;
3. trade-network exposure is most meaningful when modeled in a panel where country-specific dependency structures vary;
4. commodity-specific machinery/electronics exposure provides the strongest average PR-AUC among Random Forest feature groups, but the advantage is modest and should be reported with placebo and bootstrap caveats.

For the PhD RP, this supports the ML + NLP + Network triangle more cleanly: ML models operational vulnerability, NLP converts external events into measurable event pressure, and the network layer maps event pressure into structurally relevant exposure for each country. The current benchmark does not prove causality, but it provides a falsifiable empirical framework for studying when network-informed event exposure improves prediction and when it mainly improves interpretation.

### Network Mechanism Diagnostics

Three mechanism-oriented diagnostics were run after the formal panel benchmark to test whether the network layer has a stronger interpretation than "one more feature block."

First, a country-level dependency-concentration analysis compared machinery/electronics import concentration metrics with country-level M5 PR-AUC gains. The evidence is weak but suggestive only for top-3 concentration: `me_top3_share` has a positive Spearman correlation with M5 gains over M3 external unweighted events (`0.336`) and over M4 total-import network (`0.355`). HHI and top-1 share do not show stable positive relationships. This is not strong enough for a main claim, but it suggests a future hypothesis: network value may depend on dependency concentration beyond simple top-1 exposure.

Second, a counterfactual network-swap check replaced each target country's machinery/electronics network with donor-country dependency vectors. The true target-specific machinery/electronics network achieved mean PR-AUC `0.223`, but three donor-swapped variants had slightly higher mean PR-AUC (`donor_ARE = 0.224`, `donor_CHN = 0.224`, `donor_THA = 0.223`). This weakens a strong target-specificity claim. The current machinery/electronics network behaves like a network-family exposure map, but the evidence does not prove that every target country's exact dependency vector is uniquely optimal.

Third, a leave-one-country-out generalization test trained on 10 countries and tested on a completely unseen holdout country in 2025. In this stricter out-of-country setting, total-import network exposure performed best (`M4_total_import_network`, mean PR-AUC `0.278`), followed by external unweighted events (`0.274`), the operational baseline (`0.262`), and machinery/electronics strict network exposure (`0.249`). This is important: commodity-specific machinery/electronics exposure is strongest in the standard panel benchmark, but total-import network exposure is more robust for cross-country generalization.

### Stronger Mechanism Framing After Diagnostics

The best current framing is no longer "the network layer improves prediction" in a single universal way. The richer finding is that different network granularities serve different benchmark functions:

1. total-import network exposure acts as a broad structural prior and is more useful for leave-one-country-out generalization;
2. machinery/electronics strict network exposure acts as a commodity-channel exposure map and is strongest in the standard multi-country panel benchmark;
3. exact target-specific machinery/electronics weights are not yet uniquely validated against donor-network swaps, so the paper should avoid claiming precise causal or structural specificity;
4. the breakthrough direction is to model network granularity as a research design choice: broad networks for transfer/generalization, commodity-specific networks for channel attribution.

This gives the ML + NLP + Network framework a clearer division of labor. ML captures operational vulnerability, NLP estimates event pressure, and network structure determines the level at which event pressure should be mapped: broad dependency for cross-country robustness, or commodity-specific dependency for interpretable channel analysis.
