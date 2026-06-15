from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    ME_NETWORK_FEATURES,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_EQUAL_PLACEBO_FEATURES,
    TOTAL_NETWORK_FEATURES,
    TOTAL_RANDOM_PLACEBO_FEATURES,
    TOTAL_SHUFFLED_PLACEBO_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)
from scripts.run_panel32_network_gated_conversion_main import (  # noqa: E402
    ME_PLACEBO_BY_KIND,
    NETWORK_ROOTS,
    RANDOM_SEED,
    SEVERE_TARGET,
    add_network_gated_features,
    available,
    compact_gated_cols,
    fit_model,
    make_models,
    pooled_delta,
    safe_ap,
    top_hits,
)
from scripts.run_panel32_port_system_benchmark import (  # noqa: E402
    PORT_SYSTEM_ANOMALY,
    PORT_SYSTEM_DISTRIBUTION,
    PORT_SYSTEM_LEVELS,
    load_port_system_weekly,
)


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_network_gated_port_system_benchmark.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_network_gated_port_system_main.md"
METRICS = TABLE_DIR / "panel32_network_gated_port_system_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_network_gated_port_system_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_network_gated_port_system_predictions.csv"
DELTAS = TABLE_DIR / "panel32_network_gated_port_system_key_deltas.csv"
MATRIX = TABLE_DIR / "panel32_network_gated_port_system_claim_matrix.csv"

TOTAL_PLACEBO_BY_KIND = {
    "equal": TOTAL_EQUAL_PLACEBO_FEATURES,
    "random": TOTAL_RANDOM_PLACEBO_FEATURES,
    "shuffled": TOTAL_SHUFFLED_PLACEBO_FEATURES,
}

PORT_SYSTEM_VULNERABILITY = [
    "ps_container_neg_z52_weighted",
    "ps_container_neg_z52_max",
    "ps_container_neg_z52_gt1_ports",
    "ps_container_neg_z52_gt2_ports",
    "ps_trade_container_neg_z52_weighted",
    "ps_top3_static_ports_neg_z52",
    "ps_container_hhi",
    "ps_container_top1_share",
    "ps_container_ports_active_change_4w",
    "ps_container_neg_z52_weighted_change_4w",
    "ps_container_hhi_change_4w",
]

PORT_SYSTEM_CONTEXT = [
    "ps_top_ports",
    "ps_selected_container_coverage",
    "ps_selected_import_share_coverage",
    "ps_container_ports_active",
    "ps_container_active_ports",
    "ps_container_top3_share",
    "ps_trade_container_hhi",
    "ps_container_calls_selected_log",
    "ps_container_calls_selected_change_4w",
    "ps_container_trade_selected_log",
    "ps_container_trade_selected_change_4w",
]


def unique(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def ps_compact_gated_cols(df: pd.DataFrame, network_kind: str) -> list[str]:
    return [col for col in df.columns if col.startswith(f"psngc_{network_kind}_")]


def load_dataset() -> pd.DataFrame:
    base = pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)
    base[SEVERE_TARGET] = (
        base["next_week_container"] < (base["rolling_mean_12w"] - 2.0 * base["rolling_std_12w"].replace(0, np.nan))
    ).astype(int)
    port_system = load_port_system_weekly()
    ps_cols = [col for col in port_system.columns if col.startswith("ps_")]
    out = base.merge(port_system[["ISO3", "week"] + ps_cols], on=["ISO3", "week"], how="left")
    missing = out.loc[out["ps_top_ports"].isna(), ["ISO3", "week"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(f"Missing PortWatch selected-port rows after merge:\n{missing.head(20)}")
    ps_cols = [col for col in out.columns if col.startswith("ps_")]
    out[ps_cols] = out[ps_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = add_network_gated_features(out)
    out = add_port_system_conversion_features(out)
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DATASET, index=False)
    return out


def add_port_system_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stress = out["ps_container_neg_z52_weighted"].clip(lower=0).fillna(0.0)
    stress_max = out["ps_container_neg_z52_max"].clip(lower=0).fillna(0.0)
    trade_stress = out["ps_trade_container_neg_z52_weighted"].clip(lower=0).fillna(0.0)
    hhi = out["ps_container_hhi"].clip(lower=0).fillna(0.0)
    shortfall = out["ng_vulnerability_shortfall"].clip(lower=0).fillna(0.0)
    trend = out["ng_vulnerability_negative_trend"].clip(lower=0).fillna(0.0)
    event_vneg = out["external_very_negative_article_share"].clip(lower=0).fillna(0.0)
    event_me = out["external_me_strict_very_negative_exposure"].clip(lower=0).fillna(0.0)
    event_trade = np.log1p(out["external_trade_transport_count"].clip(lower=0).fillna(0.0))

    features = {
        "psng_vulnerability_port_stress": stress,
        "psng_vulnerability_port_stress_max": stress_max,
        "psng_vulnerability_trade_port_stress": trade_stress,
        "psng_vulnerability_shortfall_x_port_stress": shortfall * stress,
        "psng_vulnerability_trend_x_port_stress": trend * stress,
    }

    for network_kind, network_cols in NETWORK_ROOTS.items():
        network_vneg = out[network_cols[0]].clip(lower=0).fillna(0.0)
        network_trade = out[network_cols[1]].clip(lower=0).fillna(0.0)
        network_me = out[network_cols[-1]].clip(lower=0).fillna(0.0)
        features[f"psngc_{network_kind}_portstress_x_external_vneg"] = stress * event_vneg
        features[f"psngc_{network_kind}_portstress_x_network_vneg"] = stress * network_vneg
        features[f"psngc_{network_kind}_portstress_x_external_vneg_x_network_vneg"] = (
            stress * event_vneg * network_vneg
        )
        features[f"psngc_{network_kind}_shortfall_x_portstress_x_external_vneg_x_network_vneg"] = (
            shortfall * stress * event_vneg * network_vneg
        )
        features[f"psngc_{network_kind}_trend_x_portstress_x_external_vneg_x_network_vneg"] = (
            trend * stress * event_vneg * network_vneg
        )
        features[f"psngc_{network_kind}_tradestress_x_external_trade_x_network_trade"] = (
            trade_stress * event_trade * network_trade
        )
        features[f"psngc_{network_kind}_portstress_x_external_me_x_network_me"] = stress * event_me * network_me
        features[f"psngc_{network_kind}_hhi_x_external_vneg_x_network_vneg"] = hhi * event_vneg * network_vneg

    out = pd.concat([out, pd.DataFrame(features, index=out.index)], axis=1)
    new_cols = [col for col in out.columns if col.startswith("psng_") or col.startswith("psngc_")]
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def make_feature_groups(df: pd.DataFrame, country_features: list[str]) -> dict[str, list[str]]:
    base = available(df, OPERATIONAL_FEATURES) + country_features
    port_system = available(
        df,
        unique(PORT_SYSTEM_DISTRIBUTION + PORT_SYSTEM_ANOMALY + PORT_SYSTEM_LEVELS + PORT_SYSTEM_VULNERABILITY + PORT_SYSTEM_CONTEXT),
    )
    gdelt = available(df, OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES)
    true_network = available(df, TOTAL_NETWORK_FEATURES + ME_NETWORK_FEATURES)
    ps_vulnerability = available(
        df,
        [
            "psng_vulnerability_port_stress",
            "psng_vulnerability_port_stress_max",
            "psng_vulnerability_trade_port_stress",
            "psng_vulnerability_shortfall_x_port_stress",
            "psng_vulnerability_trend_x_port_stress",
        ],
    )
    base_ps = unique(base + port_system + ps_vulnerability)
    groups = {
        "PSNG0_portwatch_operational": base,
        "PSNG0b_portwatch_operational_port_system": base_ps,
        "PSNG1_portwatch_port_system_gdelt_additive": base_ps + gdelt,
        "PSNG2_portwatch_port_system_gdelt_wits_additive": base_ps + gdelt + true_network,
        "PSNG5_port_system_compact_gated_true": unique(
            base_ps + gdelt + true_network + compact_gated_cols(df, "true") + ps_compact_gated_cols(df, "true")
        ),
    }
    for kind in ["equal", "random", "shuffled"]:
        placebo_network = available(df, TOTAL_PLACEBO_BY_KIND[kind] + ME_PLACEBO_BY_KIND[kind])
        groups[f"PSNG6_{kind}_port_system_compact_gated_placebo"] = unique(
            base_ps + gdelt + placebo_network + compact_gated_cols(df, kind) + ps_compact_gated_cols(df, kind)
        )
    return {name: unique(features) for name, features in groups.items()}


def usable_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in frame.columns and pd.api.types.is_numeric_dtype(frame[f])]


def run_fold(fold, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    groups = make_feature_groups(train, country_features)
    rows = []
    predictions = []
    for group_name, raw_features in groups.items():
        features = usable_features(train, raw_features)
        for model_name, (model, fit_mode) in make_models().items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET].reset_index(drop=True), val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            main_scores = evaluate_predictions(test[TARGET].reset_index(drop=True), test_proba, threshold)
            pred = test[["ISO3", "country", "week", TARGET, SEVERE_TARGET]].copy()
            pred["fold"] = fold.name
            pred["feature_group"] = group_name
            pred["model"] = model_name
            pred["predicted_probability"] = test_proba
            pred["predicted_label"] = (test_proba >= threshold).astype(int)
            row = {
                "fold": fold.name,
                "feature_group": group_name,
                "model": model_name,
                "feature_count": len(features),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positives": int(train[TARGET].sum()),
                "validation_positives": int(validation[TARGET].sum()),
                "test_positives": int(test[TARGET].sum()),
                "test_severe_positives": int(test[SEVERE_TARGET].sum()),
                "selected_threshold": threshold,
                "validation_f1": val_f1,
                **main_scores,
                "severe_pr_auc": safe_ap(test[SEVERE_TARGET], pd.Series(test_proba, index=test.index)),
            }
            for k in [10, 25, 50]:
                row[f"main_top{k}_hits"] = top_hits(pred, TARGET, k)
                row[f"severe_top{k}_hits"] = top_hits(pred, SEVERE_TARGET, k)
            rows.append(row)
            predictions.append(pred)
    return rows, predictions


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "folds": ("fold", "nunique"),
        "mean_main_pr_auc": ("pr_auc", "mean"),
        "median_main_pr_auc": ("pr_auc", "median"),
        "mean_severe_pr_auc": ("severe_pr_auc", "mean"),
        "median_severe_pr_auc": ("severe_pr_auc", "median"),
        "mean_roc_auc": ("roc_auc", "mean"),
        "mean_f1": ("f1", "mean"),
        "feature_count": ("feature_count", "max"),
    }
    for prefix in ["main", "severe"]:
        for k in [10, 25, 50]:
            aggregations[f"{prefix}_top{k}_hits"] = (f"{prefix}_top{k}_hits", "sum")
    return (
        metrics.groupby(["feature_group", "model"], as_index=False)
        .agg(**aggregations)
        .sort_values(["mean_main_pr_auc", "main_top25_hits"], ascending=False)
    )


def hit_delta(predictions: pd.DataFrame, focus: tuple[str, str], baseline: tuple[str, str], target: str, k: int) -> int:
    focus_group, focus_model = focus
    base_group, base_model = baseline
    focus_frame = predictions.loc[
        predictions["feature_group"].eq(focus_group) & predictions["model"].eq(focus_model)
    ]
    baseline_frame = predictions.loc[
        predictions["feature_group"].eq(base_group) & predictions["model"].eq(base_model)
    ]
    delta = 0
    for fold in sorted(set(focus_frame["fold"]) & set(baseline_frame["fold"])):
        f = focus_frame.loc[focus_frame["fold"].eq(fold)]
        b = baseline_frame.loc[baseline_frame["fold"].eq(fold)]
        delta += top_hits(f, target, k) - top_hits(b, target, k)
    return delta


def make_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    model = "sklearn_gradient_boosting"
    focus = ("PSNG5_port_system_compact_gated_true", model)
    contrasts = [
        (focus, ("PSNG0_portwatch_operational", model), "ps_gated_true_vs_operational"),
        (focus, ("PSNG0b_portwatch_operational_port_system", model), "ps_gated_true_vs_port_system_operational"),
        (focus, ("PSNG1_portwatch_port_system_gdelt_additive", model), "ps_gated_true_vs_ps_gdelt_additive"),
        (focus, ("PSNG2_portwatch_port_system_gdelt_wits_additive", model), "ps_gated_true_vs_ps_wits_additive"),
        (
            focus,
            ("PSNG6_equal_port_system_compact_gated_placebo", model),
            "ps_gated_true_vs_equal_ps_placebo",
        ),
        (
            focus,
            ("PSNG6_random_port_system_compact_gated_placebo", model),
            "ps_gated_true_vs_random_ps_placebo",
        ),
        (
            focus,
            ("PSNG6_shuffled_port_system_compact_gated_placebo", model),
            "ps_gated_true_vs_shuffled_ps_placebo",
        ),
        (
            ("PSNG0b_portwatch_operational_port_system", model),
            ("PSNG0_portwatch_operational", model),
            "port_system_operational_vs_national_operational",
        ),
        (
            ("PSNG2_portwatch_port_system_gdelt_wits_additive", model),
            ("PSNG0b_portwatch_operational_port_system", model),
            "ps_wits_additive_vs_port_system_operational",
        ),
    ]
    rows = []
    for focus_tuple, baseline, contrast in contrasts:
        for label_name, target in [("main", TARGET), ("severe", SEVERE_TARGET)]:
            pr = pooled_delta(predictions, focus_tuple, baseline, target)
            rows.append(
                {
                    "contrast": contrast,
                    "focus_feature_group": focus_tuple[0],
                    "baseline_feature_group": baseline[0],
                    "model": model,
                    "label": label_name,
                    "pooled_pr_auc_delta": pr[0],
                    "ci_low": pr[1],
                    "ci_high": pr[2],
                    "p_gt_0": pr[3],
                    "top10_hit_delta": hit_delta(predictions, focus_tuple, baseline, target, 10),
                    "top25_hit_delta": hit_delta(predictions, focus_tuple, baseline, target, 25),
                    "top50_hit_delta": hit_delta(predictions, focus_tuple, baseline, target, 50),
                }
            )
    return pd.DataFrame(rows)


def make_claim_matrix(summary: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    gb = summary.loc[summary["model"].eq("sklearn_gradient_boosting")].set_index("feature_group")
    gated_vs_add = deltas.loc[
        deltas["contrast"].eq("ps_gated_true_vs_ps_wits_additive") & deltas["label"].eq("main")
    ].iloc[0]
    gated_vs_equal = deltas.loc[
        deltas["contrast"].eq("ps_gated_true_vs_equal_ps_placebo") & deltas["label"].eq("main")
    ].iloc[0]
    ps_vs_nat = deltas.loc[
        deltas["contrast"].eq("port_system_operational_vs_national_operational") & deltas["label"].eq("main")
    ].iloc[0]
    rows = [
        {
            "claim": "PortWatch selected-port system sharpens operational vulnerability.",
            "data_sources": "PortWatch only",
            "model_family": "PSNG0b vs PSNG0",
            "evidence_type": "temporal PR-AUC/top-k",
            "main_metric": f"PSNG0b mean PR-AUC {gb.loc['PSNG0b_portwatch_operational_port_system','mean_main_pr_auc']:.4f} vs PSNG0 {gb.loc['PSNG0_portwatch_operational','mean_main_pr_auc']:.4f}; pooled delta {ps_vs_nat['pooled_pr_auc_delta']:.4f}",
            "paper_bucket": "main/supporting if positive",
            "status": "computed",
        },
        {
            "claim": "Network-gated conversion benefits from richer operational vulnerability.",
            "data_sources": "PortWatch + GDELT + WITS",
            "model_family": "PSNG5 compact port-system gated conversion",
            "evidence_type": "temporal PR-AUC/top-k/placebo",
            "main_metric": f"PSNG5 mean PR-AUC {gb.loc['PSNG5_port_system_compact_gated_true','mean_main_pr_auc']:.4f}; vs PSNG2 pooled delta {gated_vs_add['pooled_pr_auc_delta']:.4f}; vs equal placebo {gated_vs_equal['pooled_pr_auc_delta']:.4f}",
            "paper_bucket": "main if better than additive/placebo; otherwise supporting/negative",
            "status": "computed",
        },
        {
            "claim": "Three-source main-paper constraint is preserved.",
            "data_sources": "PortWatch + GDELT + WITS",
            "model_family": "all PSNG groups",
            "evidence_type": "data provenance",
            "main_metric": "uses PortWatch selected-port cache plus the existing expanded32 GDELT/WITS panel; no fourth-source features",
            "paper_bucket": "main",
            "status": "satisfied",
        },
    ]
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, matrix: pd.DataFrame) -> None:
    gb = summary.loc[summary["model"].eq("sklearn_gradient_boosting")]
    content = f"""# Panel32 Network-Gated Port-System Conversion Benchmark

## Purpose

This experiment strengthens the main-paper **Network-Gated Event Conversion Model** without adding a fourth data source. It keeps the main data scope to PortWatch, GDELT, and WITS, but expands the PortWatch side from country-week operational activity to selected-port system vulnerability.

## Dataset

- File: `data/processed/multicountry32_network_gated_port_system_benchmark.csv`
- Base panel: `data/processed/multicountry32_container_event_network_benchmark.csv`
- PortWatch selected-port cache: `data/interim/portwatch_port_system_panel32_country_weekly.csv`
- Countries: {df["ISO3"].nunique()}
- Rows: {len(df)}
- Main positives: {int(df[TARGET].sum())}
- Severe 2.0-sigma positives: {int(df[SEVERE_TARGET].sum())}
- Week range: {df["week"].min().date()} to {df["week"].max().date()}

## Required Model Ladder

1. National PortWatch operational only: `PSNG0_portwatch_operational`
2. PortWatch operational + selected-port system vulnerability: `PSNG0b_portwatch_operational_port_system`
3. PortWatch + GDELT additive: `PSNG1_portwatch_port_system_gdelt_additive`
4. PortWatch + GDELT + WITS additive: `PSNG2_portwatch_port_system_gdelt_wits_additive`
5. Proposed port-system compact gated conversion: `PSNG5_port_system_compact_gated_true`
6. Equal/random/shuffled WITS port-system gated placebos: `PSNG6_*_port_system_compact_gated_placebo`

## Gradient Boosting Main Table

{gb[["feature_group", "mean_main_pr_auc", "mean_severe_pr_auc", "mean_roc_auc", "main_top10_hits", "main_top25_hits", "main_top50_hits", "severe_top10_hits", "severe_top25_hits", "feature_count"]].to_markdown(index=False)}

## Key Gradient-Boosting Deltas

{deltas.to_markdown(index=False)}

## Claim-Evidence Matrix

{matrix.to_markdown(index=False)}

## Reading

This test asks whether the event-conversion idea becomes more credible when operational vulnerability is measured at a finer PortWatch selected-port-system level. The result should be promoted only if the true WITS gated model improves over the port-system additive WITS baseline and separates from equal/random/shuffled WITS gated placebos. If it fails those controls, the selected-port system can still support operational-vulnerability measurement, but not a stronger network-gated predictive claim.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    rows = []
    prediction_frames = []
    for fold in FOLDS:
        fold_rows, fold_predictions = run_fold(fold, df)
        rows.extend(fold_rows)
        prediction_frames.extend(fold_predictions)
        print(f"Finished {fold.name}: {len(fold_rows)} rows", flush=True)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize(metrics)
    deltas = make_deltas(predictions)
    matrix = make_claim_matrix(summary, deltas)
    metrics.to_csv(METRICS, index=False)
    summary.to_csv(SUMMARY, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    deltas.to_csv(DELTAS, index=False)
    matrix.to_csv(MATRIX, index=False)
    write_report(df, summary, deltas, matrix)
    print(f"Saved dataset: {OUTPUT_DATASET}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved deltas: {DELTAS}")
    print(f"Saved matrix: {MATRIX}")
    print(f"Saved report: {REPORT}")
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    run()
