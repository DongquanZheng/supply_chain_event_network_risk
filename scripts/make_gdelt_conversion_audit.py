"""Create a case-level GDELT event-conversion audit packet.

The audit stays inside the main paper's three-source scope:
PortWatch + GDELT + WITS. It samples true positives, false positives,
false negatives, true-WITS-high cases, and placebo-high cases from existing
out-of-sample predictions and annotates each case using local metadata-derived
features. It does not fetch new data or train models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
REPORT_PATH = ROOT / "reports" / "gdelt_conversion_audit.md"

PREDICTIONS = TABLE_DIR / "panel32_gdelt_conversion_propensity_predictions.csv"
PANEL = ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
PARTNER_EVENTS = ROOT / "data" / "interim" / "gkg_partner_event_features_2021-01-01_2025-12-31_expanded32.csv"
WITS_WEIGHTS = ROOT / "data" / "interim" / "panel32_total_dependency_weights_2023.csv"
OUT_CSV = TABLE_DIR / "gdelt_conversion_audit.csv"

PRIMARY_FEATURE = "GCL4_true_wits_conversion_propensity_gated"
PRIMARY_MODEL = "sklearn_gradient_boosting"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(PREDICTIONS, parse_dates=["week"])
    pred = pred[
        pred["feature_group"].eq(PRIMARY_FEATURE) & pred["model"].eq(PRIMARY_MODEL)
    ].copy()
    if pred.empty:
        raise ValueError("Primary conversion-propensity predictions are empty.")

    panel = pd.read_csv(PANEL, parse_dates=["week"])
    partner = pd.read_csv(PARTNER_EVENTS, parse_dates=["event_week"])
    weights = pd.read_csv(WITS_WEIGHTS)
    return pred, panel, partner, weights


def add_fold_ranks(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["fold_score_rank"] = out.groupby("fold")["predicted_probability"].rank(
        method="first", ascending=False
    )
    out["top25_alert"] = out["fold_score_rank"].le(25)
    out["top50_alert"] = out["fold_score_rank"].le(50)
    out["actual_positive"] = out["abnormal_next_week_container"].astype(int).eq(1)
    out["actual_severe_positive"] = out["abnormal_next_week_container_2p0sigma"].astype(int).eq(1)
    out["prediction_outcome"] = np.select(
        [
            out["top25_alert"] & out["actual_positive"],
            out["top25_alert"] & ~out["actual_positive"],
            ~out["top50_alert"] & out["actual_positive"],
        ],
        ["true_positive_top25", "false_positive_top25", "false_negative_outside_top50"],
        default="other",
    )
    return out


def merge_panel(pred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ISO3",
        "country",
        "week",
        "portcalls_container",
        "next_week_container",
        "abnormal_threshold",
        "operational_shortfall_12w",
        "negative_trend_4w",
        "external_article_count",
        "external_very_negative_article_share",
        "external_trade_transport_count",
        "external_risk_theme_count",
        "network_partner_article_count",
        "network_very_negative_exposure",
        "equal_very_negative_exposure",
        "random_very_negative_exposure",
        "shuffled_very_negative_exposure",
        "network_trade_transport_exposure",
        "network_risk_theme_exposure",
        "me_network_strict_very_negative_exposure",
        "me_equal_strict_very_negative_exposure",
        "me_random_strict_very_negative_exposure",
        "me_shuffled_strict_very_negative_exposure",
        "network_minus_equal_very_negative",
        "network_to_equal_very_negative_ratio",
    ]
    existing = [c for c in cols if c in panel.columns]
    return pred.merge(panel[existing], on=["ISO3", "country", "week"], how="left")


def add_partner_attribution(
    cases: pd.DataFrame, partner: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    partner = partner.rename(columns={"event_week": "week", "code": "partner_code"}).copy()
    partner["partner_iso3"] = partner["partner_code"].map(code_to_iso3())
    partner = partner.dropna(subset=["partner_iso3"])

    records = []
    for case in cases[["ISO3", "week"]].drop_duplicates().itertuples(index=False):
        week_partner = partner[partner["week"].eq(case.week)]
        weight_target = weights[weights["ISO3"].eq(case.ISO3)]
        merged = weight_target.merge(week_partner, on="partner_iso3", how="left")
        for col in [
            "article_count",
            "very_negative_article_share",
            "trade_transport_count",
            "risk_theme_count",
        ]:
            merged[col] = merged[col].fillna(0)
        merged["wits_vneg_contribution"] = (
            merged["import_dependency_share"] * merged["very_negative_article_share"]
        )
        merged["wits_trade_contribution"] = (
            merged["import_dependency_share"] * merged["trade_transport_count"]
        )
        merged = merged.sort_values(
            ["wits_vneg_contribution", "wits_trade_contribution", "import_dependency_share"],
            ascending=False,
        )
        top = merged.iloc[0] if len(merged) else pd.Series(dtype=object)
        records.append(
            {
                "ISO3": case.ISO3,
                "week": case.week,
                "top_partner_iso3": top.get("partner_iso3", ""),
                "top_partner_dependency_share": top.get("import_dependency_share", np.nan),
                "top_partner_event_articles": top.get("article_count", np.nan),
                "top_partner_very_negative_share": top.get("very_negative_article_share", np.nan),
                "top_partner_trade_transport_count": top.get("trade_transport_count", np.nan),
                "top_partner_risk_theme_count": top.get("risk_theme_count", np.nan),
                "top_partner_wits_vneg_contribution": top.get("wits_vneg_contribution", np.nan),
            }
        )
    return cases.merge(pd.DataFrame(records), on=["ISO3", "week"], how="left")


def code_to_iso3() -> dict[str, str]:
    return {
        "AE": "ARE",
        "BE": "BEL",
        "BR": "BRA",
        "CA": "CAN",
        "CI": "CHL",
        "CH": "CHN",
        "EZ": "CZE",
        "EG": "EGY",
        "FR": "FRA",
        "GM": "DEU",
        "UK": "GBR",
        "ID": "IDN",
        "IN": "IND",
        "IT": "ITA",
        "JA": "JPN",
        "KS": "KOR",
        "MX": "MEX",
        "MY": "MYS",
        "NL": "NLD",
        "PK": "PAK",
        "PM": "PAN",
        "RP": "PHL",
        "PL": "POL",
        "SA": "SAU",
        "SN": "SGP",
        "SP": "ESP",
        "SW": "SWE",
        "TU": "TUR",
        "US": "USA",
        "VM": "VNM",
        "SF": "ZAF",
        "TH": "THA",
    }


def sample_cases(df: pd.DataFrame, n_each: int = 8) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    def take(category: str, data: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        subset = data.sort_values(sort_cols, ascending=ascending).head(n_each).copy()
        subset["audit_sample_category"] = category
        pieces.append(subset)

    take(
        "true_positive",
        df[df["prediction_outcome"].eq("true_positive_top25")],
        ["predicted_probability", "network_very_negative_exposure"],
        [False, False],
    )
    take(
        "false_positive",
        df[df["prediction_outcome"].eq("false_positive_top25")],
        ["predicted_probability", "network_very_negative_exposure"],
        [False, False],
    )
    take(
        "false_negative",
        df[df["prediction_outcome"].eq("false_negative_outside_top50")],
        ["predicted_probability", "external_very_negative_article_share"],
        [True, False],
    )

    true_wits_high = df[
        df["network_very_negative_exposure"].ge(df["network_very_negative_exposure"].quantile(0.9))
    ].copy()
    take(
        "true_wits_high",
        true_wits_high,
        ["network_very_negative_exposure", "predicted_probability"],
        [False, False],
    )

    exposure_cols = [
        "equal_very_negative_exposure",
        "random_very_negative_exposure",
        "shuffled_very_negative_exposure",
    ]
    df = df.copy()
    df["max_placebo_vneg_exposure"] = df[exposure_cols].max(axis=1)
    df["placebo_minus_true_vneg_exposure"] = (
        df["max_placebo_vneg_exposure"] - df["network_very_negative_exposure"]
    )
    placebo_high = df[
        df["placebo_minus_true_vneg_exposure"].ge(
            df["placebo_minus_true_vneg_exposure"].quantile(0.9)
        )
    ].copy()
    take(
        "placebo_high",
        placebo_high,
        ["placebo_minus_true_vneg_exposure", "predicted_probability"],
        [False, False],
    )

    out = pd.concat(pieces, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["ISO3", "week"], keep="first").copy()
    if len(out) < 20:
        filler = df.sort_values("predicted_probability", ascending=False).head(40)
        filler = filler[~filler.set_index(["ISO3", "week"]).index.isin(out.set_index(["ISO3", "week"]).index)]
        filler = filler.head(20 - len(out)).copy()
        filler["audit_sample_category"] = "high_score_filler"
        out = pd.concat([out, filler], ignore_index=True, sort=False)
    return out.sort_values(["audit_sample_category", "fold", "predicted_probability"], ascending=[True, True, False])


def annotate(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["logistics_event_relevance"] = np.select(
        [
            (out["external_trade_transport_count"] >= out["external_trade_transport_count"].quantile(0.75))
            | (out["network_trade_transport_exposure"] >= out["network_trade_transport_exposure"].quantile(0.75))
            | (out["me_network_strict_very_negative_exposure"] >= out["me_network_strict_very_negative_exposure"].quantile(0.75)),
            (out["external_risk_theme_count"] >= out["external_risk_theme_count"].quantile(0.5))
            | (out["network_risk_theme_exposure"] >= out["network_risk_theme_exposure"].quantile(0.5)),
        ],
        ["direct_or_trade_logistics_related", "indirect_macro_or_risk_related"],
        default="weak_or_broad_media_signal",
    )
    out["important_partner_event"] = np.where(
        (out["top_partner_dependency_share"] >= 0.05)
        & (
            (out["top_partner_event_articles"] > 0)
            | (out["top_partner_very_negative_share"] > 0)
        ),
        "yes",
        "no_or_unclear",
    )
    out["operational_vulnerability_state"] = np.where(
        out["operational_shortfall_12w"] > 0,
        "current_shortfall_positive",
        "current_shortfall_nonpositive",
    )
    out["network_audit_support"] = np.select(
        [
            (out["network_very_negative_exposure"] > out["equal_very_negative_exposure"])
            & (out["network_very_negative_exposure"] > out["random_very_negative_exposure"])
            & (out["network_very_negative_exposure"] > out["shuffled_very_negative_exposure"]),
            (out["network_very_negative_exposure"] > out["equal_very_negative_exposure"])
            | (out["network_very_negative_exposure"] > out["random_very_negative_exposure"])
            | (out["network_very_negative_exposure"] > out["shuffled_very_negative_exposure"]),
        ],
        ["supports_true_wits_over_placebos", "partial_or_mixed_support"],
        default="placebo_or_broad_media_dominates",
    )
    out["plausible_conversion_path"] = np.select(
        [
            out["actual_positive"]
            & out["operational_shortfall_12w"].gt(0)
            & out["logistics_event_relevance"].ne("weak_or_broad_media_signal")
            & out["network_audit_support"].ne("placebo_or_broad_media_dominates"),
            out["actual_positive"] & out["logistics_event_relevance"].ne("weak_or_broad_media_signal"),
            ~out["actual_positive"] & out["top25_alert"],
        ],
        ["yes_strong", "yes_weak_or_nonnetwork", "no_observed_conversion_false_alert"],
        default="unclear_or_missed_conversion",
    )
    out["audit_note"] = out.apply(make_note, axis=1)
    return out


def make_note(row: pd.Series) -> str:
    return (
        f"{row['audit_sample_category']}; {row['prediction_outcome']}; "
        f"event={row['logistics_event_relevance']}; "
        f"vulnerability={row['operational_vulnerability_state']}; "
        f"top_partner={row.get('top_partner_iso3','')} "
        f"(share={row.get('top_partner_dependency_share', np.nan):.3f}); "
        f"network_audit={row['network_audit_support']}."
    )


def write_report(audit: pd.DataFrame) -> None:
    category = audit.groupby("audit_sample_category").size().rename("cases")
    relevance = audit.groupby(["audit_sample_category", "logistics_event_relevance"]).size().unstack(fill_value=0)
    support = audit.groupby(["audit_sample_category", "network_audit_support"]).size().unstack(fill_value=0)
    conversion = audit.groupby(["audit_sample_category", "plausible_conversion_path"]).size().unstack(fill_value=0)

    lines = [
        "# GDELT Conversion Audit",
        "",
        "Date: 2026-06-15",
        "",
        "Scope: PortWatch + GDELT + WITS only. This audit uses local metadata-derived GDELT features and model outputs; it is not a full-text article annotation exercise.",
        "",
        "## Sampling Design",
        "",
        f"- Primary score: `{PRIMARY_FEATURE}` / `{PRIMARY_MODEL}`.",
        "- Case groups: true positives, false positives, false negatives, true-WITS-high cases, and placebo-high cases.",
        f"- Output rows: `{len(audit)}`.",
        "",
        "## Case Counts",
        "",
        category.to_frame().to_markdown(),
        "",
        "## Event Relevance Labels",
        "",
        relevance.to_markdown(),
        "",
        "## Network Audit Support",
        "",
        support.to_markdown(),
        "",
        "## Plausible Conversion Path",
        "",
        conversion.to_markdown(),
        "",
        "## Interpretation",
        "",
        "- Direct or trade/logistics-related event evidence appears in some high-risk and true-positive cases, but broad media/risk signals remain common.",
        "- True WITS support is mixed: several cases are better interpreted as broad or placebo-competitive event pressure rather than exact trade-network conversion.",
        "- Operational shortfall remains central for the strongest conversion-path interpretation.",
        "",
        "## Guardrail",
        "",
        "Do not cite this audit as proof that GDELT measures true disruption or that WITS proves causal propagation. Use it as a structured plausibility and attribution audit.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pred, panel, partner, weights = read_inputs()
    scored = add_fold_ranks(pred)
    enriched = merge_panel(scored, panel)
    sampled = sample_cases(enriched)
    attributed = add_partner_attribution(sampled, partner, weights)
    audit = annotate(attributed)

    columns = [
        "audit_sample_category",
        "ISO3",
        "country",
        "week",
        "fold",
        "predicted_probability",
        "fold_score_rank",
        "top25_alert",
        "prediction_outcome",
        "abnormal_next_week_container",
        "abnormal_next_week_container_2p0sigma",
        "portcalls_container",
        "next_week_container",
        "abnormal_threshold",
        "operational_shortfall_12w",
        "negative_trend_4w",
        "external_article_count",
        "external_very_negative_article_share",
        "external_trade_transport_count",
        "external_risk_theme_count",
        "network_very_negative_exposure",
        "equal_very_negative_exposure",
        "random_very_negative_exposure",
        "shuffled_very_negative_exposure",
        "me_network_strict_very_negative_exposure",
        "me_equal_strict_very_negative_exposure",
        "me_random_strict_very_negative_exposure",
        "me_shuffled_strict_very_negative_exposure",
        "top_partner_iso3",
        "top_partner_dependency_share",
        "top_partner_event_articles",
        "top_partner_very_negative_share",
        "top_partner_trade_transport_count",
        "top_partner_risk_theme_count",
        "important_partner_event",
        "logistics_event_relevance",
        "operational_vulnerability_state",
        "network_audit_support",
        "plausible_conversion_path",
        "audit_note",
    ]
    audit[columns].to_csv(OUT_CSV, index=False)
    write_report(audit[columns])
    print(f"Wrote {OUT_CSV} with {len(audit)} cases")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
