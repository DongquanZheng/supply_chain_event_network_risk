from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GDELT_TO_ISO3, ISO3_TO_GDELT  # noqa: E402


PREDICTIONS = PROJECT_ROOT / "reports" / "tables" / "panel_benchmark_predictions.csv"
DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv"
DOCS = PROJECT_ROOT / "data" / "interim" / "gkg_nlp_candidate_docs_2023-01-01_2025-12-31.csv"
TOTAL_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_total_dependency_weights_2023.csv"
ME_WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel_me_dependency_weights_2023.csv"
OUTPUT_TABLE = PROJECT_ROOT / "reports" / "tables" / "gdelt_high_risk_week_sanity_check.csv"
REPORT = PROJECT_ROOT / "reports" / "gdelt_sanity_check.md"

RISK_GROUPS = [
    "M2_own_country_news",
    "M3_external_unweighted_events",
    "M4_total_import_network",
    "M5_me_strict_network",
]

DIRECT_PATTERNS = {
    "maritime_logistics": re.compile(
        r"\b(port|ports|shipping|freight|cargo|container|vessel|maritime|logistics|"
        r"congestion|suez|panama|red sea|red-sea|tanker)\b",
        re.I,
    ),
    "manufacturing_electronics": re.compile(
        r"\b(semiconductor|semiconductors|chip|chips|electronics|machinery|machine|"
        r"factory|factories|manufactur|industrial equipment|supply chain)\b",
        re.I,
    ),
    "trade_policy": re.compile(
        r"\b(tariff|tariffs|sanction|sanctions|export control|export ban|import ban|"
        r"customs|trade war|trade restriction|trade restrictions|embargo|export|exports|import|imports)\b",
        re.I,
    ),
    "energy_transport": re.compile(
        r"\b(oil|gas|fuel|energy|lng|crude|petroleum|refinery)\b",
        re.I,
    ),
}

DISRUPTION_PATTERN = re.compile(
    r"\b(disruption|disruptions|delay|delays|shortage|shortages|congestion|strike|"
    r"strikes|attack|attacks|blocked|blockage|reroute|rerouting|shutdown|halt|"
    r"crisis|conflict|war|earthquake|flood|storm|typhoon|fire|explosion)\b",
    re.I,
)

INDIRECT_PATTERN = re.compile(
    r"\b(conflict|violence|security|border|inflation|election|government|policy|"
    r"protest|unrest|military|diplomat|central bank|currency|debt)\b",
    re.I,
)


def slug_text(url: str) -> str:
    parsed = urlparse(str(url))
    text = " ".join([parsed.netloc, parsed.path, parsed.query])
    text = unquote(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_text(row: pd.Series) -> str:
    fields = [
        slug_text(row.get("DocumentIdentifier", "")),
        str(row.get("AllNames", "")),
        str(row.get("V2Organizations", "")),
    ]
    return " ".join(fields).lower()


def classify_text(title_text: str, context_text: str, tone: float) -> tuple[str, str, int]:
    title_hits = [name for name, pattern in DIRECT_PATTERNS.items() if pattern.search(title_text)]
    context_hits = [name for name, pattern in DIRECT_PATTERNS.items() if pattern.search(context_text)]
    has_disruption = bool(DISRUPTION_PATTERN.search(title_text + " " + context_text))
    has_indirect = bool(INDIRECT_PATTERN.search(title_text + " " + context_text))
    negative = tone < -1.0

    if title_hits and (has_disruption or negative):
        return "high", "title:" + ",".join(title_hits), 3
    if title_hits:
        return "medium", "title:" + ",".join(title_hits), 2
    if context_hits and (has_disruption or negative):
        return "medium", "context:" + ",".join(context_hits), 2
    if has_disruption and has_indirect:
        return "medium", "indirect_disruption_context", 2
    if has_indirect or negative:
        return "low", "generic_risk_or_negative_tone", 1
    return "low", "weak_supply_chain_evidence", 0


def theme_tokens(themes: str) -> list[str]:
    tokens = []
    for item in str(themes).split(";"):
        if not item:
            continue
        token = item.split(",", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def top_theme_summary(frame: pd.DataFrame, n: int = 8) -> str:
    counter: Counter[str] = Counter()
    for value in frame["V2Themes"].fillna(""):
        counter.update(theme_tokens(value))
    return "; ".join(token for token, _ in counter.most_common(n))


def short_doc(row: pd.Series) -> str:
    slug = slug_text(row.get("DocumentIdentifier", ""))
    slug = slug[:120]
    source = str(row.get("SourceCommonName", ""))
    tone = float(row.get("tone", 0.0))
    return f"{source} | tone={tone:.2f} | {slug}"


def load_docs(required_weeks: set[str]) -> pd.DataFrame:
    usecols = [
        "event_week",
        "code",
        "SourceCommonName",
        "DocumentIdentifier",
        "V2Themes",
        "AllNames",
        "V2Organizations",
        "tone",
    ]
    frames = []
    for chunk in pd.read_csv(DOCS, usecols=usecols, chunksize=5000):
        mask = chunk["event_week"].astype(str).str.slice(0, 10).isin(required_weeks)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=usecols)

    docs = pd.concat(frames, ignore_index=True)
    docs["event_week"] = pd.to_datetime(docs["event_week"])
    docs["partner_iso3"] = docs["code"].map(GDELT_TO_ISO3)
    docs["title_text"] = docs["DocumentIdentifier"].fillna("").map(slug_text)
    docs["evidence_text"] = (
        docs["title_text"]
        + " "
        + docs["AllNames"].fillna("")
        + " "
        + docs["V2Organizations"].fillna("")
    ).str.lower()
    labels = [
        classify_text(title, text, float(tone))
        for title, text, tone in zip(docs["title_text"], docs["evidence_text"], docs["tone"], strict=False)
    ]
    docs["doc_relevance_label"] = [item[0] for item in labels]
    docs["doc_relevance_reason"] = [item[1] for item in labels]
    docs["doc_relevance_score"] = [item[2] for item in labels]
    docs["doc_selection_score"] = docs["doc_relevance_score"] + ((-docs["tone"]).clip(lower=0, upper=10) / 10)
    return docs


def select_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions.loc[
        predictions["fold"].eq("test_2025")
        & predictions["model"].eq("random_forest")
        & predictions["feature_group"].isin(RISK_GROUPS)
    ].copy()

    selected = []
    for group in RISK_GROUPS:
        group_rows = test.loc[test["feature_group"].eq(group)].copy()
        positives = group_rows.loc[group_rows["abnormal_next_week_container"].eq(1)]
        false_alerts = group_rows.loc[group_rows["abnormal_next_week_container"].eq(0)]
        selected.append(positives.nlargest(3, "predicted_probability"))
        selected.append(false_alerts.nlargest(2, "predicted_probability"))

    cases = pd.concat(selected, ignore_index=True)
    cases = cases.drop_duplicates(["feature_group", "week", "ISO3"])
    return cases.sort_values(["feature_group", "predicted_probability"], ascending=[True, False])


def weighted_partner_order(
    target_iso3: str,
    week_docs: pd.DataFrame,
    weights: pd.DataFrame | None,
) -> list[str]:
    partners = week_docs.loc[week_docs["partner_iso3"].ne(target_iso3), ["partner_iso3", "doc_selection_score"]]
    if weights is not None:
        partners = partners.merge(
            weights.loc[weights["ISO3"].eq(target_iso3), ["partner_iso3", "import_dependency_share"]],
            on="partner_iso3",
            how="inner",
        )
        partners["partner_score"] = partners["doc_selection_score"] * partners["import_dependency_share"]
    else:
        partners["partner_score"] = partners["doc_selection_score"]

    if partners.empty:
        return []
    return (
        partners.groupby("partner_iso3")["partner_score"]
        .mean()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )


def docs_for_case(
    case: pd.Series,
    docs: pd.DataFrame,
    total_weights: pd.DataFrame,
    me_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    week = pd.Timestamp(case["week"])
    target = str(case["ISO3"])
    feature_group = str(case["feature_group"])
    week_docs = docs.loc[docs["event_week"].eq(week)].copy()

    if feature_group == "M2_own_country_news":
        scope_docs = week_docs.loc[week_docs["partner_iso3"].eq(target)].copy()
        scope = "own_country"
        partners = target
    elif feature_group == "M4_total_import_network":
        ordered = weighted_partner_order(target, week_docs, total_weights)
        scope_docs = week_docs.loc[week_docs["partner_iso3"].isin(ordered)].copy()
        scope = "total_import_weighted_partners"
        partners = ",".join(ordered)
    elif feature_group == "M5_me_strict_network":
        ordered = weighted_partner_order(target, week_docs, me_weights)
        scope_docs = week_docs.loc[week_docs["partner_iso3"].isin(ordered)].copy()
        scope = "machinery_electronics_weighted_partners"
        partners = ",".join(ordered)
    else:
        ordered = weighted_partner_order(target, week_docs, None)
        scope_docs = week_docs.loc[week_docs["partner_iso3"].isin(ordered)].copy()
        scope = "external_unweighted_partners"
        partners = ",".join(ordered)

    if scope_docs.empty:
        return scope_docs, scope, partners

    scope_docs = scope_docs.sort_values(
        ["doc_selection_score", "doc_relevance_score", "tone"],
        ascending=[False, False, True],
    )
    return scope_docs.head(8), scope, partners


def case_label(evidence_docs: pd.DataFrame) -> tuple[str, str]:
    if evidence_docs.empty:
        return "unverifiable", "No candidate documents found in cache for this week/scope."

    label_counts = evidence_docs["doc_relevance_label"].value_counts()
    high = int(label_counts.get("high", 0))
    medium = int(label_counts.get("medium", 0))
    low = int(label_counts.get("low", 0))
    top_reasons = evidence_docs["doc_relevance_reason"].value_counts().head(3).index.tolist()

    if high >= 2:
        return "highly_relevant", f"{high} high-relevance docs; reasons: {', '.join(top_reasons)}."
    if high == 1 or medium >= 3:
        return "partly_relevant", f"{high} high and {medium} medium docs; reasons: {', '.join(top_reasons)}."
    if medium >= 1:
        return "weakly_relevant", f"{medium} medium docs but mostly generic/noisy evidence; low docs={low}."
    return "mostly_noise", f"No direct supply-chain-relevant docs among top evidence; low docs={low}."


def strict_case_label(evidence_docs: pd.DataFrame) -> tuple[str, str, int, int]:
    if evidence_docs.empty:
        return "unverifiable", "No candidate documents found in cache for this week/scope.", 0, 0

    reasons = evidence_docs["doc_relevance_reason"].fillna("")
    title_direct = reasons.str.startswith("title:")
    context_direct = reasons.str.startswith("context:")
    title_count = int(title_direct.sum())
    context_count = int(context_direct.sum())

    title_reasons = reasons.loc[title_direct].tolist()
    has_maritime = any("maritime_logistics" in reason for reason in title_reasons)
    has_trade = any("trade_policy" in reason for reason in title_reasons)
    has_energy = any("energy_transport" in reason for reason in title_reasons)
    has_manufacturing = any("manufacturing_electronics" in reason for reason in title_reasons)

    if title_count >= 3 and (has_trade or has_maritime or has_energy or has_manufacturing):
        channels = []
        if has_trade:
            channels.append("trade_policy")
        if has_maritime:
            channels.append("maritime_logistics")
        if has_energy:
            channels.append("energy_transport")
        if has_manufacturing:
            channels.append("manufacturing_electronics")
        return (
            "direct_supply_chain",
            f"{title_count} title-level direct docs across {', '.join(channels)}.",
            title_count,
            context_count,
        )

    if title_count >= 1 or context_count >= 2:
        return (
            "indirect_macro_or_disaster",
            f"{title_count} title-level and {context_count} context-level direct docs; evidence is plausible but less specific.",
            title_count,
            context_count,
        )

    medium = int(evidence_docs["doc_relevance_label"].eq("medium").sum())
    return (
        "weak_or_noisy",
        f"No title-level direct docs; medium generic disruption docs={medium}.",
        title_count,
        context_count,
    )


def build_sanity_check() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["week"])
    dataset = pd.read_csv(DATASET, parse_dates=["week"])
    total_weights = pd.read_csv(TOTAL_WEIGHTS)
    me_weights = pd.read_csv(ME_WEIGHTS)
    cases = select_cases(predictions)
    required_weeks = {pd.Timestamp(week).date().isoformat() for week in cases["week"]}
    docs = load_docs(required_weeks)

    dataset_features = [
        "ISO3",
        "week",
        "portcalls_container",
        "next_week_container",
        "abnormal_threshold",
        "external_very_negative_article_share",
        "network_very_negative_exposure",
        "me_network_strict_very_negative_exposure",
        "me_equal_strict_very_negative_exposure",
        "me_random_strict_very_negative_exposure",
    ]
    cases = cases.merge(dataset[dataset_features], on=["ISO3", "week"], how="left")

    rows = []
    for _, case in cases.iterrows():
        evidence_docs, scope, partner_iso3 = docs_for_case(case, docs, total_weights, me_weights)
        label, notes = case_label(evidence_docs)
        strict_label, strict_notes, title_direct_count, context_direct_count = strict_case_label(evidence_docs)
        rows.append(
            {
                "week": pd.Timestamp(case["week"]).date().isoformat(),
                "target_iso3": case["ISO3"],
                "target_country": case["country"],
                "feature_group": case["feature_group"],
                "model": case["model"],
                "predicted_probability": case["predicted_probability"],
                "selected_threshold": case["selected_threshold"],
                "prediction": int(case["prediction"]),
                "actual_abnormal_next_week": int(case["abnormal_next_week_container"]),
                "portcalls_container": case["portcalls_container"],
                "next_week_container": case["next_week_container"],
                "abnormal_threshold": case["abnormal_threshold"],
                "exposure_scope": scope,
                "evidence_partner_iso3": partner_iso3,
                "external_very_negative_article_share": case.get("external_very_negative_article_share", np.nan),
                "network_very_negative_exposure": case.get("network_very_negative_exposure", np.nan),
                "me_network_strict_very_negative_exposure": case.get(
                    "me_network_strict_very_negative_exposure", np.nan
                ),
                "me_equal_strict_very_negative_exposure": case.get(
                    "me_equal_strict_very_negative_exposure", np.nan
                ),
                "me_random_strict_very_negative_exposure": case.get(
                    "me_random_strict_very_negative_exposure", np.nan
                ),
                "metadata_relevance_label": label,
                "metadata_relevance_notes": notes,
                "strict_relevance_label": strict_label,
                "strict_relevance_notes": strict_notes,
                "title_direct_doc_count": title_direct_count,
                "context_direct_doc_count": context_direct_count,
                "top_themes": top_theme_summary(evidence_docs),
                "top_evidence_docs": " || ".join(short_doc(row) for _, row in evidence_docs.head(4).iterrows()),
                "evidence_doc_count_reviewed": len(evidence_docs),
                "high_relevance_doc_count": int(evidence_docs["doc_relevance_label"].eq("high").sum())
                if len(evidence_docs)
                else 0,
                "medium_relevance_doc_count": int(evidence_docs["doc_relevance_label"].eq("medium").sum())
                if len(evidence_docs)
                else 0,
            }
        )

    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> None:
    label_summary = (
        results.groupby(["feature_group", "metadata_relevance_label"], as_index=False)
        .size()
        .rename(columns={"size": "cases"})
        .sort_values(["feature_group", "metadata_relevance_label"])
    )
    outcome_summary = (
        results.groupby(["feature_group", "metadata_relevance_label"], as_index=False)
        .agg(
            cases=("week", "size"),
            actual_positive_cases=("actual_abnormal_next_week", "sum"),
            mean_probability=("predicted_probability", "mean"),
        )
        .sort_values(["feature_group", "metadata_relevance_label"])
    )
    strict_summary = (
        results.groupby(["feature_group", "strict_relevance_label"], as_index=False)
        .agg(
            cases=("week", "size"),
            actual_positive_cases=("actual_abnormal_next_week", "sum"),
            mean_probability=("predicted_probability", "mean"),
        )
        .sort_values(["feature_group", "strict_relevance_label"])
    )

    selected_cols = [
        "week",
        "target_iso3",
        "feature_group",
        "predicted_probability",
        "actual_abnormal_next_week",
        "metadata_relevance_label",
        "strict_relevance_label",
        "evidence_partner_iso3",
        "strict_relevance_notes",
        "top_themes",
    ]

    content = f"""# GDELT High-Risk Week Sanity Check

## Purpose

This report audits high-risk 2025 country-week predictions from the current panel benchmark to test whether the GDELT event layer is substantively meaningful or dominated by generic media noise.

## Method

- Candidate cases: top Random Forest predicted-risk country-weeks from `M2_own_country_news`, `M3_external_unweighted_events`, `M4_total_import_network`, and `M5_me_strict_network`.
- Evidence source: local GDELT candidate-document cache `data/interim/gkg_nlp_candidate_docs_2023-01-01_2025-12-31.csv`.
- Evidence fields: URL slug, source domain, GKG themes, names, organizations, and tone.
- Relevance label: metadata-based review, not full-text human gold labeling.

## Label Summary

{label_summary.to_markdown(index=False)}

## Outcome Summary

{outcome_summary.to_markdown(index=False)}

## Strict Relevance Summary

{strict_summary.to_markdown(index=False)}

## Reviewed Cases

{results[selected_cols].to_markdown(index=False)}

## Interpretation

This sanity check is useful for Gate 5 but should not be overclaimed. A `highly_relevant` metadata label means the cached metadata contains direct supply-chain, trade, maritime, energy, manufacturing, or logistics evidence with negative/disruption context. The stricter `strict_relevance_label` separates title-level direct supply-chain evidence from indirect macro, disaster, or geopolitical relevance. It does not prove causality, and it does not prove the article content was fully relevant because the current cache does not include article bodies.

If high-risk cases are mostly `direct_supply_chain` or `indirect_macro_or_disaster`, the event layer has face validity. If many cases are `weak_or_noisy`, the event layer should be treated as noisy and the paper should emphasize benchmark diagnostics rather than strong event-informed prediction claims.
"""
    REPORT.write_text(content, encoding="utf-8")


def run() -> None:
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results = build_sanity_check()
    results.to_csv(OUTPUT_TABLE, index=False)
    write_report(results)
    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved report: {REPORT}")
    print(results[["feature_group", "metadata_relevance_label"]].value_counts().to_string())


if __name__ == "__main__":
    run()
