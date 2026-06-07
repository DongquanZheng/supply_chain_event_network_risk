from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_nlp_candidate_docs_2023-01-01_2025-12-31.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "gkg_nlp_taxonomy_partner_week_2023-01-01_2025-12-31.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "nlp_taxonomy_summary.md"
RANDOM_SEED = 42


PATTERNS = {
    "maritime_logistics": re.compile(
        r"\b(?:port|ports|shipping|freight|cargo|container|vessel|ship|ships|"
        r"maritime|logistics|congestion|suez|panama|red sea|red-sea)\b",
        re.I,
    ),
    "machinery_electronics": re.compile(
        r"\b(?:semiconductor|semiconductors|chip|chips|electronics|electronic|"
        r"machinery|machine|machines|factory|factories|manufactur|"
        r"industrial equipment|supply chain)\b",
        re.I,
    ),
    "energy_fuel": re.compile(
        r"\b(?:oil|gas|fuel|energy|lng|crude|petroleum|tanker|refinery|"
        r"electricity|power)\b",
        re.I,
    ),
    "trade_policy": re.compile(
        r"\b(?:tariff|tariffs|sanction|sanctions|export control|export ban|"
        r"import ban|customs|trade war|trade restriction|embargo)\b",
        re.I,
    ),
    "disruption": re.compile(
        r"\b(?:disruption|disruptions|delay|delays|shortage|shortages|"
        r"congestion|strike|strikes|attack|attacks|blocked|blockage|"
        r"reroute|rerouting|shutdown|halt|crisis|conflict|war|earthquake|"
        r"flood|storm|typhoon|fire|explosion)\b",
        re.I,
    ),
}


CATEGORIES = {
    "maritime_disruption": ("maritime_logistics", "disruption"),
    "machinery_electronics_disruption": ("machinery_electronics", "disruption"),
    "energy_disruption": ("energy_fuel", "disruption"),
    "trade_policy_disruption": ("trade_policy", "disruption"),
    "broad_supply_disruption": ("disruption",),
}


def slug_text(url: str) -> str:
    parsed = urlparse(str(url))
    text = " ".join([parsed.netloc, parsed.path, parsed.query])
    text = unquote(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def build_document_text(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["DocumentIdentifier"].fillna("").map(slug_text),
        df["V2Themes"].fillna("").str.replace(r"[_;]", " ", regex=True),
        df["AllNames"].fillna("").str.slice(0, 500).str.replace(r"[;,#]", " ", regex=True),
        df["V2Organizations"].fillna("").str.slice(0, 500).str.replace(r"[;,#]", " ", regex=True),
    ]
    return (parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3]).str.lower()


def weak_labels(text: pd.Series) -> pd.DataFrame:
    flags = {
        name: text.str.contains(pattern, regex=True).astype(int)
        for name, pattern in PATTERNS.items()
    }
    labels = {}
    for category, required in CATEGORIES.items():
        value = np.ones(len(text), dtype=bool)
        for item in required:
            value &= flags[item].astype(bool).to_numpy()
        labels[category] = value.astype(int)
    return pd.DataFrame(labels)


def fit_predict_probabilities(df: pd.DataFrame, text: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    train_mask = pd.to_datetime(df["event_week"]).dt.year.eq(2023)
    output = pd.DataFrame(index=df.index)

    for category in CATEGORIES:
        y_train = labels.loc[train_mask, category]
        if y_train.nunique() < 2 or y_train.sum() < 20:
            output[f"nlp_{category}_prob"] = labels[category].astype(float)
            continue

        model = Pipeline(
            steps=[
                (
                    "hashing",
                    HashingVectorizer(
                        n_features=2**16,
                        ngram_range=(1, 2),
                        stop_words="english",
                        alternate_sign=False,
                        norm="l2",
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=50,
                        tol=1e-3,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        model.fit(text.loc[train_mask], y_train)
        output[f"nlp_{category}_prob"] = model.predict_proba(text)[:, 1]

    return output


def aggregate_partner_week(df: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    scored = pd.concat([df[["event_week", "code", "tone"]].copy(), probs], axis=1)
    scored["negative_severity"] = (-scored["tone"]).clip(lower=0, upper=10) / 10

    named_agg = {
        "article_count": ("tone", "size"),
        "nlp_candidate_avg_tone": ("tone", "mean"),
        "nlp_candidate_negative_severity": ("negative_severity", "mean"),
    }
    for col in probs.columns:
        scored[f"{col}_negative_score"] = scored[col] * scored["negative_severity"]
        named_agg[col] = (col, "mean")
        named_agg[f"{col}_negative_score"] = (f"{col}_negative_score", "mean")

    grouped = (
        scored.groupby(["event_week", "code"], as_index=False)
        .agg(**named_agg)
        .sort_values(["event_week", "code"])
    )
    return grouped


def write_report(
    path: Path,
    docs: pd.DataFrame,
    labels: pd.DataFrame,
    partner_week: pd.DataFrame,
) -> None:
    label_summary = labels.mean().sort_values(ascending=False).rename("weak_label_rate")
    content = f"""# NLP Taxonomy Feature Summary

## Input

- Candidate documents: {len(docs):,}
- Week range: {docs["event_week"].min()} to {docs["event_week"].max()}
- Partner/location codes: {docs["code"].nunique()}

## Method

GDELT GKG does not provide full article text in this table. This prototype builds document text from URL slugs, GKG themes, names, and organizations. Regex rules create weak labels, then TF-IDF + logistic regression learns taxonomy classifiers on 2023 documents and predicts probabilities for all weeks.

This is a stronger NLP layer than raw tone/theme counts, but it is still weakly supervised and should be described as a reproducible event-taxonomy proxy, not as manually validated article classification.

## Weak Label Rates

{label_summary.to_markdown()}

## Output

- Partner-week rows: {len(partner_week):,}
- Output file: `data/interim/gkg_nlp_taxonomy_partner_week_2023-01-01_2025-12-31.csv`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    docs = pd.read_csv(args.input, parse_dates=["event_week"])
    docs = docs.drop_duplicates(["event_week", "code", "DocumentIdentifier"]).reset_index(drop=True)
    docs = (
        docs.sort_values(["event_week", "code", "DocumentIdentifier"])
        .groupby(["event_week", "code"], as_index=False)
        .head(args.max_docs_per_code_week)
        .reset_index(drop=True)
    )

    text = build_document_text(docs)
    labels = weak_labels(text)
    probs = fit_predict_probabilities(docs, text, labels)
    partner_week = aggregate_partner_week(docs, probs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partner_week.to_csv(output, index=False)
    write_report(Path(args.report), docs, labels, partner_week)

    print(f"Docs: {len(docs)}")
    print(f"Partner-week rows: {len(partner_week)}")
    print(f"Saved: {output}")
    print(f"Report: {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--max-docs-per-code-week", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
