"""Compare weak structured GDELT, TF-IDF, and BERT NLP layers on expanded32.

This script keeps the PortWatch panel, WITS weights, temporal folds, labels, and
model families fixed. Only the GDELT/NLP representation changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import sys
from urllib.parse import unquote, urlparse
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    EXTERNAL_UNWEIGHTED_FEATURES,
    FOLDS,
    OPERATIONAL_FEATURES,
    OWN_NEWS_FEATURES,
    TARGET,
    TOTAL_NETWORK_FEATURES,
    add_country_dummies,
    evaluate_predictions,
    select_threshold,
    split_fold,
)
from src.panel32_config import PANEL32_GDELT_TO_ISO3  # noqa: E402


DATASET = PROJECT_ROOT / "data" / "processed" / "multicountry32_container_event_network_benchmark.csv"
DOCS = PROJECT_ROOT / "data" / "interim" / "gkg_nlp_candidate_docs_2021-01-01_2025-12-31_expanded32.csv"
WEIGHTS = PROJECT_ROOT / "data" / "interim" / "panel32_total_dependency_weights_2023.csv"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT = PROJECT_ROOT / "reports" / "panel32_gdelt_nlp_representation_benchmark.md"
METRICS = TABLE_DIR / "panel32_gdelt_nlp_representation_metrics_by_fold.csv"
SUMMARY = TABLE_DIR / "panel32_gdelt_nlp_representation_summary.csv"
PREDICTIONS = TABLE_DIR / "panel32_gdelt_nlp_representation_predictions.csv"

RANDOM_SEED = 42


@dataclass(frozen=True)
class TextConfig:
    tfidf_max_features: int = 1000
    tfidf_svd_components: int = 64
    bert_model: str = "prajjwal1/bert-tiny"
    bert_batch_size: int = 32
    max_docs_per_country_week: int = 40


def slug_text(url: str) -> str:
    parsed = urlparse(str(url))
    text = " ".join([parsed.netloc, parsed.path, parsed.query])
    text = unquote(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_field(series: pd.Series, max_chars: int) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.slice(0, max_chars)
        .str.replace(r"[_;,#|/\\-]+", " ", regex=True)
        .str.lower()
    )


def load_panel() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)


def load_docs(max_docs_per_country_week: int) -> pd.DataFrame:
    if not DOCS.exists():
        raise FileNotFoundError(
            f"Missing expanded32 candidate docs: {DOCS}. "
            "Run scripts/fetch_panel32_gdelt_candidate_docs.py first."
        )
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
    docs = pd.read_csv(DOCS, usecols=usecols, parse_dates=["event_week"])
    docs["ISO3"] = docs["code"].map(PANEL32_GDELT_TO_ISO3)
    docs = docs.dropna(subset=["ISO3"]).copy()
    docs = docs.drop_duplicates(["event_week", "ISO3", "DocumentIdentifier"])
    docs = (
        docs.sort_values(["event_week", "ISO3", "DocumentIdentifier"])
        .groupby(["event_week", "ISO3"], as_index=False)
        .head(max_docs_per_country_week)
        .reset_index(drop=True)
    )
    docs["doc_text"] = (
        docs["DocumentIdentifier"].fillna("").map(slug_text)
        + " "
        + clean_field(docs["V2Themes"], 1600)
        + " "
        + clean_field(docs["AllNames"], 600)
        + " "
        + clean_field(docs["V2Organizations"], 600)
    ).str.strip()
    return docs


def build_weekly_documents(docs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        docs.groupby(["event_week", "ISO3"], as_index=False)
        .agg(
            text=("doc_text", lambda s: " ".join(x for x in s if x)),
            candidate_doc_count=("DocumentIdentifier", "nunique"),
            candidate_source_count=("SourceCommonName", "nunique"),
            candidate_avg_tone=("tone", "mean"),
        )
        .rename(columns={"event_week": "week"})
    )
    grid = panel[["week", "ISO3"]].drop_duplicates().copy()
    out = grid.merge(weekly, on=["week", "ISO3"], how="left")
    out["text"] = out["text"].fillna("")
    for col in ["candidate_doc_count", "candidate_source_count", "candidate_avg_tone"]:
        out[col] = out[col].fillna(0.0)
    return out.sort_values(["week", "ISO3"]).reset_index(drop=True)


def fit_tfidf_embeddings(train_docs: pd.DataFrame, all_docs: pd.DataFrame, config: TextConfig) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train_docs["text"].fillna(""))
    all_x = vectorizer.transform(all_docs["text"].fillna(""))
    n_features = all_x.shape[1]
    if n_features <= 2:
        dense = all_x.toarray()
        columns = [f"dim_{i:03d}" for i in range(dense.shape[1])]
    else:
        n_components = min(config.tfidf_svd_components, n_features - 1, max(2, train_x.shape[0] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        svd.fit(train_x)
        dense = svd.transform(all_x)
        columns = [f"dim_{i:03d}" for i in range(dense.shape[1])]
    emb = all_docs[["week", "ISO3"]].copy()
    for idx, col in enumerate(columns):
        emb[col] = dense[:, idx].astype(np.float32)
    return emb


def bert_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def compute_bert_embeddings(all_docs: pd.DataFrame, config: TextConfig) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "BERT mode requires torch and transformers. Install transformers or run with --skip-bert."
        ) from exc

    if "bert" in config.bert_model.lower():
        tokenizer = BertTokenizer.from_pretrained(config.bert_model)
        model = BertModel.from_pretrained(config.bert_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model, use_fast=False)
        model = AutoModel.from_pretrained(config.bert_model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = all_docs["text"].fillna("").tolist()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), config.bert_batch_size):
            batch = texts[start : start + config.bert_batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            result = model(**encoded)
            token_emb = result.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.cpu().numpy().astype(np.float32))

    dense = np.vstack(outputs)
    emb = all_docs[["week", "ISO3"]].copy()
    for idx in range(dense.shape[1]):
        emb[f"dim_{idx:03d}"] = dense[:, idx]
    return emb


def build_text_feature_frame(
    base_rows: pd.DataFrame,
    embeddings: pd.DataFrame,
    weights: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    countries = sorted(base_rows["ISO3"].unique())
    dim_cols = [col for col in embeddings.columns if col.startswith("dim_")]
    emb = embeddings.set_index(["week", "ISO3"])[dim_cols]
    weight_map = {
        target: group.set_index("partner_iso3")["import_dependency_share"].to_dict()
        for target, group in weights.groupby("ISO3")
    }

    records = []
    zero = np.zeros(len(dim_cols), dtype=np.float32)
    for row in base_rows[["week", "ISO3"]].itertuples(index=False):
        week = row.week
        target = row.ISO3
        own = emb.loc[(week, target)].to_numpy(dtype=np.float32) if (week, target) in emb.index else zero
        partner_vectors = []
        network = np.zeros(len(dim_cols), dtype=np.float32)
        for partner in countries:
            if partner == target:
                continue
            vec = emb.loc[(week, partner)].to_numpy(dtype=np.float32) if (week, partner) in emb.index else zero
            partner_vectors.append(vec)
            network += float(weight_map.get(target, {}).get(partner, 0.0)) * vec
        external = np.mean(partner_vectors, axis=0) if partner_vectors else zero
        rec = {"week": week, "ISO3": target}
        for idx, value in enumerate(own):
            rec[f"{prefix}_own_{idx:03d}"] = value
        for idx, value in enumerate(external):
            rec[f"{prefix}_external_{idx:03d}"] = value
        for idx, value in enumerate(network):
            rec[f"{prefix}_wits_{idx:03d}"] = value
        records.append(rec)
    return pd.DataFrame(records)


def make_models() -> dict[str, tuple[object, str]]:
    return {
        "logistic": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
                ]
            ),
            "plain",
        ),
        "sklearn_gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=12,
                subsample=0.85,
                random_state=RANDOM_SEED,
            ),
            "sample_weight",
        ),
    }


def fit_model(model, fit_mode: str, x: pd.DataFrame, y: pd.Series) -> None:
    if fit_mode == "sample_weight":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["nlp_version", "feature_group", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_precision_at_25=("precision_at_25", "mean"),
            total_tp=("tp", "sum"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            total_tn=("tn", "sum"),
        )
        .sort_values(["mean_pr_auc", "mean_precision_at_25"], ascending=False)
    )


def text_cols(frame: pd.DataFrame, prefix: str, part: str) -> list[str]:
    marker = f"{prefix}_{part}_"
    return [col for col in frame.columns if col.startswith(marker)]


def run_fold_for_version(
    fold,
    df: pd.DataFrame,
    feature_frame: pd.DataFrame | None,
    nlp_version: str,
) -> tuple[list[dict], list[dict]]:
    train, validation, test = split_fold(df, fold)
    [train, validation, test], country_features = add_country_dummies(train, validation, test)
    base_features = OPERATIONAL_FEATURES + country_features

    if feature_frame is not None:
        train = train.merge(feature_frame, on=["week", "ISO3"], how="left")
        validation = validation.merge(feature_frame, on=["week", "ISO3"], how="left")
        test = test.merge(feature_frame, on=["week", "ISO3"], how="left")
        text_prefix = "tfidf" if nlp_version == "tfidf" else "bert"
        own = text_cols(train, text_prefix, "own")
        external = text_cols(train, text_prefix, "external")
        wits = text_cols(train, text_prefix, "wits")
        groups = {
            f"{nlp_version}_own_external": base_features + own + external,
            f"{nlp_version}_own_external_wits": base_features + own + external + wits,
        }
    else:
        groups = {
            "weak_structured_own_external": base_features + OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES,
            "weak_structured_own_external_wits": (
                base_features + OWN_NEWS_FEATURES + EXTERNAL_UNWEIGHTED_FEATURES + TOTAL_NETWORK_FEATURES
            ),
        }

    metric_rows = []
    prediction_rows = []
    for group_name, raw_features in groups.items():
        features = [
            feature
            for feature in dict.fromkeys(raw_features)
            if feature in train.columns and pd.api.types.is_numeric_dtype(train[feature])
        ]
        for frame in [train, validation, test]:
            frame[features] = frame[features].replace([np.inf, -np.inf], 0).fillna(0)
        for model_name, (model, fit_mode) in make_models().items():
            fit_model(model, fit_mode, train[features], train[TARGET])
            val_proba = model.predict_proba(validation[features])[:, 1]
            threshold, val_f1 = select_threshold(validation[TARGET], val_proba)
            test_proba = model.predict_proba(test[features])[:, 1]
            scores = evaluate_predictions(test[TARGET].to_numpy(), test_proba, threshold)
            metric_rows.append(
                {
                    "fold": fold.name,
                    "nlp_version": nlp_version,
                    "feature_group": group_name,
                    "model": model_name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "selected_threshold": threshold,
                    "validation_f1": val_f1,
                    "n_features": len(features),
                    **scores,
                }
            )
            pred = test[["week", "ISO3", "country", TARGET]].copy()
            pred["fold"] = fold.name
            pred["nlp_version"] = nlp_version
            pred["feature_group"] = group_name
            pred["model"] = model_name
            pred["predicted_probability"] = test_proba
            prediction_rows.extend(pred.to_dict("records"))
    return metric_rows, prediction_rows


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    config = TextConfig(
        tfidf_max_features=args.tfidf_max_features,
        tfidf_svd_components=args.tfidf_svd_components,
        bert_model=args.bert_model,
        bert_batch_size=args.bert_batch_size,
        max_docs_per_country_week=args.max_docs_per_country_week,
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    docs = load_docs(config.max_docs_per_country_week)
    weekly_docs = build_weekly_documents(docs, panel)
    weights = pd.read_csv(WEIGHTS)
    weights = weights.loc[
        weights["ISO3"].isin(panel["ISO3"].unique()) & weights["partner_iso3"].isin(panel["ISO3"].unique())
    ].copy()

    bert_embeddings = None
    if not args.skip_bert:
        if not bert_available():
            raise RuntimeError("transformers is not installed. Install it or rerun with --skip-bert.")
        bert_embeddings = compute_bert_embeddings(weekly_docs, config)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in FOLDS:
        weak_metrics, weak_predictions = run_fold_for_version(fold, panel, None, "weak_structured")
        metric_rows.extend(weak_metrics)
        prediction_rows.extend(weak_predictions)

        train, _validation, _test = split_fold(panel, fold)
        train_keys = train[["week", "ISO3"]].drop_duplicates()
        train_docs = train_keys.merge(weekly_docs, on=["week", "ISO3"], how="left")
        tfidf_embeddings = fit_tfidf_embeddings(train_docs, weekly_docs, config)
        tfidf_features = build_text_feature_frame(panel[["week", "ISO3"]].drop_duplicates(), tfidf_embeddings, weights, "tfidf")
        tfidf_metrics, tfidf_predictions = run_fold_for_version(fold, panel, tfidf_features, "tfidf")
        metric_rows.extend(tfidf_metrics)
        prediction_rows.extend(tfidf_predictions)

        if bert_embeddings is not None:
            bert_features = build_text_feature_frame(panel[["week", "ISO3"]].drop_duplicates(), bert_embeddings, weights, "bert")
            bert_metrics, bert_predictions = run_fold_for_version(fold, panel, bert_features, "bert")
            metric_rows.extend(bert_metrics)
            prediction_rows.extend(bert_predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(metrics)
    metrics.to_csv(METRICS, index=False)
    predictions.to_csv(PREDICTIONS, index=False)
    summary.to_csv(SUMMARY, index=False)
    write_report(summary, metrics, panel, docs, args.skip_bert)
    print(f"Saved metrics: {METRICS}")
    print(f"Saved summary: {SUMMARY}")
    print(f"Saved predictions: {PREDICTIONS}")
    print(f"Saved report: {REPORT}")
    print(summary.head(20).to_string(index=False))


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, panel: pd.DataFrame, docs: pd.DataFrame, skip_bert: bool) -> None:
    primary = summary[summary["model"].eq("sklearn_gradient_boosting")].copy()
    content = f"""# Panel32 GDELT NLP Representation Benchmark

## Scope

- Panel: `data/processed/multicountry32_container_event_network_benchmark.csv`
- Countries: {panel["ISO3"].nunique()}
- Rows: {len(panel):,}
- Positive labels: {int(panel[TARGET].sum()):,}
- Positive rate: {panel[TARGET].mean():.3f}
- Candidate GDELT docs: {len(docs):,}
- Week range: {panel["week"].min().date()} to {panel["week"].max().date()}

## Design

This experiment keeps PortWatch features, WITS weights, target labels, temporal folds, and model families fixed. Only the GDELT/NLP representation changes.

- `weak_structured`: existing GDELT article/tone/theme count features.
- `tfidf`: fold-safe TF-IDF + TruncatedSVD text features from GDELT URL slug, themes, names, and organizations.
- `bert`: fixed pretrained BERT-family text embeddings from the same GDELT metadata text. BERT skipped: `{skip_bert}`.

TF-IDF vocabulary and SVD are fit only on each fold's training years, then applied to validation/test. Test labels are not used for feature fitting or threshold selection.

## Primary Gradient Boosting Summary

{primary[["nlp_version", "feature_group", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_precision_at_25", "total_tp", "total_fp", "total_fn"]].to_markdown(index=False)}

## All Results

{summary[["nlp_version", "feature_group", "model", "mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_precision_at_25", "n_features" if "n_features" in summary.columns else "folds"]].to_markdown(index=False)}

## Reading Guardrail

This is a representation comparison, not a new data-source experiment. GDELT GKG does not provide full article text here; the text layer uses metadata-derived pseudo-documents. Interpret any gains as evidence for richer GDELT text representation, not full-text semantic understanding.
"""
    REPORT.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf-max-features", type=int, default=1000)
    parser.add_argument("--tfidf-svd-components", type=int, default=64)
    parser.add_argument("--bert-model", default="prajjwal1/bert-tiny")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--max-docs-per-country-week", type=int, default=40)
    parser.add_argument("--skip-bert", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
