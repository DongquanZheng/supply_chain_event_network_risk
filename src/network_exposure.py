from __future__ import annotations

import pandas as pd


def attach_iso3_codes(
    event_features: pd.DataFrame,
    gdelt_to_iso3: dict[str, str],
) -> pd.DataFrame:
    out = event_features.copy()
    out["iso3"] = out["code"].map(gdelt_to_iso3)
    return out


def compute_network_exposure(
    event_features: pd.DataFrame,
    dependency_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_with_weights = (
        event_features.merge(
            dependency_weights[["partner_iso3", "import_dependency_share"]],
            left_on="iso3",
            right_on="partner_iso3",
            how="inner",
        )
        .copy()
    )

    event_with_weights["weighted_negative_exposure"] = (
        event_with_weights["negative_article_share"]
        * event_with_weights["import_dependency_share"]
    )
    event_with_weights["weighted_very_negative_exposure"] = (
        event_with_weights["very_negative_article_share"]
        * event_with_weights["import_dependency_share"]
    )
    event_with_weights["weighted_risk_theme_exposure"] = (
        event_with_weights["risk_theme_count"]
        * event_with_weights["import_dependency_share"]
    )
    event_with_weights["weighted_trade_transport_exposure"] = (
        event_with_weights["trade_transport_count"]
        * event_with_weights["import_dependency_share"]
    )

    network_exposure = (
        event_with_weights.groupby("event_week", as_index=False)
        .agg(
            network_negative_exposure=("weighted_negative_exposure", "sum"),
            network_very_negative_exposure=("weighted_very_negative_exposure", "sum"),
            network_risk_theme_exposure=("weighted_risk_theme_exposure", "sum"),
            network_trade_transport_exposure=("weighted_trade_transport_exposure", "sum"),
            partner_article_count=("article_count", "sum"),
        )
        .sort_values("event_week")
        .reset_index(drop=True)
    )

    return event_with_weights, network_exposure


def merge_exposure_with_operational_base(
    operational_df: pd.DataFrame,
    network_exposure: pd.DataFrame,
    how: str = "left",
) -> pd.DataFrame:
    op = operational_df.copy()
    exposure = network_exposure.copy()
    op["week"] = pd.to_datetime(op["week"])
    exposure["event_week"] = pd.to_datetime(exposure["event_week"])

    return op.merge(
        exposure,
        left_on="week",
        right_on="event_week",
        how=how,
    )

