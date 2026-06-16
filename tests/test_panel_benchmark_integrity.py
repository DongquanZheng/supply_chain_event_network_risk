from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    FOLDS,
    TARGET,
    add_country_dummies,
    evaluate_predictions,
    make_feature_groups,
    precision_at_k,
    select_threshold,
    split_fold,
)
from src.network_exposure import attach_iso3_codes, compute_network_exposure  # noqa: E402


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "tiny_panel_benchmark.csv"


class PanelBenchmarkIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = pd.read_csv(FIXTURE, parse_dates=["week"]).sort_values(["week", "ISO3"]).reset_index(drop=True)

    def test_fixture_dataset_has_required_columns(self) -> None:
        required = {
            "ISO3",
            "week",
            "portcalls_container",
            TARGET,
            "external_very_negative_article_share",
            "network_very_negative_exposure",
            "me_network_strict_very_negative_exposure",
            "me_equal_strict_very_negative_exposure",
            "me_shuffled_strict_very_negative_exposure",
            "me_random_strict_very_negative_exposure",
        }
        self.assertTrue(required.issubset(set(self.dataset.columns)))

    def test_network_exposure_feature_construction(self) -> None:
        event_features = pd.DataFrame(
            {
                "event_week": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "code": ["AA", "BB"],
                "negative_article_share": [0.20, 0.50],
                "very_negative_article_share": [0.10, 0.30],
                "risk_theme_count": [2.0, 5.0],
                "trade_transport_count": [3.0, 7.0],
                "article_count": [10, 20],
            }
        )
        weights = pd.DataFrame(
            {
                "partner_iso3": ["AAA", "BBB"],
                "import_dependency_share": [0.25, 0.75],
            }
        )

        mapped = attach_iso3_codes(event_features, {"AA": "AAA", "BB": "BBB"})
        _, exposure = compute_network_exposure(mapped, weights)

        self.assertAlmostEqual(exposure.loc[0, "network_negative_exposure"], 0.425)
        self.assertAlmostEqual(exposure.loc[0, "network_very_negative_exposure"], 0.25)
        self.assertAlmostEqual(exposure.loc[0, "network_risk_theme_exposure"], 4.25)
        self.assertAlmostEqual(exposure.loc[0, "network_trade_transport_exposure"], 6.0)

    def test_temporal_folds_are_chronological(self) -> None:
        for fold in FOLDS:
            train, validation, test = split_fold(self.dataset, fold)
            self.assertLess(train["week"].max(), validation["week"].min())
            self.assertLess(validation["week"].max(), test["week"].min())
            self.assertGreater(train[TARGET].sum(), 0)
            self.assertGreater(validation[TARGET].sum(), 0)
            self.assertGreater(test[TARGET].sum(), 0)

    def test_feature_groups_do_not_include_future_or_target_columns(self) -> None:
        train, validation, test = split_fold(self.dataset, FOLDS[-1])
        [train, validation, test], country_features = add_country_dummies(train, validation, test)
        feature_groups = make_feature_groups(country_features)

        forbidden = {
            TARGET,
            "next_week_container",
            "abnormal_threshold",
        }
        for features in feature_groups.values():
            self.assertTrue(forbidden.isdisjoint(features))
            self.assertTrue(set(features).issubset(set(train.columns)))

    def test_metrics_and_threshold_logic_runs_on_fixture_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        proba = np.array([0.10, 0.80, 0.30, 0.70])

        threshold, validation_f1 = select_threshold(pd.Series(y_true), proba)
        metrics = evaluate_predictions(y_true, proba, threshold)

        self.assertGreater(validation_f1, 0)
        self.assertGreater(metrics["pr_auc"], 0.9)
        self.assertEqual(precision_at_k(y_true, proba, 2), 1.0)

    def test_placebo_columns_are_deterministic_in_fixture(self) -> None:
        first = pd.read_csv(FIXTURE)
        second = pd.read_csv(FIXTURE)
        pd.testing.assert_series_equal(
            first["me_random_strict_very_negative_exposure"],
            second["me_random_strict_very_negative_exposure"],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
