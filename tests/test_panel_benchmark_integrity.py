from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_panel_benchmark_models import (  # noqa: E402
    FOLDS,
    TARGET,
    add_country_dummies,
    load_dataset,
    make_feature_groups,
    split_fold,
)


class PanelBenchmarkIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_processed_dataset_has_required_columns(self) -> None:
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

    def test_random_placebo_columns_are_deterministic_in_cached_dataset(self) -> None:
        first = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv")
        second = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "multicountry_container_event_network_benchmark.csv")
        pd.testing.assert_series_equal(
            first["me_random_strict_very_negative_exposure"],
            second["me_random_strict_very_negative_exposure"],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
