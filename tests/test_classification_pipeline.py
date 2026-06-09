from __future__ import annotations

import unittest

import pandas as pd

from src.data.splitting import assert_patient_split_integrity, patient_level_split
from src.data.target import TARGET_COLUMN, add_los_category
from src.data.validation import (
    assert_events_within_first_24h,
    assert_matching_feature_columns,
    assert_no_leakage_columns,
    filter_first_24h_events,
)
from src.features.build_features import infer_feature_types, make_sample_dataset
from src.models.pipeline import build_classifier, fit_with_balanced_weights


class ClassificationPipelineTests(unittest.TestCase):
    def test_los_category_boundaries(self) -> None:
        df = add_los_category(pd.DataFrame({"los": [1.99, 2.0, 7.0, 7.01]}))
        self.assertEqual(df[TARGET_COLUMN].tolist(), [0, 1, 1, 2])

    def test_patient_level_split_integrity(self) -> None:
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "stay_id": range(10),
            }
        )
        split = patient_level_split(df, test_size=0.2, val_size=0.2, random_state=7)
        self.assertEqual(set(split.unique()), {"train", "val", "test"})
        assert_patient_split_integrity(df.assign(split=split))

    def test_patient_level_split_rejects_overlap(self) -> None:
        df = pd.DataFrame({"subject_id": [1, 1, 2], "split": ["train", "test", "train"]})
        with self.assertRaises(AssertionError):
            assert_patient_split_integrity(df)

    def test_no_leakage_columns_in_features(self) -> None:
        assert_no_leakage_columns(["anchor_age", "heart_rate_mean_24h", "first_careunit"])
        with self.assertRaises(AssertionError):
            assert_no_leakage_columns(["anchor_age", "los", "last_careunit", "dischtime"])

    def test_first_24h_event_filter_and_assertion(self) -> None:
        stays = pd.DataFrame(
            {
                "stay_id": [10],
                "intime": [pd.Timestamp("2200-01-01 00:00:00")],
            }
        )
        events = pd.DataFrame(
            {
                "stay_id": [10, 10, 10],
                "charttime": [
                    pd.Timestamp("2200-01-01 00:30:00"),
                    pd.Timestamp("2200-01-02 00:00:00"),
                    pd.Timestamp("2200-01-02 00:01:00"),
                ],
                "valuenum": [1.0, 2.0, 3.0],
            }
        )
        filtered = filter_first_24h_events(events, stays, "charttime")
        self.assertEqual(filtered["valuenum"].tolist(), [1.0, 2.0])
        assert_events_within_first_24h(filtered, stays, "charttime")
        with self.assertRaises(AssertionError):
            assert_events_within_first_24h(events, stays, "charttime")

    def test_matching_train_test_feature_columns_after_preprocessing(self) -> None:
        df = make_sample_dataset()
        feature_cols = [
            col
            for col in df.columns
            if col not in {"subject_id", "hadm_id", "stay_id", "intime", TARGET_COLUMN}
        ]
        train = df.iloc[:45].copy()
        test = df.iloc[45:].copy()
        numeric_cols, categorical_cols = infer_feature_types(train, feature_cols)
        model = build_classifier(numeric_cols, categorical_cols)
        fit_with_balanced_weights(model, train[feature_cols], train[TARGET_COLUMN])

        train_transformed = model.named_steps["preprocessor"].transform(train[feature_cols])
        test_transformed = model.named_steps["preprocessor"].transform(test[feature_cols])
        self.assertEqual(train_transformed.shape[1], test_transformed.shape[1])
        assert_matching_feature_columns(feature_cols, list(test[feature_cols].columns))


if __name__ == "__main__":
    unittest.main()
