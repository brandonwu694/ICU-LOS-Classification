from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.splitting import assert_patient_split_integrity, patient_level_split
from src.data.target import TARGET_COLUMN, add_los_category
from src.data.validation import (
    assert_events_within_first_24h,
    assert_matching_feature_columns,
    assert_no_leakage_columns,
    filter_first_24h_events,
)
from src.features.build_features import (
    build_inputevents_features,
    build_modeling_frame,
    build_outputevents_features,
    build_prescriptions_features,
    build_procedureevents_features,
    build_radiology_features,
    infer_feature_types,
    make_sample_dataset,
)
from src.models.pipeline import build_classifier, fit_with_balanced_weights
from src.models.pipeline import (
    build_dummy_baseline,
    build_logistic_regression_baseline,
    build_random_forest_baseline,
    tune_hist_gradient_boosting_classifier,
)


class ClassificationPipelineTests(unittest.TestCase):
    """Sanity checks for the ICU LOS classification workflow."""

    def test_los_category_boundaries(self) -> None:
        """LOS values are mapped to the expected three target classes."""
        df = add_los_category(pd.DataFrame({"los": [1.99, 2.0, 7.0, 7.01]}))
        self.assertEqual(df[TARGET_COLUMN].tolist(), [0, 1, 1, 2])

    def test_patient_level_split_integrity(self) -> None:
        """Patient-level splitting keeps each subject in only one split."""
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
        """The split checker raises an error when a patient appears twice."""
        df = pd.DataFrame({"subject_id": [1, 1, 2], "split": ["train", "test", "train"]})
        with self.assertRaises(AssertionError):
            assert_patient_split_integrity(df)

    def test_no_leakage_columns_in_features(self) -> None:
        """Known outcome and future-information columns are rejected."""
        assert_no_leakage_columns(["anchor_age", "heart_rate_mean_24h", "first_careunit"])
        with self.assertRaises(AssertionError):
            assert_no_leakage_columns(["anchor_age", "los", "last_careunit", "dischtime"])

    def test_first_24h_event_filter_and_assertion(self) -> None:
        """Timestamped events are limited to ICU hour 0 through hour 24."""
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
        """Train and test rows keep the same feature schema after preprocessing."""
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

    def test_baseline_models_fit_on_same_feature_matrix(self) -> None:
        """Baseline models fit and predict on the same feature matrix."""
        df = make_sample_dataset()
        feature_cols = [
            col
            for col in df.columns
            if col not in {"subject_id", "hadm_id", "stay_id", "intime", TARGET_COLUMN}
        ]
        train = df.iloc[:45].copy()
        test = df.iloc[45:].copy()
        numeric_cols, categorical_cols = infer_feature_types(train, feature_cols)

        logistic = build_logistic_regression_baseline(numeric_cols, categorical_cols)
        logistic.fit(train[feature_cols], train[TARGET_COLUMN])
        self.assertEqual(len(logistic.predict(test[feature_cols])), len(test))

        dummy = build_dummy_baseline()
        dummy.fit(train[feature_cols], train[TARGET_COLUMN])
        self.assertEqual(len(dummy.predict(test[feature_cols])), len(test))

        random_forest = build_random_forest_baseline(numeric_cols, categorical_cols)
        random_forest.fit(train[feature_cols], train[TARGET_COLUMN])
        self.assertEqual(len(random_forest.predict(test[feature_cols])), len(test))

    def test_group_aware_hgb_tuning_returns_fitted_pipeline(self) -> None:
        """HistGradientBoosting tuning returns a fitted patient-aware pipeline."""
        df = make_sample_dataset()
        feature_cols = [
            col
            for col in df.columns
            if col not in {"subject_id", "hadm_id", "stay_id", "intime", TARGET_COLUMN}
        ]
        train = df.iloc[:45].copy()
        numeric_cols, categorical_cols = infer_feature_types(train, feature_cols)

        tuned_model, tuning_results, best_params = tune_hist_gradient_boosting_classifier(
            numeric_cols,
            categorical_cols,
            train[feature_cols],
            train[TARGET_COLUMN],
            groups=train["subject_id"],
            n_iter=2,
            cv_splits=2,
            scoring="f1_macro",
            random_state=7,
        )

        self.assertEqual(
            type(tuned_model.named_steps["model"]).__name__,
            "HistGradientBoostingClassifier",
        )
        self.assertEqual(len(tuning_results), 2)
        self.assertTrue(best_params)

    def test_unused_raw_sources_build_first24h_stay_level_features(self) -> None:
        """Added raw source builders return one first-24-hour row per stay."""
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            base = _write_minimal_raw_fixture(raw_dir)

            builders = [
                build_inputevents_features,
                build_outputevents_features,
                build_procedureevents_features,
                build_prescriptions_features,
                build_radiology_features,
            ]
            for builder in builders:
                features = builder(raw_dir, base)
                self.assertEqual(len(features), base["stay_id"].nunique())
                self.assertFalse(features["stay_id"].duplicated().any())
                assert_no_leakage_columns([col for col in features.columns if col != "stay_id"])

            input_features = build_inputevents_features(raw_dir, base).set_index("stay_id")
            self.assertEqual(input_features.loc[10, "input_event_count_24h"], 1)
            self.assertEqual(input_features.loc[10, "input_total_volume_ml_24h"], 100)

            output_features = build_outputevents_features(raw_dir, base).set_index("stay_id")
            self.assertEqual(output_features.loc[10, "output_event_count_24h"], 1)
            self.assertEqual(output_features.loc[10, "output_total_volume_ml_24h"], 250)
            self.assertEqual(output_features.loc[10, "output_urine_volume_ml_24h"], 250)

            procedure_features = build_procedureevents_features(raw_dir, base).set_index("stay_id")
            self.assertEqual(procedure_features.loc[10, "procedure_event_count_24h"], 1)

            prescription_features = build_prescriptions_features(raw_dir, base).set_index("stay_id")
            self.assertEqual(prescription_features.loc[10, "prescription_order_count_24h"], 1)
            self.assertEqual(prescription_features.loc[10, "prescription_category_antibiotic_count_24h"], 1)

            radiology_features = build_radiology_features(raw_dir, base).set_index("stay_id")
            self.assertEqual(radiology_features.loc[10, "radiology_note_count_24h"], 1)
            self.assertEqual(radiology_features.loc[10, "radiology_mentions_line_or_tube_24h"], 1)

    def test_raw_rebuild_with_added_sources_remains_model_ready(self) -> None:
        """Raw feature rebuilds include added sources and stay model-ready."""
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            _write_minimal_raw_fixture(raw_dir)
            df, feature_cols = build_modeling_frame(raw_dir, processed_dir=None)
            assert_no_leakage_columns(feature_cols)

            source_prefixes = ["input_", "output_", "procedure_", "prescription_", "radiology_"]
            for prefix in source_prefixes:
                self.assertTrue(any(col.startswith(prefix) for col in feature_cols), prefix)

            df["split"] = patient_level_split(df, test_size=0.25, val_size=0.25, random_state=11)
            assert_patient_split_integrity(df)
            train = df[df["split"].eq("train")]
            test = df[df["split"].eq("test")]
            assert_matching_feature_columns(train[feature_cols].columns, test[feature_cols].columns)


if __name__ == "__main__":
    unittest.main()


def _write_minimal_raw_fixture(raw_dir: Path) -> pd.DataFrame:
    """Write a small raw-data fixture with in-window and out-of-window rows."""
    icu = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [100, 200, 300, 400],
            "stay_id": [10, 20, 30, 40],
            "first_careunit": ["MICU", "SICU", "CCU", "MICU"],
            "last_careunit": ["MICU", "SICU", "CCU", "MICU"],
            "intime": pd.to_datetime(
                [
                    "2200-01-01 00:00:00",
                    "2200-01-02 00:00:00",
                    "2200-01-03 00:00:00",
                    "2200-01-04 00:00:00",
                ]
            ),
            "outtime": pd.to_datetime(
                [
                    "2200-01-02 12:00:00",
                    "2200-01-05 00:00:00",
                    "2200-01-11 00:00:00",
                    "2200-01-05 12:00:00",
                ]
            ),
            "los": [1.5, 3.0, 8.0, 1.7],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [100, 200, 300, 400],
            "admittime": pd.to_datetime(
                [
                    "2199-12-31 20:00:00",
                    "2200-01-01 20:00:00",
                    "2200-01-02 20:00:00",
                    "2200-01-03 20:00:00",
                ]
            ),
            "admission_type": ["EW EMER.", "URGENT", "DIRECT EMER.", "OBSERVATION ADMIT"],
            "admission_location": ["ER", "TRANSFER", "CLINIC", "ER"],
            "insurance": ["Medicare", "Medicaid", "Other", "Medicare"],
            "language": ["English", "English", "Spanish", "English"],
            "marital_status": ["SINGLE", "MARRIED", "WIDOWED", "SINGLE"],
            "race": ["WHITE", "BLACK", "HISPANIC", "ASIAN"],
        }
    )
    patients = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "gender": ["F", "M", "F", "M"],
            "anchor_age": [50, 60, 70, 45],
        }
    )
    d_items = pd.DataFrame(
        {
            "itemid": [111, 222, 333],
            "label": ["Normal Saline", "Foley Urine", "Central Line"],
            "abbreviation": ["NS", "Urine", "Line"],
            "linksto": ["inputevents", "outputevents", "procedureevents"],
            "category": ["Fluids", "Urine", "Procedures"],
            "unitname": ["mL", "mL", ""],
            "param_type": ["Numeric", "Numeric", "Process"],
            "lownormalvalue": [None, None, None],
            "highnormalvalue": [None, None, None],
        }
    )
    inputevents = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "stay_id": [10, 10],
            "starttime": pd.to_datetime(["2200-01-01 01:00:00", "2200-01-02 01:01:00"]),
            "itemid": [111, 111],
            "amount": [100, 999],
            "amountuom": ["mL", "mL"],
            "ordercategoryname": ["04-Fluids", "04-Fluids"],
        }
    )
    outputevents = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "stay_id": [10, 10],
            "charttime": pd.to_datetime(["2200-01-01 02:00:00", "2200-01-02 02:00:00"]),
            "itemid": [222, 222],
            "value": [250, 999],
            "valueuom": ["mL", "mL"],
        }
    )
    procedureevents = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "stay_id": [10, 10],
            "starttime": pd.to_datetime(["2200-01-01 03:00:00", "2200-01-02 03:00:00"]),
            "itemid": [333, 333],
            "ordercategoryname": ["Procedures", "Procedures"],
        }
    )
    prescriptions = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "starttime": pd.to_datetime(["2200-01-01 04:00:00", "2200-01-02 04:00:00"]),
            "drug_type": ["MAIN", "MAIN"],
            "drug": ["Vancomycin", "Furosemide"],
            "route": ["IV", "IV"],
            "dose_val_rx": ["1", "1"],
        }
    )
    radiology = pd.DataFrame(
        {
            "note_id": ["n1", "n2"],
            "subject_id": [1, 1],
            "hadm_id": [100, 100],
            "note_type": ["RR", "RR"],
            "charttime": pd.to_datetime(["2200-01-01 05:00:00", "2200-01-02 05:00:00"]),
            "text": [
                "CHEST radiograph. IMPRESSION: line and tube. Pneumonia opacity.",
                "CT abdomen after first day.",
            ],
        }
    )

    tables = {
        "icustays": icu,
        "admissions": admissions,
        "patients": patients,
        "d_items": d_items,
        "inputevents": inputevents,
        "outputevents": outputevents,
        "procedureevents": procedureevents,
        "prescriptions": prescriptions,
        "radiology": radiology,
    }
    for name, table in tables.items():
        table.to_csv(raw_dir / f"{name}.csv", index=False)

    return icu[["subject_id", "hadm_id", "stay_id", "intime"]]
