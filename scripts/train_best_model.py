from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd

from config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.splitting import assert_patient_split_integrity, patient_level_split
from src.data.target import TARGET_COLUMN, TARGET_LABELS
from src.data.validation import assert_matching_feature_columns, assert_no_leakage_columns
from src.evaluation.metrics import evaluate_classifier, write_evaluation_outputs
from src.features.build_features import (
    IDENTIFIER_COLUMNS,
    build_modeling_frame,
    infer_feature_types,
    make_sample_dataset,
)
from src.models.pipeline import build_classifier, fit_with_balanced_weights


MODEL_NAME = "icu_los_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a three-class ICU LOS classifier using first-24-hour features."
    )
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Build features from data/raw instead of using existing processed first-24-hour features.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Train on a tiny synthetic dataset for demo artifact creation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    if args.sample:
        df = make_sample_dataset()
        feature_cols = [
            col
            for col in df.columns
            if col not in IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN]
        ]
        return df, feature_cols
    processed_dir = None if args.from_raw else args.processed_data_dir
    return build_modeling_frame(args.raw_data_dir, processed_dir=processed_dir)


def main() -> None:
    args = parse_args()
    modeling_df, feature_cols = _load_inputs(args)
    assert_no_leakage_columns(feature_cols)

    modeling_df = modeling_df.dropna(subset=[TARGET_COLUMN, "subject_id", "stay_id"]).copy()
    modeling_df["split"] = patient_level_split(
        modeling_df,
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_state,
    )
    assert_patient_split_integrity(modeling_df)

    numeric_cols, categorical_cols = infer_feature_types(modeling_df, feature_cols)
    train_df = modeling_df[modeling_df["split"].eq("train")].copy()
    val_df = modeling_df[modeling_df["split"].eq("val")].copy()
    test_df = modeling_df[modeling_df["split"].eq("test")].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN].astype("int64")
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN].astype("int64")
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype("int64")
    assert_matching_feature_columns(X_train.columns, X_val.columns)
    assert_matching_feature_columns(X_train.columns, X_test.columns)

    model = build_classifier(numeric_cols, categorical_cols)
    model = fit_with_balanced_weights(model, X_train, y_train)

    reports_dir = args.reports_dir / ("classification_sample" if args.sample else "classification")
    val_metrics, val_per_class, val_conf = evaluate_classifier(model, X_val, y_val)
    test_metrics, test_per_class, test_conf = evaluate_classifier(model, X_test, y_test)
    write_evaluation_outputs(reports_dir, "validation", val_metrics, val_per_class, val_conf)
    write_evaluation_outputs(reports_dir, "test", test_metrics, test_per_class, test_conf)

    test_predictions = test_df[["subject_id", "hadm_id", "stay_id", TARGET_COLUMN]].copy()
    test_predictions["predicted_los_category"] = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        for label in model.classes_:
            test_predictions[f"prob_class_{label}"] = probabilities[:, list(model.classes_).index(label)]
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    modeling_df[["subject_id", "hadm_id", "stay_id", "split", TARGET_COLUMN]].to_csv(
        reports_dir / "patient_level_split.csv",
        index=False,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    artifact_model_name = f"{MODEL_NAME}_sample" if args.sample else MODEL_NAME
    model_path = args.models_dir / f"{artifact_model_name}.joblib"
    metadata_path = args.models_dir / f"{artifact_model_name}_metadata.json"
    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target_column": TARGET_COLUMN,
        "target_labels": TARGET_LABELS,
    }
    joblib.dump(artifact, model_path)

    metadata = {
        "model_name": artifact_model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "three-class ICU length-of-stay classification",
        "target_definition": TARGET_LABELS,
        "feature_window": "ICU admission through first 24 hours only",
        "split_policy": "patient-level split by subject_id with explicit overlap assertion",
        "class_imbalance": "balanced sample weights fitted on training split only",
        "model_path": str(model_path),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "test_metrics": test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved reports: {reports_dir}")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
