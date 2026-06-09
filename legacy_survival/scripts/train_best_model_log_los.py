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
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MODELS_DIR, PROCESSED_DATA_DIR


MODEL_NAME = "hist_gradient_boosting_log_los"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the best-performing ICU LOS model and save it to models/."
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory containing modeling_dataset.parquet and readiness CSVs.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory where the trained model artifact should be saved.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
        help="Optional directory for metrics and prediction CSVs. Defaults to processed-data-dir/model_outputs.",
    )
    return parser.parse_args()


def load_training_inputs(processed_data_dir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str], pd.DataFrame]:
    modeling_path = processed_data_dir / "modeling_dataset.parquet"
    predictors_path = processed_data_dir / "modeling_ready_predictor_columns.csv"
    numeric_path = processed_data_dir / "modeling_ready_numeric_columns.csv"
    categorical_path = processed_data_dir / "modeling_ready_categorical_columns.csv"
    split_path = processed_data_dir / "modeling_train_test_split.csv"

    required_paths = [
        modeling_path,
        predictors_path,
        numeric_path,
        categorical_path,
        split_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing processed modeling artifact(s). Run notebooks 01-10 first:\n"
            f"{missing}"
        )

    modeling_df = pd.read_parquet(modeling_path)
    predictor_cols = pd.read_csv(predictors_path)["predictor_column"].tolist()
    numeric_cols = pd.read_csv(numeric_path)["numeric_predictor_column"].tolist()
    categorical_cols = pd.read_csv(categorical_path)["categorical_predictor_column"].tolist()
    split_df = pd.read_csv(split_path)

    return modeling_df, predictor_cols, numeric_cols, categorical_cols, split_df


def build_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    gb_regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gb_regressor),
        ]
    )


def score_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred_days: np.ndarray,
    test_pred_days: np.ndarray,
) -> pd.DataFrame:
    train_actual_days = train_df["duration"].to_numpy()
    test_actual_days = test_df["duration"].to_numpy()

    metrics = {
        "model": MODEL_NAME,
        "train_c_index": concordance_index(
            train_actual_days,
            train_pred_days,
            train_df["event_observed"],
        ),
        "test_c_index": concordance_index(
            test_actual_days,
            test_pred_days,
            test_df["event_observed"],
        ),
        "train_mae_days": mean_absolute_error(train_actual_days, train_pred_days),
        "test_mae_days": mean_absolute_error(test_actual_days, test_pred_days),
        "train_median_ae_days": median_absolute_error(train_actual_days, train_pred_days),
        "test_median_ae_days": median_absolute_error(test_actual_days, test_pred_days),
        "train_rmse_days": root_mean_squared_error(train_actual_days, train_pred_days),
        "test_rmse_days": root_mean_squared_error(test_actual_days, test_pred_days),
        "train_median_pred_days": np.median(train_pred_days),
        "test_median_pred_days": np.median(test_pred_days),
    }
    return pd.DataFrame([metrics])


def main() -> None:
    args = parse_args()
    processed_data_dir = args.processed_data_dir
    models_dir = args.models_dir
    model_output_dir = args.model_output_dir or processed_data_dir / "model_outputs"

    (
        modeling_df,
        ready_predictor_cols,
        ready_numeric_cols,
        ready_categorical_cols,
        split_df,
    ) = load_training_inputs(processed_data_dir)

    count_cols = [col for col in ready_predictor_cols if col.endswith("_count_24h")]
    model_predictor_cols = [col for col in ready_predictor_cols if col not in count_cols]
    numeric_cols = [col for col in ready_numeric_cols if col in model_predictor_cols]
    categorical_cols = [col for col in ready_categorical_cols if col in model_predictor_cols]

    model_df = modeling_df[
        ["subject_id", "stay_id", "duration", "event_observed"] + model_predictor_cols
    ].merge(
        split_df[["stay_id", "split"]],
        on="stay_id",
        how="inner",
    )

    if len(model_df) != len(modeling_df):
        raise ValueError("Split merge changed row count")
    if model_df["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id rows after split merge")

    train_df = model_df[model_df["split"].eq("train")].copy()
    test_df = model_df[model_df["split"].eq("test")].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split must contain both train and test rows")

    X_train = train_df[model_predictor_cols]
    X_test = test_df[model_predictor_cols]
    y_train = np.log1p(train_df["duration"])

    pipeline = build_model(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    train_pred_days = np.expm1(pipeline.predict(X_train)).clip(min=0)
    test_pred_days = np.expm1(pipeline.predict(X_test)).clip(min=0)

    metrics_df = score_predictions(train_df, test_df, train_pred_days, test_pred_days)
    test_predictions_df = test_df[
        ["subject_id", "stay_id", "duration", "event_observed"]
    ].copy()
    test_predictions_df["predicted_los_days"] = test_pred_days
    test_predictions_df["absolute_error_days"] = (
        test_predictions_df["predicted_los_days"] - test_predictions_df["duration"]
    ).abs()
    test_predictions_df["predicted_los_percentile"] = test_predictions_df[
        "predicted_los_days"
    ].rank(pct=True)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(model_output_dir / "gb_log_los_metrics.csv", index=False)
    test_predictions_df.to_csv(
        model_output_dir / "gb_log_los_test_predictions.csv",
        index=False,
    )
    pd.Series(count_cols, name="dropped_count_predictor").to_csv(
        model_output_dir / "gb_log_los_dropped_count_predictors.csv",
        index=False,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{MODEL_NAME}.joblib"
    metadata_path = models_dir / f"{MODEL_NAME}_metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": MODEL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "log1p(duration)",
        "prediction_units": "ICU length of stay in days after expm1 transform",
        "model_path": str(model_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "predictor_count": int(len(model_predictor_cols)),
        "numeric_predictor_count": int(len(numeric_cols)),
        "categorical_predictor_count": int(len(categorical_cols)),
        "dropped_count_predictor_count": int(len(count_cols)),
        "metrics": metrics_df.iloc[0].to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metrics and predictions: {model_output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
