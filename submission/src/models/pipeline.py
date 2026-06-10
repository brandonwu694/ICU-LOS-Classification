from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


HGB_PARAM_DISTRIBUTIONS = {
    "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "model__max_iter": [150, 250, 350, 500],
    "model__max_leaf_nodes": [15, 31, 45, 63],
    "model__min_samples_leaf": [10, 20, 50, 100],
    "model__l2_regularization": [0.0, 0.01, 0.05, 0.1, 0.5],
    "model__max_bins": [64, 128, 255],
}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build preprocessing fit only on training data inside each model pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
    return preprocessor


def _make_model_pipeline(numeric_cols: list[str], categorical_cols: list[str], model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ]
    )


def build_classifier(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build the selected preprocessing + gradient boosting classifier pipeline."""
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return _make_model_pipeline(numeric_cols, categorical_cols, classifier)


def build_logistic_regression_baseline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a regularized multinomial logistic regression baseline."""
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None,
        random_state=42,
    )
    return _make_model_pipeline(numeric_cols, categorical_cols, classifier)


def build_random_forest_baseline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a class-weighted random forest comparison model."""
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return _make_model_pipeline(numeric_cols, categorical_cols, classifier)


def build_dummy_baseline() -> DummyClassifier:
    """Build a majority-class baseline that ignores predictors."""
    return DummyClassifier(strategy="most_frequent")


def fit_with_balanced_weights(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y_train))
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipeline


def tune_hist_gradient_boosting_classifier(
    numeric_cols: list[str],
    categorical_cols: list[str],
    X_train,
    y_train,
    groups,
    n_iter: int = 20,
    cv_splits: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Tune the selected classifier with patient-group-aware cross-validation."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    unique_groups = np.unique(np.asarray(groups))
    effective_splits = min(cv_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two patient groups are required for tuning")

    pipeline = build_classifier(numeric_cols, categorical_cols)
    cv = StratifiedGroupKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=HGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=1,
        return_train_score=True,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y_train))
    search.fit(
        X_train,
        y_train,
        groups=np.asarray(groups),
        model__sample_weight=sample_weight,
    )
    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score",
        ascending=True,
    )
    return search.best_estimator_, results, search.best_params_
