from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_classifier(model, X, y_true, labels: list[int] | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    labels = labels or [0, 1, 2]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, labels=labels, multi_class="ovr", average="macro")
            )
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_proba, labels=labels, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = np.nan
            metrics["roc_auc_ovr_weighted"] = np.nan

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    conf = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return metrics, per_class, conf


def write_evaluation_outputs(
    output_dir: Path,
    split_name: str,
    metrics: dict,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(metrics, default=_json_default))
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )
    per_class.to_csv(output_dir / f"{split_name}_per_class_metrics.csv", index=False)
    confusion.to_csv(output_dir / f"{split_name}_confusion_matrix.csv")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
