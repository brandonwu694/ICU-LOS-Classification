from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def patient_level_split(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.Series:
    """Return train/validation/test labels with all stays for a patient kept together."""
    if subject_col not in df.columns:
        raise KeyError(f"Missing patient identifier column: {subject_col}")
    if df[subject_col].isna().any():
        raise ValueError("subject_id cannot be missing for patient-level splitting")
    if not 0 < test_size < 1 or not 0 <= val_size < 1:
        raise ValueError("test_size must be in (0, 1) and val_size must be in [0, 1)")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    groups = df[subject_col].to_numpy()
    split = pd.Series("train", index=df.index, dtype="object")

    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(first.split(df, groups=groups))
    split.iloc[test_idx] = "test"

    if val_size > 0:
        train_val = df.iloc[train_val_idx]
        relative_val_size = val_size / (1 - test_size)
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val_size,
            random_state=random_state + 1,
        )
        train_idx_rel, val_idx_rel = next(
            second.split(train_val, groups=train_val[subject_col].to_numpy())
        )
        del train_idx_rel
        val_idx = np.asarray(train_val_idx)[val_idx_rel]
        split.iloc[val_idx] = "val"

    assert_patient_split_integrity(df.assign(split=split), subject_col=subject_col)
    return split


def assert_patient_split_integrity(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    split_col: str = "split",
) -> None:
    """Raise if any subject appears in more than one split."""
    required = {subject_col, split_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing split integrity column(s): {sorted(missing)}")

    split_counts = df.groupby(subject_col, dropna=False)[split_col].nunique()
    overlapping_subjects = split_counts[split_counts > 1].index.tolist()
    if overlapping_subjects:
        preview = overlapping_subjects[:10]
        raise AssertionError(
            f"{len(overlapping_subjects)} subject_id values appear in multiple splits: {preview}"
        )
