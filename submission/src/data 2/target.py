from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "los_category"
TARGET_LABELS = {
    0: "LOS < 2 days",
    1: "2 <= LOS <= 7 days",
    2: "LOS > 7 days",
}


def add_los_category(
    df: pd.DataFrame,
    los_col: str = "los",
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create the three-class ICU length-of-stay target from LOS in days."""
    if los_col not in df.columns:
        raise KeyError(f"Missing LOS column required for target creation: {los_col}")

    out = df.copy()
    los = pd.to_numeric(out[los_col], errors="coerce")
    if los.isna().any():
        missing = int(los.isna().sum())
        raise ValueError(f"{missing} rows have missing or non-numeric LOS values")
    if (los < 0).any():
        raise ValueError("LOS values must be non-negative")

    out[target_col] = 1
    out.loc[los < 2, target_col] = 0
    out.loc[los > 7, target_col] = 2
    out[target_col] = out[target_col].astype("int64")
    return out
