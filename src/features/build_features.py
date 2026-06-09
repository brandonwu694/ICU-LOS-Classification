from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.target import TARGET_COLUMN, add_los_category
from src.data.validation import assert_no_leakage_columns


IDENTIFIER_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
BASE_FEATURE_COLUMNS = [
    "gender",
    "anchor_age",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
    "race",
    "language",
    "first_careunit",
    "icu_admit_hour",
    "icu_admit_dayofweek",
    "icu_admit_weekend",
    "hospital_to_icu_hours",
    "direct_icu_admit",
]


def read_raw_table(raw_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a MIMIC-IV CSV or CSV.GZ table from data/raw."""
    plain = raw_dir / f"{name}.csv"
    gz = raw_dir / f"{name}.csv.gz"
    path = plain if plain.exists() else gz
    if not path.exists():
        raise FileNotFoundError(f"Missing raw table: {plain} or {gz}")
    return pd.read_csv(path, usecols=columns)


def build_base_stay_features(raw_dir: Path) -> pd.DataFrame:
    """Build non-event features available at ICU admission and the LOS class target."""
    icu = read_raw_table(raw_dir, "icustays")
    admissions = read_raw_table(raw_dir, "admissions")
    patients = read_raw_table(raw_dir, "patients")

    df = icu.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    ).merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "admission_type",
                "admission_location",
                "insurance",
                "language",
                "marital_status",
                "race",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    intime = pd.to_datetime(df["intime"], errors="coerce")
    admittime = pd.to_datetime(df["admittime"], errors="coerce")
    df["icu_admit_hour"] = intime.dt.hour
    df["icu_admit_dayofweek"] = intime.dt.dayofweek
    df["icu_admit_weekend"] = df["icu_admit_dayofweek"].isin([5, 6]).astype("int64")
    df["hospital_to_icu_hours"] = (
        (intime - admittime).dt.total_seconds() / 3600
    ).clip(lower=0)
    df["direct_icu_admit"] = df["hospital_to_icu_hours"].le(6).astype("int64")

    df = add_los_category(df, los_col="los")
    keep_cols = IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN] + BASE_FEATURE_COLUMNS
    out = df[keep_cols].copy()
    feature_cols = [col for col in out.columns if col not in IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN]]
    assert_no_leakage_columns(feature_cols)
    return out


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.split("_") if part)[:60] or "unknown"


def _aggregate_numeric_events(
    events: pd.DataFrame,
    stays: pd.DataFrame,
    time_col: str,
    value_col: str,
    prefix_col: str,
    max_items: int = 25,
) -> pd.DataFrame:
    from src.data.validation import filter_first_24h_events

    required = {"stay_id", time_col, value_col, prefix_col}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"Missing event aggregation columns: {sorted(missing)}")

    events = events.copy()
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    events = events.dropna(subset=[value_col])
    events = filter_first_24h_events(events, stays[["stay_id", "intime"]], time_col=time_col)
    if events.empty:
        return pd.DataFrame({"stay_id": stays["stay_id"]})

    top_items = events[prefix_col].value_counts().head(max_items).index
    events = events[events[prefix_col].isin(top_items)].copy()
    events["feature_name"] = events[prefix_col].map(_slug)
    grouped = events.groupby(["stay_id", "feature_name"])[value_col].agg(["count", "mean", "min", "max"])
    wide = grouped.unstack("feature_name")
    wide.columns = [f"{name}_{stat}_24h" for stat, name in wide.columns]
    return wide.reset_index()


def build_optional_raw_event_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    """Build lightweight first-24-hour raw event summaries when source tables are available."""
    features = base_df[["stay_id"]].copy()
    stays = base_df[["stay_id", "intime"]].copy()

    try:
        chart = read_raw_table(raw_dir, "chartevents", ["stay_id", "charttime", "itemid", "valuenum"])
        chart_features = _aggregate_numeric_events(
            chart, stays, "charttime", "valuenum", "itemid", max_items=20
        )
        features = features.merge(chart_features, on="stay_id", how="left")
    except FileNotFoundError:
        pass

    try:
        labs = read_raw_table(raw_dir, "labevents", ["hadm_id", "charttime", "itemid", "valuenum"])
        labs = labs.merge(base_df[["hadm_id", "stay_id"]], on="hadm_id", how="inner")
        lab_features = _aggregate_numeric_events(
            labs, stays, "charttime", "valuenum", "itemid", max_items=20
        )
        features = features.merge(lab_features, on="stay_id", how="left")
    except FileNotFoundError:
        pass

    numeric_cols = [col for col in features.columns if col != "stay_id"]
    assert_no_leakage_columns(numeric_cols)
    return features


def load_processed_first24_features(processed_dir: Path) -> pd.DataFrame | None:
    """Use existing notebook-generated first-24-hour feature tables when present."""
    path = processed_dir / "modeling_dataset.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if TARGET_COLUMN not in df.columns:
        df = add_los_category(df, los_col="los")

    identifier_cols = [col for col in IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN] if col in df.columns]
    feature_cols = []
    for col in df.columns:
        if col in identifier_cols:
            continue
        if col in {"hadm_id", "subject_id", "stay_id"}:
            continue
        lower = col.lower()
        if lower in {"los", "outtime", "admittime", "duration", "event_observed", "last_careunit"}:
            continue
        if any(token in lower for token in ["dischtime", "deathtime", "dod", "expire", "discharge"]):
            continue
        if lower.endswith("_last_24h") or lower.endswith("_trend_24h"):
            continue
        feature_cols.append(col)

    assert_no_leakage_columns(feature_cols)
    return df[identifier_cols + feature_cols].copy()


def build_modeling_frame(raw_dir: Path, processed_dir: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return a modeling frame and validated feature column list."""
    if processed_dir is not None:
        processed = load_processed_first24_features(processed_dir)
        if processed is not None:
            feature_cols = [
                col
                for col in processed.columns
                if col not in IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN]
            ]
            return processed, feature_cols

    base = build_base_stay_features(raw_dir)
    event_features = build_optional_raw_event_features(raw_dir, base)
    df = base.merge(event_features, on="stay_id", how="left")
    feature_cols = [
        col for col in df.columns if col not in IDENTIFIER_COLUMNS + ["intime", TARGET_COLUMN]
    ]
    assert_no_leakage_columns(feature_cols)
    return df, feature_cols


def infer_feature_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def make_sample_dataset() -> pd.DataFrame:
    """Small synthetic dataset for tests and the public demo."""
    rng = np.random.default_rng(42)
    rows = []
    careunits = ["MICU", "SICU", "CCU"]
    admission_types = ["EW EMER.", "URGENT", "OBSERVATION ADMIT"]
    for subject_id in range(1000, 1060):
        stay_count = 1 if subject_id % 7 else 2
        for offset in range(stay_count):
            severity = rng.normal()
            los = max(0.2, 3.5 + 2.2 * severity + (subject_id % 3) * 1.4 + rng.normal(0, 1.2))
            if subject_id % 11 == 0:
                los += 5.5
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": subject_id * 10 + offset,
                    "stay_id": subject_id * 100 + offset,
                    "intime": pd.Timestamp("2200-01-01") + pd.Timedelta(days=subject_id - 1000),
                    "gender": "F" if subject_id % 2 else "M",
                    "anchor_age": int(35 + (subject_id % 45)),
                    "admission_type": admission_types[subject_id % len(admission_types)],
                    "admission_location": "EMERGENCY ROOM" if subject_id % 2 else "TRANSFER FROM HOSPITAL",
                    "insurance": "Medicare" if subject_id % 3 else "Medicaid",
                    "marital_status": "MARRIED" if subject_id % 2 else "SINGLE",
                    "race": "WHITE" if subject_id % 2 else "BLACK/AFRICAN AMERICAN",
                    "language": "English",
                    "first_careunit": careunits[subject_id % len(careunits)],
                    "icu_admit_hour": subject_id % 24,
                    "icu_admit_dayofweek": subject_id % 7,
                    "icu_admit_weekend": int(subject_id % 7 in [5, 6]),
                    "hospital_to_icu_hours": max(0, rng.normal(5, 3)),
                    "direct_icu_admit": int(subject_id % 4 != 0),
                    "heart_rate_mean_24h": 80 + severity * 8 + rng.normal(0, 3),
                    "heart_rate_max_24h": 100 + severity * 10 + rng.normal(0, 4),
                    "creatinine_mean_24h": max(0.4, 1.0 + severity * 0.3 + rng.normal(0, 0.1)),
                    "wbc_mean_24h": max(2.0, 8.0 + severity * 1.8 + rng.normal(0, 0.8)),
                    "input_event_count_24h": max(0, int(4 + severity * 2 + rng.normal(0, 1))),
                    "los": los,
                }
            )
    df = add_los_category(pd.DataFrame(rows), los_col="los")
    return df.drop(columns=["los"])
