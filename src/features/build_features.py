from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    missing_los = pd.to_numeric(icu["los"], errors="coerce").isna()
    if missing_los.any():
        print(f"Dropped {int(missing_los.sum())} ICU stays without LOS target")
        icu = icu.loc[~missing_los].copy()
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


def _empty_stay_features(stays: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"stay_id": stays["stay_id"].drop_duplicates().to_numpy()})


def _assert_one_row_per_stay(features: pd.DataFrame, source_name: str) -> None:
    if "stay_id" not in features.columns:
        raise KeyError(f"{source_name} features must include stay_id")
    if features["stay_id"].duplicated().any():
        duplicated = features.loc[features["stay_id"].duplicated(), "stay_id"].head(10).tolist()
        raise AssertionError(f"{source_name} features have duplicate stay_id rows: {duplicated}")


def _maybe_read_raw_table(raw_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    try:
        return read_raw_table(raw_dir, name, columns)
    except FileNotFoundError:
        print(f"Skipped {name}: raw file not found")
        return None


def _load_d_items_lookup(raw_dir: Path, linksto: str | None = None) -> pd.DataFrame | None:
    lookup = _maybe_read_raw_table(raw_dir, "d_items")
    if lookup is None:
        return None
    if linksto is not None and "linksto" in lookup.columns:
        lookup = lookup[lookup["linksto"].eq(linksto)].copy()
    keep = [col for col in ["itemid", "label", "category"] if col in lookup.columns]
    return lookup[keep].drop_duplicates("itemid")


def _attach_item_labels(events: pd.DataFrame, lookup: pd.DataFrame | None) -> pd.DataFrame:
    if lookup is None or "itemid" not in events.columns:
        out = events.copy()
        out["item_label"] = out.get("itemid", pd.Series(index=out.index, dtype="object")).astype(str)
        out["item_category"] = "unknown"
        return out
    out = events.merge(lookup, on="itemid", how="left")
    out["item_label"] = out.get("label", out["itemid"]).fillna(out["itemid"]).astype(str)
    if "category" in out.columns:
        out["item_category"] = out["category"].fillna("unknown").astype(str)
    else:
        out["item_category"] = "unknown"
    return out


def _filter_first24_by_stay(
    events: pd.DataFrame,
    base_df: pd.DataFrame,
    time_col: str,
    join_cols: list[str],
) -> pd.DataFrame:
    """Join source rows to ICU stays, then keep timestamps in ICU hours 0-24."""
    stay_context = base_df[["subject_id", "hadm_id", "stay_id", "intime"]].drop_duplicates()
    merged = events.merge(stay_context, on=join_cols, how="inner", suffixes=("", "_stay"))
    if "stay_id_stay" in merged.columns:
        merged["stay_id"] = merged["stay_id_stay"]
        merged = merged.drop(columns=["stay_id_stay"])
    event_time = pd.to_datetime(merged[time_col], errors="coerce")
    intime = pd.to_datetime(merged["intime"], errors="coerce")
    hours = (event_time - intime).dt.total_seconds() / 3600
    return merged.loc[hours.between(0, 24, inclusive="both")].drop(columns=["intime"])


def _add_count_pivot(
    features: pd.DataFrame,
    events: pd.DataFrame,
    group_col: str,
    prefix: str,
    max_levels: int = 10,
    binary: bool = False,
) -> pd.DataFrame:
    if events.empty or group_col not in events.columns:
        return features
    values = events[group_col].fillna("unknown").map(_slug)
    top_values = values.value_counts().head(max_levels).index
    tmp = events.assign(_feature_level=values)
    tmp = tmp[tmp["_feature_level"].isin(top_values)]
    if tmp.empty:
        return features
    pivot = tmp.groupby(["stay_id", "_feature_level"]).size().unstack(fill_value=0)
    if binary:
        pivot = pivot.gt(0).astype("int64")
        pivot.columns = [f"{prefix}_{col}_used_24h" for col in pivot.columns]
    else:
        pivot.columns = [f"{prefix}_{col}_count_24h" for col in pivot.columns]
    return features.merge(pivot.reset_index(), on="stay_id", how="left")


def _add_sum_pivot(
    features: pd.DataFrame,
    events: pd.DataFrame,
    group_col: str,
    value_col: str,
    prefix: str,
    max_levels: int = 10,
) -> pd.DataFrame:
    if events.empty or group_col not in events.columns or value_col not in events.columns:
        return features
    tmp = events.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return features
    tmp["_feature_level"] = tmp[group_col].fillna("unknown").map(_slug)
    top_values = tmp["_feature_level"].value_counts().head(max_levels).index
    tmp = tmp[tmp["_feature_level"].isin(top_values)]
    pivot = tmp.groupby(["stay_id", "_feature_level"])[value_col].sum().unstack(fill_value=0)
    pivot.columns = [f"{prefix}_{col}_total_24h" for col in pivot.columns]
    return features.merge(pivot.reset_index(), on="stay_id", how="left")


def _events_with_volume_ml(events: pd.DataFrame, value_col: str, unit_col: str) -> pd.DataFrame:
    """Keep mL/L rows and normalize the value column to mL."""
    volume_events = events[events[unit_col].fillna("").str.lower().isin(["ml", "l"])].copy()
    is_liters = volume_events[unit_col].fillna("").str.lower().eq("l")
    volume_events.loc[is_liters, value_col] = volume_events.loc[is_liters, value_col] * 1000
    return volume_events


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
    out = wide.reset_index()
    _assert_one_row_per_stay(out, "numeric event")
    return out


def build_chartevents_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    stays = base_df[["stay_id", "intime"]].copy()
    chart = _maybe_read_raw_table(raw_dir, "chartevents", ["stay_id", "charttime", "itemid", "valuenum"])
    if chart is None:
        return _empty_stay_features(stays)
    features = _aggregate_numeric_events(chart, stays, "charttime", "valuenum", "itemid", max_items=20)
    features = features.rename(columns={col: f"chart_{col}" for col in features.columns if col != "stay_id"})
    _assert_one_row_per_stay(features, "chartevents")
    return features


def build_labevents_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    stays = base_df[["stay_id", "intime"]].copy()
    labs = _maybe_read_raw_table(raw_dir, "labevents", ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"])
    if labs is None:
        return _empty_stay_features(stays)
    labs = _filter_first24_by_stay(labs, base_df, time_col="charttime", join_cols=["subject_id", "hadm_id"])
    if labs.empty:
        return _empty_stay_features(stays)
    features = _aggregate_numeric_events(labs, stays, "charttime", "valuenum", "itemid", max_items=20)
    features = features.rename(columns={col: f"lab_{col}" for col in features.columns if col != "stay_id"})
    _assert_one_row_per_stay(features, "labevents")
    return features


def build_inputevents_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    input_events = _maybe_read_raw_table(
        raw_dir,
        "inputevents",
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "itemid",
            "amount",
            "amountuom",
            "ordercategoryname",
        ],
    )
    stays = base_df[["stay_id", "intime"]].copy()
    features = _empty_stay_features(stays)
    if input_events is None:
        return features

    events = _filter_first24_by_stay(input_events, base_df, time_col="starttime", join_cols=["subject_id", "hadm_id", "stay_id"])
    if events.empty:
        return features
    lookup = _load_d_items_lookup(raw_dir, linksto="inputevents")
    events = _attach_item_labels(events, lookup)
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce")
    ml_events = _events_with_volume_ml(events, value_col="amount", unit_col="amountuom")

    summary = events.groupby("stay_id").agg(
        input_event_count_24h=("itemid", "size"),
        input_unique_item_count_24h=("itemid", "nunique"),
        input_unique_category_count_24h=("ordercategoryname", "nunique"),
    )
    if not ml_events.empty:
        summary = summary.join(
            ml_events.groupby("stay_id")["amount"].sum().rename("input_total_volume_ml_24h"),
            how="left",
        )
    features = features.merge(summary.reset_index(), on="stay_id", how="left")
    features = _add_count_pivot(features, events, "ordercategoryname", "input_category", max_levels=12)
    features = _add_sum_pivot(features, ml_events, "item_label", "amount", "input_item_volume_ml", max_levels=12)
    _assert_one_row_per_stay(features, "inputevents")
    return features


def build_outputevents_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    output_events = _maybe_read_raw_table(
        raw_dir,
        "outputevents",
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"],
    )
    stays = base_df[["stay_id", "intime"]].copy()
    features = _empty_stay_features(stays)
    if output_events is None:
        return features

    events = _filter_first24_by_stay(output_events, base_df, time_col="charttime", join_cols=["subject_id", "hadm_id", "stay_id"])
    if events.empty:
        return features
    lookup = _load_d_items_lookup(raw_dir, linksto="outputevents")
    events = _attach_item_labels(events, lookup)
    events["value"] = pd.to_numeric(events["value"], errors="coerce")
    ml_events = _events_with_volume_ml(events, value_col="value", unit_col="valueuom")
    urine_events = events[
        events["item_label"].str.contains("urine|foley|void", case=False, na=False)
        | events["item_category"].str.contains("urine", case=False, na=False)
    ].copy()

    summary = events.groupby("stay_id").agg(
        output_event_count_24h=("itemid", "size"),
        output_unique_item_count_24h=("itemid", "nunique"),
    )
    if not ml_events.empty:
        summary = summary.join(
            ml_events.groupby("stay_id")["value"].sum().rename("output_total_volume_ml_24h"),
            how="left",
        )
    if not urine_events.empty:
        urine_events["value"] = pd.to_numeric(urine_events["value"], errors="coerce")
        summary = summary.join(
            urine_events.groupby("stay_id")["value"].sum().rename("output_urine_volume_ml_24h"),
            how="left",
        )
    features = features.merge(summary.reset_index(), on="stay_id", how="left")
    features = _add_sum_pivot(features, ml_events, "item_label", "value", "output_item_volume_ml", max_levels=12)
    _assert_one_row_per_stay(features, "outputevents")
    return features


def build_procedureevents_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    procedure_events = _maybe_read_raw_table(
        raw_dir,
        "procedureevents",
        ["subject_id", "hadm_id", "stay_id", "starttime", "itemid", "ordercategoryname"],
    )
    stays = base_df[["stay_id", "intime"]].copy()
    features = _empty_stay_features(stays)
    if procedure_events is None:
        return features

    events = _filter_first24_by_stay(procedure_events, base_df, time_col="starttime", join_cols=["subject_id", "hadm_id", "stay_id"])
    if events.empty:
        return features
    lookup = _load_d_items_lookup(raw_dir, linksto="procedureevents")
    events = _attach_item_labels(events, lookup)
    summary = events.groupby("stay_id").agg(
        procedure_event_count_24h=("itemid", "size"),
        procedure_unique_item_count_24h=("itemid", "nunique"),
        procedure_unique_category_count_24h=("ordercategoryname", "nunique"),
    )
    features = features.merge(summary.reset_index(), on="stay_id", how="left")
    features = _add_count_pivot(features, events, "ordercategoryname", "procedure_category", max_levels=10)
    features = _add_count_pivot(features, events, "item_label", "procedure_item", max_levels=12, binary=True)
    _assert_one_row_per_stay(features, "procedureevents")
    return features


def _medication_category(drug: object) -> str:
    text = str(drug).lower()
    keyword_groups: list[tuple[str, tuple[str, ...]]] = [
        ("antibiotic", ("cef", "cillin", "cycline", "mycin", "penem", "floxacin", "aztreonam", "metronidazole", "vancomycin")),
        ("vasopressor_inotrope", ("norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dobutamine", "dopamine", "milrinone")),
        ("sedative_analgesic", ("propofol", "fentanyl", "midazolam", "dexmedetomidine", "hydromorphone", "morphine", "lorazepam")),
        ("anticoagulant", ("heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban")),
        ("insulin", ("insulin",)),
        ("diuretic", ("furosemide", "bumetanide", "torsemide", "chlorothiazide", "metolazone")),
        ("steroid", ("prednisone", "methylprednisolone", "hydrocortisone", "dexamethasone")),
        ("bronchodilator", ("albuterol", "ipratropium", "tiotropium", "levalbuterol")),
    ]
    for category, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def build_prescriptions_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    prescriptions = _maybe_read_raw_table(
        raw_dir,
        "prescriptions",
        ["subject_id", "hadm_id", "starttime", "drug_type", "drug", "route", "dose_val_rx"],
    )
    stays = base_df[["stay_id", "intime"]].copy()
    features = _empty_stay_features(stays)
    if prescriptions is None:
        return features

    events = _filter_first24_by_stay(prescriptions, base_df, time_col="starttime", join_cols=["subject_id", "hadm_id"])
    if events.empty:
        return features
    events["medication_category"] = events["drug"].map(_medication_category)
    summary = events.groupby("stay_id").agg(
        prescription_order_count_24h=("drug", "size"),
        prescription_unique_drug_count_24h=("drug", "nunique"),
        prescription_unique_route_count_24h=("route", "nunique"),
        prescription_unique_category_count_24h=("medication_category", "nunique"),
    )
    features = features.merge(summary.reset_index(), on="stay_id", how="left")
    features = _add_count_pivot(features, events, "medication_category", "prescription_category", max_levels=12)
    features = _add_count_pivot(features, events, "route", "prescription_route", max_levels=10)
    features = _add_count_pivot(features, events, "drug", "prescription_drug", max_levels=15, binary=True)
    _assert_one_row_per_stay(features, "prescriptions")
    return features


def _radiology_modality(text: object) -> str:
    value = str(text).lower()
    if any(keyword in value for keyword in ["ct ", "ct\n", "computed tomography"]):
        return "ct"
    if any(keyword in value for keyword in ["xray", "x-ray", "portable chest", "chest (pa", "radiograph"]):
        return "xray"
    if any(keyword in value for keyword in ["ultrasound", " us ", "doppler"]):
        return "ultrasound"
    if any(keyword in value for keyword in ["mri", "magnetic resonance"]):
        return "mri"
    return "other"


def _radiology_body_region(text: object) -> str:
    value = str(text).lower()
    if any(keyword in value for keyword in ["chest", "lung", "pulmonary"]):
        return "chest"
    if any(keyword in value for keyword in ["abdomen", "abdominal", "liver", "gallbladder", "pelvis"]):
        return "abdomen_pelvis"
    if any(keyword in value for keyword in ["head", "brain", "stroke"]):
        return "head"
    if any(keyword in value for keyword in ["spine", "cervical", "thoracic", "lumbar"]):
        return "spine"
    if any(keyword in value for keyword in ["extremity", "knee", "ankle", "foot", "hand", "wrist", "shoulder"]):
        return "extremity"
    return "other"


def build_radiology_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    radiology = _maybe_read_raw_table(
        raw_dir,
        "radiology",
        ["note_id", "subject_id", "hadm_id", "note_type", "charttime", "text"],
    )
    stays = base_df[["stay_id", "intime"]].copy()
    features = _empty_stay_features(stays)
    if radiology is None:
        return features

    events = _filter_first24_by_stay(radiology, base_df, time_col="charttime", join_cols=["subject_id", "hadm_id"])
    if events.empty:
        return features
    events["modality"] = events["text"].map(_radiology_modality)
    events["body_region"] = events["text"].map(_radiology_body_region)
    text = events["text"].fillna("").str.lower()
    events["has_impression"] = text.str.contains("impression:", regex=False).astype("int64")
    events["mentions_cxr"] = text.str.contains("chest", regex=False).astype("int64")
    events["mentions_line_or_tube"] = text.str.contains("line|tube|catheter", regex=True).astype("int64")
    events["mentions_effusion"] = text.str.contains("effusion", regex=False).astype("int64")
    events["mentions_pneumonia_or_opacity"] = text.str.contains("pneumonia|opacity|consolidation", regex=True).astype("int64")

    summary = events.groupby("stay_id").agg(
        radiology_note_count_24h=("note_id", "size"),
        radiology_unique_note_type_count_24h=("note_type", "nunique"),
        radiology_has_impression_24h=("has_impression", "max"),
        radiology_mentions_cxr_24h=("mentions_cxr", "max"),
        radiology_mentions_line_or_tube_24h=("mentions_line_or_tube", "max"),
        radiology_mentions_effusion_24h=("mentions_effusion", "max"),
        radiology_mentions_pneumonia_or_opacity_24h=("mentions_pneumonia_or_opacity", "max"),
    )
    features = features.merge(summary.reset_index(), on="stay_id", how="left")
    features = _add_count_pivot(features, events, "modality", "radiology_modality", max_levels=6)
    features = _add_count_pivot(features, events, "body_region", "radiology_body_region", max_levels=8)
    _assert_one_row_per_stay(features, "radiology")
    return features


def build_optional_raw_event_features(raw_dir: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    """Build lightweight first-24-hour raw event summaries when source tables are available."""
    features = base_df[["stay_id"]].copy()

    builders: list[Callable[[Path, pd.DataFrame], pd.DataFrame]] = [
        build_chartevents_features,
        build_labevents_features,
        build_inputevents_features,
        build_outputevents_features,
        build_procedureevents_features,
        build_prescriptions_features,
        build_radiology_features,
    ]
    for builder in builders:
        source_features = builder(raw_dir, base_df)
        _assert_one_row_per_stay(source_features, builder.__name__)
        features = features.merge(source_features, on="stay_id", how="left")

    numeric_cols = [col for col in features.columns if col != "stay_id"]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    assert_no_leakage_columns(numeric_cols)
    _assert_one_row_per_stay(features, "raw event")
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
