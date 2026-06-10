from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


LEAKAGE_COLUMNS = {
    "los",
    "outtime",
    "dischtime",
    "deathtime",
    "dod",
    "death_time",
    "death_date",
    "hospital_expire_flag",
    "event_observed",
    "duration",
    "last_careunit",
    "los_category",
}

LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(^|_)outtime($|_)",
        r"(^|_)dischtime($|_)",
        r"(^|_)deathtime($|_)",
        r"(^|_)dod($|_)",
        r"(^|_)death",
        r"expire",
        r"discharge",
        r"last_careunit",
        r"(^|_)los($|_)",
    ]
]


def assert_no_leakage_columns(columns: Iterable[str]) -> None:
    """Raise if feature columns contain known outcome or future-information fields."""
    bad = []
    for col in columns:
        lower = str(col).lower()
        if lower in LEAKAGE_COLUMNS or any(pattern.search(lower) for pattern in LEAKAGE_PATTERNS):
            bad.append(str(col))
    if bad:
        raise AssertionError(f"Leakage-prone feature columns found: {bad}")


def filter_first_24h_events(
    events: pd.DataFrame,
    stays: pd.DataFrame,
    time_col: str,
    stay_id_col: str = "stay_id",
    intime_col: str = "intime",
) -> pd.DataFrame:
    """Attach ICU intime and keep events from ICU hour 0 through hour 24 inclusive."""
    required_events = {stay_id_col, time_col}
    required_stays = {stay_id_col, intime_col}
    missing_events = required_events.difference(events.columns)
    missing_stays = required_stays.difference(stays.columns)
    if missing_events or missing_stays:
        raise KeyError(
            f"Missing event columns {sorted(missing_events)} or stay columns {sorted(missing_stays)}"
        )

    merged = events.merge(stays[[stay_id_col, intime_col]], on=stay_id_col, how="inner")
    event_time = pd.to_datetime(merged[time_col], errors="coerce")
    intime = pd.to_datetime(merged[intime_col], errors="coerce")
    hours = (event_time - intime).dt.total_seconds() / 3600
    return merged.loc[hours.between(0, 24, inclusive="both")].drop(columns=[intime_col])


def assert_events_within_first_24h(
    events: pd.DataFrame,
    stays: pd.DataFrame,
    time_col: str,
    stay_id_col: str = "stay_id",
    intime_col: str = "intime",
) -> None:
    """Raise if any event occurs before ICU admission or after ICU hour 24."""
    merged = events.merge(stays[[stay_id_col, intime_col]], on=stay_id_col, how="inner")
    event_time = pd.to_datetime(merged[time_col], errors="coerce")
    intime = pd.to_datetime(merged[intime_col], errors="coerce")
    hours = (event_time - intime).dt.total_seconds() / 3600
    invalid = ~hours.between(0, 24, inclusive="both")
    if invalid.any():
        raise AssertionError(f"{int(invalid.sum())} events fall outside ICU hours 0-24")


def assert_matching_feature_columns(train_columns: Iterable[str], other_columns: Iterable[str]) -> None:
    train = list(train_columns)
    other = list(other_columns)
    if train != other:
        missing = sorted(set(train).difference(other))
        extra = sorted(set(other).difference(train))
        raise AssertionError(
            f"Feature columns differ after preprocessing. Missing={missing}; extra={extra}"
        )
