import numpy as np
import pandas as pd

from nbt_pipeline.preprocessing.codes import add_code_labels
from nbt_pipeline.preprocessing.session import add_session_description_features
from nbt_pipeline.preprocessing.specialty import add_specialty_column


TIME_COLUMNS = [
    "into_theatre",
    "anaesthetic_start_time",
    "incision",
    "closure",
    "out_of_theatre",
    "operation_end_time",
    "recovery_time",
]


def parse_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse theatre time-of-day columns as timedeltas."""
    df = df.copy()
    for column in TIME_COLUMNS:
        if column in df:
            df[column] = pd.to_timedelta(df[column].astype("string"), errors="coerce")
    return df


def minutes_between(start, end) -> float:
    """Return minutes between two time-of-day values, allowing midnight rollover."""
    if pd.isna(start) or pd.isna(end):
        return np.nan
    delta = end - start
    if delta < pd.Timedelta(0):
        delta += pd.Timedelta(days=1)
    return delta.total_seconds() / 60


def calculate_row_duration(row: pd.Series) -> float:
    """Calculate operation duration using the most reliable available pair of times."""
    incision_to_closure = minutes_between(row.get("incision"), row.get("closure"))
    if not pd.isna(incision_to_closure):
        return incision_to_closure

    theatre_time = minutes_between(row.get("into_theatre"), row.get("out_of_theatre"))
    if not pd.isna(theatre_time):
        return theatre_time

    if not pd.isna(row.get("operation_length_mins")):
        return row.get("operation_length_mins")

    anaesthetic_to_end = minutes_between(row.get("anaesthetic_start_time"), row.get("operation_end_time"))
    if not pd.isna(anaesthetic_to_end):
        return anaesthetic_to_end

    return np.nan


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add planned-vs-actual duration features."""
    df = parse_time_columns(df)
    df["calculated_operation_length_mins"] = df.apply(calculate_row_duration, axis=1)

    if "ExpectedDurationMins" in df:
        df["duration_error_mins"] = df["calculated_operation_length_mins"] - df["ExpectedDurationMins"]
        df["is_overrun"] = df["duration_error_mins"] > 0
        df["overrun_minutes"] = df["duration_error_mins"].clip(lower=0)

    return df


def add_theatre_note_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap rule-based flags from theatre_notes without using an LLM."""
    df = df.copy()
    if "theatre_notes" not in df:
        return df

    notes = df["theatre_notes"].astype("string").fillna("")
    stripped = notes.str.strip()
    df["has_theatre_note"] = stripped.ne("") & stripped.ne(".")
    df["theatre_note_length"] = stripped.str.len()
    df["theatre_note_has_question"] = stripped.str.contains(r"\?", regex=True, na=False)
    df["theatre_note_has_plus_minus"] = stripped.str.contains(r"\+/-|±", regex=True, na=False)
    df["theatre_note_mentions_histology"] = stripped.str.contains("histology|hist", case=False, regex=True, na=False)
    df["theatre_note_mentions_biopsy"] = stripped.str.contains("biopsy", case=False, regex=True, na=False)
    df["theatre_note_mentions_xray"] = stripped.str.contains(r"x-?ray|image intensifier", case=False, regex=True, na=False)
    return df


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable analysis/model features."""
    df = add_code_labels(df)
    df = add_session_description_features(df)
    df = add_specialty_column(df)
    df = add_duration_features(df)
    df = add_theatre_note_flags(df)
    return df
