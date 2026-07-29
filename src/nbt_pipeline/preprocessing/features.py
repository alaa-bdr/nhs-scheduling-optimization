import pandas as pd

from nbt_pipeline.preprocessing.codes import add_code_labels
from nbt_pipeline.preprocessing.session import add_session_description_features
from nbt_pipeline.preprocessing.specialty import add_specialty_column
from nbt_pipeline.preprocessing.theatre import add_theatre_room_features
from nbt_pipeline.preprocessing.time import validate_operation_length_rule


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add planned-vs-actual duration features."""
    df = df.copy()

    if "operation_length_mins" in df:
        df["calculated_operation_length_mins"] = df["operation_length_mins"]

        try:
            validated = validate_operation_length_rule(df)
            df["operation_length_rule_valid"] = validated["operation_length_rule_valid"]
            df["time_sequence_valid"] = validated["time_sequence_valid"]
        except KeyError:
            pass

    if "ExpectedDurationMins" in df and "calculated_operation_length_mins" in df:
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
    df = add_theatre_room_features(df)
    df = add_session_description_features(df)
    df = add_specialty_column(df)
    df = add_duration_features(df)
    df = add_theatre_note_flags(df)
    return df
