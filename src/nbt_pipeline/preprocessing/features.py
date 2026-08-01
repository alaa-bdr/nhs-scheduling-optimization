import pandas as pd

from nbt_pipeline.preprocessing.codes import add_code_labels
from nbt_pipeline.preprocessing.session import add_session_description_features
from nbt_pipeline.preprocessing.specialty import add_specialty_column
from nbt_pipeline.preprocessing.theatre import add_theatre_room_features
from nbt_pipeline.preprocessing.time import add_theatre_flow_time_features


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add planned-vs-actual duration features."""
    df = df.copy()

    if "operation_length_mins" in df:
        df["calculated_operation_length_mins"] = df["operation_length_mins"]

        try:
            df = add_theatre_flow_time_features(df)
        except KeyError:
            pass

    if "ExpectedDurationMins" in df and "calculated_operation_length_mins" in df:
        df["duration_error_mins"] = df["calculated_operation_length_mins"] - df["ExpectedDurationMins"]
        df["is_overrun"] = df["duration_error_mins"] > 0
        df["overrun_minutes"] = df["duration_error_mins"].clip(lower=0)
        df["underrun_minutes"] = (-df["duration_error_mins"]).clip(lower=0)

    return df


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable analysis/model features."""
    df = add_code_labels(df)
    df = add_theatre_room_features(df)
    df = add_session_description_features(df)
    df = add_specialty_column(df)
    df = add_duration_features(df)
    return df
