import pandas as pd

from nbt_pipeline.preprocessing.codes import add_code_labels
from nbt_pipeline.preprocessing.outliers import add_duration_timing_review_flag
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
        df["duration_tolerance_mins"] = df["ExpectedDurationMins"].mul(0.10).clip(lower=10)

        valid_duration_comparison = (
            df["duration_error_mins"].notna()
            & df["duration_tolerance_mins"].notna()
        )
        df["meaningful_overrun_flag"] = (
            df["duration_error_mins"] > df["duration_tolerance_mins"]
        ).where(valid_duration_comparison, pd.NA).astype("boolean")
        df["meaningful_underrun_flag"] = (
            df["duration_error_mins"] < -df["duration_tolerance_mins"]
        ).where(valid_duration_comparison, pd.NA).astype("boolean")

        df["duration_status"] = pd.Series("missing_duration", index=df.index, dtype="string")
        df.loc[valid_duration_comparison, "duration_status"] = "within_tolerance"
        df.loc[df["meaningful_overrun_flag"].fillna(False), "duration_status"] = "meaningful_overrun"
        df.loc[df["meaningful_underrun_flag"].fillna(False), "duration_status"] = "meaningful_underrun"

    return df


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable analysis/model features."""
    df = add_code_labels(df)
    df = add_theatre_room_features(df)
    df = add_session_description_features(df)
    df = add_specialty_column(df)
    df = add_duration_features(df)
    if {
        "ExpectedDurationMins",
        "operation_length_mins",
        "operation_length_rule_valid",
        "time_sequence_valid",
    }.issubset(df.columns):
        df = add_duration_timing_review_flag(df)
    return df
