from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


TIME_COLUMNS = [
    "into_theatre",
    "anaesthetic_start_time",
    "incision",
    "closure",
    "out_of_theatre",
    "operation_end_time",
    "recovery_time",
]


def _minute_part(value) -> float:
    """Return the minute-like component from the dataset's theatre time values."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, time):
        return float(value.minute)

    text = str(value).strip()
    if not text:
        return np.nan

    first_part = text.split(":")[0]
    try:
        return float(first_part)
    except ValueError:
        return np.nan


def _clock_minutes(value) -> float:
    """Return minutes from midnight for normal clock-time values."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, time):
        return float(value.hour * 60 + value.minute)

    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour * 60 + parsed.minute)


def _format_clock(total_minutes: float) -> str | pd.NA:
    if pd.isna(total_minutes):
        return pd.NA

    total_minutes = int(total_minutes) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _latest_clock_time_with_minute(minute_value: float, anchor_minutes: float) -> float:
    if pd.isna(minute_value) or pd.isna(anchor_minutes):
        return np.nan

    candidate = (int(anchor_minutes) // 60) * 60 + int(minute_value)
    while candidate > anchor_minutes:
        candidate -= 60
    return float(candidate)


def _first_clock_time_with_minute_between(
    minute_value: float,
    lower_minutes: float,
    upper_minutes: float,
) -> float:
    if pd.isna(minute_value) or pd.isna(lower_minutes) or pd.isna(upper_minutes):
        return np.nan

    candidate = (int(lower_minutes) // 60) * 60 + int(minute_value)
    while candidate < lower_minutes:
        candidate += 60

    if candidate > upper_minutes:
        return np.nan
    return float(candidate)


def _nearest_clock_time_after(
    minute_value: float,
    anchor_minutes: float,
    upper_minutes: float,
) -> float:
    """Return the matching minute in the anchor hour or following hour."""
    candidate = _first_clock_time_with_minute_between(
        minute_value, anchor_minutes, upper_minutes
    )
    if pd.isna(candidate) or candidate - anchor_minutes >= 60:
        return np.nan
    return candidate


def _nearest_clock_time_before(
    minute_value: float,
    anchor_minutes: float,
    lower_minutes: float,
) -> float:
    """Return the matching minute in the anchor hour or preceding hour."""
    candidate = _latest_clock_time_with_minute(minute_value, anchor_minutes)
    if pd.isna(candidate) or candidate < lower_minutes or anchor_minutes - candidate >= 60:
        return np.nan
    return candidate


def time_column_format_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missingness and clock-like formatting for theatre time columns."""
    rows = []
    for column in TIME_COLUMNS:
        if column not in df:
            continue

        series = df[column]
        clock_minutes = series.map(_clock_minutes)
        hour_values = clock_minutes // 60

        rows.append(
            {
                "column": column,
                "non_null": series.notna().sum(),
                "missing": series.isna().sum(),
                "missing_pct": round(series.isna().mean() * 100, 2),
                "hour_0_values": (hour_values == 0).sum(),
                "hour_1_or_more_values": (hour_values >= 1).sum(),
                "unique_values": series.nunique(dropna=True),
            }
        )

    return pd.DataFrame(rows)


def validate_operation_length_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the provisional ordered theatre-time reconstruction.

    The dataset appears to store most theatre event columns as minute-of-hour values,
    while `out_of_theatre` is usually a full clock time. This helper uses
    `out_of_theatre` as the anchor. It applies three working assumptions that
    require confirmation from NBT:

    operation_length_mins = operation_end_time - into_theatre

    - operation end is the nearest matching minute before out of theatre;
    - incision is the nearest matching minute after entering theatre;
    - closure is the nearest matching minute before operation end.

    Each nearest-minute interval must be shorter than 60 minutes. The complete
    inferred order must be:

    into theatre <= anaesthetic start <= incision <= closure
    <= operation end <= out of theatre
    """
    required = {"into_theatre", "operation_end_time", "out_of_theatre", "operation_length_mins"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    out_clock = result["out_of_theatre"].map(_clock_minutes)
    operation_end_minute = result["operation_end_time"].map(_minute_part)
    into_minute = result["into_theatre"].map(_minute_part)

    operation_end_clock = [
        _latest_clock_time_with_minute(minute, anchor)
        for minute, anchor in zip(operation_end_minute, out_clock)
    ]
    operation_end_clock = pd.Series(operation_end_clock, index=result.index)

    into_clock = operation_end_clock - result["operation_length_mins"]
    duration_is_compatible = (
        result["operation_length_mins"].notna()
        & into_minute.notna()
        & operation_end_clock.notna()
        & ((into_clock % 60) == into_minute)
    )

    result["into_theatre_inferred"] = into_clock.map(_format_clock)
    result["operation_end_time_inferred"] = operation_end_clock.map(_format_clock)
    result["operation_length_rule_valid"] = duration_is_compatible

    anaesthetic_clock = pd.Series(
        [
            _nearest_clock_time_after(minute, lower, upper)
            for minute, lower, upper in zip(
                result["anaesthetic_start_time"].map(_minute_part),
                into_clock,
                operation_end_clock,
            )
        ],
        index=result.index,
    )
    incision_clock = pd.Series(
        [
            _nearest_clock_time_after(minute, lower, upper)
            for minute, lower, upper in zip(
                result["incision"].map(_minute_part),
                into_clock,
                operation_end_clock,
            )
        ],
        index=result.index,
    )
    closure_clock = pd.Series(
        [
            _nearest_clock_time_before(minute, upper, lower)
            for minute, lower, upper in zip(
                result["closure"].map(_minute_part),
                into_clock,
                operation_end_clock,
            )
        ],
        index=result.index,
    )

    result["anaesthetic_start_time_inferred"] = anaesthetic_clock.map(_format_clock)
    result["incision_inferred"] = incision_clock.map(_format_clock)
    result["closure_inferred"] = closure_clock.map(_format_clock)

    result["time_sequence_valid"] = (
        result["operation_length_rule_valid"]
        & anaesthetic_clock.notna()
        & incision_clock.notna()
        & closure_clock.notna()
        & (into_clock <= anaesthetic_clock)
        & (anaesthetic_clock <= incision_clock)
        & (incision_clock <= closure_clock)
        & (closure_clock <= operation_end_clock)
        & (operation_end_clock <= out_clock)
    )

    source_complete = result[
        [
            "into_theatre",
            "anaesthetic_start_time",
            "incision",
            "closure",
            "operation_end_time",
            "out_of_theatre",
            "operation_length_mins",
        ]
    ].notna().all(axis=1)
    result["time_reconstruction_status"] = pd.Series(
        "provisional_rule_failed", index=result.index, dtype="string"
    )
    result.loc[~source_complete, "time_reconstruction_status"] = "missing_source_value"
    result.loc[
        source_complete & ~result["operation_length_rule_valid"],
        "time_reconstruction_status",
    ] = "invalid_operation_window"
    result.loc[
        result["time_sequence_valid"], "time_reconstruction_status"
    ] = "valid_provisional_sequence"

    return result


def _hour_band(hour_value: float) -> str | pd.NA:
    if pd.isna(hour_value):
        return pd.NA

    hour = int(hour_value)
    return f"{hour:02d}:00-{hour:02d}:59"


def add_theatre_flow_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day and theatre-flow features from the provisional timeline.

    These features are based on the inferred clock timeline used by
    `validate_operation_length_rule`. They are intended for assumption-sensitive
    EDA only until NBT confirms the local timestamp meanings and nearest-hour
    rules. The raw time columns remain unchanged.
    """
    result = validate_operation_length_rule(df)

    out_clock = result["out_of_theatre"].map(_clock_minutes)
    operation_end_minute = result["operation_end_time"].map(_minute_part)
    into_minute = result["into_theatre"].map(_minute_part)

    operation_end_clock = pd.Series(
        [
            _latest_clock_time_with_minute(minute, anchor)
            for minute, anchor in zip(operation_end_minute, out_clock)
        ],
        index=result.index,
    )
    into_clock = operation_end_clock - result["operation_length_mins"]

    anaesthetic_clock = result["anaesthetic_start_time_inferred"].map(_clock_minutes)
    incision_clock = result["incision_inferred"].map(_clock_minutes)
    closure_clock = result["closure_inferred"].map(_clock_minutes)

    # Restore day offsets lost when inferred clocks were formatted as HH:MM.
    for clock in [anaesthetic_clock, incision_clock]:
        clock += np.floor(into_clock / (24 * 60)) * (24 * 60)
        clock += (clock < into_clock) * (24 * 60)
    closure_clock += np.floor(operation_end_clock / (24 * 60)) * (24 * 60)
    closure_clock -= (closure_clock > operation_end_clock) * (24 * 60)

    valid_duration = result["operation_length_rule_valid"].fillna(False)
    valid_sequence = result["time_sequence_valid"].fillna(False)

    operation_start_hour = ((into_clock % (24 * 60)) // 60).where(valid_duration)
    result["operation_start_hour"] = operation_start_hour.astype("Float64")
    result["operation_start_hour_band"] = operation_start_hour.map(_hour_band)

    result["post_operation_theatre_time_mins"] = (out_clock - operation_end_clock).where(valid_duration)
    result["theatre_occupancy_mins"] = (out_clock - into_clock).where(valid_duration)
    result["theatre_to_anaesthetic_start_mins"] = (anaesthetic_clock - into_clock).where(valid_sequence)
    result["anaesthetic_to_incision_mins"] = (incision_clock - anaesthetic_clock).where(valid_sequence)
    result["incision_to_closure_mins"] = (closure_clock - incision_clock).where(valid_sequence)
    result["closure_to_operation_end_mins"] = (operation_end_clock - closure_clock).where(valid_sequence)

    flow_columns = [
        "post_operation_theatre_time_mins",
        "theatre_occupancy_mins",
        "theatre_to_anaesthetic_start_mins",
        "anaesthetic_to_incision_mins",
        "incision_to_closure_mins",
        "closure_to_operation_end_mins",
    ]
    for column in flow_columns:
        result[column] = result[column].where(result[column] >= 0)

    return result
