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
    Reconstruct likely operation start/end clock times and test the recorded duration.

    The dataset appears to store most theatre event columns as minute-of-hour values,
    while `out_of_theatre` is usually a full clock time. This helper uses
    `out_of_theatre` as the anchor, then checks whether:

    operation_length_mins = operation_end_time - into_theatre
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

    for column in ["anaesthetic_start_time", "incision", "closure"]:
        minute_values = result[column].map(_minute_part)
        result[f"{column}_inferred"] = [
            _format_clock(
                _first_clock_time_with_minute_between(minute, lower, upper)
            )
            for minute, lower, upper in zip(minute_values, into_clock, operation_end_clock)
        ]

    result["time_sequence_valid"] = (
        result["operation_length_rule_valid"]
        & result["anaesthetic_start_time_inferred"].notna()
        & result["incision_inferred"].notna()
        & result["closure_inferred"].notna()
    )

    return result
