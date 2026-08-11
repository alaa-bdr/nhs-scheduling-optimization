from __future__ import annotations

import numpy as np
import pandas as pd


def _positive_log_iqr_fences(
    series: pd.Series,
    multiplier: float = 1.5,
) -> tuple[float, float]:
    """Return two-sided IQR fences for positive, right-skewed values."""
    values = pd.to_numeric(series, errors="coerce")
    values = values[values.gt(0)].dropna()
    if values.empty:
        return np.nan, np.nan

    logged = np.log(values)
    q1, q3 = logged.quantile([0.25, 0.75])
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        centre = float(values.median())
        return centre, centre

    return (
        float(np.exp(q1 - multiplier * iqr)),
        float(np.exp(q3 + multiplier * iqr)),
    )


def add_duration_timing_review_flag(
    df: pd.DataFrame,
    *,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag unusual duration records that also fail provisional timing validation.

    Expected duration, recorded operation length, and their ratio are screened
    independently using two-sided log-IQR fences. A row is flagged only when at
    least one screen is unusual and either provisional timing-validation flag is
    false. The function preserves all source values and rows.
    """
    required_columns = {
        "ExpectedDurationMins",
        "operation_length_mins",
        "operation_length_rule_valid",
        "time_sequence_valid",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than zero")

    result = df.copy()
    expected = pd.to_numeric(result["ExpectedDurationMins"], errors="coerce")
    recorded = pd.to_numeric(result["operation_length_mins"], errors="coerce")
    valid_pair = expected.gt(0) & recorded.gt(0)
    ratio = (recorded / expected.where(expected.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )

    expected_lower, expected_upper = _positive_log_iqr_fences(
        expected[valid_pair], iqr_multiplier
    )
    recorded_lower, recorded_upper = _positive_log_iqr_fences(
        recorded[valid_pair], iqr_multiplier
    )
    ratio_lower, ratio_upper = _positive_log_iqr_fences(
        ratio[valid_pair], iqr_multiplier
    )

    expected_outlier = valid_pair & (
        expected.lt(expected_lower) | expected.gt(expected_upper)
    )
    recorded_outlier = valid_pair & (
        recorded.lt(recorded_lower) | recorded.gt(recorded_upper)
    )
    ratio_outlier = valid_pair & (
        ratio.lt(ratio_lower) | ratio.gt(ratio_upper)
    )
    timing_failure = (
        result["operation_length_rule_valid"].eq(False).fillna(False)
        | result["time_sequence_valid"].eq(False).fillna(False)
    )

    result["duration_timing_review_flag"] = (
        timing_failure & (expected_outlier | recorded_outlier | ratio_outlier)
    ).astype("boolean")
    return result
