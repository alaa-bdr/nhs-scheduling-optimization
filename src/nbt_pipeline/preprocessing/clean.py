import pandas as pd

from nbt_pipeline.preprocessing.codes import VALID_CODE_VALUES, _normalise_code


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype, missingness, and cardinality information for each column."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": df.notna().sum().values,
            "missing": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).round(2).values,
            "unique_values": df.nunique(dropna=True).values,
        }
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values by column."""
    summary = df.isna().sum().rename("missing_count").to_frame()
    summary["missing_pct"] = (summary["missing_count"] / len(df) * 100).round(2)
    return summary.sort_values("missing_pct", ascending=False)


def blank_like_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count empty strings and dot-only placeholders in text columns."""
    rows = []
    for column in df.select_dtypes(include="object").columns:
        cleaned = df[column].astype("string").str.strip()
        rows.append(
            {
                "column": column,
                "empty_strings": (cleaned == "").sum(),
                "single_dots": (cleaned == ".").sum(),
                "unique_values": cleaned.nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows).sort_values(["single_dots", "empty_strings"], ascending=False)


def clean_blank_text(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty strings to missing values while preserving meaningful dots in notes."""
    df = df.copy()
    text_columns = df.select_dtypes(include="object").columns
    for column in text_columns:
        stripped = df[column].astype("string").str.strip()
        df[column] = df[column].where(stripped != "", pd.NA)
    return df


def clean_suspicious_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert clearly invalid numeric values to missing values."""
    df = df.copy()

    if "ExpectedDurationMins" in df:
        df.loc[df["ExpectedDurationMins"] <= 0, "ExpectedDurationMins"] = pd.NA

    if "operation_length_mins" in df:
        df.loc[df["operation_length_mins"] <= 0, "operation_length_mins"] = pd.NA

    if "age_at_operation" in df:
        invalid_age = (
            (df["age_at_operation"] <= 0)
            | (df["age_at_operation"] > 125)
        )
        df.loc[invalid_age, "age_at_operation"] = pd.NA

    return df


def clean_invalid_coded_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert unexpected values in known coded columns to missing values."""
    df = df.copy()

    for column, valid_values in VALID_CODE_VALUES.items():
        if column not in df:
            continue

        codes = df[column].map(_normalise_code)
        invalid_mask = codes.notna() & ~codes.isin(valid_values)
        df.loc[invalid_mask, column] = pd.NA

    return df


def clean_invalid_procedure_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise valid OPCS-like procedure codes and remove invalid formats."""
    df = df.copy()
    column = "actual_proc_1_procedure_code"

    if column not in df:
        return df

    codes = df[column].astype("string").str.strip().str.upper()
    valid_format = codes.str.fullmatch(r"[A-Z]\d{3}")

    df[column] = codes
    df.loc[codes.notna() & ~valid_format.fillna(False), column] = pd.NA
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning steps that are safe for the raw dataset."""
    df = clean_blank_text(df)
    df = clean_suspicious_numeric_values(df)
    df = clean_invalid_coded_values(df)
    df = clean_invalid_procedure_codes(df)
    return df


def cleaning_change_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Summarise where cleaning changed values to missing."""
    rows = []
    shared_columns = before.columns.intersection(after.columns)

    for column in shared_columns:
        before_series = before[column]
        after_series = after[column]
        changed_to_missing = before_series.notna() & after_series.isna()

        if changed_to_missing.any():
            rows.append(
                {
                    "column": column,
                    "changed_to_nan": int(changed_to_missing.sum()),
                    "missing_before": int(before_series.isna().sum()),
                    "missing_after": int(after_series.isna().sum()),
                    "missing_pct_after": round(after_series.isna().mean() * 100, 2),
                }
            )

    columns = [
        "column",
        "changed_to_nan",
        "missing_before",
        "missing_after",
        "missing_pct_after",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values("changed_to_nan", ascending=False)
