import pandas as pd


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


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning steps that are safe for the raw dataset."""
    return clean_blank_text(df)
