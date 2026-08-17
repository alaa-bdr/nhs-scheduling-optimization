import pandas as pd

from nbt_pipeline.preprocessing.clean import clean_dataset
from nbt_pipeline.preprocessing.features import add_feature_columns
from nbt_pipeline.preprocessing.load import load_nbt_smallset
from nbt_pipeline.preprocessing.selection import (
    drop_analysis_columns,
    remove_exact_source_duplicates,
)


def build_preprocessed_dataset() -> pd.DataFrame:
    """Load, clean, decode, and feature-engineer the NBT small-set dataset."""
    df = load_nbt_smallset()
    df = clean_dataset(df)
    df = add_feature_columns(df)
    return df


def build_analysis_dataset() -> pd.DataFrame:
    """Build the cleaned analysis dataset with unsuitable columns removed."""
    deduplicated = remove_exact_source_duplicates(build_preprocessed_dataset())
    return drop_analysis_columns(deduplicated)
