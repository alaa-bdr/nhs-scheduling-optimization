import pandas as pd

from nbt_pipeline.preprocessing.clean import clean_dataset
from nbt_pipeline.preprocessing.features import add_feature_columns
from nbt_pipeline.preprocessing.load import load_nbt_smallset


def build_preprocessed_dataset() -> pd.DataFrame:
    """Load, clean, decode, and feature-engineer the NBT small-set dataset."""
    df = load_nbt_smallset()
    df = clean_dataset(df)
    df = add_feature_columns(df)
    return df
