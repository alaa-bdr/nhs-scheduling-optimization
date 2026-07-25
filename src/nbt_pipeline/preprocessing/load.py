from pathlib import Path

import pandas as pd

from nbt_pipeline.config import RAW_DATA_PATH


def load_nbt_smallset(path: str | Path = RAW_DATA_PATH, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Load the raw NBT small-set Excel file."""
    return pd.read_excel(path, sheet_name=sheet_name)
