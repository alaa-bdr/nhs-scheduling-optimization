import pandas as pd

from nbt_pipeline.config import RAW_DATA_PATH


def load_nbt_smallset(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Sheet1")
