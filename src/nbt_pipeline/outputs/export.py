from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe as Excel or CSV based on the file extension."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    return output_path
