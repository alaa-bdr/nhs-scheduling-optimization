"""Prepare train/validation/test datasets for modelling.

Every model in this project must load the files produced here so that
all group members train and evaluate on identical data.

Target: operation_length_mins (how long the operation actually took).
Features: only information known BEFORE the operation starts.
"""

from pathlib import Path

import pandas as pd

from nbt_pipeline.preprocessing import build_preprocessed_dataset

RANDOM_SEED = 42
TARGET = "operation_length_mins"
OUTPUT_DIR = Path("data/modelling")

# Pre-operative features only. Reconstructed timing fields are excluded
# because the reconstruction rule is not yet confirmed by NBT.
FEATURE_COLUMNS = [
    "ExpectedDurationMins",
    "age_at_operation",
    "ASAScore",
    "sex_national_code",
    "admission_type",
    "intended_management",
    "PriorityLevelCode",
    "anaesthetic_desc",
    "procedure_code_chapter",
    "procedure_code_group",
    "theatre_area",
    "TheatreRoom",
    "session_specialty",
    "session_time_band",
]


def build_datasets(output_dir: Path = OUTPUT_DIR) -> None:
    """Build and save the train, validation, and test splits."""
    df = build_preprocessed_dataset()
    print(f"Preprocessed rows: {len(df)}")

    # Eligibility: the target must be present to train or evaluate on a row.
    df = df.dropna(subset=[TARGET])
    print(f"Rows with a recorded target: {len(df)}")

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = sorted(set(FEATURE_COLUMNS) - set(available))
    if missing:
        print(f"Warning, feature columns not found: {missing}")

    modelling_df = df[available + [TARGET]].copy()
    for col in modelling_df.select_dtypes(include="object").columns:
        modelling_df[col] = modelling_df[col].astype("string")

    # 70 / 15 / 15 split with a fixed seed for reproducibility.
    train = modelling_df.sample(frac=0.70, random_state=RANDOM_SEED)
    remainder = modelling_df.drop(train.index)
    validation = remainder.sample(frac=0.50, random_state=RANDOM_SEED)
    test = remainder.drop(validation.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train.parquet")
    validation.to_parquet(output_dir / "validation.parquet")
    test.to_parquet(output_dir / "test.parquet")

    print(f"Features used: {len(available)}")
    print(f"train={len(train)}  validation={len(validation)}  test={len(test)}")
    print(f"Saved to {output_dir.resolve()}")


if __name__ == "__main__":
    build_datasets()
