from nbt_pipeline.config import PROCESSED_DATA_DIR
from nbt_pipeline.outputs import save_dataframe
from nbt_pipeline.preprocessing import build_preprocessed_dataset


def main() -> None:
    df = build_preprocessed_dataset()
    output_path = save_dataframe(df, PROCESSED_DATA_DIR / "nbt_smallset_preprocessed.xlsx")
    print(f"Saved preprocessed dataset to: {output_path}")


if __name__ == "__main__":
    main()
