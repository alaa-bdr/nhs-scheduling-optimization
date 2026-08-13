from nbt_pipeline.config import PROCESSED_DATA_DIR
from nbt_pipeline.outputs import save_dataframe
from nbt_pipeline.preprocessing import build_analysis_dataset


def main() -> None:
    df = build_analysis_dataset()
    output_path = save_dataframe(df, PROCESSED_DATA_DIR / "nbt_smallset_analysis.xlsx")
    print(f"Saved cleaned analysis dataset to: {output_path}")


if __name__ == "__main__":
    main()
