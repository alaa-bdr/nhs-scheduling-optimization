from nbt_pipeline.config import PROCESSED_DATA_DIR
from nbt_pipeline.extraction.pipeline import extract_theatre_notes
from nbt_pipeline.preprocessing.load import load_nbt_smallset
from nbt_pipeline.preprocessing.opcs_decode import decode_opcs_column


def run() -> None:
    df = load_nbt_smallset()
    df = decode_opcs_column(df)
    df = extract_theatre_notes(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_DIR / "nbt_smallset_extracted.parquet")


if __name__ == "__main__":
    run()
