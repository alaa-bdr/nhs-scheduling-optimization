import argparse
import json

from nbt_pipeline.extraction.pipeline import extract_theatre_notes
from nbt_pipeline.preprocessing.load import load_nbt_smallset


def format_value(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def print_extraction_result(result) -> None:
    extracted_columns = [column for column in result.columns if column.startswith("theatre_notes_")]

    for row_number, (_, row) in enumerate(result.iterrows(), start=1):
        print(f"\n=== Theatre Note {row_number} ===")
        print(row["theatre_notes"])
        print()

        rows = [
            {
                "field": column.replace("theatre_notes_", ""),
                "value": format_value(row[column]),
            }
            for column in extracted_columns
        ]
        print(__import__("pandas").DataFrame(rows).to_string(index=False))


def run() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny theatre-note extraction test.")
    parser.add_argument("--rows", type=int, default=5, help="Number of non-empty theatre notes to test.")
    args = parser.parse_args()

    df = load_nbt_smallset()
    sample = df[df["theatre_notes"].notna()].head(args.rows).copy()
    result = extract_theatre_notes(sample)
    print_extraction_result(result)


if __name__ == "__main__":
    run()
