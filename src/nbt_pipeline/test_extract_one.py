import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from nbt_pipeline.extraction.pipeline import extract_theatre_notes
from nbt_pipeline.preprocessing.load import load_nbt_smallset


def format_value(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None or pd.isna(value):
        return "null"
    return str(value)


def print_extraction_result(result) -> None:
    output = result.copy()
    output = output.map(format_value)

    with pd.option_context("display.max_columns", None, "display.max_colwidth", 80, "display.width", 320):
        table = output.to_string(index=False)
        encoding = sys.stdout.encoding or "utf-8"
        print(table.encode(encoding, errors="replace").decode(encoding))


def run() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny theatre-note extraction test.")
    parser.add_argument("--rows", type=int, default=20, help="Number of non-empty theatre notes to test.")
    parser.add_argument("--output", help="Path to save the joined extraction result.")
    args = parser.parse_args()

    df = load_nbt_smallset()
    sample = df[df["theatre_notes"].notna()].head(args.rows).copy()
    result = extract_theatre_notes(sample)
    output_path = Path(args.output or f"result/theatre_notes_test_{args.rows}_rows.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".xlsx":
        result.to_excel(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    print(f"Saved joined result to: {output_path}")
    print_extraction_result(result)


if __name__ == "__main__":
    run()
