import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "NBT-SmallSet.xlsx"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "result"

EXTRACTION_MODEL = os.environ.get("NBT_EXTRACTION_MODEL", "claude-sonnet-4-6")
EXTRACTION_BATCH_SIZE = int(os.environ.get("NBT_EXTRACTION_BATCH_SIZE", "20"))
