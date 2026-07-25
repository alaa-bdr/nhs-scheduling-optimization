from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "NBT-SmallSet.xlsx"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULT_DIR = PROJECT_ROOT / "result"
