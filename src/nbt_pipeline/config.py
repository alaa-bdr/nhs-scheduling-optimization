import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "NBT-SmallSet.xlsx"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "result"
CREWAI_STORAGE_DIR = PROJECT_ROOT / ".crewai_storage"
CREWAI_CHROMA_DIR = CREWAI_STORAGE_DIR / "chromadb"
CREWAI_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CREWAI_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CREWAI_STORAGE_DIR", str(CREWAI_STORAGE_DIR))
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

try:
    from crewai.rag.chromadb import config as chromadb_config
    from crewai.rag.chromadb import constants as chromadb_constants

    chromadb_constants.DEFAULT_STORAGE_PATH = str(CREWAI_CHROMA_DIR)
    chromadb_config.DEFAULT_STORAGE_PATH = str(CREWAI_CHROMA_DIR)
except ImportError:
    pass

EXTRACTION_MODEL = os.environ.get("NBT_EXTRACTION_MODEL", "claude-sonnet-4-6")
EXTRACTION_BATCH_SIZE = int(os.environ.get("NBT_EXTRACTION_BATCH_SIZE", "20"))
EXTRACTION_CREW_MAX_ITER = int(
    os.environ.get("NBT_EXTRACTION_CREW_MAX_ITER", os.environ.get("NBT_EXTRACTION_AGENT_MAX_ITER", "1"))
)
EXTRACTION_CREW_MAX_RPM = int(
    os.environ.get("NBT_EXTRACTION_CREW_MAX_RPM", os.environ.get("NBT_EXTRACTION_AGENT_MAX_RPM", "10"))
)
EXTRACTION_EMBEDDING_MODEL = os.environ.get("NBT_EXTRACTION_EMBEDDING_MODEL", "gemini-embedding-001")
EXTRACTION_USE_PDF_KNOWLEDGE = os.environ.get("NBT_EXTRACTION_USE_PDF_KNOWLEDGE", "false").lower() == "true"
