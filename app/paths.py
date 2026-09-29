"""Application locations, independent of the process's working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEM_DIR = DATA_DIR / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUT_DIR = OUTPUT_DIR / "paintings"
DRAFT_DIR = OUTPUT_DIR / "painting_drafts"
MAP_DIR = OUTPUT_DIR / "maps"
# Read compatibility for the old maps/<id>/<filename> layout.
# New exports use maps/<DEM parent>/<id>_<filename>; browsing never migrates files.
LEGACY_OUT_DIR = OUTPUT_DIR / "maps"
LEGACY_DRAFT_DIR = OUTPUT_DIR / "drafts"
LEGACY_MIRRORED_DIR = OUTPUT_DIR
MODEL_DIR = PROJECT_ROOT / "models"
POSTER_DIR = PROJECT_ROOT / "poster"
REFERENCE_DIR = DATA_DIR / "reference_data"
PROFESSOR_MAPS_DIR = REFERENCE_DIR / "professor_geologic_maps"

TEST_MAPS_FILE = OUTPUT_DIR / "test_maps.json"
