"""Central path definitions. Everything resolves from the project root,
regardless of where you run python from."""

from pathlib import Path

# src/mrd/paths.py -> parents[0]=mrd, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FEATURES = DATA_DIR / "features"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_CACHE = DATA_DIR / "cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

_ALL_DIRS = [
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    DATA_FEATURES,
    DATA_EXTERNAL,
    DATA_CACHE,
    REPORTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
]


def ensure_dirs() -> None:
    """Create all expected directories if they don't exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
