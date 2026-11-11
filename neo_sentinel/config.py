"""Project wide constants and paths. Everything else imports from here."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_DATA_PATH = DATA_DIR / "neo.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "neo_sentinel.joblib"
DEFAULT_CARD_PATH = MODELS_DIR / "model_card.json"

RANDOM_STATE = 42

# Columns in the NASA NeoWs CSV (Kaggle: sameepvani/nasa-nearest-earth-objects).
RAW_COLUMNS = [
    "id",
    "name",
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "orbiting_body",
    "sentry_object",
    "absolute_magnitude",
    "hazardous",
]

# Model inputs. The two diameter columns are deliberately NOT here: in NeoWs
# they are computed from absolute magnitude with a fixed formula (see
# physics.py), so they carry no information that absolute_magnitude lacks.
FEATURES = ["absolute_magnitude", "relative_velocity", "miss_distance"]
TARGET = "hazardous"
GROUP = "id"  # one asteroid appears in many rows; splits are made per asteroid

FEATURE_LABELS = {
    "absolute_magnitude": "Absolute magnitude (H)",
    "relative_velocity": "Relative velocity (km/h)",
    "miss_distance": "Miss distance (km)",
}

# Screening goal: catch at least this share of hazardous objects on validation.
DEFAULT_TARGET_RECALL = 0.95


def model_path() -> Path:
    """Model file used by the API and the app. Override with NEO_MODEL_PATH."""
    return Path(os.environ.get("NEO_MODEL_PATH", DEFAULT_MODEL_PATH))
