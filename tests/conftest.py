"""
Test setup.

If no real trained pipeline exists yet (models/pipeline.joblib, produced by
src/train.py on the real dataset), this fixture creates a small synthetic
one so the API tests can run end to end without requiring the full NASA
dataset to be downloaded first. If a real pipeline already exists, it is
left untouched and used as is.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import FEATURE_COLUMNS  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


@pytest.fixture(scope="session", autouse=True)
def ensure_pipeline_exists():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if PIPELINE_PATH.exists():
        yield
        return

    rng = np.random.RandomState(42)
    n = 200
    X = rng.rand(n, len(FEATURE_COLUMNS))
    y = (X[:, 0] + X[:, 2] > 1.0).astype(int)  # arbitrary but learnable synthetic rule

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression().fit(X_scaled, y)
    pipeline = Pipeline([("scaler", scaler), ("model", model)])

    joblib.dump(
        {"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS, "model_name": "synthetic_test_model"},
        PIPELINE_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump({"test_metrics": {"note": "synthetic model used for tests only"}}, f)

    yield

    # Clean up the synthetic artifacts so a real training run isn't shadowed.
    PIPELINE_PATH.unlink(missing_ok=True)
    METRICS_PATH.unlink(missing_ok=True)
