"""
End to end check of src/train.py on a small synthetic dataset shaped like
the NASA CSV. Writes artifacts to a temp folder so real models are untouched.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import train  # noqa: E402
from preprocess import DataValidationError, clean  # noqa: E402


def _synthetic_csv(path: Path, n: int = 400, text_labels: bool = False) -> Path:
    rng = np.random.RandomState(0)
    dmin = rng.uniform(0.01, 1.0, n)
    df = pd.DataFrame(
        {
            "id": range(n),
            "name": [f"obj{i}" for i in range(n)],
            "est_diameter_min": dmin,
            "est_diameter_max": dmin * 2.2,
            "relative_velocity": rng.uniform(5000, 120000, n),
            "miss_distance": rng.uniform(1e5, 7e7, n),
            "orbiting_body": "Earth",
            "sentry_object": False,
            "absolute_magnitude": rng.uniform(15, 30, n),
        }
    )
    hazardous = (df["est_diameter_max"] > 1.0) & (df["absolute_magnitude"] < 24)
    df["hazardous"] = hazardous.map({True: "True", False: "False"}) if text_labels else hazardous
    df.to_csv(path, index=False)
    return path


def test_train_end_to_end(tmp_path):
    csv = _synthetic_csv(tmp_path / "neo.csv")
    out = tmp_path / "models"

    report = train.main(str(csv), models_dir=out, n_iter=2, shap_plot=False)

    assert (out / "pipeline.joblib").exists()
    assert (out / "metrics.json").exists()
    assert (out / "feature_importance.json").exists()

    bundle = joblib.load(out / "pipeline.joblib")
    assert bundle["model_name"] == report["selected_model"]
    assert bundle["model_name"] in report["validation_metrics"]

    sample = pd.DataFrame([[0.1, 0.25, 45000.0, 5e6, 21.5]], columns=bundle["feature_columns"])
    proba = bundle["pipeline"].predict_proba(sample)[0, 1]
    assert 0.0 <= proba <= 1.0

    metrics = json.loads((out / "metrics.json").read_text())
    cm = metrics["test_metrics"]["confusion_matrix"]
    assert sum(cm.values()) == metrics["split_sizes"]["test"]


def test_text_labels_are_parsed_correctly(tmp_path):
    csv = _synthetic_csv(tmp_path / "neo.csv", text_labels=True)
    df = clean(pd.read_csv(csv))
    # "False" must become 0, not True
    assert set(df["hazardous"].unique()) == {0, 1}
    assert df["hazardous"].mean() < 0.9


def test_bad_labels_raise():
    df = pd.DataFrame(
        {
            "est_diameter_min": [0.1],
            "est_diameter_max": [0.2],
            "relative_velocity": [1000.0],
            "miss_distance": [1e6],
            "absolute_magnitude": [20.0],
            "hazardous": ["maybe"],
        }
    )
    with pytest.raises(DataValidationError):
        clean(df)
