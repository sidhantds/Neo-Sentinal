"""Shared fixtures.

The synthetic dataset mimics the real one's structure: several rows per
asteroid with one label each, diameters computed from H, and a label that
depends on H <= 22 plus a hidden orbit distance that miss distance only
partly reveals. Tests never touch the committed model in models/ except
test_shipped_model.py, which checks it on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neo_sentinel.physics import AU_KM, diameter_range_km


def make_dataset(n_objects: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_objects):
        h = rng.uniform(15, 30)
        moid_au = rng.uniform(0.0, 0.5)
        hazardous = h <= 22 and moid_au <= 0.05
        for _ in range(rng.integers(1, 6)):
            miss = (moid_au + rng.uniform(0.001, 0.2)) * AU_KM
            rows.append(
                {
                    "id": 1_000_000 + i,
                    "name": f"({2000 + i % 25} AB{i})",
                    "relative_velocity": rng.uniform(5_000, 120_000),
                    "miss_distance": miss,
                    "orbiting_body": "Earth",
                    "sentry_object": False,
                    "absolute_magnitude": round(h, 2),
                    "hazardous": hazardous,
                }
            )
    df = pd.DataFrame(rows)
    dmin, dmax = diameter_range_km(df["absolute_magnitude"])
    df.insert(2, "est_diameter_min", dmin)
    df.insert(3, "est_diameter_max", dmax)
    return df


@pytest.fixture(scope="session")
def dataset_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "neo.csv"
    make_dataset().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_model(dataset_csv, tmp_path_factory):
    """A small model trained by the real training code on synthetic data."""
    from neo_sentinel.train import train

    out = tmp_path_factory.mktemp("models") / "neo_sentinel.joblib"
    train(
        dataset_csv,
        out,
        target_recall=0.9,
        n_iter=2,
        cv_folds=2,
        only=["logistic_regression", "hist_gradient_boosting"],
    )
    return out


@pytest.fixture()
def use_model(trained_model, monkeypatch):
    monkeypatch.setenv("NEO_MODEL_PATH", str(trained_model))
    return trained_model


VALID = {"absolute_magnitude": 21.5, "relative_velocity": 45_000.0, "miss_distance": 5_000_000.0}
