"""Guards the model committed in models/: it must load, be versioned, beat
the baseline, respect physics, and give sensible answers."""

import pandas as pd
import pytest

from neo_sentinel.bundle import ModelBundle
from neo_sentinel.config import DEFAULT_MODEL_PATH, FEATURES
from neo_sentinel.data import DATASET_SHA256
from neo_sentinel.metrics import physics_violation_rate

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="no committed model")


@pytest.fixture(scope="module")
def bundle():
    return ModelBundle.load(DEFAULT_MODEL_PATH)


def test_trained_on_the_verified_dataset(bundle):
    assert bundle.metadata["data_sha256"] == DATASET_SHA256
    assert bundle.version.endswith(DATASET_SHA256[:8])


def test_beats_the_brightness_rule_on_test(bundle):
    model, rule = bundle.card["test"], bundle.card["test_baseline_brightness_rule"]
    assert model["average_precision"] > rule["average_precision"]
    assert model["flagged_share"] < rule["flagged_share"]
    assert model["recall"] >= 0.9


def test_respects_physics(bundle):
    grid = pd.DataFrame(
        [
            {"absolute_magnitude": h, "relative_velocity": v, "miss_distance": d}
            for h in range(15, 31)
            for v in (2e4, 6e4, 1.2e5)
            for d in (1e6, 1e7, 3e7, 7e7)
        ]
    )
    assert physics_violation_rate(bundle.estimator, grid[FEATURES].to_numpy(), FEATURES) == 0.0


def test_sensible_extremes(bundle):
    frame = pd.DataFrame(
        [
            {"absolute_magnitude": 18.0, "relative_velocity": 80_000.0, "miss_distance": 2_000_000.0},
            {"absolute_magnitude": 28.0, "relative_velocity": 20_000.0, "miss_distance": 60_000_000.0},
        ]
    )
    big_close, small_far = bundle.score(frame)["hazard_probability"]
    assert big_close > 0.5
    assert small_far < 0.01
