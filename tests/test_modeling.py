import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from neo_sentinel.config import FEATURES
from neo_sentinel.metrics import calibration_table, choose_threshold, evaluate, physics_violation_rate
from neo_sentinel.models import BrightnessRule
from neo_sentinel.splitting import group_split
from tests.conftest import make_dataset


def test_group_split_keeps_each_asteroid_in_one_part():
    df = make_dataset(400)
    tr, va, te = group_split(df["hazardous"].astype(int), df["id"])
    ids = [set(df["id"].iloc[idx]) for idx in (tr, va, te)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert len(tr) + len(va) + len(te) == len(df)
    assert 0.6 < len(tr) / len(df) < 0.8
    # stratified: every part contains hazardous objects
    for idx in (tr, va, te):
        assert df["hazardous"].iloc[idx].any()


def test_group_split_rejects_tiny_data():
    with pytest.raises(ValueError):
        group_split([0, 1, 0], [1, 2, 3])


def test_choose_threshold_meets_recall_target_with_smallest_list():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)
    for target in (0.8, 0.9, 0.95):
        thr = choose_threshold(y, p, target)
        assert evaluate(y, p, thr)["recall"] >= target
        # a slightly higher threshold must miss the target
        assert evaluate(y, p, thr + 1e-3)["recall"] < target + 0.01


def test_evaluate_reports_flagged_share_and_confusion():
    m = evaluate([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert m["flagged_share"] == 0.5
    assert m["confusion_matrix"] == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1}


def test_calibration_table_bins_cover_all_rows():
    rows = calibration_table(np.r_[np.zeros(50), np.ones(50)], np.linspace(0, 1, 100), bins=5)
    assert sum(r["count"] for r in rows) == 100
    assert rows[0]["observed_rate"] < rows[-1]["observed_rate"]


class AntiPhysics:
    """Says smaller objects are MORE dangerous, which breaks the physics check."""

    def predict_proba(self, X):
        p = np.clip((np.asarray(X)[:, 0] - 10) / 25, 0, 1)
        return np.column_stack([1 - p, p])


def test_physics_check_catches_a_model_that_contradicts_physics():
    X = make_dataset(100)[FEATURES].to_numpy()
    assert physics_violation_rate(AntiPhysics(), X, FEATURES) > 0.9
    assert physics_violation_rate(BrightnessRule().fit(X), X, FEATURES) == 0.0
    constant = DummyClassifier(strategy="prior").fit(X, np.r_[0, 1, np.zeros(len(X) - 2)])
    assert physics_violation_rate(constant, X, FEATURES) == 0.0


def test_brightness_rule_baseline():
    X = np.array([[21.0, 1e4, 1e6], [23.0, 1e4, 1e6]])
    assert BrightnessRule().fit(X).predict(X).tolist() == [1, 0]
