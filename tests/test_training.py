import json

import pandas as pd
import pytest

from neo_sentinel import experiments
from neo_sentinel.bundle import TIERS, ModelBundle
from neo_sentinel.config import FEATURES
from neo_sentinel.data import DATASET_SHA256, file_sha256
from tests.conftest import VALID


def test_training_writes_bundle_and_card(trained_model, dataset_csv):
    assert trained_model.exists()
    card = json.loads(trained_model.with_name("model_card.json").read_text())

    assert card["model_name"] in card["validation"]
    assert "brightness_rule" in card["validation"]  # baseline always compared
    assert card["metadata"]["data_sha256"] == file_sha256(dataset_csv)
    assert card["metadata"]["model_version"].endswith(card["metadata"]["data_sha256"][:8])
    assert sum(card["test"]["confusion_matrix"].values()) == card["split"]["rows"]["test"]
    assert set(card["permutation_importance_test"]) == set(FEATURES)
    # the selected model passed the physics check
    assert card["validation"][card["model_name"]]["physics_violations"] <= 0.01


def test_model_learns_the_synthetic_rule(trained_model):
    card = json.loads(trained_model.with_name("model_card.json").read_text())
    assert card["test"]["roc_auc"] > 0.85


def test_bundle_round_trip_and_column_order(trained_model, tmp_path):
    bundle = ModelBundle.load(trained_model)
    copy = ModelBundle.load(bundle.save(tmp_path / "copy.joblib"))

    frame = pd.DataFrame([VALID])
    shuffled = frame[list(reversed(frame.columns))]
    a = bundle.score(frame)["hazard_probability"].iloc[0]
    assert copy.score(shuffled)["hazard_probability"].iloc[0] == a


def test_tiers_follow_threshold(trained_model):
    bundle = ModelBundle.load(trained_model)
    assert bundle.tier(0.0) == "low"
    assert bundle.tier(bundle.threshold) != "low"
    assert bundle.tier(0.99) == "high"
    assert [bundle.tier(p) for p in sorted([0.0, 0.25, 0.45, 0.9])] == sorted(
        [bundle.tier(p) for p in [0.0, 0.25, 0.45, 0.9]], key=TIERS.index
    )


def test_loading_something_else_fails(tmp_path):
    import joblib

    joblib.dump({"pipeline": "old format"}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="not a NEO Sentinel model bundle"):
        ModelBundle.load(tmp_path / "old.joblib")


def test_history_experiment_runs(dataset_csv, tmp_path):
    out = tmp_path / "exp.json"
    report = experiments.run(dataset_csv, out_path=out)
    assert out.exists()
    assert set(report["results"]) == {"single_approach", "with_history"}


def test_leakage_experiment_shows_overlap_only_for_row_split(dataset_csv, tmp_path):
    report = experiments.run_leakage(dataset_csv, out_path=tmp_path / "leak.json")
    assert report["results"]["random_rows"]["test_asteroids_also_in_training"] > 0.3
    assert report["results"]["by_asteroid"]["test_asteroids_also_in_training"] == 0.0


def test_dataset_checksum_constant_is_well_formed():
    assert len(DATASET_SHA256) == 64
