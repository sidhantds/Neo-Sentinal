import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import (  # noqa: E402
    DataValidationError,
    clean,
    split_features_target,
    validate_prediction_input,
    validate_raw_schema,
)


def make_raw_df(**overrides) -> pd.DataFrame:
    base = {
        "id": [1, 2],
        "name": ["a", "b"],
        "est_diameter_min": [0.1, 0.2],
        "est_diameter_max": [0.2, 0.3],
        "relative_velocity": [40000.0, 50000.0],
        "miss_distance": [1000000.0, 2000000.0],
        "orbiting_body": ["Earth", "Earth"],
        "sentry_object": [False, False],
        "absolute_magnitude": [21.0, 19.5],
        "hazardous": [False, True],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_raw_schema_passes_on_valid_columns():
    df = make_raw_df()
    validate_raw_schema(df)  # should not raise


def test_validate_raw_schema_fails_on_missing_column():
    df = make_raw_df().drop(columns=["miss_distance"])
    with pytest.raises(DataValidationError):
        validate_raw_schema(df)


def test_clean_drops_identifier_and_constant_columns():
    df = make_raw_df()
    cleaned = clean(df)
    for col in ["id", "name", "orbiting_body", "sentry_object"]:
        assert col not in cleaned.columns


def test_clean_drops_rows_with_missing_values():
    df = make_raw_df(est_diameter_min=[0.1, None])
    cleaned = clean(df)
    assert len(cleaned) == 1


def test_clean_drops_rows_with_negative_physical_values():
    df = make_raw_df(miss_distance=[1000000.0, -5.0])
    cleaned = clean(df)
    assert len(cleaned) == 1


def test_clean_casts_target_to_int():
    df = make_raw_df()
    cleaned = clean(df)
    assert set(cleaned["hazardous"].unique()) <= {0, 1}


def test_split_features_target_shapes():
    df = clean(make_raw_df())
    X, y = split_features_target(df)
    assert "hazardous" not in X.columns
    assert len(X) == len(y)


def test_validate_prediction_input_rejects_inverted_diameter_range():
    record = {
        "est_diameter_min": 0.5,
        "est_diameter_max": 0.2,
        "relative_velocity": 40000.0,
        "miss_distance": 1000000.0,
        "absolute_magnitude": 21.0,
    }
    with pytest.raises(DataValidationError):
        validate_prediction_input(record)


def test_validate_prediction_input_accepts_valid_record():
    record = {
        "est_diameter_min": 0.1,
        "est_diameter_max": 0.2,
        "relative_velocity": 40000.0,
        "miss_distance": 1000000.0,
        "absolute_magnitude": 21.0,
    }
    validate_prediction_input(record)  # should not raise
