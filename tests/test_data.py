import numpy as np
import pandas as pd
import pytest

from neo_sentinel.data import (
    DataValidationError,
    clean_features,
    clean_training_data,
    describe_dataset,
    parse_label,
)
from neo_sentinel.physics import brightness_rule, diameter_range_km
from tests.conftest import make_dataset


def test_diameter_formula_matches_real_nasa_rows():
    # Two rows copied from the NASA CSV: (H, est_diameter_min, est_diameter_max)
    for h, dmin, dmax in [(16.73, 1.1982708007, 2.6794149658), (20.0, 0.2658, 0.5943468684)]:
        got_min, got_max = diameter_range_km(h)
        assert got_min == pytest.approx(dmin, rel=1e-4)
        assert got_max == pytest.approx(dmax, rel=1e-4)


def test_brightness_rule_boundary():
    assert brightness_rule([21.9, 22.0, 22.1]).tolist() == [True, True, False]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([True, False], [1, 0]),
        (["True", "False"], [1, 0]),  # astype(bool) would make "False" True
        ([" true ", "FALSE"], [1, 0]),
        (["1", "0"], [1, 0]),
        ([1.0, 0.0], [1, 0]),
    ],
)
def test_parse_label(values, expected):
    assert parse_label(pd.Series(values)).tolist() == expected


def test_parse_label_rejects_unknown_values():
    with pytest.raises(DataValidationError):
        parse_label(pd.Series(["maybe", "True"]))


def test_clean_features_drops_invalid_rows_and_keeps_extra_columns():
    df = pd.DataFrame(
        {
            "name": ["ok", "neg", "nan", "text", "inf"],
            "absolute_magnitude": [20, 20, None, 20, 20],
            "relative_velocity": [1e4, -5, 1e4, "fast", 1e4],
            "miss_distance": [1e6, 1e6, 1e6, 1e6, np.inf],
        }
    )
    valid, bad = clean_features(df)
    assert valid["name"].tolist() == ["ok"]
    assert bad.sum() == 4
    assert valid["relative_velocity"].dtype.kind == "f"


def test_clean_features_requires_columns():
    with pytest.raises(DataValidationError, match="miss_distance"):
        clean_features(pd.DataFrame({"absolute_magnitude": [20], "relative_velocity": [1e4]}))


def test_clean_training_data_removes_duplicates_and_reports():
    df = make_dataset(50)
    doubled = pd.concat([df, df.iloc[:5]])
    cleaned, report = clean_training_data(doubled)
    assert len(cleaned) == len(df)
    assert report.dropped["exact_duplicates"] == 5
    assert set(cleaned["hazardous"].unique()) <= {0, 1}


def test_conflicting_labels_for_one_asteroid_stop_training():
    df = make_dataset(20)
    # give the first asteroid a second pass with the opposite label
    first = df.iloc[[0]]
    second_pass = first.assign(hazardous=not bool(first["hazardous"].iloc[0]), miss_distance=123.0)
    df = pd.concat([df, second_pass])
    with pytest.raises(DataValidationError, match="conflicting"):
        clean_training_data(df)


def test_describe_dataset_detects_diameters_are_derived():
    facts = describe_dataset(clean_training_data(make_dataset(100))[0])
    assert facts["unique_asteroids"] == 100
    assert facts["rows"] > facts["unique_asteroids"]
    assert facts["diameter_max_relative_error_vs_formula"] < 1e-9
