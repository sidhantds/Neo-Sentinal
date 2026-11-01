"""
Data validation and preprocessing for the NEO Sentinel project.

This module is imported by both src/train.py (at training time) and
app/main.py (at prediction time), so the exact same validation and
feature logic runs in both places. That is what prevents training-serving
skew.
"""

from __future__ import annotations

import pandas as pd

RAW_REQUIRED_COLUMNS = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]

TARGET_COLUMN = "hazardous"

# Columns dropped before modeling, and why:
#   id, name            -> identifiers, no predictive signal, risk of leakage via lookup
#   orbiting_body       -> constant ("Earth") in this dataset, zero variance
#   sentry_object       -> near-constant flag with negligible signal, excluded to keep
#                          the feature set lean and interpretable
DROP_COLUMNS = ["id", "name", "orbiting_body", "sentry_object"]

FEATURE_COLUMNS = RAW_REQUIRED_COLUMNS  # the final feature set used by the model


class DataValidationError(ValueError):
    """Raised when the input data does not meet the schema the pipeline expects."""


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_raw_schema(df)
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    missing = set(RAW_REQUIRED_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning applied once, at training time, on the raw CSV."""
    df = df.copy()

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    before = len(df)
    df = df.dropna(subset=RAW_REQUIRED_COLUMNS + [TARGET_COLUMN])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in required columns.")

    # Guard against non-physical values (negative sizes, negative distances).
    for col in ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance"]:
        before = len(df)
        df = df[df[col] >= 0]
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with negative '{col}'.")

    df[TARGET_COLUMN] = _parse_target(df[TARGET_COLUMN])

    return df


def _parse_target(series: pd.Series) -> pd.Series:
    """
    Convert the hazardous label to 0/1. A plain astype(bool) would turn the
    string "False" into True, so text labels are mapped explicitly.
    """
    if series.dtype == bool:
        return series.astype(int)
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = sorted(series[parsed.isna()].astype(str).unique())[:5]
        raise DataValidationError(f"Unrecognised values in '{TARGET_COLUMN}': {bad}")
    return parsed.astype(int)


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def validate_prediction_input(record: dict) -> None:
    """
    Validates a single incoming prediction record before it reaches the
    saved pipeline. Pydantic in app/main.py already enforces types and
    simple bounds; this function enforces the cross-field physical
    constraint that Pydantic field validators cannot express as cleanly.
    """
    if record["est_diameter_min"] > record["est_diameter_max"]:
        raise DataValidationError(
            "est_diameter_min cannot be greater than est_diameter_max."
        )
