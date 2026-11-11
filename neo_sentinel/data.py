"""Loading, validating, cleaning and downloading the NASA NEO dataset.

Used by training and by the batch screening path in the app, so a CSV is
cleaned the same way no matter where it enters the system.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neo_sentinel.config import DEFAULT_DATA_PATH, FEATURES, GROUP, TARGET
from neo_sentinel.physics import diameter_range_km

log = logging.getLogger(__name__)

# SHA-256 of the 90,836 row CSV published on Kaggle. The download is rejected
# if the file differs, so every training run starts from the same bytes.
DATASET_SHA256 = "61b67ccfd0c4b01b0c628ee7c9a32cba0149b5e1fe56619bbae1373bbd5b3e66"
KAGGLE_SLUG = "sameepvani/nasa-nearest-earth-objects"
# Public GitHub copies of the same Kaggle file, checked against DATASET_SHA256.
MIRROR_URLS = [
    "https://raw.githubusercontent.com/siddharthaborgohain/NearestEarthObjects/main/neo.csv",
    "https://raw.githubusercontent.com/Ethjackson/NASA-Nearest-Earth-Objects-Supervised-Learning/main/neo.csv",
]

REQUIRED_TRAINING_COLUMNS = FEATURES + [TARGET, GROUP]


class DataValidationError(ValueError):
    """The data does not match the schema the pipeline expects."""


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "dropped": self.dropped}


# ---------------------------------------------------------------- validation


def require_columns(df: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")


def parse_label(series: pd.Series) -> pd.Series:
    """Turn the hazardous label into 0/1.

    A plain astype(bool) would turn the string "False" into True, so text
    labels are mapped explicitly and anything unrecognised is an error.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        bad = sorted(series[parsed.isna()].astype(str).unique())[:5]
        raise DataValidationError(f"Unrecognised values in '{TARGET}': {bad}")
    return parsed.astype(int)


def invalid_feature_rows(df: pd.DataFrame) -> pd.Series:
    """True for rows whose features are missing, non numeric or non physical."""
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(numeric).all(axis=1)
    bad |= numeric["relative_velocity"] <= 0
    bad |= numeric["miss_distance"] <= 0
    return bad


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce feature columns to numbers. Returns (valid rows, mask of invalid rows).

    Shared by training and batch screening. Extra columns are kept untouched.
    """
    require_columns(df, FEATURES)
    bad = invalid_feature_rows(df)
    out = df.loc[~bad].copy()
    out[FEATURES] = out[FEATURES].apply(pd.to_numeric)
    return out, bad


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean the raw CSV for training."""
    require_columns(df, REQUIRED_TRAINING_COLUMNS)
    report = CleaningReport(rows_in=len(df))

    df = df.dropna(subset=[TARGET, GROUP])
    report.dropped["missing_label_or_id"] = report.rows_in - len(df)

    before = len(df)
    df, _ = clean_features(df)
    report.dropped["invalid_features"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report.dropped["exact_duplicates"] = before - len(df)

    df[TARGET] = parse_label(df[TARGET])

    # The label belongs to the asteroid, not the approach. If one asteroid has
    # two different labels the file is corrupt and training should stop.
    conflicting = df.groupby(GROUP)[TARGET].nunique()
    if (conflicting > 1).any():
        ids = conflicting[conflicting > 1].index[:5].tolist()
        raise DataValidationError(f"Asteroids with conflicting labels: {ids}")

    report.rows_out = len(df)
    return df.reset_index(drop=True), report


def describe_dataset(df: pd.DataFrame) -> dict:
    """Facts about the cleaned data that shape the modelling decisions."""
    facts = {
        "rows": int(len(df)),
        "unique_asteroids": int(df[GROUP].nunique()),
        "hazardous_share_rows": round(float(df[TARGET].mean()), 4),
        "hazardous_share_asteroids": round(float(df.groupby(GROUP)[TARGET].first().mean()), 4),
        "max_rows_per_asteroid": int(df.groupby(GROUP).size().max()),
    }
    if {"est_diameter_min", "est_diameter_max"} <= set(df.columns):
        dmin, dmax = diameter_range_km(df["absolute_magnitude"])
        rel_err = np.maximum(
            np.abs(dmin - df["est_diameter_min"]) / df["est_diameter_min"],
            np.abs(dmax - df["est_diameter_max"]) / df["est_diameter_max"],
        )
        facts["diameter_max_relative_error_vs_formula"] = float(rel_err.max())
    return facts


def load_training_data(path: str | Path = DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Run `python -m neo_sentinel download` first.")
    raw = pd.read_csv(path)
    df, report = clean_training_data(raw)
    return df, report


# ---------------------------------------------------------------- download


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_from_url(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "neo-sentinel"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _download_from_kaggle(dest: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # optional dependency

    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_download_files(KAGGLE_SLUG, path=tmp, unzip=True)
        csvs = list(Path(tmp).glob("*.csv"))
        if not csvs:
            raise RuntimeError("Kaggle download contained no CSV")
        shutil.move(str(csvs[0]), dest)


def download_dataset(dest: str | Path = DEFAULT_DATA_PATH, force: bool = False) -> Path:
    """Fetch the dataset, verify its checksum, and return the local path.

    Tries Kaggle first (needs a configured API token), then the public
    mirrors. A file is only accepted if its SHA-256 matches DATASET_SHA256.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if file_sha256(dest) == DATASET_SHA256:
            log.info("Dataset already present and verified at %s", dest)
            return dest
        log.warning("Existing %s has an unexpected checksum, downloading again", dest)

    sources = [("kaggle", lambda tmp: _download_from_kaggle(tmp))]
    sources += [(url, lambda tmp, u=url: _download_from_url(u, tmp)) for url in MIRROR_URLS]

    errors = []
    for name, fetch in sources:
        tmp = dest.with_suffix(".part")
        try:
            log.info("Trying %s", name)
            fetch(tmp)
            digest = file_sha256(tmp)
            if digest != DATASET_SHA256:
                raise RuntimeError(f"checksum mismatch ({digest[:12]}...)")
            tmp.replace(dest)
            log.info("Dataset saved to %s (checksum verified)", dest)
            return dest
        except Exception as exc:  # try the next source
            errors.append(f"{name}: {exc}")
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        "Could not download the dataset from any source.\n  "
        + "\n  ".join(errors)
        + f"\nDownload it manually from https://www.kaggle.com/datasets/{KAGGLE_SLUG} "
        f"and save it as {dest}."
    )
