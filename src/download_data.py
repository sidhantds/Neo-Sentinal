"""
Download the NASA Nearest Earth Objects dataset.

Source: NASA Open API (NeoWs - Near Earth Object Web Service), compiled into a
static CSV and published on Kaggle by sameepvani.

Dataset page: https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
Rows: 90,836
Columns: id, name, est_diameter_min, est_diameter_max, relative_velocity,
         miss_distance, orbiting_body, sentry_object, absolute_magnitude, hazardous

This script pulls the CSV with the Kaggle API. You need a Kaggle account and an
API token (kaggle.json) placed at ~/.kaggle/kaggle.json or set as the
KAGGLE_USERNAME / KAGGLE_KEY environment variables.

Usage:
    python src/download_data.py
"""

import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_SLUG = "sameepvani/nasa-nearest-earth-objects"
EXPECTED_CSV_NAMES = ["neo.csv", "neo_v2.csv", "nasa.csv"]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except (ImportError, OSError) as exc:
        print(
            "The 'kaggle' package or your API credentials are not set up.\n"
            "Install it with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json "
            "(download it from https://www.kaggle.com/settings -> API -> Create New Token).\n\n"
            "Alternatively, skip this script and manually download the CSV from:\n"
            f"https://www.kaggle.com/datasets/{DATASET_SLUG}\n"
            f"and place it at {DATA_DIR / 'neo.csv'}\n\n"
            f"Original error: {exc}"
        )
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset {DATASET_SLUG} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(DATA_DIR), unzip=False)

    zip_path = DATA_DIR / "nasa-nearest-earth-objects.zip"
    if not zip_path.exists():
        # Kaggle sometimes names the zip after the last path segment only.
        candidates = list(DATA_DIR.glob("*.zip"))
        if not candidates:
            print("Download completed but no zip file was found. Check the data/ folder manually.")
            sys.exit(1)
        zip_path = candidates[0]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    found = [p for p in DATA_DIR.glob("*.csv") if p.name.lower() in EXPECTED_CSV_NAMES]
    if not found:
        all_csv = list(DATA_DIR.glob("*.csv"))
        if all_csv:
            found = [all_csv[0]]

    if found:
        target = DATA_DIR / "neo.csv"
        if found[0] != target:
            found[0].rename(target)
        print(f"Dataset ready at: {target}")
    else:
        print("Extraction finished but no CSV was located. Inspect the data/ folder manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
