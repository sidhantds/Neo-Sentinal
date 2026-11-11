"""Input monitoring: is live traffic still shaped like the training data?

At training time we store a profile of each feature (decile edges and the
share of rows in each bin). The API keeps a rolling window of the inputs it
receives and compares the two with the population stability index (PSI).

PSI rule of thumb: below 0.1 stable, 0.1 to 0.25 worth watching, above 0.25
the inputs have shifted enough that the model's numbers should not be
trusted without retraining.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import pandas as pd

PSI_WATCH = 0.1
PSI_DRIFT = 0.25
MIN_SAMPLES = 100
_EPS = 1e-4


def reference_profile(X: pd.DataFrame, bins: int = 10) -> dict:
    profile = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        profile[col] = {
            "edges": [float(e) for e in edges],
            "expected": [float(c) for c in counts / counts.sum()],
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(np.quantile(values, 0.01)),
            "p99": float(np.quantile(values, 0.99)),
        }
    return profile


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < PSI_WATCH:
        return "stable"
    if value < PSI_DRIFT:
        return "watch"
    return "drift"


def out_of_range(record: dict, profile: dict) -> list[str]:
    """Plain language warnings for inputs outside what the model was trained on."""
    warnings = []
    for col, ref in profile.items():
        value = record.get(col)
        if value is None:
            continue
        if value < ref["min"] or value > ref["max"]:
            warnings.append(
                f"{col}={value:g} is outside the training range "
                f"[{ref['min']:g}, {ref['max']:g}]; treat this score with caution."
            )
    return warnings


class DriftMonitor:
    """Thread safe rolling window of recent inputs."""

    def __init__(self, profile: dict, window: int = 5000):
        self.profile = profile
        self.window = window
        self._rows: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self.total_seen = 0

    def add(self, records: list[dict]) -> None:
        with self._lock:
            for r in records:
                self._rows.append([r[c] for c in self.profile])
            self.total_seen += len(records)

    def report(self) -> dict:
        with self._lock:
            rows = np.array(self._rows, dtype=float).reshape(-1, len(self.profile))
        n = len(rows)
        result = {
            "window_size": n,
            "window_capacity": self.window,
            "total_requests_scored": self.total_seen,
            "thresholds": {"watch": PSI_WATCH, "drift": PSI_DRIFT},
            "features": {},
        }
        if n < MIN_SAMPLES:
            result["status"] = "insufficient_data"
            result["message"] = f"Need at least {MIN_SAMPLES} scored objects, have {n}."
            return result

        worst = 0.0
        for i, (col, ref) in enumerate(self.profile.items()):
            edges = np.asarray(ref["edges"])
            counts = np.bincount(np.searchsorted(edges, rows[:, i], side="right"), minlength=len(edges) + 1)
            value = psi(ref["expected"], counts / n)
            worst = max(worst, value)
            result["features"][col] = {
                "psi": round(value, 4),
                "status": psi_status(value),
                "live_mean": round(float(rows[:, i].mean()), 4),
            }
        result["status"] = psi_status(worst)
        return result
