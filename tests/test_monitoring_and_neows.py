import json
from pathlib import Path

import numpy as np
import pandas as pd

from neo_sentinel.monitoring import DriftMonitor, out_of_range, psi, reference_profile
from neo_sentinel.neows import parse_feed

FIXTURES = Path(__file__).parent / "fixtures"


def _profile():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.uniform(0, 10, 5000)})
    return reference_profile(X)


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25] * 4, [0.25] * 4) == 0.0
    assert psi([0.25] * 4, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_drift_monitor_states():
    profile = _profile()
    monitor = DriftMonitor(profile, window=1000)
    assert monitor.report()["status"] == "insufficient_data"

    rng = np.random.default_rng(1)
    monitor.add([{"a": a, "b": b} for a, b in zip(rng.normal(0, 1, 500), rng.uniform(0, 10, 500), strict=True)])
    assert monitor.report()["status"] == "stable"

    monitor.add([{"a": a, "b": 5.0} for a in rng.normal(3, 1, 1000)])  # shifted, fills the window
    report = monitor.report()
    assert report["status"] == "drift"
    assert report["features"]["a"]["status"] == "drift"
    assert report["window_size"] == 1000
    assert report["total_requests_scored"] == 1500


def test_out_of_range_warning():
    profile = _profile()
    assert out_of_range({"a": 0.0, "b": 5.0}, profile) == []
    warnings = out_of_range({"a": 0.0, "b": 50.0}, profile)
    assert len(warnings) == 1 and "b=50" in warnings[0]


def test_parse_neows_feed():
    payload = json.loads((FIXTURES / "neows_feed.json").read_text())
    df = parse_feed(payload)
    assert len(df) == 2  # the Mars approach is skipped
    assert df["close_approach_date"].tolist() == ["2024-01-01", "2024-01-02"]  # sorted by date
    big = df.set_index("name").loc["433 Example (1898 DQ)"]
    assert big["relative_velocity"] == 47_115.85
    assert big["absolute_magnitude"] == 19.4
    assert big["nasa_hazardous"]
    for col in ["absolute_magnitude", "relative_velocity", "miss_distance"]:
        assert df[col].dtype.kind == "f"


def test_parse_empty_feed():
    assert parse_feed({"near_earth_objects": {}}).empty
