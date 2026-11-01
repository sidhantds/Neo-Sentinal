"""
Runs the Streamlit app headless and checks the main flows work.
Uses the synthetic pipeline from conftest.py when no real model exists.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "NEO Sentinel"


def test_single_prediction_shows_result():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click().run()  # the form's "Assess hazard risk" button
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Hazard probability" in labels
    assert "Risk tier" in labels


def test_inverted_diameter_is_rejected():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.number_input[0].set_value(2.0)  # min diameter larger than max
    at.button[0].click().run()
    assert not at.exception
    assert any("est_diameter_min" in e.value for e in at.error)
