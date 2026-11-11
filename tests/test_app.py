"""Runs the Streamlit app headless."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture()
def app(use_model):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_loads(app):
    assert app.title[0].value == "NEO Sentinel"
    assert len(app.tabs) == 4


def test_single_check_shows_result(app):
    app.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Hazard probability", "Risk tier", "Review list"} <= set(labels)


def test_invalid_input_shows_error(app):
    app.number_input[2].set_value(-5.0)  # negative miss distance
    app.button[0].click().run()
    assert not app.exception
    assert any("miss_distance" in e.value for e in app.error)
