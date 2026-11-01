import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VALID_PAYLOAD = {
    "est_diameter_min": 0.1,
    "est_diameter_max": 0.25,
    "relative_velocity": 45000.0,
    "miss_distance": 5000000.0,
    "absolute_magnitude": 21.5,
}


@pytest.fixture()
def client():
    # Imported here, not at module level, so the session-scoped
    # ensure_pipeline_exists fixture in conftest.py has already written
    # models/pipeline.joblib before the app's startup event tries to load it.
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input_returns_expected_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["hazardous"], bool)
    assert 0.0 <= body["hazard_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "moderate", "elevated", "high"}
    assert "model_name" in body


def test_predict_rejects_negative_values(client):
    bad_payload = dict(VALID_PAYLOAD, relative_velocity=-100.0)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["miss_distance"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_inverted_diameter_range(client):
    bad_payload = dict(VALID_PAYLOAD, est_diameter_min=1.0, est_diameter_max=0.2)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_name" in body
    assert "feature_columns" in body
