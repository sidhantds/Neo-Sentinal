import json

import pytest
from fastapi.testclient import TestClient

from neo_sentinel.api.app import app
from tests.conftest import VALID


@pytest.fixture()
def client(use_model):
    with TestClient(app) as c:
        yield c


def test_health_reports_version(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": body["model_version"]}
    assert body["model_version"]


def test_predict(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["hazard_probability"] <= 1
    assert body["risk_tier"] in {"low", "moderate", "elevated", "high"}
    assert body["flagged"] == (body["hazard_probability"] >= body["threshold"])
    assert body["estimated_diameter_km"]["min"] < body["estimated_diameter_km"]["max"]
    assert body["brightness_rule_flag"] is True
    assert r.headers["X-Model-Version"] == body["model_version"]


@pytest.mark.parametrize(
    "change",
    [
        {"relative_velocity": -1},
        {"miss_distance": 0},
        {"absolute_magnitude": 99},
        {"absolute_magnitude": "bright"},
        {"unexpected_field": 1},
    ],
)
def test_predict_rejects_bad_input(client, change):
    assert client.post("/predict", json={**VALID, **change}).status_code == 422


def test_predict_rejects_missing_field(client):
    payload = dict(VALID)
    del payload["miss_distance"]
    assert client.post("/predict", json=payload).status_code == 422


def test_out_of_range_input_is_scored_with_a_warning(client):
    body = client.post("/predict", json={**VALID, "absolute_magnitude": 3.0}).json()
    assert body["warnings"] and "outside the training range" in body["warnings"][0]


def test_batch_keeps_order_and_matches_single(client):
    objects = [VALID, {**VALID, "absolute_magnitude": 28.0}, {**VALID, "miss_distance": 7e7}]
    body = client.post("/predict/batch", json={"objects": objects}).json()
    assert body["summary"]["total"] == 3
    assert sum(body["summary"]["tiers"].values()) == 3
    single = client.post("/predict", json=VALID).json()
    assert body["results"][0]["hazard_probability"] == single["hazard_probability"]


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"objects": []}).status_code == 422
    assert client.post("/predict/batch", json={"objects": [VALID] * 1001}).status_code == 422


def test_model_card(client):
    card = client.get("/model").json()
    assert {"metadata", "validation", "test", "threshold"} <= set(card)


def test_drift_endpoint(client):
    client.post("/predict/batch", json={"objects": [VALID] * 200})
    report = client.get("/monitoring/drift").json()
    assert report["status"] == "drift"  # 200 identical inputs look nothing like training data


def test_prediction_log(use_model, tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("NEO_PREDICTION_LOG", str(log_file))
    with TestClient(app) as c:
        c.post("/predict/batch", json={"objects": [VALID, VALID]})
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["absolute_magnitude"] == VALID["absolute_magnitude"]


def test_no_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=VALID).status_code == 503
        assert c.get("/model").status_code == 503
