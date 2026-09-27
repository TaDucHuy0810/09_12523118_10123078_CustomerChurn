import json
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai-models"))

from app.backend import main as backend
from service import main as ai_service

SCHEMA = json.loads((ROOT / "ai-models" / "models" / "schema.json").read_text(encoding="utf-8"))


def valid_features():
    features = {}
    for item in SCHEMA["features"]:
        if item["type"] == "number":
            features[item["name"]] = (item["min"] + item["max"]) / 2
        else:
            features[item["name"]] = item["values"][0]
    return features


class FormFeatureParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.features = []

    def handle_starttag(self, tag, attrs):
        if tag in {"input", "select"}:
            feature = dict(attrs).get("data-feature")
            if feature:
                self.features.append(feature)


def test_frontend_fields_match_schema():
    parser = FormFeatureParser()
    parser.feed((ROOT / "app" / "frontend" / "site" / "index.html").read_text(encoding="utf-8"))
    expected = [item["name"] for item in SCHEMA["features"]]
    assert parser.features == expected


def test_ai_service_accepts_complete_schema_payload():
    response = TestClient(ai_service.app).post("/predict", json={"features": valid_features()})
    assert response.status_code == 200
    assert response.json()["model_version"] == ai_service.metadata["model_version"]


def test_ai_health_reports_port_uptime_and_model_status():
    response = TestClient(ai_service.app).get("/health")
    assert response.status_code == 200
    assert response.json()["port"] == 8001
    assert response.json()["uptime_seconds"] >= 0
    assert response.json()["model_loaded"] is True


def test_ai_service_rejects_missing_required_feature():
    features = valid_features()
    features.pop("Gender")
    response = TestClient(ai_service.app).post("/predict", json={"features": features})
    assert response.status_code == 422
    assert any("Missing required feature: Gender" in error for error in response.json()["detail"]["errors"])


class DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "prediction": 1,
            "label": "Churn",
            "probability": 0.75,
            "model_version": "1.0.0",
            "request_id": self.request_id,
            "latency_ms": 5.0,
        }


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json, headers):
        response = DummyResponse()
        response.request_id = headers["X-Request-ID"]
        return response


def test_backend_rejects_bad_features_before_calling_ai(monkeypatch):
    monkeypatch.setattr(backend, "client", None)
    monkeypatch.setattr(backend.httpx, "AsyncClient", DummyAsyncClient)
    features = valid_features()
    features.pop("CLTV")
    response = TestClient(backend.app).post("/api/predict", json={"features": features})
    assert response.status_code == 422
    assert any("Missing required feature: CLTV" in error for error in response.json()["detail"]["errors"])


def test_backend_forwards_request_id_and_returns_prediction(monkeypatch):
    monkeypatch.setattr(backend, "client", None)
    monkeypatch.setattr(backend.httpx, "AsyncClient", DummyAsyncClient)
    backend.history.clear()
    response = TestClient(backend.app).post(
        "/api/predict",
        headers={"X-Request-ID": "test-request-123"},
        json={"features": valid_features()},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "test-request-123"
    assert backend.history[-1]["request_id"] == "test-request-123"
