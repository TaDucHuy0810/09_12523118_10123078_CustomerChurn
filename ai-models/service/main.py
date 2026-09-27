from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict
from src.schema_validation import validate_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s ai-service req=%(request_id)s %(message)s")
logger = logging.getLogger(__name__)
STARTED_AT = time.monotonic()
SERVICE_PORT = int(os.getenv("PORT", "8001"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", "ai-models/models/model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "ai-models/models/metadata.json"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", str(MODEL_PATH.parent / "schema.json")))
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
expected_features = [feature["name"] for feature in schema["features"]]
available_model_names = [item["name"] for item in metadata.get("models", [])]
if not available_model_names:
    available_model_names = [metadata["model_name"]]
models = {}
for model_name in available_model_names:
    model_path = MODEL_PATH.parent / f"{model_name}.joblib"
    if not model_path.exists() and model_name == metadata["model_name"]:
        model_path = MODEL_PATH
    loaded_model = joblib.load(model_path)
    model_features = list(getattr(loaded_model, "feature_names_in_", expected_features))
    if model_features != expected_features:
        raise RuntimeError(f"model feature order does not match schema.json: {model_name}")
    models[model_name] = loaded_model

app = FastAPI(title="Customer Churn AI Service", version=metadata["model_version"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, object]
    model: str | None = None


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-service",
        "port": SERVICE_PORT,
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 2),
        "model_loaded": bool(models),
    }


@app.get("/model-info")
def model_info():
    return {
        "default_model": metadata["model_name"],
        "models": metadata.get("models", [{"name": metadata["model_name"], "metrics": metadata["metrics"]}]),
    }


def _predict(payload: PredictionRequest, route_model: str | None, x_request_id: str | None):
    request_id = x_request_id or str(uuid4())
    started = time.perf_counter()
    validation_errors = validate_features(payload.features, schema)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={"errors": validation_errors, "request_id": request_id},
        )
    model_name = route_model or payload.model or metadata["model_name"]
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"unknown model: {model_name}")
    selected_model = models[model_name]
    try:
        frame = pd.DataFrame([payload.features]).reindex(columns=expected_features)
        prediction = int(selected_model.predict(frame)[0])
        probability = float(selected_model.predict_proba(frame)[0][1]) if hasattr(selected_model, "predict_proba") else None
    except Exception as error:
        logger.error("prediction_failed error=%s", error, extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="features do not match the model schema") from error
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("predict class=%s probability=%s latency_ms=%s", prediction, probability, elapsed_ms, extra={"request_id": request_id})
    return {
        "prediction": prediction,
        "label": "Churn" if prediction else "No churn",
        "probability": probability,
        "model_version": metadata["model_version"],
        "model_name": model_name,
        "request_id": request_id,
        "latency_ms": elapsed_ms,
    }


@app.post("/predict")
def predict(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    return _predict(payload, None, x_request_id)


@app.post("/predict/logistic")
def predict_logistic(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    return _predict(payload, "logistic_regression", x_request_id)


@app.post("/predict/knn")
def predict_knn(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    return _predict(payload, "knn", x_request_id)


@app.post("/predict/decision-tree")
def predict_decision_tree(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    return _predict(payload, "decision_tree", x_request_id)


@app.post("/predict/naive-bayes")
def predict_naive_bayes(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    return _predict(payload, "naive_bayes", x_request_id)
