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
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
expected_features = [feature["name"] for feature in schema["features"]]
model_features = list(getattr(model, "feature_names_in_", expected_features))
if model_features != expected_features:
    raise RuntimeError("model feature order does not match schema.json")

app = FastAPI(title="Customer Churn AI Service", version=metadata["model_version"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, object]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-service",
        "port": SERVICE_PORT,
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 2),
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict")
def predict(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    request_id = x_request_id or str(uuid4())
    started = time.perf_counter()
    validation_errors = validate_features(payload.features, schema)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={"errors": validation_errors, "request_id": request_id},
        )
    try:
        frame = pd.DataFrame([payload.features]).reindex(columns=expected_features)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1]) if hasattr(model, "predict_proba") else None
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
        "request_id": request_id,
        "latency_ms": elapsed_ms,
    }
