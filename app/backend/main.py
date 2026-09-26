from __future__ import annotations

import logging
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from src.schema_validation import validate_features

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s backend req=%(request_id)s %(message)s")
logger = logging.getLogger(__name__)
STARTED_AT = time.monotonic()
SERVICE_PORT = int(os.getenv("PORT", "8000"))
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8001").rstrip("/")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "customer_churn")
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "ai-models" / "models" / "schema.json"
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
history: list[dict] = []
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=300) if MongoClient else None

app = FastAPI(title="Customer Churn Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, object]


@app.get("/health")
def health():
    dependencies = {"mongo": "not_configured", "ai_service": "unknown"}

    if client:
        try:
            client.admin.command("ping")
            dependencies["mongo"] = "healthy"
        except Exception as error:
            logger.warning("mongo_health_failed error=%s", error)
            dependencies["mongo"] = "unavailable"

    try:
        with httpx.Client(timeout=2) as http_client:
            response = http_client.get(f"{AI_SERVICE_URL}/health")
            response.raise_for_status()
        dependencies["ai_service"] = "healthy"
    except httpx.HTTPError as error:
        logger.warning("ai_health_failed error=%s", error)
        dependencies["ai_service"] = "unavailable"

    is_healthy = all(
        status == "healthy"
        for status in dependencies.values()
    )
    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": "backend",
        "port": SERVICE_PORT,
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 2),
        "ai_service_url": AI_SERVICE_URL,
        "dependencies": dependencies,
    }


@app.post("/api/predict")
async def predict(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    request_id = x_request_id or str(uuid4())
    validation_errors = validate_features(payload.features, SCHEMA)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={"errors": validation_errors, "request_id": request_id},
        )
    logger.info("validate_ok -> call ai-service", extra={"request_id": request_id})
    try:
        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.post(
                f"{AI_SERVICE_URL}/predict",
                json={"features": payload.features},
                headers={"X-Request-ID": request_id},
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=400, detail=error.response.text) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="AI service unavailable") from error

    result = response.json()
    record = {"request_id": request_id, "created_at": datetime.now(UTC).isoformat(), "result": result}
    history.append(record)
    if client:
        try:
            client[MONGODB_DATABASE].predictions.insert_one(record)
        except Exception as error:
            logger.warning("history_persist_failed error=%s", error, extra={"request_id": request_id})
    logger.info("200 OK saved_history", extra={"request_id": request_id})
    return result


@app.get("/api/history")
def get_history():
    if client:
        try:
            records = list(
                client[MONGODB_DATABASE].predictions.find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(50)
            )
            return {"items": list(reversed(records))}
        except Exception as error:
            logger.error("history_read_failed error=%s", error)
            raise HTTPException(status_code=503, detail="history storage unavailable") from error
    return {"items": history[-50:]}
