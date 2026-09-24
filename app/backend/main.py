from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s backend req=%(request_id)s %(message)s")
logger = logging.getLogger(__name__)
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8001").rstrip("/")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "customer_churn")
history: list[dict] = []
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=300) if MongoClient else None

app = FastAPI(title="Customer Churn Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, object]


@app.get("/health")
def health():
    return {"status": "healthy", "service": "backend", "ai_service_url": AI_SERVICE_URL}


@app.post("/api/predict")
async def predict(payload: PredictionRequest, x_request_id: str | None = Header(default=None)):
    request_id = x_request_id or str(uuid4())
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
    return {"items": history[-50:]}
