"""Concurrent local load test for the Customer Churn prediction API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "ai-models" / "models" / "schema.json"


def build_payload(schema_path: Path) -> dict[str, dict[str, Any]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    features = {}
    for item in schema["features"]:
        if item["type"] == "number":
            features[item["name"]] = (item["min"] + item["max"]) / 2
        else:
            features[item["name"]] = item["values"][0]
    return {"features": features}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


async def run_load_test(base_url: str, users: int, duration_seconds: int, payload: dict) -> dict:
    latencies: list[float] = []
    status_codes: list[int] = []
    errors: list[str] = []
    attempts = 0
    deadline = time.perf_counter() + duration_seconds
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)

    async with httpx.AsyncClient(timeout=15, limits=limits) as client:
        health = await client.get(f"{base_url}/health")
        health.raise_for_status()
        warmup = await client.post(f"{base_url}/api/predict", json=payload)
        warmup.raise_for_status()

        started = time.perf_counter()
        deadline = started + duration_seconds

        async def worker() -> None:
            nonlocal attempts
            while time.perf_counter() < deadline:
                request_started = time.perf_counter()
                attempts += 1
                try:
                    response = await client.post(f"{base_url}/api/predict", json=payload)
                    status_codes.append(response.status_code)
                    if response.is_success:
                        latencies.append((time.perf_counter() - request_started) * 1000)
                    else:
                        errors.append(f"HTTP {response.status_code}")
                except httpx.HTTPError as error:
                    errors.append(type(error).__name__)

        await asyncio.gather(*(worker() for _ in range(users)))
        elapsed = time.perf_counter() - started

    total = attempts
    return {
        "base_url": base_url,
        "users": users,
        "duration_seconds": round(elapsed, 2),
        "requests": total,
        "successful_requests": len(latencies),
        "errors": total - len(latencies),
        "error_rate_percent": round(100 * (total - len(latencies)) / total, 2) if total else 0.0,
        "throughput_requests_per_second": round(total / elapsed, 2) if elapsed else 0.0,
        "p50_ms": round(percentile(latencies, 0.50), 2) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
        "status_codes": sorted(set(status_codes)),
        "sample_errors": errors[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    if args.users < 1 or args.duration < 1:
        parser.error("--users and --duration must be positive")
    payload = build_payload(args.schema)
    result = asyncio.run(run_load_test(args.base_url.rstrip("/"), args.users, args.duration, payload))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
