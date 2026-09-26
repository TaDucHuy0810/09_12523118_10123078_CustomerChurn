# Performance and Public Deployment Record

## Concurrent local API test

| Item                | Result                                                          |
| ------------------- | --------------------------------------------------------------- |
| Date                | 2026-09-26                                                      |
| Route               | `POST /api/predict` through frontend-facing backend port        |
| Users               | 10 concurrent clients                                           |
| Duration            | 60.13 seconds                                                   |
| Requests            | 2,799 successful requests                                       |
| Throughput          | 46.55 requests/second                                           |
| Client-observed p50 | 204.64 ms                                                       |
| Client-observed p95 | 289.35 ms                                                       |
| Errors              | 0 (0%)                                                          |
| Payload             | All 20 required schema fields                                   |
| Local ports         | Frontend 3001, backend 8000, AI 8001                            |
| Host                | Windows 11 Home; AMD Ryzen 7 6800HS; 13.7 GB RAM; Docker 29.7.2 |
| Git base            | `9bd3b6c` plus uncommitted working-tree changes                 |

This was a warmed, single-host local test, not a production capacity claim. Latencies include client, Docker networking and persistence to local MongoDB. Re-run against the public URLs and target deployment environment before presenting as production performance.

Run from the repository root with the stack started:

```powershell
python scripts/load_test.py --base-url http://localhost:8000 --users 10 --duration 60
```

The script builds a valid payload from `ai-models/models/schema.json` and prints throughput, p50/p95, HTTP status and error rate. Use `--base-url` for a different reachable API URL; the script defaults to 10 clients and 60 seconds.

## Docker smoke checks

Verified after rebuild:

- Compose configuration and all images build successfully.
- Frontend `/health` returns HTTP 200.
- Backend and AI `/health` report healthy; AI reports `model_loaded: true`.
- Full 20-feature prediction through the App succeeds; a missing required feature returns 422.
- A successful request is retrievable from `GET /api/history` after reading MongoDB.
- Health responses include status, port and uptime; AI reports `model_loaded: true`.
- Tests: `python -m pytest tests -q` (10 passed).

## Public quick tunnels

Verified on 2026-09-26. These are temporary, account-less Cloudflare quick tunnels; they have no uptime guarantee and stop working when their `cloudflared` processes exit.

| Time (GMT+7) | Component  | Public URL                                              | Verification                                            |
| ------------ | ---------- | ------------------------------------------------------- | ------------------------------------------------------- |
| 16:57        | App        | https://antarctica-film-edt-hospitals.trycloudflare.com | Frontend HTTP 200; full prediction succeeded            |
| 16:58        | AI service | https://northwest-sku-til-determine.trycloudflare.com   | `/health`, `/model-info`, `/docs`, `/predict` succeeded |

Start tunnels from separate terminals while Docker Compose stays running:

```powershell
cloudflared tunnel --url http://localhost:3001
cloudflared tunnel --url http://localhost:8001
```

When either process restarts, get its new URL, smoke-test it, and update `README.md` and this log. Do not leave an expired quick-tunnel URL as the active demo address.
