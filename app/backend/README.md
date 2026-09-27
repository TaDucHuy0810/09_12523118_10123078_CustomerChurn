# Backend

Backend FastAPI nhận request từ frontend, gọi AI service và lưu lịch sử dự
đoán vào MongoDB (đồng thời giữ bản sao gần nhất trong bộ nhớ).

## Endpoint

- `GET /health`
- `POST /api/predict`
- `GET /api/history`

## Chạy bằng Docker

```powershell
docker compose up --build backend
```

Backend đọc `AI_SERVICE_URL`, `MONGODB_URI` và `MONGODB_DATABASE` từ biến môi
trường. `X-Request-ID` được truyền xuyên suốt để đối chiếu log giữa backend và
AI service.
