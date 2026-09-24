# Performance test

Chưa có kết quả public/load test được xác nhận trong workspace hiện tại. Sau khi chạy Compose trên máy triển khai, ghi lại:

- thời điểm và commit;
- số user đồng thời và thời lượng;
- request/second;
- p50/p95 latency của `/api/predict`;
- error rate;
- cấu hình máy và trạng thái warm-up.

Smoke test tối thiểu:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
```
