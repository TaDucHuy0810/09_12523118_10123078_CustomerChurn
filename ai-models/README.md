# AI Models

Thư mục này chứa artifact model và FastAPI service phục vụ dự đoán.

## Artifact

- `models/model.joblib`: pipeline tốt nhất đã fit, dùng làm model mặc định.
- `models/{logistic_regression,knn,decision_tree,naive_bayes}.joblib`: bốn pipeline có thể gọi độc lập.
- `models/schema.json`: schema feature.
- `models/metadata.json`: metric và thông tin lần train.
- `models/checkpoints.json`: checkpoint của từng lần đóng gói model.
- `model_comparison.csv`: bảng so sánh model.

## Chạy service ngoài Docker

Từ thư mục gốc:

```powershell
Set-Location ai-models
uvicorn service.main:app --host 0.0.0.0 --port 8001
Set-Location ..
```

Trong Docker, service được khởi động bằng
[Dockerfile](Dockerfile) và đọc `MODEL_PATH`, `METADATA_PATH`.

Endpoint: `/health`, `/model-info`, `/predict`, `/predict/logistic`,
`/predict/knn`, `/predict/decision-tree`, `/predict/naive-bayes`, `/docs`.
