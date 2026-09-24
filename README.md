# Customer Churn Prediction

Đồ án Machine Learning phân loại khả năng khách hàng rời mạng bằng dữ liệu Telco Customer Churn.

## Bài toán và dữ liệu

- Classification, target `Churn Label` (`No=0`, `Yes=1`).
- Dataset: `data/telco_churn.csv`, 7.043 dòng ban đầu; nguồn và ghi chú license ở [DATA.md](DATA.md).
- EDA: 7 hình trong [docs/figures](docs/figures), chạy bằng `python notebooks/eda.py`. Giải thích từng hình ở [docs/EDA.md](docs/EDA.md).

## Machine Learning

```powershell
pip install -r requirements.txt
python src/train_model.py
```

Pipeline dùng split 80/20 có stratify trước khi fit imputer, OneHotEncoder và StandardScaler. Các cột target-derived/geographic bị loại để tránh leakage. Có Dummy baseline, Logistic Regression, KNN, Decision Tree và Random Forest. Logistic Regression được tuning bằng GridSearchCV trên tập train; tất cả model được đánh giá bằng 5-fold Stratified CV.

Artifact sau huấn luyện:

- `ai-models/models/model.joblib`: pipeline + model cuối cùng.
- `ai-models/models/schema.json`: feature, kiểu dữ liệu, miền giá trị.
- `ai-models/models/metadata.json`: model version, metric, CV, split và phiên bản thư viện.
- `ai-models/model_comparison.csv`: metric test, CV, confusion matrix và thời gian.

Tạo biểu đồ metric và confusion matrix:

```powershell
python src/evaluate_models.py
python src/confusion_matrix.py
```

## Chạy sản phẩm bằng Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Mở frontend tại `http://localhost:3000`. API docs: backend `http://localhost:8000/docs`, AI service `http://localhost:8001/docs`. Health là `/health`, model info là `http://localhost:8001/model-info`.

Luồng là Frontend -> Backend `/api/predict` -> AI service `/predict` -> MongoDB history. `request_id` được truyền xuyên suốt và xuất hiện trong log. Địa chỉ service lấy từ `.env`.

## Kiểm thử

```powershell
python -m py_compile src/train_model.py ai-models/service/main.py app/backend/main.py
pytest
```

Khi server chạy:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
```

Kết quả load/performance test phải ghi vào `docs/performance.md` sau khi chạy trên máy triển khai thật; repo chưa thể tự tạo public URL hoặc kết quả tunnel hợp lệ.

## Cấu trúc chính

```text
app/frontend/       Dockerized Nginx frontend
app/backend/        Backend proxy, history and request logging
ai-models/models/   model.joblib, schema.json, metadata.json
ai-models/service/  AI service
data/               raw dataset
docs/               report, figures and explanations
docker-compose.yml  frontend, backend, AI service and MongoDB
```

`docs/baocao.docx` đã có. Nhóm còn cần bổ sung thủ công slide, tên thành viên, link public, nhật ký tunnel và performance test. Lịch sử branch/merge phải được tạo bởi các thành viên trên GitHub, không thể hợp lệ hóa bằng commit giả.
