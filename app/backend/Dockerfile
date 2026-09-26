FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/backend /app/app/backend
COPY src/schema_validation.py /app/src/schema_validation.py
COPY ai-models/models/schema.json /app/model_artifacts/schema.json
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
