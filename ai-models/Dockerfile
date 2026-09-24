FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai-models/models /app/ai-models/models
COPY ai-models/service /app/service
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]
