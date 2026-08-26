FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/app/data/fraud_agent_memory.db
RUN mkdir -p /app/data

EXPOSE 8000

CMD uvicorn main:api --host 0.0.0.0 --port ${PORT:-8000}