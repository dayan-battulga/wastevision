# Multi-stage Docker build for WasteVision inference server

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY . .

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
