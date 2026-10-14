# ─────────────────────────────────────────────────────────────────
# Waste Detection API – CPU-only Dockerfile
# Base: python:3.11-slim  (Debian Bookworm slim)
# Runs as non-root user 'appuser'
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies (libgomp for ONNX Runtime, libgl for Pillow)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── dependency layer (cached until requirements.txt changes) ───────
WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── application layer ──────────────────────────────────────────────
WORKDIR /app

# Create non-root user
RUN useradd --create-home --uid 1001 appuser

# Copy application source
COPY app/       ./app/
COPY config/    ./config/

# Create models directory; the actual .onnx is mounted at runtime
RUN mkdir -p models && chown -R appuser:appuser /app

USER appuser

# ── runtime ────────────────────────────────────────────────────────
EXPOSE 8000

# 2 uvicorn workers – tune WEB_CONCURRENCY or --workers to CPU count
ENV WEB_CONCURRENCY=2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Health check (waits 10 s for startup, then polls every 30 s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", \
     "uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${WEB_CONCURRENCY} \
        --log-level info"]
