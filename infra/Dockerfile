# =============================================================================
# Dockerfile — MLB AI Inference API (producción)
# Multi-stage: builder instala deps, runtime es mínimo (~400 MB final)
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bullseye AS builder

WORKDIR /build

# Instala dependencias de compilación necesarias para polars, lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia solo requirements primero (aprovecha cache de Docker)
COPY requirements.txt .

# Instala en directorio aislado para copiar al stage final
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bullseye AS runtime

# Dependencias de sistema mínimas (lightgbm necesita libgomp en runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copia dependencias instaladas desde builder
COPY --from=builder /install /usr/local

# Copia código fuente (solo lo necesario)
COPY src/               ./src/
COPY api/               ./api/
COPY predict_tonight.py ./predict_tonight.py

# Modelo y datos se montan como volumen o se descargan desde S3 en startup
# No se incluyen en la imagen para mantenerla pequeña y sin datos sensibles
ENV MODEL_PATH=/app/models/at_bat_predictor.pkl \
    SILVER_DIR=/app/data/silver/plate_appearances \
    LOG_LEVEL=INFO \
    MAX_WORKERS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cambia a usuario no-root ANTES de exponer cualquier puerto
USER appuser

EXPOSE 8000

# Healthcheck: ALB lo usa para registrar/deregistrar targets
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Gunicorn con worker class uvicorn (async) para producción
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info", \
     "--access-log"]
