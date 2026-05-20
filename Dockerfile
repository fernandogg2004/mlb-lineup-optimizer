# =============================================================================
# MLB Lineup Optimizer — Production API Container
# Multi-stage build: builder installs all Python deps + compiles C extensions;
# runtime copies only the installed site-packages and application source.
#
# Final image size target: < 2.5 GB (LightGBM + Numba + Ray dominate)
# Base: python:3.11-slim (Debian Bookworm, glibc 2.36, no Alpine to keep
#       binary-wheel compatibility for lightgbm, numba, and ray)
#
# Build:
#   docker build -t mlb-optimizer:latest .
#
# Run (local):
#   docker run -p 8000:8000 \
#     -e AT_BAT_MODEL_PATH=/models/at_bat_predictor.pkl \
#     -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#     -v $(pwd)/models:/models:ro \
#     mlb-optimizer:latest
#
# Environment variables consumed at runtime:
#   AT_BAT_MODEL_PATH         Path to the pickled AtBatPredictor inside the container
#   MODEL_VERSION             Human-readable version tag logged in every response
#   MLFLOW_TRACKING_URI       MLflow server for the retraining pipeline
#   MLB_S3_BUCKET             S3 bucket for Delta Lake artifact store
#   PROMETHEUS_MULTIPROC_DIR  Shared tmpfs dir for multi-worker Prometheus metrics
#   ANTHROPIC_API_KEY         Required by the RAG explainer agent (Phase 4)
#   OPENAI_API_KEY            Required by embedding client + RAG evaluator
#   PINECONE_API_KEY          Required by scouting retriever
#   CORS_ORIGINS              Comma-separated allowed origins (e.g. https://dashboard.mlb.ai)
#   OPTIMIZER_THREADS         Thread pool size for GA optimization (default: 4)
#   PORT                      Uvicorn listen port (default: 8000)
# =============================================================================

# ---- Stage 1: dependency builder ----------------------------------------
FROM python:3.11-slim AS builder

# System packages needed to compile C extensions:
#   gcc/g++   — general C compilation
#   libgomp1  — OpenMP for LightGBM parallel tree building
#   llvm-dev  — required at compile time by numba (links against LLVM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        llvm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirement files first so Docker layer caching avoids
# re-installing packages when only application code changes.
COPY requirements_api.txt ./

# Install all packages into /install (--prefix) so we can copy cleanly
# into the runtime stage without build tools.
RUN pip install --no-cache-dir --upgrade pip==24.1.2 && \
    pip install --no-cache-dir --prefix=/install \
        -r requirements_api.txt


# ---- Stage 2: runtime image ---------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="MLB Lineup Optimizer API" \
      org.opencontainers.image.description="Monte Carlo + GA lineup optimization with RAG explainability" \
      org.opencontainers.image.version="1.0.0"

# Runtime system libraries needed by C extensions at execution time:
#   libgomp1  — OpenMP runtime for LightGBM + Numba parallel loops
#   curl      — used by the HEALTHCHECK instruction below
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root user
RUN groupadd --gid 10001 mlb && \
    useradd  --uid 10001 --gid 10001 --no-create-home --shell /sbin/nologin mlb

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Application source — copy only what the API needs at runtime
WORKDIR /app
COPY src/  ./src/
COPY app/  ./app/

# Prometheus multi-process directory (shared across uvicorn workers)
# Numba JIT cache directory (needs write access; source files are read-only)
# Matplotlib config directory (avoids permission warnings)
RUN mkdir -p /tmp/prometheus_multiproc /tmp/numba_cache /tmp/matplotlib && \
    chown -R mlb:mlb /tmp/prometheus_multiproc /tmp/numba_cache /tmp/matplotlib

# Model artifacts directory (mounted as read-only volume in production)
RUN mkdir -p /models && chown mlb:mlb /models

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

# Kubernetes liveness probe: lightweight TCP check
# Kubernetes readiness probe: full /health endpoint (model loaded check)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

USER mlb
EXPOSE ${PORT}

# Production command: gunicorn manages worker processes; each worker is a
# uvicorn ASGI server. Worker count = (2 × CPU_CORES) + 1 is the rule of thumb
# for I/O-bound APIs; for CPU-bound optimization, cap at physical cores.
# Override CMD with AT_BAT_MODEL_PATH and MODEL_VERSION env vars.
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
