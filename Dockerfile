# ============================================================================
# Dockerfile — AumOS LLM Serving Service
# Multi-stage build: builder → runtime
# ============================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --prefix=/install --no-warn-script-location .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Security: non-root user
RUN groupadd -r aumos && useradd -r -g aumos -d /app -s /sbin/nologin aumos

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/

WORKDIR /app

# Set ownership
RUN chown -R aumos:aumos /app

USER aumos

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/live'); r.raise_for_status()" || exit 1

# Start service
CMD ["uvicorn", "aumos_llm_serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
