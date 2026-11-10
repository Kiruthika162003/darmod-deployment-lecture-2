## Multi-stage Dockerfile for Maple (Vector / DARMOD demo).
##
## Stage 1 (deps) installs Python dependencies into a wheel cache.
## Stage 2 (runtime) is a slim base that copies only the wheels and source.
## Result is ~250 MB instead of ~900 MB on a single-stage build.

# ---- Stage 1: build deps ----
FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --wheel-dir=/wheels -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# Cloud Run requires the container to listen on $PORT and to NOT need a TTY.
WORKDIR /app

# Install deps from pre-built wheels (fast, deterministic).
COPY --from=deps /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Application code last so source edits don't bust the dep layer.
COPY backend ./backend
COPY frontend ./frontend

# Drop privileges. Cloud Run is happy with any non-root UID.
RUN useradd --create-home --uid 1001 maple
USER maple

EXPOSE 8080

# Gunicorn with a uvicorn worker class. One worker is plenty for the lecture
# demo; raise --workers for real traffic. Threads handle Cloud Run's mixed
# I/O workload (Gemini calls + HTTP fetches).
CMD ["sh", "-c", "exec gunicorn backend.main:app \
      --bind :${PORT} \
      --workers 1 \
      --worker-class uvicorn.workers.UvicornWorker \
      --threads 8 \
      --timeout 120 \
      --access-logfile -"]
