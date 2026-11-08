FROM python:3.10-slim-bookworm AS base

# Flush logs immediately and avoid generating bytecode files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# CTranslate2 requires the OpenMP runtime on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying source code to improve layer caching.
COPY requirements-lock.txt requirements.txt requirements-dev.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m pip check

# Create an unprivileged application user and a writable model cache.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache


FROM base AS test

# Test tools are installed only in the test image.
RUN python -m pip install -r requirements-dev.txt \
    && python -m pip check

COPY --chown=app:app *.py ./
COPY --chown=app:app static/ ./static/
COPY --chown=app:app tests/ ./tests/

USER app

# These tests replace model inference and require no model download.
CMD ["python", "-m", "pytest", "-v", "-p", "no:cacheprovider"]


FROM base AS runtime

# Copy only application modules and interface assets.
COPY --chown=app:app *.py ./
COPY --chown=app:app static/ ./static/

USER app

EXPOSE 8000

# One worker preserves the intended single-model, single-processing policy.
# Bind to all container interfaces so Docker's port mapping can reach the app.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]