FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ROOT=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/data

WORKDIR /app

RUN mkdir -p /data

COPY pyproject.toml README.md GUIDE.md .env.example ./
COPY src ./src
COPY data ./data
COPY models ./models

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-build-isolation -e . && \
    DATA_DIR="" python -c "from src.engine.recommender import ModelHub; ModelHub().train_all('all')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=4)" || exit 1

CMD ["sh", "-c", "dkmodel --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
