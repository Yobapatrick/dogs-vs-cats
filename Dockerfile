# --- Build stage --------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps required by Pillow and OpenCV-like operations
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Runtime stage ------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed wheels from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

COPY --chown=appuser:appuser . .

# Default: open a shell. Override with `docker run ... train.py --config ...`
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--config", "configs/cnn.yaml"]
