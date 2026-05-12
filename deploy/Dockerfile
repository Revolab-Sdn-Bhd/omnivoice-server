# OmniVoice TTS Server — Vast.ai Serverless
# CUDA 12.4 + Python 3.11, installs omnivoice-server from PyPI.
#
# Build:
#   docker build -t omnivoice-server:latest deploy/
#
# Run locally:
#   docker run --gpus all -p 8880:8880 omnivoice-server:latest
#
# Push to registry for Vast.ai:
#   docker tag omnivoice-server:latest your-registry/omnivoice-server:latest
#   docker push your-registry/omnivoice-server:latest

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

WORKDIR /app

# Copy project files for installation
COPY pyproject.toml uv.lock README.md ./

# Install omnivoice-server and its dependencies via uv
RUN uv pip install --system --no-dev-dependencies -e ".[benchmark]" 2>&1 || \
    uv pip install --system -e ".[benchmark]"

# Copy source code and voice profiles
COPY omnivoice_server/ omnivoice_server/

# Copy PyWorker
COPY deploy/worker.py /app/worker.py

# Install Vast.ai PyWorker SDK
RUN uv pip install --system vastai 2>&1 || pip install vastai

# Environment defaults for serverless
ENV OMNIVOICE_HOST=0.0.0.0 \
    OMNIVOICE_PORT=8880 \
    OMNIVOICE_DEVICE=cuda \
    OMNIVOICE_MAX_CONCURRENT=4 \
    OMNIVOICE_LOG_LEVEL=info \
    OMNIVOICE_RESPONSE_CACHE_ENABLED=false

# PyWorker launches the model server and proxies requests
EXPOSE 8880

CMD ["python", "/app/worker.py"]
