# Multi-stage: builder with CUDA toolkit, runtime slim
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS builder

WORKDIR /build

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY omnivoice_server/ omnivoice_server/
RUN uv sync --frozen --no-dev

# --- Runtime stage ---
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Runtime deps: ffmpeg for audio encoding, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy app source
COPY omnivoice_server/ omnivoice_server/
COPY pyproject.toml uv.lock ./

ENV PATH="/app/.venv/bin:$PATH"
ENV OMNIVOICE_LOG_LEVEL=info

RUN mkdir -p /app/profiles

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8880/live || exit 1

ENTRYPOINT ["python", "-m", "omnivoice_server"]
CMD ["--host", "0.0.0.0", "--port", "8880", "--device", "cuda", "--profile-dir", "/app/profiles"]
