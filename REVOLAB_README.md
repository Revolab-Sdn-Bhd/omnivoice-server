# OmniVoice Server — Deployment Guide

OpenAI-compatible TTS server wrapping [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Quick Reference

| Item | Value |
|------|-------|
| Default port | 8880 |
| Health check | `GET /live` (process alive), `GET /ready` (model loaded), `GET /health` (full status) |
| Metrics | `GET /metrics/prometheus` (Prometheus text format) |
| Auth | Bearer token via `OMNIVOICE_API_KEY` env var (empty = no auth) |
| Model download | ~3GB from HuggingFace on first startup |
| Min hardware | 8GB RAM (CPU), 4GB VRAM (GPU) |

## Using Custom Checkpoints

Point to your own fine-tuned weights on HuggingFace:

```bash
# Bare metal
uv run omnivoice-server --model-id Revolab/omnivoice

# Docker
OMNIVOICE_MODEL_ID=Revolab/omnivoice docker compose up -d

# Or set in docker-compose.yml / env file
```

The server will download from the specified repo on first startup. Subsequent starts use the HuggingFace cache. To use a local path instead:

```bash
uv run omnivoice-server --model-id /data/models/omnivoice-checkpoint
```

## Option 1: Docker (recommended)

```bash
# Clone
git clone https://github.com/khursanirevo/omnivoice-server.git
cd omnivoice-server

# Start (GPU)
docker compose up -d

# Start (CPU only — change OMNIVOICE_DEVICE in docker-compose.yml)
```

The `docker-compose.yml` is ready to go. Key env vars to tune:

```yaml
environment:
  - OMNIVOICE_HOST=0.0.0.0
  - OMNIVOICE_PORT=8880
  - OMNIVOICE_DEVICE=cuda        # or "cpu"
  - OMNIVOICE_NUM_STEP=16        # 1-64, higher=better quality, slower
  - OMNIVOICE_MAX_CONCURRENT=2   # parallel inference threads
  - OMNIVOICE_API_KEY=           # optional auth
  - OMNIVOICE_LOG_LEVEL=info
```

GPU requires the NVIDIA runtime + `nvidia-container-toolkit` installed.

## Option 2: Bare metal

**Use `uv` — not pip.** All commands below use `uv`.

```bash
# Prerequisites: Python 3.10+, uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone
git clone https://github.com/khursanirevo/omnivoice-server.git
cd omnivoice-server

# Install PyTorch with CUDA first (required before project install)
uv pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Sync project dependencies
uv sync

# Run
uv run omnivoice-server --host 0.0.0.0 --port 8880 --device cuda

# Or use the module directly
uv run python -m omnivoice_server --host 0.0.0.0 --port 8880 --device cuda
```

### Systemd

The service file uses `uv run` to launch the server.

```bash
# Copy service file
sudo cp omnivoice-server.service /etc/systemd/system/

# Create env file if needed
sudo mkdir -p /etc/omnivoice
sudo tee /etc/omnivoice/env <<EOF
OMNIVOICE_API_KEY=your-key-here
OMNIVOICE_NUM_STEP=16
EOF

# Edit paths in the service file to match your install location
# The ExecStart should use: uv run python -m omnivoice_server --host 0.0.0.0 --port 8880 --device cuda

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now omnivoice-server
sudo journalctl -u omnivoice-server -f   # view logs
```

## API Endpoints

### Generate speech

```bash
curl -X POST http://HOST:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "omnivoice",
    "input": "Hello, this is a test.",
    "voice": "alloy"
  }' \
  --output speech.wav
```

### Streaming

Add `"stream": true` for chunked PCM output (lower perceived latency).

### Voice cloning

```bash
curl -X POST http://HOST:8880/v1/audio/speech/clone \
  -F "text=Say this in the cloned voice" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=What was said in the reference" \
  --output cloned.wav
```

### Health checks

```bash
curl http://HOST:8880/live    # 200 = process alive
curl http://HOST:8880/ready   # 200 = model loaded, 503 = still loading
curl http://HOST:8880/health  # JSON status with cache/queue stats
```

## Configuration Reference

All env vars use `OMNIVOICE_` prefix. CLI flags override env vars.

| Env Var | Default | Description |
|---------|---------|-------------|
| `OMNIVOICE_HOST` | `127.0.0.1` | Bind host (`0.0.0.0` in Docker) |
| `OMNIVOICE_PORT` | `8880` | Bind port |
| `OMNIVOICE_DEVICE` | `cpu` | `cpu`, `cuda` |
| `OMNIVOICE_NUM_STEP` | `16` | Inference steps (1-64). 16=fast, 32=quality |
| `OMNIVOICE_MAX_CONCURRENT` | `2` | Max parallel inference calls |
| `OMNIVOICE_API_KEY` | `""` | Bearer token. Empty = no auth |
| `OMNIVOICE_MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace repo or local path. Use your custom checkpoint: `Revolab/omnivoice` |
| `OMNIVOICE_PROFILE_DIR` | `~/.omnivoice/profiles` | Voice profiles directory |
| `OMNIVOICE_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `OMNIVOICE_REQUEST_TIMEOUT_S` | `120` | Max seconds per request |
| `OMNIVOICE_LOUDNESS_TARGET_LUFS` | `-16.0` | Output loudness. `null` to disable |
| `OMNIVOICE_RESPONSE_CACHE_ENABLED` | `true` | Cache repeated identical requests |
| `OMNIVOICE_RESPONSE_CACHE_MAX_GB` | `5.0` | Max disk for response cache |
| `OMNIVOICE_MAX_QUEUE_DEPTH` | `64` | Max pending requests. 503 when exceeded |
| `OMNIVOICE_WORKERS` | `1` | Worker processes (opt-in multi-worker) |
| `OMNIVOICE_BATCH_ENABLED` | `false` | Request batching (experimental) |
| `OMNIVOICE_BATCH_MAX_SIZE` | `4` | Max requests per batch |
| `OMNIVOICE_BATCH_TIMEOUT_MS` | `50` | Max wait before processing batch |
| `OMNIVOICE_COMPILE_MODE` | `none` | `none`, `default`, `reduce-overhead`, `max-autotune` |
| `OMNIVOICE_COMPILE_CACHE_DIR` | `null` | Persistent torch.compile cache dir |

## Performance Tuning

### GPU (production)

Best config: `torch.compile(mode="max-autotune")` on CUDA FP16.

```bash
uv run omnivoice-server --device cuda --compile-mode max-autotune --num-step 16
```

Verified on H200: **99ms at step=16, 41ms at step=4**.

### CPU (dev/testing)

Best config: OpenVINO + torchao int8dq.

```bash
uv pip install openvino torchao
uv run omnivoice-server --device cpu --quantization int8dq --compile-mode default
```

Note: OpenVINO backend is used automatically when `--compile-mode` is set and `openvino` is installed (it registers as a `torch.compile` backend).

### step count guide

| `num_step` | GPU (H200) | CPU (Xeon) | Quality |
|-----------|------------|------------|---------|
| 4 | 41ms | 0.98s | Acceptable |
| 8 | 61ms | 1.66s | Good |
| 16 | 99ms | 4.43s | High |
| 32 | ~200ms | ~9s | Best |

## Audio Formats

| Format | Response param | Content-Type |
|--------|---------------|--------------|
| WAV (default) | `wav` | `audio/wav` |
| PCM (streaming) | `pcm` | `audio/pcm` |
| MP3 | `mp3` | `audio/mpeg` |
| Opus | `opus` | `audio/ogg; codecs=opus` |

MP3/Opus require `ffmpeg` installed on the server.

## Voice Presets

OpenAI-compatible preset names: `alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`.

Custom voices via `instructions` field: `"female, british accent, young adult, high pitch"`.

## Web UI

The server includes a built-in frontend at `http://HOST:8880/`. It provides:

- Voice presets (alloy, nova, echo, etc.) or custom instructions
- Voice cloning from uploaded reference audio
- Text input with quality/speed/format controls
- Built-in audio player with download

No extra setup needed — served automatically by the FastAPI server.

## Monitoring

- **Prometheus**: `GET /metrics/prometheus` — counters, gauges, histograms
- **Health**: `GET /health` — JSON with model status, cache stats, queue depth
- **Liveness**: `GET /live` — always 200 if process is up
- **Readiness**: `GET /ready` — 200 when model loaded, 503 during startup

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model download fails | Set `HF_HUB_CACHE` to a writable dir, or pre-download with `uv run python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"` |
| CUDA OOM | Reduce `--max-concurrent 1` or use `--device cpu` |
| First request slow | Model compilation. Use `--compile-cache-dir` with precompiled kernels |
| 503 Queue Full | Increase `--max-queue-depth` or add GPU workers |
| Auth failing | Check `OMNIVOICE_API_KEY` matches `Authorization: Bearer <key>` header |

## Architecture

```
Client (OpenAI SDK / HTTP)
  │
  ▼
FastAPI (uvicorn)
  ├── Auth middleware (optional Bearer token)
  ├── Request ID middleware (X-Request-ID)
  ├── Response cache (disk LRU)
  ├── Request deduplication
  └── Backpressure (queue depth limit)
       │
       ▼
  InferenceService
    ├── ThreadPoolExecutor → OmniVoice model
    │     └── Qwen3-0.6B LLM + HiggsAudioV2 codec
    ├── torch.compile (optional)
    ├── TorchAO quantization (optional)
    └── Batching (experimental)
```
