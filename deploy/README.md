# Deploy OmniVoice TTS Server to Vast.ai Serverless

## Overview

Deploy the OmniVoice TTS server as a Vast.ai Serverless endpoint. The PyWorker (`worker.py`) starts the FastAPI server as a subprocess and proxies requests through Vast's routing layer.

## Prerequisites

- [Vast.ai CLI](https://cloud.vast.ai/) account with API key
- Docker (for building and pushing image)
- GitHub Container Registry (GHCR) access

## 1. Build & Push Docker Image

```bash
# Login to GHCR (one-time)
echo $GITHUB_TOKEN | docker login ghcr.io -u khursanirevo --password-stdin

# From project root
docker build -t ghcr.io/khursanirevo/omnivoice-server:latest -f deploy/Dockerfile .

# Push to GHCR
docker push ghcr.io/khursanirevo/omnivoice-server:latest
```

## 2. Create Endpoint

```bash
vastai create endpoint \
  --name "omnivoice-tts" \
  --type "custom"
```

## 3. Create Workergroup

```bash
vastai create workergroup \
  --endpoint-id <ENDPOINT_ID> \
  --image "your-registry/omnivoice-server:latest" \
  --gpu-count 1 \
  --min-workers 0 \
  --max-workers 4 \
  --target-utilization 70 \
  --env "OMNIVOICE_MODEL_ID=Revolab/omnivoice" \
  --env "OMNIVOICE_MAX_CONCURRENT=4" \
  --env "OMNIVOICE_DEVICE=cuda" \
  --env "OMNIVOICE_API_KEY=your-secret-key" \
  --ports "8880/http"
```

### GPU Requirements

- **Minimum**: 16 GB VRAM (e.g., RTX 4090, A10G)
- **Recommended**: 24 GB VRAM (e.g., A5000, L40S, A100)
- The model loads in FP16 and uses ~10-12 GB VRAM at rest

## 4. Test the Endpoint

```bash
ENDPOINT_URL="https://<your-endpoint>.vast.ai"

# Health check
curl "$ENDPOINT_URL/health"

# List available voices
curl "$ENDPOINT_URL/v1/audio/voices"

# Generate speech (OpenAI-compatible)
curl "$ENDPOINT_URL/v1/audio/speech" \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello, this is a test of the TTS server.",
    "voice": "anwar",
    "response_format": "mp3"
  }' \
  --output test.mp3
```

## 5. Available Voices

| Voice ID | Gender | Description |
|----------|--------|-------------|
| anwar | Male | Standard male |
| Farid_595 | Male | Male variant |
| hendrick-6 | Male | Male variant |
| wafiy_5 | Male | Male variant |
| Dania_0 | Female | Standard female |
| Pearl_Happy | Female | Happy tone |
| NorinaYahya | Female | Female variant |
| SitiNora | Female | Female variant |
| Aisyah | Female | Female variant |
| Sofea | Female | Female variant |
| Pearl_Angry1 | Female | Angry tone |
| Pearl_Angry2 | Female | Angry variant |
| Pearl_Sad | Female | Sad tone |
| Pearl_Surprise | Female | Surprise tone |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS |
| `/v1/audio/voices` | GET | List available voices |
| `/v1/models` | GET | List models |
| `/generate` | POST | SepBox-compatible TTS |
| `/voices` | GET | Voice management |

## Autoscaling

The PyWorker configures benchmark-driven autoscaling via `vastai`:
- **Workload metric**: Input text length (characters)
- **Benchmark**: Runs 3 iterations at concurrency 4 on cold start
- **Queue timeout**: 300s max wait time
- Vast scales workers based on measured throughput vs target utilization

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIVOICE_MODEL_ID` | `Revolab/omnivoice` | HuggingFace model ID |
| `OMNIVOICE_DEVICE` | `cuda` | Inference device |
| `OMNIVOICE_MAX_CONCURRENT` | `4` | Max parallel inferences |
| `OMNIVOICE_API_KEY` | (empty) | Auth key. Empty = no auth |
| `OMNIVOICE_LOG_LEVEL` | `info` | Logging level |
| `OMNIVOICE_PORT` | `8880` | Server port |
| `OMNIVOICE_NUM_STEP` | `16` | Diffusion steps (8-32) |
| `OMNIVOICE_GUIDANCE_SCALE` | `2.0` | CFG scale |
| `OMNIVOICE_BATCH_ENABLED` | `false` | Enable request batching |

## Troubleshooting

**Server won't start**: Check `MODEL_LOG_FILE=/tmp/omnivoice-server.log` in the container.

**CUDA OOM**: Reduce `OMNIVOICE_MAX_CONCURRENT` to 1-2, or use a GPU with more VRAM.

**Slow cold start**: Model download + warmup takes 2-5 minutes on first run. Subsequent starts use HuggingFace cache.
