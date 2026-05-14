# RevoVoice Server

OpenAI-compatible HTTP server for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS with voice cloning, streaming, and a built-in web UI.

## Setup

```bash
git clone https://github.com/khursanirevo/omnivoice-server.git
cd omnivoice-server
bash scripts/install.sh        # auto-detects GPU driver, installs matching PyTorch
uv run omnivoice-server        # downloads Revolab/omnivoice (~3GB) on first run
```

Server starts at `http://127.0.0.1:8880`. To use a different checkpoint:

```bash
uv run omnivoice-server --model your-org/your-model
```

## Web UI

Open `http://127.0.0.1:8880/` in your browser. The UI lets you:
- Browse and search speakers with audio preview
- Generate speech with duration control and speed modes
- Stream audio in real-time via WebSocket
- View latency metrics and model/voices revision info

## API

### OpenAI-compatible TTS

```bash
# List available voices
curl http://localhost:8880/v1/audio/voices

# Generate speech
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "revovoice", "input": "Hello world.", "voice": "anwar"}' \
  --output speech.wav

# Stream via SSE
curl -X POST "http://localhost:8880/v1/audio/speech?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"model": "revovoice", "input": "Hello world.", "voice": "anwar"}'
```

### SepBox-compatible TTS

```bash
# Full synthesis → WAV
curl -X POST http://localhost:8880/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world.",
    "voice_ref_path": "anwar",
    "language": "en",
    "num_step": 10,
    "duration": 3.5
  }' \
  --output speech.wav

# SSE streaming
curl -X POST http://localhost:8880/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world.", "voice_ref_path": "anwar"}'
```

### Voice cloning

Save a speaker profile (reusable, survives restarts):

```bash
curl -X POST http://localhost:8880/v1/voices/profiles \
  -F "profile_id=my_speaker" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Exact transcript of the reference audio"
```

Use it:

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "revovoice", "input": "Hello.", "voice": "my_speaker"}' \
  --output out.wav
```

### WebSocket streaming (Cartesia-compatible)

Connect to `ws://localhost:8880/tts/websocket` and send JSON messages:

```json
{"voice": {"mode": "id", "id": "anwar"}, "language": "en", "context_id": "ctx-1", "transcript": "Text to speak.", "continue": true}
```

Cancel:

```json
{"cancel": true, "context_id": "ctx-1"}
```

Audio is returned as base64-encoded int16 PCM at 24kHz in `audio_chunk` messages, followed by a `done` message.

### Voices management

Voices are synced from a HuggingFace dataset (`Revolab/voices` by default).

```bash
# List all voices with metadata
curl http://localhost:8880/voices

# List as speakers (legacy alias)
curl http://localhost:8880/speakers

# Refresh from HuggingFace
curl -X POST http://localhost:8880/voices/refresh

# Get voice audio file
curl http://localhost:8880/voices/anwar/audio --output anwar.wav

# Upload a new voice
curl -X POST http://localhost:8880/voices \
  -F "voice_name=my_voice" \
  -F "ref_text=Transcript of the audio" \
  -F "audio_file=@voice.wav"
```

### Utilities

```bash
# Get test quotes for TTS evaluation
curl http://localhost:8880/api/quotes

# Normalize text for TTS
curl -X POST http://localhost:8880/api/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "RM1,500 pada 18 Feb", "language": "ms"}'

# Estimate audio duration
curl -X POST http://localhost:8880/api/estimate-duration \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world.", "voice_id": "anwar"}'

# Health check (includes model/voices revision hashes)
curl http://localhost:8880/health
```

## Configuration

All settings via env vars (`OMNIVOICE_` prefix) or CLI flags.

| Env Var | Default | Description |
|---------|---------|-------------|
| `OMNIVOICE_HOST` | `127.0.0.1` | Bind host |
| `OMNIVOICE_PORT` | `8880` | Port |
| `OMNIVOICE_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `OMNIVOICE_MODEL_ID` | `Revolab/omnivoice` | HuggingFace repo or local path |
| `OMNIVOICE_MODEL_REVISION` | `""` | Git revision to load from HuggingFace |
| `OMNIVOICE_NUM_STEP` | `16` | Diffusion steps (1–64) |
| `OMNIVOICE_MAX_CONCURRENT` | `2` | Parallel inference slots |
| `OMNIVOICE_WORKERS` | `1` | Worker processes (multi-GPU) |
| `OMNIVOICE_API_KEY` | `""` | Bearer token (empty = no auth) |
| `OMNIVOICE_PROFILE_DIR` | `~/.local/share/omnivoice/profiles` | Speaker profiles directory |
| `OMNIVOICE_VOICES_HF_REPO` | `Revolab/voices` | HuggingFace dataset for voice audio |
| `OMNIVOICE_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `OMNIVOICE_COMPILE_MODE` | `none` | `none`, `default`, `reduce-overhead`, `max-autotune` |
| `OMNIVOICE_COMPILE_CACHE_DIR` | `null` | Persist compiled kernels across restarts |
| `OMNIVOICE_QUANTIZATION` | `none` | `fp8wo`, `fp8dq`, `int8wo`, `int8dq` |
| `OMNIVOICE_REQUEST_TIMEOUT_S` | `120` | Max seconds per request before 504 |
| `OMNIVOICE_RESPONSE_CACHE_ENABLED` | `true` | Cache repeated identical requests |
| `OMNIVOICE_RESPONSE_CACHE_MAX_GB` | `5.0` | Max disk for response cache |
| `OMNIVOICE_BATCH_ENABLED` | `false` | Enable request batching |
| `OMNIVOICE_BATCH_MAX_SIZE` | `4` | Max requests per batch |
| `OMNIVOICE_BATCH_TIMEOUT_MS` | `50` | Max wait before processing partial batch |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| **OpenAI-compatible** | | |
| `POST` | `/v1/audio/speech` | TTS (supports `?stream=true` for SSE) |
| `GET` | `/v1/audio/voices` | List voices |
| `GET` | `/v1/models` | List models |
| **SepBox TTS** | | |
| `POST` | `/generate` | Full TTS → WAV |
| `POST` | `/generate/stream` | SSE streaming TTS |
| `GET` | `/api/quotes` | Test sentences for TTS evaluation |
| `POST` | `/api/normalize` | Text normalization for TTS |
| `POST` | `/api/estimate-duration` | Estimate audio duration |
| **Voices** | | |
| `GET` | `/voices` | List voices with metadata |
| `GET` | `/speakers` | List speakers (legacy alias) |
| `POST` | `/voices/refresh` | Refresh from HuggingFace |
| `GET` | `/voices/{id}/audio` | Get voice audio file |
| `POST` | `/voices` | Upload a new voice |
| **Voice profiles** | | |
| `POST` | `/v1/voices/profiles` | Save speaker profile |
| `GET` | `/v1/voices/profiles/{id}` | Get profile |
| `PATCH` | `/v1/voices/profiles/{id}` | Update profile |
| `DELETE` | `/v1/voices/profiles/{id}` | Delete profile |
| **WebSocket** | | |
| `WS` | `/tts/websocket` | Cartesia-compatible streaming TTS |
| **Other** | | |
| `GET` | `/health` | Health check with model/voices revision |
| `GET` | `/` | Web UI |

## Performance

For production: `--compile-mode max-autotune` compiles Triton kernels on first boot (~2 min). Persist them with `--compile-cache-dir ./torch_compile_cache` so subsequent restarts load instantly.

Quantization (`--quantization fp8wo` or `int8wo`) reduces VRAM usage at minimal quality loss on FP8-capable GPUs.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA not detected | Run `bash scripts/install.sh` — re-detects and reinstalls correct torch variant |
| CUDA OOM | Lower `--max-concurrent` or use `--device cpu` |
| First request slow | Kernel compilation on first run. Use `--compile-cache-dir` to persist |
| 503 Queue Full | Raise `--max-queue-depth` or add inference slots |
| Auth failing | Check `OMNIVOICE_API_KEY` matches `Authorization: Bearer <key>` |
| Browser shows old UI | Hard-refresh (Ctrl+Shift+R). Server sends `no-cache` headers. |

## Development

```bash
bash scripts/install.sh     # install runtime deps
uv sync --extra dev         # add dev deps (pytest, ruff, mypy)
uv run pytest
uv run ruff check omnivoice_server/
```

## License

MIT — built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by k2-fsa.
