# Setup Guide

Complete setup from scratch for RevoVoice server and revo-norm text normalization.

## Prerequisites

- **Python 3.10+**
- **uv** — [install](https://docs.astral.sh/uv/getting-started/installation/)
- **NVIDIA GPU** with CUDA 11.8+ drivers (recommended), or CPU-only
- **Git**

## 1. Clone the repo

```bash
git clone https://github.com/khursanirevo/omnivoice-server.git
cd omnivoice-server
```

## 2. Install RevoVoice server

The install script auto-detects your GPU and installs the matching PyTorch:

```bash
bash scripts/install.sh
```

What it does:
1. Detects CUDA version from `nvidia-smi` (falls back to CPU if no GPU)
2. Runs `uv sync` to install all Python dependencies
3. Reinstalls `torch`, `torchaudio`, `torchcodec` with the correct CUDA variant
4. Verifies the installation

If you don't have `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Install revo-norm (text normalization)

revo-norm is a TTS text normalization library for English and Malay. It converts written text into spoken form (e.g., `"RM100"` → `"seratus ringgit"`, `"25C"` → `"twenty five celsius"`).

The server uses it optionally — if installed, text is normalized before synthesis; if not, raw text is passed through.

### Option A: From PyPI (recommended)

```bash
uv pip install revo-norm
```

### Option B: From source

```bash
cd tasks/synthetic-dialogue-dataset/revo-norm
uv sync --all-extras
uv pip install -e .
cd ../../..
```

### Verify

```bash
uv run python -c "from revo_norm import normalize_text; print(normalize_text('RM100 untuk 5 unit', language='ms'))"
# Should print: "seratus ringgit untuk lima unit"
```

### What revo-norm normalizes

| Input | Output | Feature |
|-------|--------|---------|
| `RM100` | `seratus ringgit` | Currency |
| `3:30 pm` | `three thirty p m` | Time |
| `15/08/2025` | `fifteenth of August 2025` | Date |
| `user@example.com` | `user at example dot com` | Email |
| `25C` | `twenty five celsius` | Temperature |
| `911111-01-1111` | digit-by-digit | IC number |
| `JSON` | `jay son` | Pronunciation |
| `ML` | `ML` (preserved) | Acronym |

Supports profiles: `minimal`, `basic`, `standard` (default), `aggressive`, `technical_doc`.

## 4. Run the server

```bash
uv run omnivoice-server
```

On first run, it downloads:
- **OmniVoice model** (~3GB) from `Revolab/omnivoice`
- **Voice audio** from `Revolab/voices` dataset

Subsequent starts use the HuggingFace cache.

Server starts at **http://127.0.0.1:8880**.

### Common CLI flags

```bash
# Bind to all interfaces on a custom port
uv run omnivoice-server --host 0.0.0.0 --port 8000

# Use a different model
uv run omnivoice-server --model your-org/your-model

# Specific model revision
uv run omnivoice-server --model-revision abc12345

# Enable debug logging
uv run omnivoice-server --log-level debug

# Fewer diffusion steps (faster, lower quality)
uv run omnivoice-server --num-step 8
```

All flags can also be set via environment variables with `OMNIVOICE_` prefix:

```bash
OMNIVOICE_PORT=8000 OMNIVOICE_LOG_LEVEL=debug uv run omnivoice-server
```

## 5. Verify it works

```bash
# Health check
curl http://localhost:8880/health

# Generate speech
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "revovoice", "input": "Hello world.", "voice": "anwar"}' \
  --output speech.wav

# List available voices
curl http://localhost:8880/voices

# Test text normalization (requires revo-norm)
curl -X POST http://localhost:8880/api/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "RM1,500 pada 18 Feb", "language": "ms"}'
```

Or open **http://localhost:8880/** in your browser for the web UI.

## Configuration

Key settings (env vars with `OMNIVOICE_` prefix, or CLI flags):

| Setting | Default | Description |
|---------|---------|-------------|
| `OMNIVOICE_HOST` | `127.0.0.1` | Bind host |
| `OMNIVOICE_PORT` | `8880` | Port |
| `OMNIVOICE_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `OMNIVOICE_MODEL_ID` | `Revolab/omnivoice` | HuggingFace model repo |
| `OMNIVOICE_NUM_STEP` | `16` | Diffusion steps (higher = better quality, slower) |
| `OMNIVOICE_MAX_CONCURRENT` | `2` | Parallel inference slots |
| `OMNIVOICE_VOICES_HF_REPO` | `Revolab/voices` | HuggingFace voices dataset |

Full list in [README.md](README.md).

## Running with systemd (production)

```bash
sudo cp omnivoice-server.service /etc/systemd/system/
# Edit User, Group, and ExecStart path in the service file
sudo systemctl daemon-reload
sudo systemctl enable --now omnivoice-server
sudo journalctl -u omnivoice-server -f
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check omnivoice_server/

# Run tests
uv run pytest

# Type checking
uv run mypy omnivoice_server/
```

### revo-norm development

```bash
cd tasks/synthetic-dialogue-dataset/revo-norm
uv sync --all-extras
uv run pytest                    # 149 tests
uv run ruff check revo_norm/     # lint
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA not detected | Run `bash scripts/install.sh` — re-detects and reinstalls correct torch |
| `ModuleNotFoundError: omnivoice` | Run `uv sync` |
| `ModuleNotFoundError: revo_norm` | Run `uv pip install revo-norm` (optional, text normalization) |
| CUDA OOM | Lower `--max-concurrent` or use `--device cpu` |
| First request is slow | Model warmup. Use `--compile-cache-dir` to persist compiled kernels |
| Download fails (firewall) | Set `HF_HUB_OFFLINE=1` if model is already cached, or configure `HTTPS_PROXY` |
| Permission error on HF cache | Check write access to `~/.cache/huggingface/` or `HF_HOME` |
| Browser shows old UI | Hard-refresh (Ctrl+Shift+R). Server sends no-cache headers. |
