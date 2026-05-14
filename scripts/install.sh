#!/usr/bin/env bash
# Install omnivoice-server with the correct PyTorch CUDA variant for this machine.
# Usage: ./scripts/install.sh
set -euo pipefail

# ── Check system dependencies ────────────────────────────────────────────────
check_system_deps() {
    local missing=()

    if ! command -v python3 &>/dev/null; then
        missing+=("python3 (3.10+)")
    else
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
            missing+=("python3 >= 3.10 (found $PY_VER)")
        fi
    fi

    if ! command -v uv &>/dev/null; then
        missing+=("uv (install: https://docs.astral.sh/uv/getting-started/installation/)")
    fi

    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "✗ Missing system dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install them first, then re-run this script."
        exit 1
    fi
}

# ── Check/install FFmpeg (needed by torchcodec) ──────────────────────────────
check_ffmpeg() {
    if command -v ffmpeg &>/dev/null; then
        echo "→ FFmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
        return
    fi

    echo "→ FFmpeg not found. Attempting to install..."
    if command -v apt-get &>/dev/null; then
        echo "  Installing via apt..."
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v dnf &>/dev/null; then
        echo "  Installing via dnf..."
        sudo dnf install -y ffmpeg
    elif command -v brew &>/dev/null; then
        echo "  Installing via brew..."
        brew install ffmpeg
    else
        echo "⚠ FFmpeg not found and cannot auto-install."
        echo "  Install it manually: https://ffmpeg.org/download.html"
        echo "  torchcodec (audio decoding) will not work without it."
    fi
}

# ── Detect CUDA version from driver ──────────────────────────────────────────
detect_cuda() {
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+") || true
    fi

    if [ -z "${CUDA_VER:-}" ]; then
        TORCH_INDEX="cpu"
        echo "→ No NVIDIA GPU detected, installing CPU build"
        return
    fi

    MAJOR="${CUDA_VER%%.*}"
    MINOR="${CUDA_VER#*.}"; MINOR="${MINOR%%.*}"

    # Map driver CUDA version to best PyTorch CUDA index
    # PyTorch releases lag behind CUDA — use latest supported index
    if   [ "$MAJOR" -ge 13 ];                        then TORCH_INDEX="cu126"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 6 ]; then TORCH_INDEX="cu126"
    elif [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 4 ]; then TORCH_INDEX="cu124"
    elif [ "$MAJOR" -eq 12 ];                        then TORCH_INDEX="cu121"
    elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then TORCH_INDEX="cu118"
    else
        echo "→ CUDA $CUDA_VER too old for a supported PyTorch build, falling back to CPU"
        TORCH_INDEX="cpu"
    fi

    if [ "$MAJOR" -ge 13 ]; then
        echo "→ Detected CUDA $CUDA_VER → PyTorch $TORCH_INDEX (cu130 not yet stable, using cu126)"
    else
        echo "→ Detected CUDA $CUDA_VER → PyTorch $TORCH_INDEX"
    fi
}

# ── Main install ─────────────────────────────────────────────────────────────
echo "=== RevoVoice Server Setup ==="
echo ""

check_system_deps
check_ffmpeg
detect_cuda

# Install all non-torch deps
echo "→ Running uv sync..."
uv sync

# Reinstall torch stack from the correct CUDA index
if [ "$TORCH_INDEX" != "cpu" ]; then
    PYTORCH_URL="https://download.pytorch.org/whl/$TORCH_INDEX"
    echo "→ Installing torch+$TORCH_INDEX from $PYTORCH_URL"

    # Pin torch to a known-good version range to avoid bleeding-edge breakage
    uv pip install \
        "torch>=2.5.0,<2.12" \
        "torchaudio>=2.5.0" \
        "torchcodec>=0.1.0" \
        --index-url "$PYTORCH_URL" \
        --reinstall-package torch \
        --reinstall-package torchaudio \
        --reinstall-package torchcodec
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo "→ Verifying installation..."
uv run python -c "
import torch
print(f'   torch:          {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU:            {torch.cuda.get_device_name(0)}')
    print(f'   Device count:   {torch.cuda.device_count()}')

try:
    import torchcodec
    print(f'   torchcodec:     {torchcodec.__version__}')
except Exception as e:
    print(f'   torchcodec:     FAILED - {e}')

try:
    import torchaudio
    print(f'   torchaudio:     {torchaudio.__version__}')
except Exception as e:
    print(f'   torchaudio:     FAILED - {e}')
"

echo ""
echo "✓ Install complete"
echo ""
echo "Next steps:"
echo "  uv run omnivoice-server              # start server"
echo "  uv run omnivoice-server --help       # see all options"
echo "  See SETUP.md for full documentation"
