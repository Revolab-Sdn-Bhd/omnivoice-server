"""
Audio encoding helpers.
All functions are pure (no side effects) and synchronous.
"""

from __future__ import annotations

# FIX: io and torchaudio were imported a second time in the middle of the file,
# after validate_audio_bytes. Moved all imports to top — single import block.
import io

import torch
import torchaudio

SAMPLE_RATE = 24_000


def normalize_loudness(
    tensor: torch.Tensor,
    target_lufs: float = -16.0,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Normalize audio tensor to target integrated LUFS via pyloudnorm."""
    import numpy as np
    import pyloudnorm as pyln

    # (1, T) or (T,) → numpy (samples, channels) for pyloudnorm
    audio = tensor.detach().cpu().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    np_audio = audio.numpy().T  # (channels, T) → (T, channels)

    meter = pyln.Meter(sample_rate)
    try:
        current_lufs = meter.integrated_loudness(np_audio)
    except ValueError:
        return tensor  # Audio too short for LUFS measurement

    if not np.isfinite(current_lufs):
        return tensor  # Silence or too short to measure

    normalized = pyln.normalize.loudness(np_audio, current_lufs, target_lufs)
    # (T, channels) → (channels, T) → tensor (1, T)
    result = torch.from_numpy(normalized.T).to(tensor.device)
    return result.to(tensor.dtype)


def tensor_to_wav_bytes(tensor: torch.Tensor) -> bytes:
    """Convert (1, T) float32 tensor to 16-bit PCM WAV bytes via stdlib wave."""
    import struct
    import wave

    cpu_tensor = tensor.cpu()
    if cpu_tensor.dim() == 1:
        cpu_tensor = cpu_tensor.unsqueeze(0)

    pcm = (cpu_tensor.squeeze(0).clamp(-1.0, 1.0) * 32767).to(torch.int16)
    raw_bytes = struct.pack(f"<{pcm.numel()}h", *pcm.tolist())

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


def _postprocess(
    tensor: torch.Tensor,
    target_lufs: float = -23.0,
) -> torch.Tensor:
    """Normalize loudness."""
    tensor = normalize_loudness(tensor, target_lufs=target_lufs)
    return tensor


def tensors_to_wav_bytes(
    tensors: list[torch.Tensor],
    target_lufs: float = -23.0,
) -> bytes:
    """
    Concatenate multiple (1, T) tensors into a single WAV.
    Applies LUFS normalization.
    """
    def _to_tensor(t):
        if isinstance(t, torch.Tensor):
            return t.cpu()
        import numpy as np
        return torch.from_numpy(t).float() if isinstance(t, np.ndarray) else torch.as_tensor(t).float()

    converted = [_to_tensor(t) for t in tensors]
    if len(converted) == 1:
        combined = converted[0]
    else:
        combined = torch.cat(converted, dim=-1)
    combined = _postprocess(combined, target_lufs=target_lufs)
    return tensor_to_wav_bytes(combined)


def tensor_to_pcm16_bytes(tensor) -> bytes:
    """
    Convert (1, T) float32 tensor to raw PCM int16 bytes.
    Used for streaming — no WAV header, continuous byte stream.
    """
    import numpy as np

    if isinstance(tensor, np.ndarray):
        flat = tensor.squeeze().astype(np.float32)
        return (flat * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    flat = tensor.squeeze(0).cpu()  # (T,)
    return (flat * 32767).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()


def _encode_pcm_via_ffmpeg(
    pcm_bytes: bytes, codec: str, fmt: str, bitrate: str | None = None
) -> bytes:
    """Encode raw PCM int16 bytes to an audio format via ffmpeg pipe."""
    import subprocess

    cmd = [
        "ffmpeg", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", "-", "-c:a", codec, "-f", fmt, "-",
    ]
    if bitrate:
        cmd.extend(["-b:a", bitrate])
    proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {codec} encoding failed: {proc.stderr.decode()[:300]}"
        )
    return proc.stdout


def tensor_to_mp3_bytes(tensor: torch.Tensor, bitrate: str = "128k") -> bytes:
    """Convert (1, T) float32 tensor to MP3 bytes via ffmpeg."""
    flat = tensor.squeeze(0).cpu()
    pcm = (flat * 32767).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()
    return _encode_pcm_via_ffmpeg(pcm, "libmp3lame", "mp3", bitrate)


def tensor_to_opus_bytes(tensor: torch.Tensor, bitrate: str = "24k") -> bytes:
    """Convert (1, T) float32 tensor to Opus OGG bytes via ffmpeg."""
    flat = tensor.squeeze(0).cpu()
    pcm = (flat * 32767).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()
    return _encode_pcm_via_ffmpeg(pcm, "libopus", "ogg", bitrate)


FORMAT_MEDIA_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
}


def encode_tensors(tensors: list[torch.Tensor], fmt: str) -> tuple[bytes, str]:
    """Encode tensor list to bytes. Returns (audio_bytes, media_type)."""
    if fmt == "wav":
        return tensors_to_wav_bytes(tensors), FORMAT_MEDIA_TYPES["wav"]
    if fmt == "pcm":
        return (
            b"".join(tensor_to_pcm16_bytes(t) for t in tensors),
            FORMAT_MEDIA_TYPES["pcm"],
        )

    # Combine tensors for compressed formats
    combined = torch.cat([t.cpu() for t in tensors], dim=-1)

    if fmt == "mp3":
        return tensor_to_mp3_bytes(combined), FORMAT_MEDIA_TYPES["mp3"]
    if fmt == "opus":
        return tensor_to_opus_bytes(combined), FORMAT_MEDIA_TYPES["opus"]

    raise ValueError(f"Unsupported audio format: {fmt}")


def read_upload_bounded(data: bytes, max_bytes: int, field_name: str = "ref_audio") -> bytes:
    """
    Validates upload size after reading.
    """
    if len(data) == 0:
        raise ValueError(f"{field_name} is empty")
    if len(data) > max_bytes:
        mb = len(data) / 1024 / 1024
        limit_mb = max_bytes / 1024 / 1024
        raise ValueError(f"{field_name} too large: {mb:.1f} MB (limit: {limit_mb:.0f} MB)")
    return data


def validate_audio_bytes(data: bytes, field_name: str = "ref_audio") -> None:
    """
    Lightweight validation: check that bytes are parseable as audio.
    Does NOT decode the full file — only reads metadata.
    Uses stdlib wave module for WAV files, falls back to torchaudio.load
    for other formats.
    """
    # Try stdlib wave first (handles WAV robustly, no torchcodec dependency)
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if frames == 0:
                raise ValueError(f"{field_name}: audio file has 0 frames")
            if rate < 8000:
                raise ValueError(f"{field_name}: sample rate {rate}Hz too low (min 8000Hz)")
            return
    except wave.Error:
        pass  # Not a WAV file, try torchaudio below

    try:
        buf = io.BytesIO(data)
        waveform, sample_rate = torchaudio.load(buf)
        if waveform.shape[-1] == 0:
            raise ValueError(f"{field_name}: audio file has 0 frames")
        if sample_rate < 8000:
            raise ValueError(f"{field_name}: sample rate {sample_rate}Hz too low (min 8000Hz)")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"{field_name}: could not parse as audio file. "
            "Supported formats: WAV, MP3, FLAC, OGG. "
            f"Original error: {e}"
        ) from e
