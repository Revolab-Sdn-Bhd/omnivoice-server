"""
Tests for audio utilities.
"""

from __future__ import annotations

import io
import struct

import pytest
import torch
import torchaudio

from omnivoice_server.utils.audio import (
    read_upload_bounded,
    tensor_to_pcm16_bytes,
    tensor_to_wav_bytes,
    tensors_to_wav_bytes,
    validate_audio_bytes,
)


def test_tensor_to_wav_bytes():
    """Convert tensor to WAV bytes with RIFF header."""
    tensor = torch.randn(1, 24000)  # 1 second at 24kHz
    wav_bytes = tensor_to_wav_bytes(tensor)

    # Check WAV magic bytes
    assert wav_bytes[:4] == b"RIFF"
    assert b"WAVE" in wav_bytes[:12]

    # Verify it's parseable
    buf = io.BytesIO(wav_bytes)
    waveform, sample_rate = torchaudio.load(buf)
    assert sample_rate == 24000
    assert waveform.shape[0] == 1


def test_tensors_to_wav_bytes_single():
    """Single tensor should produce valid WAV with post-processing."""
    tensor = torch.randn(1, 24000)
    wav_bytes = tensors_to_wav_bytes([tensor])

    assert wav_bytes[:4] == b"RIFF"
    buf = io.BytesIO(wav_bytes)
    waveform, sample_rate = torchaudio.load(buf)
    assert sample_rate == 24000
    # Post-processing trims 0.5s (12000 samples) from front
    assert waveform.shape[1] == 12000


def test_tensors_to_wav_bytes_multiple():
    """Multiple tensors should be concatenated, then trimmed."""
    t1 = torch.randn(1, 12000)
    t2 = torch.randn(1, 12000)
    wav_bytes = tensors_to_wav_bytes([t1, t2])

    buf = io.BytesIO(wav_bytes)
    waveform, sample_rate = torchaudio.load(buf)
    # 12000 + 12000 - 12000 (0.5s trim) = 12000
    assert waveform.shape[1] == 12000


def test_tensor_to_pcm16_bytes():
    """Convert tensor to raw PCM int16 bytes (no WAV header)."""
    tensor = torch.randn(1, 100)
    pcm_bytes = tensor_to_pcm16_bytes(tensor)

    # Should be 2 bytes per sample (int16)
    assert len(pcm_bytes) == 100 * 2

    # Should NOT have WAV header
    assert pcm_bytes[:4] != b"RIFF"


def test_read_upload_bounded_valid():
    """Valid upload within size limit should pass."""
    data = b"x" * 1000
    result = read_upload_bounded(data, max_bytes=2000)
    assert result == data


def test_read_upload_bounded_empty():
    """Empty upload should raise ValueError."""
    with pytest.raises(ValueError, match="is empty"):
        read_upload_bounded(b"", max_bytes=1000)


def test_read_upload_bounded_too_large():
    """Upload exceeding size limit should raise ValueError."""
    data = b"x" * 3000
    with pytest.raises(ValueError, match="too large"):
        read_upload_bounded(data, max_bytes=2000)


def test_validate_audio_bytes_valid_wav():
    """Valid WAV bytes should pass validation."""
    # Build minimal valid WAV without torchaudio (torchcodec breaks on BytesIO)
    num_frames = 1000
    data_size = num_frames * 2  # 16-bit mono
    audio_bytes = (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
        + b"\x00" * data_size
    )

    # Should not raise
    validate_audio_bytes(audio_bytes)


def test_validate_audio_bytes_invalid_format():
    """Invalid audio format should raise ValueError."""
    invalid_bytes = b"This is not audio data"

    with pytest.raises(ValueError, match="could not parse as audio file"):
        validate_audio_bytes(invalid_bytes)


def test_validate_audio_bytes_empty_audio():
    """Audio file with 0 frames should raise ValueError."""
    # Create WAV with 0 samples
    tensor = torch.randn(1, 0)
    buf = io.BytesIO()

    try:
        torchaudio.save(buf, tensor, 24000, format="wav")
    except RuntimeError:
        # Some torchaudio versions don't support saving empty tensors
        pytest.skip("torchaudio version doesn't support empty tensor save")

    buf.seek(0)
    audio_bytes = buf.read()

    # Different PyTorch versions return different error messages
    with pytest.raises(ValueError, match="has 0 frames|could not parse"):
        validate_audio_bytes(audio_bytes)


def test_validate_audio_bytes_low_sample_rate():
    """Audio with sample rate below 8000Hz should raise ValueError."""
    # Build WAV at 4000Hz without torchaudio (torchcodec breaks on BytesIO)
    num_frames = 1000
    data_size = num_frames * 2
    audio_bytes = (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, 1, 4000, 8000, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
        + b"\x00" * data_size
    )

    with pytest.raises(ValueError, match="sample rate.*too low"):
        validate_audio_bytes(audio_bytes)


def test_validate_audio_bytes_custom_field_name():
    """Custom field name should appear in error messages."""
    invalid_bytes = b"not audio"

    with pytest.raises(ValueError, match="my_field"):
        validate_audio_bytes(invalid_bytes, field_name="my_field")
