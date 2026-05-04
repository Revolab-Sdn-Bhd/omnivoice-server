"""Tests for voice management endpoints."""

from __future__ import annotations

import io


def test_list_voices_empty(client):
    """GET /voices returns empty voices list when no WAV files."""
    resp = client.get("/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert isinstance(data["voices"], list)


def test_list_voices_with_wav(client, voice_with_file):
    """GET /voices lists WAV files from voices/ directory."""
    resp = client.get("/voices")
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    ids = [v["id"] for v in voices]
    assert "test_voice" in ids


def test_upload_voice(client, sample_audio_bytes):
    """POST /voices uploads a new WAV file."""
    resp = client.post(
        "/voices",
        data={"voice_name": "new_voice"},
        files={"audio_file": ("new.wav", io.BytesIO(sample_audio_bytes), "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "new_voice"
    assert data["name"] == "New Voice"


def test_upload_voice_invalid_name(client, sample_audio_bytes):
    """POST /voices rejects names with no valid characters."""
    resp = client.post(
        "/voices",
        data={"voice_name": "!@#$%"},
        files={"audio_file": ("f.wav", io.BytesIO(sample_audio_bytes), "audio/wav")},
    )
    assert resp.status_code == 400


def test_upload_voice_appears_in_list(client, sample_audio_bytes):
    """Uploaded voice appears in GET /voices."""
    client.post(
        "/voices",
        data={"voice_name": "uploaded"},
        files={"audio_file": ("f.wav", io.BytesIO(sample_audio_bytes), "audio/wav")},
    )
    resp = client.get("/voices")
    ids = [v["id"] for v in resp.json()["voices"]]
    assert "uploaded" in ids
