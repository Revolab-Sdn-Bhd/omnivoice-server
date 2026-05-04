"""
Tests for /generate, /api/generate_tts, /api/quotes, and /v1/models endpoints.
"""

from __future__ import annotations


def test_generate_returns_wav(client, voice_with_file):
    """POST /generate returns WAV."""
    resp = client.post(
        "/generate",
        json={"text": "Hello world", "voice_ref_path": "test_voice"},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_generate_voice_not_found(client):
    """POST /generate returns 404 for unknown voice path."""
    resp = client.post(
        "/generate",
        json={"text": "Hello", "voice_ref_path": "nonexistent"},
    )
    assert resp.status_code == 404


def test_generate_accepts_all_params(client, voice_with_file):
    """POST /generate accepts all T3-style parameters."""
    resp = client.post(
        "/generate",
        json={
            "text": "Hello",
            "language": "en",
            "voice_ref_path": "test_voice",
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "frequency_penalty": 0.2,
            "min_p": 0.1,
            "max_tokens": 500,
        },
    )
    assert resp.status_code == 200


def test_legacy_generate_tts(client, voice_with_file):
    """POST /api/generate_tts returns WAV."""
    resp = client.post(
        "/api/generate_tts",
        data={"text": "Hello world", "speaker_name": "test_voice"},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_legacy_generate_tts_speaker_name_with_ext(client, voice_with_file):
    """POST /api/generate_tts normalizes speaker_name.wav."""
    resp = client.post(
        "/api/generate_tts",
        data={"text": "Hello", "speaker_name": "test_voice.wav"},
    )
    assert resp.status_code == 200


def test_legacy_generate_tts_rejects_temp_prefix(client, voice_with_file):
    """POST /api/generate_tts rejects [TEMP] prefixed names."""
    resp = client.post(
        "/api/generate_tts",
        data={"text": "Hello", "speaker_name": "[TEMP]_voice"},
    )
    assert resp.status_code == 400


def test_legacy_generate_tts_voice_not_found(client):
    """POST /api/generate_tts returns 404 for unknown voice."""
    resp = client.post(
        "/api/generate_tts",
        data={"text": "Hello", "speaker_name": "nonexistent"},
    )
    assert resp.status_code == 404


def test_legacy_generate_tts_stream(client, voice_with_file):
    """POST /api/generate_tts_stream returns SSE events."""
    resp = client.post(
        "/api/generate_tts_stream",
        data={"text": "Hello world.", "speaker_name": "test_voice"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_get_quotes(client):
    """GET /api/quotes returns list of test sentences."""
    resp = client.get("/api/quotes")
    assert resp.status_code == 200
    data = resp.json()
    assert "quotes" in data
    assert len(data["quotes"]) > 0


def test_get_models(client):
    """GET /v1/models returns sepbox-tts model."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "sepbox-tts" in ids
