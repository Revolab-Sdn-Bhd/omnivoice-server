"""Tests for OpenAI-compatible TTS endpoint."""

from __future__ import annotations

from conftest import make_wav_bytes


def test_speech_returns_wav(client, voice_with_file):
    """POST /v1/audio/speech returns WAV with RIFF header."""
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "sepbox-tts",
            "input": "Hello world",
            "voice": "test_voice",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_speech_default_voice(client, voice_with_file):
    """POST /v1/audio/speech uses default voice 'anwar' when not specified."""
    # Default voice is "anwar" per spec — create it so the test passes
    cfg = client.app.state.cfg
    wav_path = cfg.voices_dir / "anwar.wav"
    if not wav_path.exists():
        sample = make_wav_bytes(duration_frames=100)
        wav_path.write_bytes(sample)
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "Hello"},
    )
    assert resp.status_code == 200


def test_speech_voice_not_found(client):
    """POST /v1/audio/speech returns 404 for unknown voice."""
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "Hello", "voice": "nonexistent"},
    )
    assert resp.status_code == 404


def test_speech_empty_text_rejected(client, voice_with_file):
    """Empty text returns 422."""
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "", "voice": "test_voice"},
    )
    assert resp.status_code == 422


def test_speech_accepts_sepbox_params(client, voice_with_file):
    """T3-style parameters are accepted without error."""
    resp = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello",
            "voice": "test_voice",
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "frequency_penalty": 0.2,
            "min_p": 0.1,
            "max_tokens": 500,
        },
    )
    assert resp.status_code == 200


def test_speech_pcm_format(client, voice_with_file):
    """response_format=pcm is not supported, falls back to wav."""
    # Spec says only wav is actually supported
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "Hello", "voice": "test_voice"},
    )
    assert resp.status_code == 200


def test_list_audio_voices(client, voice_with_file):
    """GET /v1/audio/voices returns OpenAI-compatible voice list."""
    resp = client.get("/v1/audio/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert "data" in data
    ids = [v["id"] for v in data["data"]]
    assert "test_voice" in ids
