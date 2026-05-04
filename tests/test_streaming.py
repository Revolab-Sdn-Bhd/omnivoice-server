"""Tests for streaming TTS endpoints."""

from __future__ import annotations

import json


def test_generate_stream_returns_sse(client, voice_with_file):
    """POST /generate/stream returns SSE events with base64 audio."""
    resp = client.post(
        "/generate/stream",
        json={"text": "Hello world.", "voice_ref_path": "test_voice"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    body = resp.text
    events = [line for line in body.strip().split("\n") if line.startswith("data: ")]
    assert len(events) >= 2  # At least one audio chunk + final event

    # Check first audio event
    first_data = json.loads(events[0][6:])
    assert "chunk_index" in first_data
    assert first_data["audio"] != ""
    assert first_data["sample_rate"] == 24000
    assert first_data["dtype"] == "float32"
    assert first_data["is_final"] is False

    # Check final event
    last_data = json.loads(events[-1][6:])
    assert last_data["is_final"] is True
    assert last_data["audio"] == ""


def test_generate_stream_multiple_sentences(client, voice_with_file):
    """Multiple sentences produce multiple audio chunks."""
    text = "First sentence. Second sentence. Third sentence."
    resp = client.post(
        "/generate/stream",
        json={"text": text, "voice_ref_path": "test_voice"},
    )
    assert resp.status_code == 200
    events = [line for line in resp.text.strip().split("\n") if line.startswith("data: ")]
    assert len(events) >= 2


def test_speech_stream_sse(client, voice_with_file):
    """POST /v1/audio/speech?stream=true returns SSE events."""
    resp = client.post(
        "/v1/audio/speech?stream=true",
        json={"input": "Hello world.", "voice": "test_voice"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = [line for line in resp.text.strip().split("\n") if line.startswith("data: ")]
    assert len(events) >= 2

    # Verify final event
    last_data = json.loads(events[-1][6:])
    assert last_data["is_final"] is True


def test_streaming_empty_text(client, voice_with_file):
    """Empty text returns 422."""
    resp = client.post(
        "/generate/stream",
        json={"text": "", "voice_ref_path": "test_voice"},
    )
    assert resp.status_code == 422
