"""
POST /api/generate_tts        — Legacy ChatterBox TTS (form data, returns WAV)
POST /api/generate_tts_stream — Legacy ChatterBox streaming TTS (form data, SSE)
GET  /api/quotes              — Curated test sentences
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..utils.audio import tensors_to_wav_bytes
from ..utils.text import split_sentences

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_inference(request: Request) -> InferenceService:
    return request.app.state.inference_svc


def _get_metrics(request: Request) -> MetricsService:
    return request.app.state.metrics_svc


def _get_cfg(request: Request):
    return request.app.state.cfg


# ── speaker_name normalization ─────────────────────────────────────────────────


def _normalize_speaker_name(speaker_name: str) -> str:
    """Normalize speaker_name to a stem.

    Accepts bare stem ("anwar"), with extension ("anwar.wav"),
    or full path ("/app/voices/anwar.wav"). Extracts the stem.
    """
    if speaker_name.startswith("[TEMP]"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="[TEMP] prefixed names are not allowed",
        )
    stem = Path(speaker_name).stem
    return re.sub(r"[^a-zA-Z0-9_\-]", "", stem)


def _resolve_voice(speaker_name: str, cfg) -> tuple[str | None, str | None]:
    """Resolve speaker_name to (audio_path, ref_text)."""
    stem = _normalize_speaker_name(speaker_name)
    wav_path = cfg.voices_dir / f"{stem}.wav"
    if not wav_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{stem}' not found",
        )
    txt_path = wav_path.with_suffix(".txt")
    ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
    return str(wav_path), ref_text


def _tensor_to_base64_float32(tensor) -> str:
    """Convert tensor to base64-encoded raw float32 PCM bytes."""
    import numpy as np

    flat = tensor.squeeze(0).cpu().float().numpy()
    return base64.b64encode(flat.astype(np.float32).tobytes()).decode("ascii")


# ── Common form fields ─────────────────────────────────────────────────────────

TTS_FORM_DEFAULTS = dict(
    speaker_name=Form(..., description="Voice name"),
    preset_name=Form(default="neutral", description="Ignored (compatibility only)"),
    temperature=Form(default=0.3, ge=0.0, le=2.0),
    top_p=Form(default=1.0, ge=0.0, le=1.0),
    repetition_penalty=Form(default=2.0, ge=0.0),
    frequency_penalty=Form(default=0.3, ge=0.0),
    min_p=Form(default=0.05, ge=0.0, le=1.0),
    max_tokens=Form(default=1000, ge=1),
)


# ── POST /api/generate_tts ────────────────────────────────────────────────────


@router.post("/api/generate_tts")
async def generate_tts(
    text: str = Form(..., min_length=1, max_length=10_000),
    speaker_name: str = Form(..., description="Voice name"),
    preset_name: str = Form(default="neutral", description="Ignored (compatibility only)"),
    temperature: float = Form(default=0.3, ge=0.0, le=2.0),
    top_p: float = Form(default=1.0, ge=0.0, le=1.0),
    repetition_penalty: float = Form(default=2.0, ge=0.0),
    frequency_penalty: float = Form(default=0.3, ge=0.0),
    min_p: float = Form(default=0.05, ge=0.0, le=1.0),
    max_tokens: int = Form(default=1000, ge=1),
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Legacy ChatterBox-compatible TTS endpoint. Returns WAV bytes."""
    audio_path, ref_text = _resolve_voice(speaker_name, cfg)

    syn_req = SynthesisRequest(
        text=text,
        mode="clone",
        ref_audio_path=audio_path,
        ref_text=ref_text,
        speed=1.0,
        class_temperature=temperature if temperature != 0.3 else None,
    )

    try:
        timeout_s = cfg.request_timeout_s
        result = await asyncio.wait_for(
            inference_svc.synthesize(syn_req), timeout=timeout_s,
        )
        metrics_svc.record_success(result.latency_s)
    except QueueFullError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except asyncio.TimeoutError:
        metrics_svc.record_timeout()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Synthesis timed out after {cfg.request_timeout_s}s",
        )
    except Exception as e:
        metrics_svc.record_error()
        logger.exception("Legacy TTS failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )

    wav_bytes = tensors_to_wav_bytes(result.tensors)
    return Response(content=wav_bytes, media_type="audio/wav")


# ── POST /api/generate_tts_stream ─────────────────────────────────────────────


@router.post("/api/generate_tts_stream")
async def generate_tts_stream(
    text: str = Form(..., min_length=1, max_length=10_000),
    speaker_name: str = Form(..., description="Voice name"),
    preset_name: str = Form(default="neutral", description="Ignored (compatibility only)"),
    temperature: float = Form(default=0.3, ge=0.0, le=2.0),
    top_p: float = Form(default=1.0, ge=0.0, le=1.0),
    repetition_penalty: float = Form(default=2.0, ge=0.0),
    frequency_penalty: float = Form(default=0.3, ge=0.0),
    min_p: float = Form(default=0.05, ge=0.0, le=1.0),
    max_tokens: int = Form(default=1000, ge=1),
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Legacy ChatterBox-compatible streaming TTS. Returns SSE events."""
    audio_path, ref_text = _resolve_voice(speaker_name, cfg)

    return StreamingResponse(
        _stream_legacy_sse(
            text, audio_path, ref_text, temperature, inference_svc, metrics_svc, cfg,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_legacy_sse(
    text: str,
    audio_path: str | None,
    ref_text: str | None,
    temperature: float,
    inference_svc: InferenceService,
    metrics_svc: MetricsService,
    cfg,
) -> AsyncIterator[str]:
    """SSE generator for legacy streaming TTS."""
    sentences = split_sentences(text, max_chars=cfg.stream_chunk_max_chars)
    if not sentences:
        event = json.dumps({
            "chunk_index": 0,
            "audio": "",
            "sample_rate": 24000,
            "dtype": "float32",
            "is_final": True,
        })
        yield f"data: {event}\n\n"
        return

    chunk_index = 0
    for sentence in sentences:
        syn_req = SynthesisRequest(
            text=sentence,
            mode="clone",
            ref_audio_path=audio_path,
            ref_text=ref_text,
            speed=1.0,
            class_temperature=temperature if temperature != 0.3 else None,
        )

        try:
            result = await inference_svc.synthesize(syn_req)
            metrics_svc.record_success(result.latency_s)
        except asyncio.TimeoutError:
            metrics_svc.record_timeout()
            error_event = json.dumps({
                "error": "Streaming chunk timed out",
                "is_final": True,
            })
            yield f"data: {error_event}\n\n"
            return
        except Exception:
            metrics_svc.record_error()
            logger.exception("Legacy streaming chunk failed")
            error_event = json.dumps({
                "error": "Streaming chunk failed",
                "is_final": True,
            })
            yield f"data: {error_event}\n\n"
            return

        for tensor in result.tensors:
            audio_b64 = _tensor_to_base64_float32(tensor)
            event = json.dumps({
                "chunk_index": chunk_index,
                "audio": audio_b64,
                "sample_rate": 24000,
                "dtype": "float32",
                "is_final": False,
            })
            yield f"data: {event}\n\n"
            chunk_index += 1

    final_event = json.dumps({
        "chunk_index": chunk_index,
        "audio": "",
        "sample_rate": 24000,
        "dtype": "float32",
        "is_final": True,
    })
    yield f"data: {final_event}\n\n"


# ── GET /api/quotes ───────────────────────────────────────────────────────────


QUOTES = [
    "How are you doing today?",
    "My account number is AA212CC8819000ZZ.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "I have an appointment scheduled for next Tuesday at 3 PM.",
    "Please transfer five hundred ringgit to account number 1234567890.",
    "The weather forecast predicts heavy rainfall in the northern regions.",
    "Welcome to our customer service hotline, how may I assist you?",
    "Your order has been shipped and will arrive within three to five business days.",
    "Saya ingin membuat aduan mengenain perkhidmatan yang diterima semalam.",
    "Terima kasih kerana menghubungi kami, ada apa yang boleh saya bantu?",
    "This is a longer passage used for testing multi-sentence synthesis. "
    "It contains several sentences with varying lengths and complexities. "
    "The purpose is to evaluate how well the system handles extended text input "
    "while maintaining natural prosody and consistent voice quality throughout.",
]


@router.get("/api/quotes")
async def get_quotes():
    """Returns a curated list of test sentences for TTS evaluation."""
    return {"quotes": QUOTES}
