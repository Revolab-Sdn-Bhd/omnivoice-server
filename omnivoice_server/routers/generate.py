"""
POST /generate        — Full TTS → WAV file
POST /generate/stream — SSE streaming TTS

SepBox-compatible TTS endpoints that accept T3-style parameters
and map them to OmniVoice's SynthesisRequest internally.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..utils.audio import tensors_to_wav_bytes
from ..utils.text import split_sentences

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    """SepBox-compatible TTS request body."""

    text: str = Field(..., min_length=1, max_length=10_000)
    language: str = Field(default="en", pattern=r"^(en|ms|mixed)$")
    voice_ref_path: str | None = Field(default=None)
    voice_ref_audio: str | None = Field(default=None, description="Base64-encoded WAV")
    chunk_size: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=2.0, ge=0.0)
    frequency_penalty: float = Field(default=0.3, ge=0.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_inference(request: Request) -> InferenceService:
    return request.app.state.inference_svc


def _get_metrics(request: Request) -> MetricsService:
    return request.app.state.metrics_svc


def _get_cfg(request: Request):
    return request.app.state.cfg


def _resolve_voice_path(
    voice_ref_path: str | None,
    voice_ref_audio: str | None,
    cfg,
) -> tuple[str | None, str | None]:
    """Resolve voice reference to (audio_path, ref_text).

    Returns (temp_file_path_or_real_path, ref_text_or_empty).
    If voice_ref_audio is provided, decodes base64 and writes to temp file.
    """
    if voice_ref_audio:
        raw = base64.b64decode(voice_ref_audio)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name, ""

    if voice_ref_path:
        p = Path(voice_ref_path)
        # If just a stem like "anwar", resolve to voices/anwar.wav
        if not p.is_file() and not p.suffix:
            p = cfg.voices_dir / f"{voice_ref_path}.wav"
        if not p.is_file():
            p = Path(voice_ref_path)

        # Look for companion .txt for ref_text
        txt_path = p.with_suffix(".txt")
        ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
        return str(p), ref_text

    return None, None


def _build_synthesis_req(body: TTSRequest, cfg) -> SynthesisRequest:
    """Map SepBox TTSRequest to OmniVoice SynthesisRequest."""
    audio_path, ref_text = _resolve_voice_path(
        body.voice_ref_path, body.voice_ref_audio, cfg
    )

    if audio_path:
        mode = "clone"
    else:
        mode = "design"

    return SynthesisRequest(
        text=body.text,
        mode=mode,
        ref_audio_path=audio_path,
        ref_text=ref_text,
        speed=1.0,
        language=body.language if body.language != "en" else None,
        # Map temperature to class_temperature (closest equivalent)
        class_temperature=body.temperature if body.temperature != 0.3 else None,
        # The following are accepted for API compat but OmniVoice doesn't use them
        # top_p, repetition_penalty, frequency_penalty, min_p, max_tokens
    )


def _tensor_to_base64_float32(tensor) -> str:
    """Convert tensor to base64-encoded raw float32 PCM bytes."""
    import numpy as np

    flat = tensor.squeeze(0).cpu().float().numpy()
    return base64.b64encode(flat.astype(np.float32).tobytes()).decode("ascii")


# ── POST /generate ────────────────────────────────────────────────────────────


@router.post("/generate")
async def generate(
    request: Request,
    body: TTSRequest,
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Run the full TTS pipeline and return a complete WAV file."""
    if len(body.text) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text exceeds 10,000 characters",
        )

    syn_req = _build_synthesis_req(body, cfg)

    if syn_req.ref_audio_path:
        p = Path(syn_req.ref_audio_path)
        if not p.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice path not found: {body.voice_ref_path}",
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
        logger.exception("Synthesis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )

    wav_bytes = tensors_to_wav_bytes(result.tensors)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
    )


# ── POST /generate/stream ─────────────────────────────────────────────────────


@router.post("/generate/stream")
async def generate_stream(
    request: Request,
    body: TTSRequest,
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Stream TTS audio as Server-Sent Events (SSE)."""
    if len(body.text) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text exceeds 10,000 characters",
        )

    return StreamingResponse(
        _stream_sse(body, inference_svc, metrics_svc, cfg),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_sse(
    body: TTSRequest,
    inference_svc: InferenceService,
    metrics_svc: MetricsService,
    cfg,
) -> AsyncIterator[str]:
    """SSE generator yielding base64-encoded float32 PCM chunks."""
    sentences = split_sentences(body.text, max_chars=cfg.stream_chunk_max_chars)
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

    base_req = _build_synthesis_req(body, cfg)
    chunk_index = 0
    for sentence in sentences:
        syn_req = SynthesisRequest(
            text=sentence,
            mode=base_req.mode,
            ref_audio_path=base_req.ref_audio_path,
            ref_text=base_req.ref_text,
            speed=1.0,
            language=base_req.language,
            class_temperature=base_req.class_temperature,
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
            logger.exception("Streaming chunk failed")
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

    # Final event
    final_event = json.dumps({
        "chunk_index": chunk_index,
        "audio": "",
        "sample_rate": 24000,
        "dtype": "float32",
        "is_final": True,
    })
    yield f"data: {final_event}\n\n"
