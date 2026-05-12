"""
GET  /v1/audio/voices — OpenAI-compatible voice list
POST /v1/audio/speech — OpenAI-compatible TTS (with optional SSE streaming)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..utils.audio import encode_tensors, tensors_to_wav_bytes
from ..utils.text import normalize_for_tts

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────


class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible TTS request body."""

    model: str = Field(default="revovoice")
    input: str = Field(..., min_length=1, max_length=10_000)
    voice: str = Field(default="anwar")
    response_format: Literal["wav", "mp3", "opus"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=2.0, ge=0.0)
    frequency_penalty: float = Field(default=0.3, ge=0.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1)
    chunk_size: int = Field(default=0, ge=0)
    language: str = Field(default="en", pattern=r"^(en|ms|mixed)$")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_inference(request: Request) -> InferenceService:
    return request.app.state.inference_svc


def _get_metrics(request: Request) -> MetricsService:
    return request.app.state.metrics_svc


def _get_cfg(request: Request):
    return request.app.state.cfg


def _resolve_voice(voice_id: str, cfg) -> tuple[str | None, str | None]:
    """Resolve voice ID to (wav_path, ref_text)."""
    wav_path = cfg.voices_dir / f"{voice_id}.wav"
    if not wav_path.is_file():
        return None, None
    txt_path = wav_path.with_suffix(".txt")
    ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
    return str(wav_path), ref_text


def _build_synthesis_req(body: OpenAISpeechRequest, cfg) -> SynthesisRequest:
    """Map OpenAI speech request to OmniVoice SynthesisRequest."""
    audio_path, ref_text = _resolve_voice(body.voice, cfg)

    if audio_path:
        mode = "clone"
    else:
        mode = "design"

    return SynthesisRequest(
        text=normalize_for_tts(body.input, language=body.language),
        mode=mode,
        ref_audio_path=audio_path,
        ref_text=ref_text,
        speed=body.speed,
        class_temperature=body.temperature if body.temperature != 0.3 else None,
    )


def _tensor_to_base64_float32(tensor) -> str:
    """Convert tensor to base64-encoded raw float32 PCM bytes."""
    import numpy as np

    flat = tensor.squeeze(0).cpu().float().numpy()
    return base64.b64encode(flat.astype(np.float32).tobytes()).decode("ascii")


# ── GET /v1/audio/voices ─────────────────────────────────────────────────────


@router.get("/audio/voices", tags=["OpenAI-compatible"])
async def list_audio_voices(cfg=Depends(_get_cfg)):
    """OpenAI-compatible voice list."""
    voices_dir: Path = cfg.voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices = []
    for wav in sorted(voices_dir.glob("*.wav")):
        voices.append({
            "id": wav.stem,
            "name": wav.stem.replace("_", " ").replace("-", " ").title(),
        })
    return {
        "object": "list",
        "data": voices,
    }


# ── POST /v1/audio/speech ────────────────────────────────────────────────────


@router.post("/audio/speech", tags=["OpenAI-compatible"])
async def create_speech(
    request: Request,
    body: OpenAISpeechRequest,
    stream: bool = Query(default=False),
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Generate speech from text. Supports streaming via ?stream=true."""
    client = request.client.host if request.client else "unknown"
    t0 = time.monotonic()

    if len(body.input) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input exceeds 10,000 characters",
        )

    # Validate voice exists
    audio_path, _ = _resolve_voice(body.voice, cfg)
    if not audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{body.voice}' not found",
        )

    logger.info(
        "[speech] client=%s voice=%s stream=%s text=%r",
        client, body.voice, stream, body.input,
    )

    if stream:
        return StreamingResponse(
            _stream_sse(body, inference_svc, metrics_svc, cfg, client, t0),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    syn_req = _build_synthesis_req(body, cfg)

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

    elapsed = time.monotonic() - t0
    logger.info(
        "[speech] client=%s voice=%s synthesis=%.3fs chars=%d",
        client, body.voice, elapsed, len(body.input),
    )

    if body.response_format == "wav":
        audio_bytes = tensors_to_wav_bytes(
            result.tensors, target_lufs=cfg.target_lufs,
        )
        return Response(content=audio_bytes, media_type="audio/wav")

    audio_bytes, media_type = encode_tensors(result.tensors, body.response_format)
    return Response(content=audio_bytes, media_type=media_type)


# ── SSE streaming ─────────────────────────────────────────────────────────────


async def _stream_sse(
    body: OpenAISpeechRequest,
    inference_svc: InferenceService,
    metrics_svc: MetricsService,
    cfg,
    client: str,
    t0: float,
) -> AsyncIterator[str]:
    """SSE generator for streaming TTS."""
    from ..utils.text import split_sentences

    sentences = split_sentences(body.input, max_chars=cfg.stream_chunk_max_chars)
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
    ttfc_logged = False

    for i, sentence in enumerate(sentences):
        logger.info("[speech] client=%s voice=%s sentence[%d] %r", client, body.voice, i, sentence)
        t_sent = time.monotonic()

        syn_req = SynthesisRequest(
            text=normalize_for_tts(sentence, language=body.language),
            mode=base_req.mode,
            ref_audio_path=base_req.ref_audio_path,
            ref_text=base_req.ref_text,
            speed=base_req.speed,
            class_temperature=base_req.class_temperature,
        )

        try:
            result = await inference_svc.synthesize(syn_req)
            metrics_svc.record_success(result.latency_s)
        except asyncio.TimeoutError:
            metrics_svc.record_timeout()
            error_event = json.dumps({"error": "Streaming chunk timed out", "is_final": True})
            yield f"data: {error_event}\n\n"
            return
        except Exception:
            metrics_svc.record_error()
            logger.exception("Streaming chunk failed")
            error_event = json.dumps({"error": "Streaming chunk failed", "is_final": True})
            yield f"data: {error_event}\n\n"
            return

        logger.info(
            "[speech] client=%s voice=%s sentence[%d] synthesis=%.3fs",
            client, body.voice, i, time.monotonic() - t_sent,
        )

        for tensor in result.tensors:
            if not ttfc_logged:
                logger.info(
                    "[speech] client=%s voice=%s ttfc=%.3fs",
                    client, body.voice, time.monotonic() - t0,
                )
                ttfc_logged = True
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

    logger.info(
        "[speech] client=%s voice=%s total=%.3fs chunks=%d",
        client, body.voice, time.monotonic() - t0, chunk_index,
    )
    final_event = json.dumps({
        "chunk_index": chunk_index,
        "audio": "",
        "sample_rate": 24000,
        "dtype": "float32",
        "is_final": True,
    })
    yield f"data: {final_event}\n\n"
