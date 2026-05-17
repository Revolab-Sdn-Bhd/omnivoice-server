"""
GET  /v1/audio/voices — OpenAI-compatible voice list
POST /v1/audio/speech — OpenAI-compatible TTS (with optional SSE streaming)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ..observability.tracer import (
    build_synthesis_output,
    get_current_trace_id,
    get_observe,
    update_current_trace,
)
from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..utils.audio import encode_tensors, tensors_to_wav_bytes
from ._shared import (  # noqa: I001
    build_synthesis_request,
    tensor_to_base64_float32,
)
from ._shared import (
    get_cfg as _get_cfg,
)
from ._shared import (
    get_inference_svc as _get_inference,
)
from ._shared import (
    get_metrics_svc as _get_metrics,
)

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
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


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
@get_observe()(capture_output=False)
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

    update_current_trace(metadata={
        "voice": body.voice,
        "format": body.response_format,
        "stream": stream,
        "language": body.language,
    })

    if len(body.input) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input exceeds 10,000 characters",
        )

    logger.info(
        "[speech] client=%s voice=%s stream=%s text=%r",
        client, body.voice, stream, body.input,
    )

    if stream:
        trace_id = get_current_trace_id()
        return StreamingResponse(
            _stream_sse(body, inference_svc, metrics_svc, cfg, client, t0, trace_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    syn_req = build_synthesis_request(
        text=body.input,
        cfg=cfg,
        voice_ref=body.voice,
        speed=body.speed,
        class_temperature=body.temperature if body.temperature != 0.3 else None,
        seed=body.seed,
        _trace_id=get_current_trace_id(),
    )

    if not syn_req.ref_audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{body.voice}' not found",
        )

    try:
        timeout_s = cfg.request_timeout_s
        result = await asyncio.wait_for(
            inference_svc.synthesize(syn_req), timeout=timeout_s,
        )
        metrics_svc.record_success(result.latency_s)
    except QueueFullError as e:
        update_current_trace(metadata={"error": f"QueueFullError: {e}"})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except asyncio.TimeoutError:
        metrics_svc.record_timeout()
        update_current_trace(metadata={"error": f"TimeoutError: {cfg.request_timeout_s}s"})
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Synthesis timed out after {cfg.request_timeout_s}s",
        )
    except Exception as e:
        metrics_svc.record_error()
        logger.exception("Synthesis failed")
        update_current_trace(metadata={"error": f"{type(e).__name__}: {e}"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "[speech] client=%s voice=%s synthesis=%.3fs chars=%d",
        client, body.voice, elapsed, len(body.input),
    )

    # Attach audio + metrics to root trace (following ChatterBox pattern)
    wav_for_trace = tensors_to_wav_bytes(result.tensors)
    trace_output = build_synthesis_output(
        tensors=result.tensors,
        wav_bytes=wav_for_trace,
        latency_s=result.latency_s,
        text=body.input,
        voice=body.voice,
        mode="clone",
        speed=body.speed,
        device="cuda",
        extra={
            "format": body.response_format,
            "total_e2e_ms": round(elapsed * 1000, 2),
            "inference_ms": round(result.latency_s * 1000, 2),
        },
    )
    update_current_trace(output=trace_output)

    # Slack notification (non-blocking, sampled)
    from ..observability.slack_notifier import send_tts_notification

    trace_id = get_current_trace_id()
    send_tts_notification(
        text=body.input,
        voice=body.voice,
        mode="clone",
        endpoint="v1/audio/speech",
        trace_output=trace_output,
        trace_id=trace_id,
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
    trace_id: str | None = None,
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

    base_req = build_synthesis_request(
        text=body.input,
        cfg=cfg,
        voice_ref=body.voice,
        speed=body.speed,
        class_temperature=body.temperature if body.temperature != 0.3 else None,
        seed=body.seed,
    )
    chunk_index = 0
    ttfc_logged = False

    for i, sentence in enumerate(sentences):
        logger.info("[speech] client=%s voice=%s sentence[%d] %r", client, body.voice, i, sentence)
        t_sent = time.monotonic()

        syn_req = SynthesisRequest(
            text=sentence,
            mode=base_req.mode,
            ref_audio_path=base_req.ref_audio_path,
            ref_text=base_req.ref_text,
            speed=base_req.speed,
            class_temperature=base_req.class_temperature,
            _trace_id=trace_id,
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
            audio_b64 = tensor_to_base64_float32(tensor)
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
