"""
/v1/audio/speech        - OpenAI-compatible TTS (instructions-driven design)
/v1/audio/speech/clone  - One-shot voice cloning (multipart upload)
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..services.profiles import ProfileService
from ..utils.audio import encode_tensors, normalize_loudness, tensor_to_pcm16_bytes
from ..utils.text import split_sentences
from ..voice_presets import DEFAULT_DESIGN_INSTRUCTIONS, OPENAI_VOICE_PRESETS

logger = logging.getLogger(__name__)
router = APIRouter()


def _write_trace(
    trace_dir: Path,
    audio_bytes: bytes,
    ext: str,
    meta: dict,
) -> None:
    """Write audio + JSON sidecar to trace_dir. Non-fatal on error."""
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:22]
        stem = trace_dir / ts
        stem.with_suffix(f".{ext}").write_bytes(audio_bytes)
        stem.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug("Trace written: %s", stem.name)
    except OSError as e:
        logger.warning("Trace write failed: %s", e)


def _build_trace_meta(
    *,
    request: Request,
    cfg,
    text: str,
    voice: str | None,
    profile_id: str | None,
    mode: str,
    instruct: str | None,
    ref_text: str | None,
    latency_s: float,
    duration_s: float,
    num_step: int,
    guidance_scale: float,
    speed: float,
    fmt: str,
    audio_size_bytes: int,
) -> dict:
    rtf = round(latency_s / duration_s, 4) if duration_s > 0 else None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request.headers.get("X-Request-ID", ""),
        "text": text,
        "text_chars": len(text),
        "voice": voice,
        "profile_id": profile_id,
        "mode": mode,
        "instruct": instruct,
        "ref_text": ref_text,
        "latency_s": round(latency_s, 3),
        "duration_s": round(duration_s, 3),
        "rtf": rtf,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "speed": speed,
        "format": fmt,
        "audio_size_bytes": audio_size_bytes,
        "device": cfg.device,
        "model_id": cfg.model_id,
    }


ACCEPT_FORMAT_MAP: dict[str, str] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "opus",
    "audio/opus": "opus",
    "audio/pcm": "pcm",
}


def _resolve_format(
    response_format: str | None,
    accept: str | None,
) -> str:
    """Resolve output format: explicit param > Accept header > default wav."""
    if response_format is not None:
        return response_format
    if accept:
        for mime, fmt in ACCEPT_FORMAT_MAP.items():
            if mime in accept:
                return fmt
    return "wav"


class SpeechRequest(BaseModel):
    """OpenAI TTS API compatible request body."""

    model: str = Field(default="omnivoice")
    input: str = Field(..., min_length=1, max_length=10_000)
    voice: str = Field(default="auto")
    speaker: str | None = Field(default=None)
    instructions: str | None = Field(default=None)
    response_format: Literal["wav", "pcm", "mp3", "opus"] | None = Field(
        default=None,
        description=(
            "Output audio format. Auto-detected from Accept header if not set."
        ),
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream: bool = Field(default=False)
    num_step: int | None = Field(default=None, ge=8, le=32)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=10.0)
    denoise: bool | None = Field(default=None)
    t_shift: float | None = Field(default=None, ge=0.0, le=2.0)
    position_temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    duration: float | None = Field(default=None, ge=0.1, le=60.0)
    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'en', 'vi', 'zh') for multilingual pronunciation",
    )
    timeout: float | None = Field(
        default=None, ge=1.0, le=300.0,
        description="Max wait in seconds. Uses server default if not set.",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ("omnivoice", "tts-1", "tts-1-hd"):
            logger.debug(f"model='{v}' mapped to omnivoice")
        return v


def _get_inference(request: Request) -> InferenceService:
    return request.app.state.inference_svc


def _get_profiles(request: Request) -> ProfileService:
    return request.app.state.profile_svc


def _get_metrics(request: Request) -> MetricsService:
    return request.app.state.metrics_svc


def _get_cfg(request: Request):
    return request.app.state.cfg


def _resolve_synthesis_mode(
    body: SpeechRequest,
    profile_svc: ProfileService,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Resolve synthesis mode. Returns (mode, instruct, ref_audio_path, ref_text, profile_id)."""
    from ..services.profiles import ProfileNotFoundError

    instructions = body.instructions.strip() if body.instructions else None
    speaker = body.speaker.strip() if body.speaker else None
    voice = body.voice.strip() if body.voice else None

    if instructions:
        return "design", instructions, None, None, None

    candidate = speaker or voice

    # Preset names are lowercase — compare case-insensitively
    if candidate and candidate.lower() in OPENAI_VOICE_PRESETS:
        return "design", OPENAI_VOICE_PRESETS[candidate.lower()], None, None, None

    # Strip "clone:" prefix for backwards compatibility with list_voices IDs
    profile_id = candidate
    if profile_id and profile_id.startswith("clone:"):
        profile_id = profile_id[6:]

    if profile_id:
        try:
            ref_audio_path = str(profile_svc.get_ref_audio_path(profile_id))
            ref_text = profile_svc.get_ref_text(profile_id)
            return "clone", None, ref_audio_path, ref_text, profile_id
        except ProfileNotFoundError:
            pass

    return "design", DEFAULT_DESIGN_INSTRUCTIONS, None, None, None


@router.post(
    "/audio/speech",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic_wav": {
                            "summary": "Basic WAV output",
                            "value": {
                                "input": "Hello, welcome to the service.",
                                "voice": "alloy",
                            },
                        },
                        "opus_format": {
                            "summary": "Opus compressed output",
                            "value": {
                                "input": "The quick brown fox jumps.",
                                "response_format": "opus",
                            },
                        },
                        "custom_voice": {
                            "summary": "Custom voice instructions",
                            "value": {
                                "input": "Good morning everyone.",
                                "instructions": "A warm, deep male voice, "
                                "speaking slowly and clearly.",
                                "response_format": "mp3",
                            },
                        },
                        "streaming": {
                            "summary": "Streaming PCM output",
                            "value": {
                                "input": "This is a longer text that "
                                "benefits from sentence-level streaming.",
                                "stream": True,
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Audio file in requested format",
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                    "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                    "audio/ogg": {"schema": {"type": "string", "format": "binary"}},
                    "audio/pcm": {"schema": {"type": "string", "format": "binary"}},
                },
            }
        },
    },
)
async def create_speech(
    request: Request,
    body: SpeechRequest,
    inference_svc: InferenceService = Depends(_get_inference),
    profile_svc: ProfileService = Depends(_get_profiles),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """Generate speech from text."""
    fmt = _resolve_format(body.response_format, request.headers.get("Accept"))
    mode, instruct, ref_audio_path, ref_text, profile_id = _resolve_synthesis_mode(
        body, profile_svc
    )

    embedding_cache_path = (
        str(profile_svc.get_embedding_cache_path(profile_id)) if profile_id else None
    )

    req = SynthesisRequest(
        text=body.input,
        mode=mode,
        instruct=instruct,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        embedding_cache_path=embedding_cache_path,
        speed=body.speed,
        num_step=body.num_step,
        guidance_scale=body.guidance_scale,
        denoise=body.denoise,
        t_shift=body.t_shift,
        position_temperature=body.position_temperature,
        class_temperature=body.class_temperature,
        duration=body.duration,
        language=body.language,
    )

    if body.stream:
        return StreamingResponse(
            _stream_sentences(body.input, req, inference_svc, metrics_svc, cfg),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Channels": "1",
                "X-Audio-Bit-Depth": "16",
                "X-Audio-Format": "pcm-int16-le",
            },
        )

    # Check response cache (non-streaming only)
    cache = getattr(request.app.state, "response_cache", None)
    cache_key = None
    if cache is not None:
        from ..services.response_cache import ResponseCache

        cache_key = ResponseCache.build_key(
            text=body.input,
            mode=mode,
            instruct=instruct,
            profile_id=profile_id,
            speed=body.speed,
            num_step=body.num_step or cfg.num_step,
            guidance_scale=(
                body.guidance_scale if body.guidance_scale is not None
                else cfg.guidance_scale
            ),
            denoise=body.denoise if body.denoise is not None else cfg.denoise,
            t_shift=body.t_shift if body.t_shift is not None else cfg.t_shift,
            pt=(
                body.position_temperature
                if body.position_temperature is not None
                else cfg.position_temperature
            ),
            ct=(
                body.class_temperature
                if body.class_temperature is not None
                else cfg.class_temperature
            ),
            duration=body.duration,
            fmt=fmt,
            lufs=cfg.loudness_target_lufs,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            audio_bytes, metadata = cached
            metrics_svc.record_success(0.0)
            return Response(
                content=audio_bytes,
                media_type=metadata.get("media_type", "audio/wav"),
                headers={
                    "X-Audio-Duration-S": str(metadata.get("duration_s", 0)),
                    "X-Synthesis-Latency-S": "0",
                    "X-Cache": "HIT",
                },
            )

    # Run inference (with optional per-request timeout)
    try:
        timeout_s = body.timeout or cfg.request_timeout_s
        result = await asyncio.wait_for(
            inference_svc.synthesize(req), timeout=timeout_s,
        )
        metrics_svc.record_success(result.latency_s)
    except QueueFullError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "1"},
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

    # Normalize loudness if configured
    tensors = result.tensors
    if cfg.loudness_target_lufs is not None:
        tensors = [normalize_loudness(t, cfg.loudness_target_lufs) for t in tensors]

    audio_bytes, media_type = encode_tensors(tensors, fmt)

    # Save to cache
    if cache is not None and cache_key is not None:
        cache.put(cache_key, audio_bytes, {
            "media_type": media_type,
            "duration_s": result.duration_s,
            "text": body.input,
        })

    # Trace
    if cfg.trace_dir:
        _write_trace(cfg.trace_dir, audio_bytes, fmt, _build_trace_meta(
            request=request,
            cfg=cfg,
            text=body.input,
            voice=body.voice,
            profile_id=profile_id,
            mode=mode,
            instruct=instruct,
            ref_text=ref_text,
            latency_s=result.latency_s,
            duration_s=result.duration_s,
            num_step=body.num_step or cfg.num_step,
            guidance_scale=body.guidance_scale if body.guidance_scale is not None
            else cfg.guidance_scale,
            speed=body.speed,
            fmt=fmt,
            audio_size_bytes=len(audio_bytes),
        ))

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-Audio-Duration-S": str(round(result.duration_s, 3)),
            "X-Synthesis-Latency-S": str(round(result.latency_s, 3)),
        },
    )


async def _stream_sentences(
    text: str,
    base_req: SynthesisRequest,
    inference_svc: InferenceService,
    metrics_svc: MetricsService,
    cfg,
) -> AsyncIterator[bytes]:
    """Sentence-level streaming generator."""
    sentences = split_sentences(text, max_chars=cfg.stream_chunk_max_chars)

    if not sentences:
        return

    for sentence in sentences:
        req = SynthesisRequest(
            text=sentence,
            mode=base_req.mode,
            instruct=base_req.instruct,
            ref_audio_path=base_req.ref_audio_path,
            ref_text=base_req.ref_text,
            embedding_cache_path=base_req.embedding_cache_path,
            speed=base_req.speed,
            num_step=base_req.num_step,
            guidance_scale=base_req.guidance_scale,
            denoise=base_req.denoise,
            t_shift=base_req.t_shift,
            position_temperature=base_req.position_temperature,
            class_temperature=base_req.class_temperature,
            duration=base_req.duration,
            language=base_req.language,
        )
        try:
            result = await inference_svc.synthesize(req)
            metrics_svc.record_success(result.latency_s)
            for tensor in result.tensors:
                yield tensor_to_pcm16_bytes(tensor)
        except asyncio.TimeoutError:
            metrics_svc.record_timeout()
            logger.warning(f"Streaming chunk timed out: '{sentence[:50]}...'")
            return
        except Exception:
            metrics_svc.record_error()
            logger.exception(f"Streaming chunk failed: '{sentence[:50]}...'")
            return


@router.post(
    "/audio/speech/clone",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "basic_clone": {
                            "summary": "Clone voice from reference audio",
                            "value": {
                                "text": "This is the text to synthesize.",
                                "ref_audio": "(binary WAV file upload)",
                            },
                        },
                        "with_ref_text": {
                            "summary": "Clone with reference transcript",
                            "value": {
                                "text": "Hello from the cloned voice.",
                                "ref_audio": "(binary WAV file upload)",
                                "ref_text": "The transcript of the reference audio.",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def create_speech_clone(
    request: Request,
    text: str = Form(..., min_length=1, max_length=10_000),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(..., description="Transcript of the reference audio. Required."),
    speed: float = Form(default=1.0, ge=0.25, le=4.0),
    num_step: int | None = Form(default=None, ge=1, le=64),
    guidance_scale: float | None = Form(default=None, ge=0.0, le=10.0),
    denoise: bool | None = Form(default=None),
    t_shift: float | None = Form(default=None, ge=0.0, le=2.0),
    position_temperature: float | None = Form(default=None, ge=0.0, le=10.0),
    class_temperature: float | None = Form(default=None, ge=0.0, le=2.0),
    duration: float | None = Form(default=None, ge=0.1, le=60.0),
    language: str | None = Form(
        default=None,
        description="Language code (e.g., 'en', 'vi', 'zh') for multilingual pronunciation",
    ),
    inference_svc: InferenceService = Depends(_get_inference),
    metrics_svc: MetricsService = Depends(_get_metrics),
    cfg=Depends(_get_cfg),
):
    """One-shot voice cloning. Upload reference audio + text to synthesize."""
    # Fail-fast: reject oversized uploads before reading body
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            cl_bytes = int(content_length)
            if cl_bytes > cfg.max_ref_audio_bytes:
                cl_mb = cl_bytes / 1024 / 1024
                limit_mb = cfg.max_ref_audio_bytes / 1024 / 1024
                logger.warning(
                    f"Rejected upload: Content-Length {cl_mb:.1f}MB > limit {limit_mb:.0f}MB"
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Upload too large: {cl_mb:.1f}MB exceeds limit of {limit_mb:.0f}MB",
                )
        except ValueError:
            pass  # Invalid Content-Length header — let body validation handle it

    from ..utils.audio import read_upload_bounded, validate_audio_bytes

    raw = await ref_audio.read()
    try:
        audio_bytes = read_upload_bounded(raw, cfg.max_ref_audio_bytes)
        validate_audio_bytes(audio_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "ref_audio.wav")
        Path(tmp_path).write_bytes(audio_bytes)

        req = SynthesisRequest(
            text=text,
            mode="clone",
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            t_shift=t_shift,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            duration=duration,
            language=language,
        )
        try:
            result = await inference_svc.synthesize(req)
            metrics_svc.record_success(result.latency_s)
        except QueueFullError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
                headers={"Retry-After": "1"},
            )
        except asyncio.TimeoutError:
            metrics_svc.record_timeout()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Synthesis timed out after {cfg.request_timeout_s}s",
            )
        except Exception as e:
            metrics_svc.record_error()
            logger.exception("Clone synthesis failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Synthesis failed: {e}",
            )

        audio_out, media_type = encode_tensors(result.tensors, "wav")

        if cfg.trace_dir:
            _write_trace(cfg.trace_dir, audio_out, "wav", _build_trace_meta(
                request=request,
                cfg=cfg,
                text=text,
                voice=None,
                profile_id=None,
                mode="clone",
                instruct=None,
                ref_text=ref_text,
                latency_s=result.latency_s,
                duration_s=result.duration_s,
                num_step=num_step or cfg.num_step,
                guidance_scale=guidance_scale if guidance_scale is not None else cfg.guidance_scale,
                speed=speed,
                fmt="wav",
                audio_size_bytes=len(audio_out),
            ))

        return Response(
            content=audio_out,
            media_type=media_type,
            headers={
                "X-Audio-Duration-S": str(round(result.duration_s, 3)),
                "X-Synthesis-Latency-S": str(round(result.latency_s, 3)),
            },
        )
