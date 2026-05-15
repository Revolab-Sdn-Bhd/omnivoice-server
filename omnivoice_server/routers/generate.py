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
from pydantic import BaseModel, ConfigDict, Field

from ..services.inference import InferenceService, QueueFullError, SynthesisRequest
from ..services.metrics import MetricsService
from ..utils.audio import tensors_to_wav_bytes
from ..utils.text import split_sentences

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Prompt presets ──────────────────────────────────────────────────────────────

QUOTES: dict[str, list[str]] = {
    "Tone": [
        "How are you doing today?",
        "What time is the meeting?",
        "Can you help me with this task?",
        "Do you know where the nearest restaurant is?",
        "Are you sure about this decision?",
        "Cik Nurul Amira ada email registered sebagai nurul.amira89@gmail.com, betul ya?",
        "Okay, terima kasih. By the way, saya perasan ada satu transaction yang saya tak kenal. Boleh check tak RM1,500 transaction pada 18 Februari?",  # noqa: E501
        "Apa khabar? Bagaimana keadaan kamu?",
        "Boleh saya tolong?",
        "That's amazing!",
        "I can't believe it!",
        "Wow, what an incredible result!",
        "This is fantastic news!",
        "Congratulations on your success!",
        "Syabas! Tahniah!",
        "Hebat sangat!",
    ],
    "Digits": [
        "My account number is AA212CC8819000ZZ.",
        "Your Windows 11 home activation code is YTMG3-N6DKC-DKB77-7M9GH-8HVX7.",
        "The activation key for Office 365 has been sent to cheelam1829@revolab.ai – also check your spam folder.",  # noqa: E501
        "Nombor akaun Encik Hafiz ialah 1640-8821-4490, berdaftar atas nama Hafiz bin Kamal.",
    ],
    "Entity": [
        "nurul.amira89@gmail.com",
        "No. 8, Jalan Setia 3/4, Setia Alam, 40170 Shah Alam",
        "Encik Rajesh",
        "Cik Nurul Amira",
        "Farah Nabila binti Yazid",
        "Hafiz bin Kamal",
        "Arif Catering Services",
        "Maybank Customer Service",
        "18 Februari",
        "Nicholas Chua Jun Kit",
        "Thinesh anak lelaki Narayanasamy",
        "university of Malaya",
        "FSKTM",
        "mamak KLCC",
        "D-King",
    ],
    "Code-mixing": [
        "Eh, jom makan nasi lemak dekat mamak KLCC near D-King lepas kerja nanti.",
        "So easy also dunno meh?",
        "They got sell Nasi Lemak lah, Roti Canai lah, Chapatti lah, everything got lah.",
        "Kalau nak beli rumah, better check your DSR dulu – Debt Service Ratio kena below 70%.",
        "Company bagi bonus dua bulan gaji tahun ni, so I plan nak bayar hutang PTPTN dulu.",
    ],
    "Quoted words": [
        "Dia 'on the way' ke 'meeting' tapi tersekat dalam 'jam'!",
        "Kereta itu mempunyai 'suspension' yang 'sporty' dan 'handling' yang mantap.",
    ],
    "Long text": [
        "Kami perlukan selfie dengan IC dan signature di borang KYC untuk akaun business bawah nama Arif Catering Services.",  # noqa: E501
        "Farah Nabila binti Yazid ialah kawan Nicholas Chua Jun Kit and Thinesh anak lelaki Narayanasamy. So 3 of them went to university of Malaya to study computer science specialized in AI at FSKTM.",  # noqa: E501
    ],
    "General": [
        "Hi",
        "Okay, terima kasih sebab call Maybank Customer Service. Have a nice day!",
        "Baik, terima kasih.",
        "The file sha1 hash is 4f2d5e7a3b9c8d1e6f4a2b0d3c9e8f7a1d2e3c4f",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",  # noqa: E501
        "Spread love everywhere you go. Let no one ever come to you without leaving happier. - Mother Teresa",  # noqa: E501
    ],
}


@router.get("/api/quotes", tags=["TTS"])
async def get_quotes():
    """Returns categorized test sentences for TTS evaluation."""
    # Flatten for backward compat
    all_quotes = [q for qs in QUOTES.values() for q in qs]
    return {"quotes": all_quotes, "categories": QUOTES}


@router.post("/api/normalize", tags=["TTS"])
async def normalize_text_endpoint(request: Request):
    """Normalize text for TTS and return the result."""
    body = await request.json()
    text = body.get("text", "")
    language = body.get("language", "en")
    if not text:
        return {"normalized": "", "original": text}
    from ..utils.text import normalize_for_tts
    normalized = normalize_for_tts(text, language=language)
    return {"normalized": normalized, "original": text}


FRAME_RATE = 25


@router.post("/api/estimate-duration", tags=["TTS"])
async def estimate_duration_endpoint(request: Request):
    """Estimate audio duration for given text and speaker."""
    from omnivoice.utils.duration import RuleDurationEstimator

    body = await request.json()
    text = body.get("text", "")
    voice_id = body.get("voice_id", "")

    if not text or not voice_id:
        return {"estimated_duration_s": 0, "estimated_frames": 0, "frame_rate": FRAME_RATE}

    cfg = request.app.state.cfg
    wav_path = cfg.voices_dir / f"{voice_id}.wav"
    if not wav_path.is_file():
        return {"estimated_duration_s": 0, "estimated_frames": 0, "frame_rate": FRAME_RATE}

    # Get ref text and duration from voice metadata
    from .voices import _load_voice_meta
    meta = _load_voice_meta(cfg.voices_dir, voice_id)
    ref_text = meta.get("transcript", "")
    ref_duration_s = meta.get("duration", 0)

    if not ref_text or ref_duration_s <= 0:
        return {"estimated_duration_s": 0, "estimated_frames": 0, "frame_rate": FRAME_RATE}

    ref_frames = int(ref_duration_s * FRAME_RATE)
    estimator = RuleDurationEstimator()
    ref_weight = estimator.calculate_total_weight(ref_text)
    target_weight = estimator.calculate_total_weight(text)
    est_frames = estimator.estimate_duration(text, ref_text, ref_frames)
    est_duration = est_frames / FRAME_RATE

    return {
        "estimated_duration_s": round(est_duration, 2),
        "estimated_frames": int(est_frames),
        "frame_rate": FRAME_RATE,
        "ref_duration_s": ref_duration_s,
        "ref_frames": ref_frames,
        "formula": {
            "target_weight": round(target_weight, 1),
            "ref_weight": round(ref_weight, 1),
            "speed_factor": round(ref_weight / ref_frames, 3) if ref_frames > 0 else 0,
            "est_frames": (
                f"{target_weight:.0f} / ({ref_weight:.0f}"
                f" / {ref_frames}) = {int(est_frames)}"
            ),
            "est_duration": f"{int(est_frames)} / {FRAME_RATE} = {est_duration:.2f}s",
        },
    }


# ── Request model ──────────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    """SepBox-compatible TTS request body."""

    model_config = ConfigDict(extra="forbid")

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
    num_step: int | None = Field(default=None, ge=2, le=32)
    speed: float = Field(default=1.0, ge=0.5, le=3.0)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    denoise: bool | None = Field(default=None)
    t_shift: float | None = Field(default=None, ge=0.0, le=1.0)
    position_temperature: float | None = Field(default=None, ge=0.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    layer_penalty_factor: float | None = Field(default=None)
    preprocess_prompt: bool | None = Field(default=None)
    postprocess_output: bool | None = Field(default=None)
    audio_chunk_duration: float | None = Field(default=None)
    audio_chunk_threshold: float | None = Field(default=None)
    target_lufs: float = Field(default=-23.0, ge=-60.0, le=0.0)
    duration: float | None = Field(
        default=None, ge=0.1, le=60.0,
        description="Fixed output duration in seconds. None = auto-estimate.",
    )
    instruct: str | None = Field(
        default=None, description="Voice design instruction",
    )


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

        # Look for companion .txt or .json for ref_text
        txt_path = p.with_suffix(".txt")
        json_path = p.with_suffix(".json")
        if txt_path.exists():
            ref_text = txt_path.read_text().strip()
        elif json_path.exists():
            try:
                ref_text = json.loads(json_path.read_text()).get("transcript", "")
            except (json.JSONDecodeError, OSError):
                ref_text = ""
        else:
            ref_text = ""
        return str(p), ref_text

    return None, None


def _build_synthesis_req(body: TTSRequest, cfg) -> SynthesisRequest:
    """Map SepBox TTSRequest to OmniVoice SynthesisRequest."""
    audio_path, ref_text = _resolve_voice_path(
        body.voice_ref_path, body.voice_ref_audio, cfg
    )

    if audio_path:
        mode = "clone"
    elif body.instruct:
        mode = "design"
    else:
        mode = "design"

    normalized_text = body.text

    return SynthesisRequest(
        text=normalized_text,
        mode=mode,
        instruct=body.instruct,
        ref_audio_path=audio_path,
        ref_text=ref_text,
        speed=body.speed,
        duration=body.duration,
        num_step=body.num_step,
        language=body.language if body.language != "en" else None,
        guidance_scale=body.guidance_scale,
        denoise=body.denoise,
        t_shift=body.t_shift,
        position_temperature=body.position_temperature,
        class_temperature=body.class_temperature,
        layer_penalty_factor=body.layer_penalty_factor,
        preprocess_prompt=body.preprocess_prompt,
        postprocess_output=body.postprocess_output,
        audio_chunk_duration=body.audio_chunk_duration,
        audio_chunk_threshold=body.audio_chunk_threshold,
    )


def _tensor_to_base64_float32(tensor) -> str:
    """Convert tensor to base64-encoded raw float32 PCM bytes."""
    import numpy as np

    if isinstance(tensor, np.ndarray):
        flat = tensor.squeeze()
    else:
        flat = tensor.squeeze(0).cpu().float().numpy()
    return base64.b64encode(flat.astype(np.float32).tobytes()).decode("ascii")


# ── POST /generate ────────────────────────────────────────────────────────────


@router.post("/generate", tags=["TTS"])
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

    wav_bytes = tensors_to_wav_bytes(
        result.tensors, target_lufs=body.target_lufs,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Latency-S": f"{result.latency_s:.3f}",
            "X-Audio-Duration-S": f"{result.duration_s:.3f}",
            "X-RTF": (
                f"{result.latency_s / result.duration_s:.3f}"
                if result.duration_s > 0
                else "inf"
            ),
        },
    )


# ── POST /generate/stream ─────────────────────────────────────────────────────


@router.post("/generate/stream", tags=["TTS"])
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
            speed=base_req.speed,
            language=base_req.language,
            num_step=base_req.num_step,
            guidance_scale=base_req.guidance_scale,
            denoise=base_req.denoise,
            t_shift=base_req.t_shift,
            position_temperature=base_req.position_temperature,
            class_temperature=base_req.class_temperature,
            layer_penalty_factor=base_req.layer_penalty_factor,
            preprocess_prompt=base_req.preprocess_prompt,
            postprocess_output=base_req.postprocess_output,
            audio_chunk_duration=base_req.audio_chunk_duration,
            audio_chunk_threshold=base_req.audio_chunk_threshold,
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
