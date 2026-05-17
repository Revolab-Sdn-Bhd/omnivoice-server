"""
Shared utilities for all TTS routers.

Voice resolution, request construction, and dependency helpers
live here so inference pipeline changes propagate from one place.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from fastapi import Request

from ..services.inference import InferenceService, SynthesisRequest
from ..services.metrics import MetricsService


def get_inference_svc(request: Request) -> InferenceService:
    return request.app.state.inference_svc


def get_metrics_svc(request: Request) -> MetricsService:
    return request.app.state.metrics_svc


def get_cfg(request: Request):
    return request.app.state.cfg


def resolve_voice(
    voice_ref: str | None,
    cfg,
    *,
    voice_ref_audio: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve voice reference to (audio_path, ref_text).

    Handles:
    - base64-encoded WAV (voice_ref_audio) -> temp file
    - voice ID stem (e.g. "anwar") -> voices_dir/anwar.wav
    - explicit file path -> used directly
    Returns (None, None) if voice cannot be resolved.
    """
    if voice_ref_audio:
        raw = base64.b64decode(voice_ref_audio)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(raw)
        tmp.close()
        return tmp.name, ""

    if not voice_ref:
        return None, None

    p = Path(voice_ref)
    if not p.is_file() and not p.suffix:
        p = cfg.voices_dir / f"{voice_ref}.wav"

    if not p.is_file():
        return None, None

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


def build_synthesis_request(
    text: str,
    cfg,
    *,
    voice_ref: str | None = None,
    voice_ref_audio: str | None = None,
    instruct: str | None = None,
    speed: float = 1.0,
    duration: float | None = None,
    num_step: int | None = None,
    language: str | None = None,
    guidance_scale: float | None = None,
    denoise: bool | None = None,
    t_shift: float | None = None,
    position_temperature: float | None = None,
    class_temperature: float | None = None,
    layer_penalty_factor: float | None = None,
    preprocess_prompt: bool | None = None,
    postprocess_output: bool | None = None,
    audio_chunk_duration: float | None = None,
    audio_chunk_threshold: float | None = None,
    embedding_cache_path: str | None = None,
    _trace_id: str | None = None,
) -> SynthesisRequest:
    """Build SynthesisRequest with resolved voice and determined mode."""
    audio_path, ref_text = resolve_voice(voice_ref, cfg, voice_ref_audio=voice_ref_audio)

    if audio_path:
        mode = "clone"
    elif instruct:
        mode = "design"
    else:
        mode = "design"

    return SynthesisRequest(
        text=text,
        mode=mode,
        instruct=instruct,
        ref_audio_path=audio_path,
        ref_text=ref_text,
        speed=speed,
        duration=duration,
        num_step=num_step,
        language=language,
        guidance_scale=guidance_scale,
        denoise=denoise,
        t_shift=t_shift,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        layer_penalty_factor=layer_penalty_factor,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
        embedding_cache_path=embedding_cache_path,
        _trace_id=_trace_id,
    )


def tensor_to_base64_float32(tensor) -> str:
    """Convert tensor to base64-encoded raw float32 PCM bytes."""
    import numpy as np

    if isinstance(tensor, np.ndarray):
        flat = tensor.squeeze()
    else:
        flat = tensor.squeeze(0).cpu().float().numpy()
    return base64.b64encode(flat.astype(np.float32).tobytes()).decode("ascii")
