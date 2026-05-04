"""
GET  /voices             — list WAV files from voices/ dir
POST /voices             — upload a new voice WAV
GET  /api/speakers       — legacy alias (key "speakers")
POST /api/create_speaker — legacy alias (param "speaker_name")

Profile management endpoints kept for OmniVoice clone support:
  /v1/voices/profiles/*
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_cfg(request: Request):
    return request.app.state.cfg


def _get_profiles(request: Request):
    return request.app.state.profile_svc


def _scan_voices(voices_dir: Path) -> list[dict]:
    """Scan voices_dir for .wav files and return voice entries."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices = []
    for wav in sorted(voices_dir.glob("*.wav")):
        stem = wav.stem
        name = stem.replace("_", " ").replace("-", " ").title()
        voices.append({
            "id": stem,
            "name": name,
            "path": str(wav),
        })
    return voices


# ── GET /voices ────────────────────────────────────────────────────────────────


@router.get("/voices")
async def list_voices(cfg=Depends(_get_cfg)):
    """List all available voice WAV files from the voices/ directory."""
    voices = _scan_voices(cfg.voices_dir)
    return {"voices": voices}


# ── POST /voices ──────────────────────────────────────────────────────────────


@router.post("/voices", status_code=status.HTTP_201_CREATED)
async def upload_voice(
    voice_name: str = Form(..., description="Name for the voice (alphanumeric, dash, underscore)"),
    audio_file: UploadFile = File(..., description="Voice reference WAV audio"),
    cfg=Depends(_get_cfg),
):
    """Upload a new WAV file and register it as a voice."""
    # Sanitize name
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", voice_name)
    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="voice_name contains no valid characters",
        )

    # Read and validate size
    raw = await audio_file.read()
    max_bytes = 50 * 1024 * 1024  # 50 MB per spec
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(raw) / 1024 / 1024:.1f}MB exceeds 50MB limit",
        )

    # Save WAV file
    voices_dir: Path = cfg.voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)
    dest = voices_dir / f"{sanitized}.wav"
    try:
        dest.write_bytes(raw)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save voice file: {e}",
        )

    return {
        "id": sanitized,
        "name": sanitized.replace("_", " ").replace("-", " ").title(),
        "path": str(dest.resolve()),
    }


# ── GET /api/speakers (legacy) ────────────────────────────────────────────────


@router.get("/api/speakers")
async def list_speakers(cfg=Depends(_get_cfg)):
    """Legacy alias for GET /voices. Uses key 'speakers' instead of 'voices'."""
    voices = _scan_voices(cfg.voices_dir)
    return {"speakers": voices}


# ── POST /api/create_speaker (legacy) ─────────────────────────────────────────


@router.post("/api/create_speaker", status_code=status.HTTP_201_CREATED)
async def create_speaker(
    speaker_name: str = Form(..., description="Name for the voice"),
    audio_file: UploadFile = File(..., description="Voice reference WAV audio"),
    cfg=Depends(_get_cfg),
):
    """Legacy alias for POST /voices. Uses speaker_name instead of voice_name."""
    # Reuse the upload_voice logic by calling it with the same params
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "", speaker_name)
    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="speaker_name contains no valid characters",
        )

    raw = await audio_file.read()
    max_bytes = 50 * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(raw) / 1024 / 1024:.1f}MB exceeds 50MB limit",
        )

    voices_dir: Path = cfg.voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)
    dest = voices_dir / f"{sanitized}.wav"
    try:
        dest.write_bytes(raw)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save voice file: {e}",
        )

    return {
        "id": sanitized,
        "name": sanitized.replace("_", " ").replace("-", " ").title(),
        "path": str(dest.resolve()),
    }


# ── Profile management (kept for OmniVoice clone support) ─────────────────────


@router.post("/v1/voices/profiles", status_code=status.HTTP_201_CREATED)
async def create_profile(
    request: Request,
    profile_id: str = Form(
        ...,
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Unique identifier. Alphanumeric, dashes, underscores only.",
    ),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(..., description="Transcript of the reference audio. Required."),
    overwrite: bool = Form(default=False),
    profile_svc=Depends(_get_profiles),
):
    """Save a voice cloning profile."""
    from ..services.profiles import ProfileAlreadyExistsError
    from ..utils.audio import read_upload_bounded, validate_audio_bytes

    cfg = request.app.state.cfg
    raw = await ref_audio.read()
    try:
        audio_bytes = read_upload_bounded(raw, cfg.max_ref_audio_bytes)
        validate_audio_bytes(audio_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )

    try:
        meta = profile_svc.save_profile(
            profile_id=profile_id,
            audio_bytes=audio_bytes,
            ref_text=ref_text,
            overwrite=overwrite,
        )
    except ProfileAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    return meta


@router.get("/v1/voices/profiles/{profile_id}")
async def get_profile(
    profile_id: str,
    profile_svc=Depends(_get_profiles),
):
    """Get a voice profile."""
    profiles = profile_svc.list_profiles()
    profile = next((p for p in profiles if p["profile_id"] == profile_id), None)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found",
        )
    return profile


@router.delete("/v1/voices/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    profile_id: str,
    profile_svc=Depends(_get_profiles),
):
    """Delete a voice profile."""
    from ..services.profiles import ProfileNotFoundError

    try:
        profile_svc.delete_profile(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found",
        )


@router.patch("/v1/voices/profiles/{profile_id}", status_code=status.HTTP_200_OK)
async def update_profile(
    profile_id: str,
    request: Request,
    ref_audio: UploadFile | None = File(default=None),
    ref_text: str | None = Form(default=None),
    profile_svc=Depends(_get_profiles),
):
    """Update an existing profile."""
    from ..services.profiles import ProfileNotFoundError
    from ..utils.audio import read_upload_bounded, validate_audio_bytes

    try:
        existing_path = profile_svc.get_ref_audio_path(profile_id)
    except ProfileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found",
        )

    if ref_audio is None and ref_text is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide at least one of: ref_audio, ref_text",
        )

    if ref_audio is not None and ref_text is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="ref_text is required when updating ref_audio — provide the transcript.",
        )

    if ref_audio is not None:
        cfg = request.app.state.cfg
        raw = await ref_audio.read()
        try:
            audio_bytes = read_upload_bounded(raw, cfg.max_ref_audio_bytes)
            validate_audio_bytes(audio_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(e),
            )
        meta = profile_svc.save_profile(
            profile_id=profile_id,
            audio_bytes=audio_bytes,
            ref_text=ref_text,
            overwrite=True,
        )
    else:
        audio_bytes = existing_path.read_bytes()
        meta = profile_svc.save_profile(
            profile_id=profile_id,
            audio_bytes=audio_bytes,
            ref_text=ref_text,
            overwrite=True,
        )

    return meta
