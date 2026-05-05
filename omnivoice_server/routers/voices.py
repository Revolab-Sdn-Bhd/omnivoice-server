"""
GET  /voices             — list WAV files from voices/ dir (with metadata)
POST /voices             — upload a new voice WAV (with optional metadata)

Voice metadata is stored in companion `{voice_id}.meta.json` files alongside the WAV.
Metadata fields: gender, language, description, tags.

Profile management endpoints kept for OmniVoice clone support:
  /v1/voices/profiles/*
"""

from __future__ import annotations

import json
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


def _load_voice_meta(voices_dir: Path, voice_id: str) -> dict:
    """Load companion metadata for a voice, or return empty dict."""
    meta_path = voices_dir / f"{voice_id}.meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse metadata for voice '%s'", voice_id)
    return {}


def _scan_voices(voices_dir: Path) -> list[dict]:
    """Scan voices_dir for .wav files and return voice entries with metadata."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices = []
    for wav in sorted(voices_dir.glob("*.wav")):
        stem = wav.stem
        name = stem.replace("_", " ").replace("-", " ").title()
        meta = _load_voice_meta(voices_dir, stem)
        voices.append({
            "id": stem,
            "name": meta.get("name", name),
            "path": str(wav),
            "gender": meta.get("gender"),
            "language": meta.get("language"),
            "description": meta.get("description"),
            "tags": meta.get("tags", []),
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
    ref_text: str = Form(..., min_length=1, description="Transcript of the reference audio"),
    audio_file: UploadFile = File(..., description="Voice reference WAV audio"),
    gender: str = Form(default="", description="Voice gender: male, female, neutral"),
    language: str = Form(
        default="",
        description="Comma-separated language codes (e.g., ms, en, zh)",
    ),
    description: str = Form(default="", description="Description of the voice"),
    tags: str = Form(default="", description="Comma-separated tags"),
    cfg=Depends(_get_cfg),
):
    """Upload a new WAV file and register it as a voice with optional metadata."""
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

    # Save companion transcript
    txt_path = voices_dir / f"{sanitized}.txt"
    txt_path.write_text(ref_text.strip())

    # Save companion metadata
    display_name = sanitized.replace("_", " ").replace("-", " ").title()
    meta = {"name": display_name}
    if gender:
        meta["gender"] = gender
    if language:
        meta["language"] = [lang.strip() for lang in language.split(",") if lang.strip()]
    if description:
        meta["description"] = description
    if tags:
        meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    meta_path = voices_dir / f"{sanitized}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "id": sanitized,
        "name": display_name,
        "path": str(dest.resolve()),
        "gender": meta.get("gender"),
        "language": meta.get("language"),
        "description": meta.get("description"),
        "tags": meta.get("tags", []),
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
