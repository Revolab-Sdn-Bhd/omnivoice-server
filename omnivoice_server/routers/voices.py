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
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_voices_dir(cfg) -> Path:
    """Resolve the correct voices subdirectory from an HF snapshot path."""
    hf_path = cfg.voices_dir
    # Walk up to find the snapshot root (parent of dataset/ or data/)
    if hf_path.name in ("dataset", "data"):
        hf_path = hf_path.parent
    if any((hf_path / "dataset").glob("*.wav")):
        return hf_path / "dataset"
    elif any((hf_path / "data").glob("*.wav")):
        return hf_path / "data"
    return hf_path


def _get_cfg(request: Request):
    return request.app.state.cfg


def _get_profiles(request: Request):
    return request.app.state.profile_svc


def _load_voice_meta(voices_dir: Path, voice_id: str) -> dict:
    """Load companion metadata for a voice, or return empty dict."""
    meta = {}
    # Try companion .json (HF dataset format with transcript)
    json_path = voices_dir / f"{voice_id}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            meta["transcript"] = data.get("transcript", "")
            meta["language_detected"] = data.get("language_detected", "")
            meta["source"] = data.get("source", "")
            if data.get("speaker_display_name"):
                meta["display_name"] = data["speaker_display_name"].replace("_", " ").replace("-", " ")
            if "gender" in data and data["gender"]:
                meta["gender"] = data["gender"]
            if "language" in data and data["language"]:
                meta["language"] = data["language"]
            if "tags" in data and data["tags"]:
                meta["tags"] = data["tags"]
            if "duration" in data:
                meta["duration"] = data["duration"]
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse JSON for voice '%s'", voice_id)
    # Override with .meta.json if present (user uploads)
    meta_path = voices_dir / f"{voice_id}.meta.json"
    if meta_path.exists():
        try:
            meta.update(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse metadata for voice '%s'", voice_id)
    return meta


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
            "name": meta.get("display_name", meta.get("name", name)),
            "path": str(wav),
            "transcript": meta.get("transcript", ""),
            "language_detected": meta.get("language_detected", ""),
            "source": meta.get("source", ""),
            "gender": meta.get("gender"),
            "language": meta.get("language"),
            "description": meta.get("description"),
            "tags": meta.get("tags", []),
            "duration": meta.get("duration"),
        })
    return voices


# ── GET /voices/{voice_id}/audio ───────────────────────────────────────────────


@router.get("/voices/{voice_id}/audio", tags=["Voices"])
async def get_voice_audio(voice_id: str, cfg=Depends(_get_cfg)):
    """Serve the original WAV audio file for a voice."""
    wav_path = cfg.voices_dir / f"{voice_id}.wav"
    if not wav_path.is_file():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return FileResponse(str(wav_path), media_type="audio/wav")


# ── GET /voices ────────────────────────────────────────────────────────────────


@router.get("/voices", tags=["Voices"])
async def list_voices(cfg=Depends(_get_cfg)):
    """List all available voice WAV files from the voices/ directory."""
    voices = _scan_voices(cfg.voices_dir)
    return {"voices": voices}


@router.get("/speakers", tags=["Voices"])
async def list_speakers(cfg=Depends(_get_cfg)):
    """Legacy alias for GET /voices. Returns speakers key instead of voices."""
    voices = _scan_voices(cfg.voices_dir)
    return {"speakers": voices}


@router.post("/voices/refresh", tags=["Voices"])
async def refresh_voices(request: Request):
    """Re-check HuggingFace for new voices and update if changed."""
    cfg = request.app.state.cfg
    if not cfg.voices_hf_repo:
        return {"voices": _scan_voices(cfg.voices_dir), "updated": False, "reason": "no_hf_repo"}

    current_dir = str(cfg.voices_dir)

    import asyncio

    from huggingface_hub import snapshot_download

    loop = asyncio.get_running_loop()
    new_path = await loop.run_in_executor(
        None,
        lambda: snapshot_download(
            cfg.voices_hf_repo,
            repo_type="dataset",
            force_download=True,
        ),
    )

    new_voices_dir = _resolve_voices_dir_from_snapshot(Path(new_path))

    if str(new_voices_dir) == current_dir:
        voices = _scan_voices(cfg.voices_dir)
        return {"voices": voices, "updated": False}

    cfg.voices_dir = new_voices_dir
    logger.info("Voices updated: %s -> %s", current_dir, new_voices_dir)
    voices = _scan_voices(cfg.voices_dir)
    return {"voices": voices, "updated": True}


def _resolve_voices_dir_from_snapshot(hf_path: Path) -> Path:
    """Pick the right subdirectory from an HF snapshot."""
    if any((hf_path / "dataset").glob("*.wav")):
        return hf_path / "dataset"
    elif any((hf_path / "data").glob("*.wav")):
        return hf_path / "data"
    return hf_path


# ── POST /voices ──────────────────────────────────────────────────────────────


@router.post("/voices", status_code=status.HTTP_201_CREATED, tags=["Voices"])
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


@router.post("/v1/voices/profiles", status_code=status.HTTP_201_CREATED, tags=["Voice Profiles"])
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


@router.get("/v1/voices/profiles/{profile_id}", tags=["Voice Profiles"])
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


@router.delete("/v1/voices/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Voice Profiles"])
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


@router.patch("/v1/voices/profiles/{profile_id}", status_code=status.HTTP_200_OK, tags=["Voice Profiles"])
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
