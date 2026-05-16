"""
FastAPI application factory.
All shared state lives on app.state — no module-level globals.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path


def _resolve_hf_revision(repo_id: str, repo_type: str) -> str:
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id and repo.repo_type == repo_type:
                rev = next(iter(repo.revisions))
                return rev.commit_hash[:8]
    except Exception:
        pass
    return ""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .observability.tracer import (
    flush_blocking,
    flush_in_background,
    get_langfuse_client,
    is_enabled,
    join_background_flushes,
)
from .routers import generate, health, models, speech, voices, websocket
from .services.inference import InferenceService, SynthesisRequest
from .services.metrics import MetricsService
from .services.model import ModelService
from .services.profiles import ProfileService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Settings = app.state.cfg

    # Startup config validation
    _validate_config(cfg)

    # Initialize Langfuse tracing
    langfuse = get_langfuse_client(
        host=cfg.langfuse_base_url or None,
    )
    if langfuse:
        try:
            langfuse.auth_check()
            logger.info("Langfuse connected and tracing enabled")
        except Exception as e:
            logger.warning("Langfuse auth failed: %s", e)
    else:
        logger.info("Langfuse not configured (missing credentials)")

    app.state.start_time = time.time()
    app.state.metrics_svc = MetricsService()

    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    app.state.profile_svc = ProfileService(profile_dir=cfg.profile_dir)

    # Download voices from HuggingFace if configured
    if cfg.voices_hf_repo:
        from huggingface_hub import snapshot_download
        logger.info("Downloading voices from %s...", cfg.voices_hf_repo)
        hf_cache = snapshot_download(cfg.voices_hf_repo, repo_type="dataset")
        cfg.voices_revision_hash = _resolve_hf_revision(cfg.voices_hf_repo, "dataset")
        hf_path = Path(hf_cache)
        # HF repo structure may use "data/" or "dataset/"
        if (hf_path / "data" / "*.wav").exists() or any((hf_path / "data").glob("*.wav")):
            cfg.voices_dir = hf_path / "data"
        elif any((hf_path / "dataset").glob("*.wav")):
            cfg.voices_dir = hf_path / "dataset"
        else:
            cfg.voices_dir = hf_path
        logger.info("Voices ready at %s", cfg.voices_dir)
    else:
        cfg.voices_dir.mkdir(parents=True, exist_ok=True)

    # Auto-register profiles from voice_samples/
    _auto_register_voice_samples(app.state.profile_svc)

    # Response cache
    if cfg.response_cache_enabled:
        from .services.response_cache import ResponseCache

        app.state.response_cache = ResponseCache(
            cfg.profile_dir / "response_cache", cfg.response_cache_max_gb
        )
        logger.info("Response cache enabled (%.1f GB)", cfg.response_cache_max_gb)
    else:
        app.state.response_cache = None

    if cfg.workers == 1:
        # Single-worker mode: load model here, use ThreadPoolExecutor
        model_svc = ModelService(cfg)
        app.state.model_svc = model_svc
        executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent)
        app.state.executor = executor
        app.state.inference_svc = InferenceService(model_svc, cfg, executor=executor)

        await model_svc.load()
    else:
        # Multi-worker mode.
        # IMPORTANT SAFETY INVARIANT: This code path only runs inside forked worker
        # processes. The parent process never calls create_app() or runs this lifespan —
        # it goes directly to worker_mgr.monitor() in cli.py. CUDA initialization here
        # is safe because it happens AFTER fork, in an isolated child process.
        model_svc = ModelService(cfg)
        app.state.model_svc = model_svc
        app.state.executor = None
        app.state.inference_svc = InferenceService(model_svc, cfg, executor=None)

        await model_svc.load()

        # Slot 0 worker: measure peak VRAM and write to temp file for parent
        worker_slot = int(os.environ.get("OMNIVOICE_WORKER_SLOT", "-1"))
        if worker_slot == 0 and cfg.device == "cuda":
            import json

            import torch

            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / 1024 / 1024
            free_bytes, total_bytes = torch.cuda.mem_get_info()

            vram_file = "/tmp/omnivoice_vram_measurement.json"

            with open(vram_file, "w") as f:
                json.dump({
                    "peak_vram_mb": round(peak_mb, 2),
                    "total_vram_mb": round(total_bytes / 1024 / 1024, 2),
                }, f)
            logger.info("VRAM measurement written: %.0f MB -> %s", peak_mb, vram_file)

    # Auto-benchmark and warmup (CUDA only — benchmark has no meaning on CPU/MPS)
    if app.state.model_svc.is_loaded and cfg.device == "cuda":
        loop = asyncio.get_running_loop()
        model = app.state.model_svc.model
        cache_dir = str(cfg.compile_cache_dir) if cfg.compile_cache_dir else None

        if cfg.batch_enabled:
            from .services.gpu_benchmark import find_optimal_batch_size

            # Run GPU benchmark to find optimal batch size
            bench = await loop.run_in_executor(
                None, find_optimal_batch_size, model, cfg.num_step, cache_dir,
            )
            app.state.gpu_benchmark = bench
            optimal_bs = bench["optimal_batch_size"]

            optimal_timeout = bench["optimal_batch_timeout_ms"]
            updates = {}
            if cfg.batch_max_size == 4:  # default value, not user-set
                updates["batch_max_size"] = optimal_bs
            if cfg.batch_timeout_ms == 50:  # default value, not user-set
                updates["batch_timeout_ms"] = optimal_timeout
            if updates:
                cfg = cfg.model_copy(update=updates)
                app.state.cfg = cfg
                app.state.inference_svc._cfg = cfg
            logger.info(
                "Batch config: max_size=%d, timeout=%dms",
                cfg.batch_max_size, cfg.batch_timeout_ms,
            )
        else:
            logger.info("Batching disabled, using single-request mode")

        # Warmup to prime CUDA kernels
        try:
            warmup_text = "Warmup inference for CUDA kernel compilation."
            for _ in range(3):
                await app.state.inference_svc.synthesize(
                    SynthesisRequest(text=warmup_text, mode="auto")
                )
            logger.info("Warmup complete")
        except Exception:
            logger.warning("Warmup encountered errors (non-fatal)")

    # Start batch scheduler if enabled
    if hasattr(app.state.inference_svc, "start_batch_scheduler"):
        app.state.inference_svc.start_batch_scheduler()

    yield

    # Shutdown
    if langfuse:
        logger.info("Flushing Langfuse traces...")
        flush_blocking()
        join_background_flushes()
        logger.info("Langfuse shutdown complete")

    if hasattr(app.state.inference_svc, "stop_batch_scheduler"):
        app.state.inference_svc.stop_batch_scheduler()
    if getattr(app.state, "executor", None):
        app.state.executor.shutdown(wait=False)


def _auto_register_voice_samples(profile_svc: ProfileService) -> None:
    """Register audio files in voice_samples/ that have a companion .txt transcript.

    Supports .wav and .mp3. Audio is converted to 24 kHz mono WAV before saving.
    """
    import pathlib

    import torchaudio

    samples_dir = pathlib.Path("voice_samples")
    if not samples_dir.is_dir():
        return

    from .services.profiles import ProfileAlreadyExistsError

    audio_files = sorted(
        f for f in samples_dir.iterdir()
        if f.suffix.lower() in (".wav", ".mp3", ".m4a")
    )

    for audio in audio_files:
        txt = audio.with_suffix(".txt")
        if not txt.exists():
            logger.warning(
                "voice_samples/%s has no companion .txt — skipping auto-register", audio.name
            )
            continue

        # Sanitize stem to a valid profile_id (mirrors ProfileService._profile_path)
        profile_id = "".join(c for c in audio.stem if c.isalnum() or c in "-_")
        if not profile_id:
            logger.warning(
                "voice_samples/%s: cannot derive a valid profile_id — skipping", audio.name
            )
            continue

        ref_text = txt.read_text().strip()

        # Convert to 24 kHz mono WAV bytes via temp file (torchaudio needs a path)
        try:
            import tempfile
            waveform, sr = torchaudio.load(str(audio))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            torchaudio.save(tmp_path, waveform, 24000)
            audio_bytes = pathlib.Path(tmp_path).read_bytes()
            pathlib.Path(tmp_path).unlink()
        except Exception as e:
            logger.warning("voice_samples/%s: audio load failed (%s) — skipping", audio.name, e)
            continue

        try:
            profile_svc.save_profile(profile_id, audio_bytes, ref_text=ref_text, overwrite=False)
            logger.info(
                "Auto-registered voice profile '%s' from voice_samples/%s", profile_id, audio.name
            )
        except ProfileAlreadyExistsError:
            logger.debug("Voice profile '%s' already exists — skipping", profile_id)


def _validate_config(cfg: Settings) -> None:
    """Fail-fast validation of config combinations."""
    if cfg.device == "cpu" and cfg.workers > 1:
        logger.warning("Multi-worker with CPU device has no benefit; using workers=1")
        cfg.workers = 1
    if cfg.device not in ("cuda",) and cfg.compile_mode != "none":
        logger.warning("torch.compile only benefits CUDA; disabling")
        cfg.compile_mode = "none"  # type: ignore[assignment]


def _status_to_code(status_code: int) -> str:
    _map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "validation_error",
        500: "inference_failed",
        503: "model_not_ready",
        504: "timeout",
    }
    return _map.get(status_code, f"http_{status_code}")


def _build_ws_schema() -> dict:
    """OpenAPI schema fragment for WS /tts/websocket (not auto-generated by FastAPI)."""
    return {
        "/tts/websocket": {
            "get": {
                "summary": "Cartesia-compatible streaming TTS",
                "description": (
                    "WebSocket endpoint for real-time streaming TTS. "
                    "Send JSON generate/cancel messages; receive audio_chunk, done, error messages.\n\n"
                    "**Generate message:**\n"
                    '```json\n{"voice": {"mode": "id", "id": "anwar"}, "language": "en", '
                    '"context_id": "ctx-1", "transcript": "Text to speak.", "continue": true}\n```\n\n'
                    "**Cancel message:**\n"
                    '```json\n{"cancel": true, "context_id": "ctx-1"}\n```\n\n'
                    "**Audio chunk response:**\n"
                    '```json\n{"type": "audio_chunk", "data": "<base64 int16 PCM>", '
                    '"done": false, "status_code": 206, "context_id": "ctx-1"}\n```\n\n'
                    "**Done response:**\n"
                    '```json\n{"type": "done", "done": true, "status_code": 200, '
                    '"context_id": "ctx-1"}\n```\n\n'
                    "Audio encoding: int16 PCM (`pcm_s16le`) at 24kHz, base64-encoded."
                ),
                "tags": ["WebSocket"],
                "responses": {
                    "101": {"description": "WebSocket upgrade — protocol as described above"},
                },
                "parameters": [],
            },
        },
    }


def create_app(cfg: Settings) -> FastAPI:
    app = FastAPI(
        title="RevoVoice",
        description="OpenAI-compatible TTS server for OmniVoice",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.cfg = cfg

    # ── OpenAPI override (inject WebSocket docs) ─────────────────────────────
    _openapi_cache = None

    def custom_openapi():
        nonlocal _openapi_cache
        if _openapi_cache is not None:
            return _openapi_cache
        from fastapi.openapi.utils import get_openapi
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["paths"].update(_build_ws_schema())
        _openapi_cache = schema
        return schema

    app.openapi = custom_openapi

    # ── Request ID middleware ────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get(
            "X-Request-ID", uuid.uuid4().hex[:16]
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if is_enabled():
            flush_in_background()
        return response

    # ── Auth middleware ───────────────────────────────────────────────────────
    if cfg.api_key:

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth for health and static
            if request.url.path in (
                "/health", "/",
                "/v1/models", "/v1/audio/voices",
            ) or (
                request.url.path.startswith("/static/")
                or request.url.path.startswith("/share/")
                or request.url.path == "/api/share"
                or request.url.path.startswith("/api/history")
                or request.url.path.startswith("/api/stats")
            ):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {cfg.api_key}":
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid or missing API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = exc.errors()
        hints = []
        for d in details:
            loc = ".".join(str(l) for l in d.get("loc", []))
            msg = d.get("msg", "")
            if d.get("type") == "extra_forbidden":
                hints.append(f"Unexpected field: {loc}")
            else:
                hints.append(f"{loc}: {msg}")
        message = "Request validation failed — " + "; ".join(hints)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": message,
                    "detail": details,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _status_to_code(exc.status_code),
                    "message": exc.detail,
                }
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Health (no prefix — serves /health directly)
    app.include_router(health.router)

    # OpenAI-compatible (prefix /v1)
    app.include_router(models.router, prefix="/v1")
    app.include_router(speech.router, prefix="/v1")

    # SepBox-compatible TTS (no prefix — serves /generate directly)
    app.include_router(generate.router)

    # Voice management (no prefix — serves /voices directly)
    app.include_router(voices.router)

    # WebSocket
    app.include_router(websocket.router)

    # ── Share endpoints ──────────────────────────────────────────────────────

    @app.post("/api/share", include_in_schema=False)
    async def create_share(request: Request):
        """Save generated audio + metadata and return a shareable link."""
        body = await request.json()
        audio_b64 = body.get("audio_base64")
        text = body.get("text", "")
        speaker_name = body.get("speaker_name", "")
        fmt = body.get("format", "wav")
        created_at = body.get("created_at", "")

        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_base64 is required")

        shared_dir: Path = cfg.profile_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)

        share_id = uuid.uuid4().hex[:12]

        # Save audio file
        audio_bytes = base64.b64decode(audio_b64)
        audio_path = shared_dir / f"{share_id}.{fmt}"
        audio_path.write_bytes(audio_bytes)

        # Save metadata
        duration = body.get("duration", 0.0)
        meta = {
            "text": text,
            "speaker_name": speaker_name,
            "format": fmt,
            "created_at": created_at,
            "duration": duration,
        }
        meta_path = shared_dir / f"{share_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))

        share_url = str(request.url_for("get_share_page", share_id=share_id))
        return {"share_id": share_id, "url": share_url}

    @app.get(
        "/share/{share_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
        name="get_share_page",
    )
    async def get_share_page(share_id: str):
        """Serve a minimal mobile-friendly audio player page."""
        shared_dir: Path = cfg.profile_dir / "shared"
        meta_path = shared_dir / f"{share_id}.meta.json"

        # Find audio file (could be wav/mp3/opus)
        audio_file = None
        for ext in ("wav", "mp3", "opus"):
            candidate = shared_dir / f"{share_id}.{ext}"
            if candidate.exists():
                audio_file = candidate
                break

        if not audio_file or not meta_path.exists():
            raise HTTPException(status_code=404, detail="Share not found")

        meta = json.loads(meta_path.read_text())
        text = meta.get("text", "")
        speaker_name = meta.get("speaker_name", "")
        audio_url = f"/share/{share_id}/audio"
        audio_fmt = meta.get("format", "wav")
        dl_name = f"{speaker_name}.{audio_fmt}"
        duration = meta.get("duration", 0.0)

        # Increment view counter
        try:
            meta["views"] = meta.get("views", 0) + 1
            meta_path.write_text(json.dumps(meta, ensure_ascii=False))
        except Exception:
            pass

        views_count = meta.get("views", 0)
        created_date = meta.get("created_at", "")

        # Format duration as m:ss
        dur_m = int(duration) // 60
        dur_s = int(duration) % 60
        dur_str = f"{dur_m}:{dur_s:02d}"

        # Format created_at as "Generated MMM D, YYYY"
        date_str = ""
        if created_date:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(created_date.replace("Z", "+00:00"))
                date_str = dt.strftime("Generated %b %-d, %Y")
            except Exception:
                date_str = f"Generated {created_date[:10]}"

        # Truncated text for OG description
        og_desc = (text[:100] + "..." if len(text) > 100 else text) if text else "Shared audio"

        # Audio MIME type
        audio_mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg"}.get(audio_fmt, "audio/wav")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>{speaker_name} — Revolab</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙️</text></svg>">
<meta property="og:title" content="{speaker_name} — Revolab">
<meta property="og:description" content="{og_desc}">
<meta property="og:type" content="music.song">
<meta property="og:audio" content="{audio_url}">
<meta property="og:audio:type" content="{audio_mime}">
<meta property="og:site_name" content="Revolab">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{speaker_name} — Revolab">
<meta name="twitter:description" content="{og_desc}">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ height: 100%; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px 16px;
  }}
  .container {{
    width: 100%;
    max-width: 480px;
    text-align: center;
  }}
  .speaker-name {{
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }}
  .speaker-icon {{ font-size: 22px; }}
  .spoken-text {{
    font-size: 15px;
    line-height: 1.6;
    color: #aaa;
    margin-bottom: 6px;
    padding: 0 8px;
    font-style: italic;
  }}
  .created-date {{
    font-size: 12px;
    color: #555;
    margin-bottom: 28px;
    margin-top: 4px;
  }}
  .duration-badge {{
    display: inline-block;
    font-size: 13px;
    color: #aaa;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 4px 14px;
    margin-bottom: 16px;
    font-variant-numeric: tabular-nums;
  }}
  #waveform {{
    width: 100%;
    height: 80px;
    margin-bottom: 8px;
    border-radius: 8px;
    overflow: hidden;
    background: #111;
  }}
  .controls {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-bottom: 28px;
  }}
  .play-btn {{
    width: 56px;
    height: 56px;
    min-width: 48px;
    min-height: 48px;
    border-radius: 50%;
    border: 2px solid #333;
    background: #1a1a1a;
    color: #fff;
    font-size: 22px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }}
  .play-btn:hover {{ border-color: #555; background: #222; }}
  .play-btn:active {{ transform: scale(0.95); }}
  .time-display {{
    font-size: 13px;
    color: #888;
    font-variant-numeric: tabular-nums;
    min-width: 90px;
  }}
  .btn-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .download-btn, .copy-btn {{
    display: inline-block;
    padding: 12px 28px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    color: #ccc;
    font-size: 14px;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s ease;
  }}
  .download-btn:hover, .copy-btn:hover {{ border-color: #555; color: #fff; background: #222; }}
  .download-btn:active, .copy-btn:active {{ transform: scale(0.97); }}
  .cta-link {{
    display: inline-block;
    margin-top: 16px;
    font-size: 13px;
    color: #666;
    text-decoration: none;
    transition: color 0.2s ease;
  }}
  .cta-link:hover {{ color: #999; }}
  .branding {{
    margin-top: 36px;
    font-size: 12px;
    color: #555;
  }}
  .branding a {{ color: #666; text-decoration: none; }}
  .branding a:hover {{ color: #888; }}
  .loading {{ color: #555; font-size: 14px; padding: 40px 0; }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .container > * {{
    animation: fadeIn 0.4s ease forwards;
  }}
  .container > *:nth-child(2) {{ animation-delay: 0.05s; }}
  .container > *:nth-child(3) {{ animation-delay: 0.1s; }}
  .container > *:nth-child(4) {{ animation-delay: 0.15s; }}
  .container > *:nth-child(5) {{ animation-delay: 0.2s; }}
  .container > *:nth-child(6) {{ animation-delay: 0.25s; }}
  .container > *:nth-child(7) {{ animation-delay: 0.3s; }}
</style>
</head>
<body>
<div class="container">
  <div class="speaker-name"><span class="speaker-icon">&#127908;</span> {speaker_name}</div>
  <div class="spoken-text">&ldquo;{text}&rdquo;</div>
  <div class="created-date">{date_str}{' &middot; ' + str(views_count) + ' views' if views_count else ''}</div>
  <div class="duration-badge">{dur_str}</div>
  <div id="waveform"></div>
  <div class="controls">
    <button class="play-btn" id="playBtn">&#9654;</button>
    <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
  </div>
  <div class="btn-row">
    <a class="download-btn" href="{audio_url}" download="{dl_name}">&#11015;&nbsp; Download</a>
    <button class="copy-btn" onclick="copyLink()">&#128279; Copy Link</button>
  </div>
  <a class="cta-link" href="/">Try Revolab &rarr;</a>
  <div class="branding">Powered by <a href="/">Revolab</a></div>
</div>
<script src="https://unpkg.com/wavesurfer.js@7"></script>
<script>
function copyLink() {{
  navigator.clipboard.writeText(window.location.href);
  var btn = document.querySelector('.copy-btn');
  btn.textContent = '✓ Copied!';
  setTimeout(function() {{ btn.innerHTML = '&#128279; Copy Link'; }}, 2000);
}}
(function() {{
  var playBtn = document.getElementById('playBtn');
  var timeDisplay = document.getElementById('timeDisplay');

  function formatTime(sec) {{
    var m = Math.floor(sec / 60);
    var s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }}

  var ws = WaveSurfer.create({{
    container: '#waveform',
    waveColor: '#444',
    progressColor: '#888',
    cursorColor: '#aaa',
    height: 80,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    url: '{audio_url}',
  }});

  var isPlaying = false;

  ws.on('ready', function() {{
    timeDisplay.textContent = '0:00 / ' + formatTime(ws.getDuration());
  }});

  ws.on('audioprocess', function() {{
    var ct = formatTime(ws.getCurrentTime());
    var dur = formatTime(ws.getDuration());
    timeDisplay.textContent = ct + ' / ' + dur;
  }});

  ws.on('seeking', function() {{
    var ct = formatTime(ws.getCurrentTime());
    var dur = formatTime(ws.getDuration());
    timeDisplay.textContent = ct + ' / ' + dur;
  }});

  ws.on('play', function() {{
    isPlaying = true;
    playBtn.innerHTML = '&#9646;&#9646;';
  }});

  ws.on('pause', function() {{
    isPlaying = false;
    playBtn.innerHTML = '&#9654;';
  }});

  ws.on('finish', function() {{
    isPlaying = false;
    playBtn.innerHTML = '&#9654;';
  }});

  playBtn.addEventListener('click', function() {{
    ws.playPause();
  }});
}})();
</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    @app.get("/share/{share_id}/audio", include_in_schema=False)
    async def get_share_audio(share_id: str):
        """Serve the raw audio file for a shared clip."""
        shared_dir: Path = cfg.profile_dir / "shared"

        for ext in ("wav", "mp3", "opus"):
            candidate = shared_dir / f"{share_id}.{ext}"
            if candidate.exists():
                from starlette.responses import FileResponse

                mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/opus"}
                return FileResponse(
                    str(candidate),
                    media_type=mime_map.get(ext, "application/octet-stream"),
                    headers={"Cache-Control": "public, max-age=86400"},
                )

        raise HTTPException(status_code=404, detail="Audio not found")

    # ── Speaker stats helper ──────────────────────────────────────────────────

    def _increment_speaker_stats(speaker_name: str):
        """Increment usage counter for a speaker in the stats file."""
        stats_dir: Path = cfg.profile_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / "speaker_usage.json"
        try:
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
            else:
                stats = {}
            stats[speaker_name] = stats.get(speaker_name, 0) + 1
            stats_path.write_text(json.dumps(stats, ensure_ascii=False))
        except Exception:
            pass

    # ── Speaker stats endpoint ────────────────────────────────────────────────

    @app.get("/api/stats/speakers", include_in_schema=False)
    async def get_speaker_stats():
        """Return usage counts per speaker."""
        stats_dir: Path = cfg.profile_dir / "stats"
        stats_path = stats_dir / "speaker_usage.json"
        if not stats_path.exists():
            return {"stats": {}}
        try:
            stats = json.loads(stats_path.read_text())
            return {"stats": stats}
        except Exception:
            return {"stats": {}}

    # ── History endpoints ─────────────────────────────────────────────────────

    @app.post("/api/history", include_in_schema=False)
    async def create_history_entry(request: Request):
        """Save a generated audio to persistent history."""
        body = await request.json()
        audio_b64 = body.get("audio_base64")
        text = body.get("text", "")
        speaker_name = body.get("speaker_name", "")
        fmt = body.get("format", "wav")
        created_at = body.get("created_at", "")
        version = body.get("version", 0)
        duration = body.get("duration", 0.0)

        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_base64 is required")

        history_dir: Path = cfg.profile_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        entry_id = uuid.uuid4().hex[:12]

        # Save audio file
        audio_bytes = base64.b64decode(audio_b64)
        audio_path = history_dir / f"{entry_id}.{fmt}"
        audio_path.write_bytes(audio_bytes)

        # Save metadata
        meta = {
            "id": entry_id,
            "text": text,
            "speaker_name": speaker_name,
            "format": fmt,
            "created_at": created_at,
            "version": version,
            "duration": duration,
        }
        meta_path = history_dir / f"{entry_id}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))

        _increment_speaker_stats(speaker_name)

        return {"id": entry_id, "saved": True}

    @app.get("/api/history", include_in_schema=False)
    async def list_history_entries():
        """List all history entries sorted by created_at descending."""
        history_dir: Path = cfg.profile_dir / "history"
        if not history_dir.exists():
            return {"entries": []}

        entries = []
        for meta_path in history_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                entries.append({
                    "id": meta.get("id", meta_path.stem),
                    "text": meta.get("text", ""),
                    "speaker_name": meta.get("speaker_name", ""),
                    "format": meta.get("format", "wav"),
                    "created_at": meta.get("created_at", ""),
                    "version": meta.get("version", 0),
                    "duration": meta.get("duration", 0.0),
                })
            except Exception:
                logger.warning("Failed to read history meta: %s", meta_path)

        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return {"entries": entries}

    @app.get("/api/history/{entry_id}/audio", include_in_schema=False)
    async def get_history_audio(entry_id: str):
        """Serve the raw audio file for a history entry."""
        history_dir: Path = cfg.profile_dir / "history"

        for ext in ("wav", "mp3", "opus"):
            candidate = history_dir / f"{entry_id}.{ext}"
            if candidate.exists():
                from starlette.responses import FileResponse

                mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/opus"}
                return FileResponse(
                    str(candidate),
                    media_type=mime_map.get(ext, "application/octet-stream"),
                    headers={"Cache-Control": "public, max-age=3600"},
                )

        raise HTTPException(status_code=404, detail="Audio not found")

    @app.delete("/api/history/{entry_id}", include_in_schema=False)
    async def delete_history_entry(entry_id: str):
        """Delete a history entry (audio + metadata)."""
        history_dir: Path = cfg.profile_dir / "history"
        deleted = False

        # Delete audio file(s)
        for ext in ("wav", "mp3", "opus"):
            candidate = history_dir / f"{entry_id}.{ext}"
            if candidate.exists():
                candidate.unlink()
                deleted = True

        # Delete metadata
        meta_path = history_dir / f"{entry_id}.meta.json"
        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        if not deleted:
            raise HTTPException(status_code=404, detail="History entry not found")

        return {"deleted": True}

    # ── Frontend ──────────────────────────────────────────────────────────────
    import pathlib
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend():
        from starlette.responses import HTMLResponse as HR
        content = (static_dir / "index.html").read_text()
        return HR(content=content, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    return app
