"""
FastAPI application factory.
All shared state lives on app.state — no module-level globals.
"""

from __future__ import annotations

import asyncio
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
    get_langfuse_client,
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
        return response

    # ── Auth middleware ───────────────────────────────────────────────────────
    if cfg.api_key:

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth for health and static
            if request.url.path in (
                "/health", "/",
                "/v1/models", "/v1/audio/voices",
            ) or request.url.path.startswith("/static/"):
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
