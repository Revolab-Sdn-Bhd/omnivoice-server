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

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .routers import health, models, speech, voices
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

    app.state.start_time = time.time()
    app.state.metrics_svc = MetricsService()

    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    app.state.profile_svc = ProfileService(profile_dir=cfg.profile_dir)

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

    # Auto-benchmark and warmup
    if app.state.model_svc.is_loaded:
        from .services.gpu_benchmark import find_optimal_batch_size

        # Run GPU benchmark to find optimal batch size (blocking, run in executor)
        loop = asyncio.get_running_loop()
        model = app.state.model_svc.model
        cache_dir = str(cfg.compile_cache_dir) if cfg.compile_cache_dir else None
        bench = await loop.run_in_executor(
            None, find_optimal_batch_size, model, cfg.num_step, cache_dir,
        )
        app.state.gpu_benchmark = bench
        optimal_bs = bench["optimal_batch_size"]

        # Override batch_max_size if not explicitly set by user
        if not cfg.batch_enabled:
            logger.info("Batching disabled, using single-request mode")
        else:
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

        # Warmup with optimal batch size to prime compiled kernels
        try:
            warmup_texts = [
                "Warmup inference for CUDA kernel compilation.",
                "Second warmup pass to ensure all kernels are cached.",
            ]
            for _ in range(3):
                await app.state.inference_svc.synthesize(
                    SynthesisRequest(text=warmup_texts[0], mode="auto")
                )
            # Batch warmup
            for _ in range(2):
                await app.state.inference_svc.synthesize(
                    SynthesisRequest(text=warmup_texts[0], mode="auto")
                )
                await app.state.inference_svc.synthesize(
                    SynthesisRequest(text=warmup_texts[1], mode="auto")
                )
            logger.info(
                "Warmup complete — optimal_batch=%d, throughput=%.1f req/s",
                optimal_bs, bench["optimal_throughput_req_s"],
            )
        except Exception:
            logger.warning("Warmup encountered errors (non-fatal)")

    # Start batch scheduler if enabled
    if hasattr(app.state.inference_svc, "start_batch_scheduler"):
        app.state.inference_svc.start_batch_scheduler()

    yield

    # Shutdown
    if hasattr(app.state.inference_svc, "stop_batch_scheduler"):
        app.state.inference_svc.stop_batch_scheduler()
    if getattr(app.state, "executor", None):
        app.state.executor.shutdown(wait=False)


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


def create_app(cfg: Settings) -> FastAPI:
    app = FastAPI(
        title="omnivoice-server",
        description="OpenAI-compatible HTTP server for OmniVoice TTS",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.cfg = cfg

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
            # Skip auth for health, metrics, and model listing
            if request.url.path in (
                "/health", "/live", "/ready",
                "/metrics", "/metrics/prometheus", "/v1/models",
                "/", "/static/index.html",
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
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "detail": exc.errors(),
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
    app.include_router(speech.router, prefix="/v1")
    app.include_router(voices.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")
    app.include_router(health.router)

    # ── Frontend ──────────────────────────────────────────────────────────────
    import pathlib
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend():
        return (static_dir / "index.html").read_text()

    return app
