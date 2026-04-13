"""Health and metrics endpoints."""

from __future__ import annotations

import time

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Readiness check. Returns 503 while model is loading, 200 when ready."""
    cfg = request.app.state.cfg
    model_svc = request.app.state.model_svc
    ram_mb = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)

    if not model_svc.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "ready": False,
                "model_loaded": False,
                "memory_rss_mb": ram_mb,
            },
        )

    uptime_s = round(time.monotonic() - request.app.state.start_time, 1)
    result = {
        "status": "healthy",
        "ready": True,
        "model_loaded": True,
        "uptime_s": uptime_s,
        "model_id": cfg.model_id,
        "memory_rss_mb": ram_mb,
    }

    # Include GPU benchmark results if available
    bench = getattr(request.app.state, "gpu_benchmark", None)
    if bench:
        result["gpu_benchmark"] = {
            "optimal_batch_size": bench["optimal_batch_size"],
            "throughput_req_s": bench["optimal_throughput_req_s"],
            "vram_used_mb": bench["vram_used_mb"],
            "vram_total_mb": bench["vram_total_mb"],
        }

    return result


@router.get("/metrics")
async def metrics(request: Request):
    """Request metrics and current memory usage."""
    metrics_svc = request.app.state.metrics_svc
    snapshot = metrics_svc.snapshot()
    snapshot["ram_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    return snapshot
