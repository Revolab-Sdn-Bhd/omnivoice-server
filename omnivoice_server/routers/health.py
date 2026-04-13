"""Health, readiness, and metrics endpoints."""

from __future__ import annotations

import time

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()


@router.get("/live")
async def liveness():
    """Liveness probe — process is alive."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(request: Request):
    """Readiness probe — model loaded and ready to serve."""
    model_svc = request.app.state.model_svc
    if not model_svc.is_loaded:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


@router.get("/health")
async def health(request: Request):
    """Full health status with benchmark, cache, and queue info."""
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
            "optimal_batch_timeout_ms": bench["optimal_batch_timeout_ms"],
            "throughput_req_s": bench["optimal_throughput_req_s"],
            "single_request_latency_ms": bench["single_request_latency_ms"],
            "vram_used_mb": bench["vram_used_mb"],
            "vram_total_mb": bench["vram_total_mb"],
        }

    # Queue depth
    inference_svc = getattr(request.app.state, "inference_svc", None)
    if inference_svc is not None:
        result["pending_requests"] = inference_svc.pending_count
        result["max_queue_depth"] = request.app.state.cfg.max_queue_depth

    # Response cache stats
    cache = getattr(request.app.state, "response_cache", None)
    if cache is not None:
        result["response_cache"] = cache.stats()

    return result


@router.get("/metrics")
async def metrics(request: Request):
    """Request metrics and current memory usage."""
    metrics_svc = request.app.state.metrics_svc
    snapshot = metrics_svc.snapshot()
    snapshot["ram_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    return snapshot


@router.get("/metrics/prometheus")
async def prometheus_metrics(request: Request):
    """Prometheus-format metrics."""
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    # Use module-level singletons to avoid re-registration
    if not hasattr(prometheus_metrics, "_initialized"):
        prometheus_metrics._req_total = Counter(
            "omnivoice_requests_total", "Total requests", ["status"]
        )
        prometheus_metrics._latency = Histogram(
            "omnivoice_latency_seconds", "Synthesis latency",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        prometheus_metrics._pending = Gauge(
            "omnivoice_pending_requests", "Pending inference requests"
        )
        prometheus_metrics._cache_hits = Counter(
            "omnivoice_cache_hits_total", "Cache hits"
        )
        prometheus_metrics._cache_misses = Counter(
            "omnivoice_cache_misses_total", "Cache misses"
        )
        prometheus_metrics._initialized = True

    metrics_svc = request.app.state.metrics_svc
    snap = metrics_svc.snapshot()

    prometheus_metrics._req_total.labels(status="success").inc(
        snap["requests_success"]
    )
    prometheus_metrics._req_total.labels(status="error").inc(
        snap["requests_error"]
    )
    prometheus_metrics._req_total.labels(status="timeout").inc(
        snap["requests_timeout"]
    )

    # Pending
    inference_svc = getattr(request.app.state, "inference_svc", None)
    if inference_svc is not None:
        prometheus_metrics._pending.set(inference_svc.pending_count)

    # Cache
    cache = getattr(request.app.state, "response_cache", None)
    if cache is not None:
        cs = cache.stats()
        prometheus_metrics._cache_hits.inc(cs["response_cache_hits"])
        prometheus_metrics._cache_misses.inc(cs["response_cache_misses"])

    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
