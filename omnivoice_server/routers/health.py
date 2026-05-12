"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health(request: Request):
    """Server health check — verifies model is loaded."""
    model_svc = request.app.state.model_svc
    workers_total = request.app.state.cfg.workers

    if not model_svc.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "workers_healthy": 0,
                "workers_total": workers_total,
            },
        )

    return {
        "status": "ok",
        "workers_healthy": workers_total,
        "workers_total": workers_total,
    }
