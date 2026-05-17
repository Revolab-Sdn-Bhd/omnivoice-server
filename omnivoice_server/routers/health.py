"""Health check and model revision management."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
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
        "model_id": request.app.state.cfg.model_id,
        "model_revision": request.app.state.model_svc.model_revision_hash or None,
        "voices_revision": request.app.state.cfg.voices_revision_hash or None,
        "device": request.app.state.cfg.device,
    }


@router.get("/api/model/revisions", tags=["Model"])
async def list_model_revisions(request: Request):
    """List all commits for the current model from HuggingFace."""
    from huggingface_hub import HfApi

    cfg = request.app.state.cfg
    try:
        api = HfApi()
        refs = api.list_repo_commits(cfg.model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list repo commits: {e}")

    revisions = []
    for ref in refs:
        revisions.append({
            "commit_hash": ref.commit_id,
            "short_hash": ref.commit_id[:8],
            "title": ref.title,
            "created_at": ref.created_at.isoformat() if ref.created_at else None,
        })

    return {
        "model_id": cfg.model_id,
        "current_revision": request.app.state.model_svc.model_revision_hash,
        "revisions": revisions,
    }


class SwitchRevisionRequest(BaseModel):
    revision: str


@router.post("/api/model/switch-revision", tags=["Model"])
async def switch_model_revision(request: Request, body: SwitchRevisionRequest):
    """Switch to a different model revision (downloads if needed, then reloads)."""
    model_svc = request.app.state.model_svc
    cfg = request.app.state.cfg

    current = model_svc.model_revision_hash
    target = body.revision

    if current and target[:8] == current[:8]:
        return {"status": "already_active", "revision": current}

    logger.info("Switching model revision: %s → %s", current, target[:8])

    cfg.model_revision = target

    loop = asyncio.get_running_loop()
    try:
        await model_svc.reload()
    except Exception as e:
        logger.exception("Failed to reload model with revision %s", target[:8])
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")

    return {
        "status": "switched",
        "revision": model_svc.model_revision_hash,
    }
