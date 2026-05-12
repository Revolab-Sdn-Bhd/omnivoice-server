"""GET /v1/models — OpenAI-compatible model listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

MODEL_CREATED = 1714800000


@router.get("/models", tags=["OpenAI-compatible"])
async def list_models():
    """Returns the single model this server provides."""
    return {
        "object": "list",
        "data": [
            {
                "id": "revovoice",
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "revolab",
            },
        ],
    }
