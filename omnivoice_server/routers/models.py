"""GET /v1/models — OpenAI-compatible model listing endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

MODEL_CREATED = 1714800000


@router.get("/models")
async def list_models():
    """Returns the single model this server provides."""
    return {
        "object": "list",
        "data": [
            {
                "id": "sepbox-tts",
                "object": "model",
                "created": MODEL_CREATED,
                "owned_by": "sepbox",
            },
        ],
    }
