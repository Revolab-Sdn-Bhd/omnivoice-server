"""
WS /tts/websocket — Cartesia-compatible streaming TTS WebSocket endpoint.

Supports concurrent context_ids, sentence-level processing, and
cancel messages. Audio is sent as base64-encoded int16 PCM.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.inference import SynthesisRequest
from ..utils.text import split_sentences

logger = logging.getLogger(__name__)
router = APIRouter()


def _tensor_to_base64_int16(tensor) -> str:
    """Convert tensor to base64-encoded int16 PCM bytes."""
    import numpy as np

    flat = tensor.squeeze(0).cpu().float().numpy()
    pcm = (flat * 32767).clip(-32768, 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _resolve_voice_path(voice_id: str, cfg) -> tuple[str | None, str | None]:
    """Resolve voice_id to (wav_path, ref_text)."""
    if not voice_id:
        voice_id = "anwar"
    wav_path = cfg.voices_dir / f"{voice_id}.wav"
    if not wav_path.is_file():
        return None, None
    txt_path = wav_path.with_suffix(".txt")
    ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
    return str(wav_path), ref_text


@router.websocket("/tts/websocket")
async def tts_websocket(ws: WebSocket):
    """Cartesia-compatible WebSocket endpoint for real-time streaming TTS."""
    await ws.accept()

    # Track cancellation per context_id
    cancelled_contexts: set[str] = set()
    # Track in-flight tasks per context_id for cancellation
    active_tasks: dict[str, list[asyncio.Task]] = {}

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(ws, "Invalid JSON", "")
                continue

            # Handle cancel message
            if msg.get("cancel"):
                ctx_id = msg.get("context_id", "")
                cancelled_contexts.add(ctx_id)
                # Cancel active tasks for this context
                for task in active_tasks.pop(ctx_id, []):
                    task.cancel()
                await _send_done(ws, ctx_id)
                continue

            # Handle generate message
            ctx_id = msg.get("context_id", "")
            transcript = msg.get("transcript", "")
            cont = msg.get("continue", True)
            language = msg.get("language", "en")
            voice_cfg = msg.get("voice", {})
            voice_id = voice_cfg.get("id", "anwar") if isinstance(voice_cfg, dict) else "anwar"

            # Clear cancellation for this context on new message
            cancelled_contexts.discard(ctx_id)

            if not transcript:
                if not cont:
                    await _send_done(ws, ctx_id)
                continue

            # Launch synthesis in background task
            task = asyncio.create_task(
                _process_transcript(
                    ws, msg, ctx_id, transcript, language, voice_id,
                    cancelled_contexts, active_tasks,
                )
            )
            active_tasks.setdefault(ctx_id, []).append(task)
            task.add_done_callback(
                lambda t, cid=ctx_id: _cleanup_task(active_tasks, cid, t)
            )

            if not cont:
                # Wait for completion then send done
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                await _send_done(ws, ctx_id)

    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        # Cancel all active tasks
        for tasks in active_tasks.values():
            for task in tasks:
                task.cancel()


async def _process_transcript(
    ws: WebSocket,
    msg: dict[str, Any],
    ctx_id: str,
    transcript: str,
    language: str,
    voice_id: str,
    cancelled_contexts: set[str],
    active_tasks: dict[str, list[asyncio.Task]],
) -> None:
    """Process a single transcript message: split sentences, synthesize, stream."""
    cfg = ws.app.state.cfg
    inference_svc = ws.app.state.inference_svc

    audio_path, ref_text = _resolve_voice_path(voice_id, cfg)
    if not audio_path:
        await _send_error(ws, f"Voice '{voice_id}' not found", ctx_id)
        return

    sentences = split_sentences(transcript, max_chars=cfg.stream_chunk_max_chars)

    for sentence in sentences:
        # Check cancellation
        if ctx_id in cancelled_contexts:
            return

        syn_req = SynthesisRequest(
            text=sentence,
            mode="clone",
            ref_audio_path=audio_path,
            ref_text=ref_text,
            speed=1.0,
            language=language if language != "en" else None,
        )

        try:
            result = await inference_svc.synthesize(syn_req)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("WebSocket synthesis failed")
            await _send_error(ws, str(e), ctx_id)
            return

        for tensor in result.tensors:
            if ctx_id in cancelled_contexts:
                return
            audio_b64 = _tensor_to_base64_int16(tensor)
            chunk_msg = json.dumps({
                "type": "audio_chunk",
                "data": audio_b64,
                "done": False,
                "status_code": 206,
                "context_id": ctx_id,
            })
            await ws.send_text(chunk_msg)


def _cleanup_task(
    active_tasks: dict[str, list[asyncio.Task]],
    ctx_id: str,
    task: asyncio.Task,
) -> None:
    """Remove completed task from active_tasks."""
    tasks = active_tasks.get(ctx_id)
    if tasks:
        try:
            tasks.remove(task)
        except ValueError:
            pass
        if not tasks:
            active_tasks.pop(ctx_id, None)


async def _send_done(ws: WebSocket, ctx_id: str) -> None:
    """Send done message."""
    msg = json.dumps({
        "type": "done",
        "done": True,
        "status_code": 200,
        "context_id": ctx_id,
    })
    try:
        await ws.send_text(msg)
    except Exception:
        pass


async def _send_error(ws: WebSocket, error: str, ctx_id: str) -> None:
    """Send error message."""
    msg = json.dumps({
        "type": "error",
        "error": error,
        "done": True,
        "status_code": 500,
        "context_id": ctx_id,
    })
    try:
        await ws.send_text(msg)
    except Exception:
        pass
