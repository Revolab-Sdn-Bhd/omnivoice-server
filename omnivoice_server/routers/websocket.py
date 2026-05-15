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
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.inference import SynthesisRequest
from ..utils.text import split_sentences

logger = logging.getLogger(__name__)
router = APIRouter()


def _tensor_to_base64_int16(tensor) -> str:
    """Convert tensor to base64-encoded int16 PCM bytes."""
    import numpy as np

    if isinstance(tensor, np.ndarray):
        flat = tensor.squeeze().astype(np.float32)
    else:
        flat = tensor.squeeze(0).cpu().float().numpy()
    pcm = (flat * 32767).clip(-32768, 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


SAMPLE_RATE = 24_000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # int16 mono = 2 bytes per sample


async def _send_buffered_chunks(
    ws: WebSocket,
    pcm_buffer: bytearray,
    ctx_id: str,
    cancelled_contexts: set[str],
    chunk_idx: int,
    flush: bool = False,
) -> int:
    """Send complete 1-second chunks from buffer. Returns updated chunk_idx."""
    while len(pcm_buffer) >= BYTES_PER_SECOND or (flush and pcm_buffer):
        if ctx_id in cancelled_contexts:
            return chunk_idx
        chunk_size = BYTES_PER_SECOND if len(pcm_buffer) >= BYTES_PER_SECOND else len(pcm_buffer)
        chunk_bytes = bytes(pcm_buffer[:chunk_size])
        del pcm_buffer[:chunk_size]
        audio_b64 = base64.b64encode(chunk_bytes).decode("ascii")
        chunk_msg = json.dumps({
            "type": "audio_chunk",
            "data": audio_b64,
            "done": False,
            "status_code": 206,
            "context_id": ctx_id,
        })
        try:
            await ws.send_text(chunk_msg)
        except (WebSocketDisconnect, AssertionError):
            return chunk_idx
        chunk_idx += 1
    return chunk_idx


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
    client = ws.client.host if ws.client else "unknown"
    logger.info("[ws] connected client=%s", client)

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
            num_step = msg.get("num_step")
            voice_cfg = msg.get("voice", {})
            voice_id = voice_cfg.get("id", "anwar") if isinstance(voice_cfg, dict) else "anwar"
            mode = msg.get("mode", "auto")
            instruct = msg.get("instruct")
            speed = msg.get("speed", 1.0)
            guidance_scale = msg.get("guidance_scale")
            denoise = msg.get("denoise")
            t_shift = msg.get("t_shift")
            duration = msg.get("duration")
            position_temperature = msg.get("position_temperature")
            class_temperature = msg.get("class_temperature")
            layer_penalty_factor = msg.get("layer_penalty_factor")
            preprocess_prompt = msg.get("preprocess_prompt")
            postprocess_output = msg.get("postprocess_output")
            audio_chunk_duration = msg.get("audio_chunk_duration")
            audio_chunk_threshold = msg.get("audio_chunk_threshold")

            # Clear cancellation for this context on new message
            cancelled_contexts.discard(ctx_id)

            if not transcript:
                if not cont:
                    await _send_done(ws, ctx_id)
                continue

            t0 = time.monotonic()
            logger.info(
                "[ws] recv ctx=%s voice=%s text=%r",
                ctx_id, voice_id, transcript,
            )

            preceding = list(active_tasks.get(ctx_id, []))

            # Launch synthesis in background task
            task = asyncio.create_task(
                _process_transcript(
                    ws, msg, ctx_id, transcript, language, voice_id,
                    cancelled_contexts, active_tasks, t0, preceding,
                    mode=mode,
                    instruct=instruct,
                    num_step=num_step,
                    speed=speed,
                    guidance_scale=guidance_scale,
                    denoise=denoise,
                    t_shift=t_shift,
                    duration=duration,
                    position_temperature=position_temperature,
                    class_temperature=class_temperature,
                    layer_penalty_factor=layer_penalty_factor,
                    preprocess_prompt=preprocess_prompt,
                    postprocess_output=postprocess_output,
                    audio_chunk_duration=audio_chunk_duration,
                    audio_chunk_threshold=audio_chunk_threshold,
                )
            )
            active_tasks.setdefault(ctx_id, []).append(task)

            def _make_cleanup(cid: str):
                def _cb(t: asyncio.Task) -> None:
                    _cleanup_task(active_tasks, cid, t)
                return _cb
            task.add_done_callback(_make_cleanup(ctx_id))

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
        logger.info("[ws] disconnected client=%s", client)
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
    t0: float,
    preceding: list[asyncio.Task],
    *,
    mode: str = "auto",
    instruct: str | None = None,
    num_step: int | None = None,
    speed: float = 1.0,
    guidance_scale: float | None = None,
    denoise: bool | None = None,
    t_shift: float | None = None,
    duration: float | None = None,
    position_temperature: float | None = None,
    class_temperature: float | None = None,
    layer_penalty_factor: float | None = None,
    preprocess_prompt: bool | None = None,
    postprocess_output: bool | None = None,
    audio_chunk_duration: float | None = None,
    audio_chunk_threshold: float | None = None,
) -> None:
    """Process a single transcript message: split sentences, synthesize, stream."""
    import numpy as np

    cfg = ws.app.state.cfg
    inference_svc = ws.app.state.inference_svc

    audio_path, ref_text = _resolve_voice_path(voice_id, cfg)
    if not audio_path:
        await _send_error(ws, f"Voice '{voice_id}' not found", ctx_id)
        return

    sentences = split_sentences(transcript, max_chars=cfg.stream_chunk_max_chars)

    # Synthesize all sentences first (runs concurrently with preceding tasks).
    pcm_buffer = bytearray()
    for i, sentence in enumerate(sentences):
        if ctx_id in cancelled_contexts:
            return

        logger.info("[ws] ctx=%s sentence[%d] %r", ctx_id, i, sentence)
        t_sent = time.monotonic()

        syn_req = SynthesisRequest(
            text=sentence,
            mode=mode,
            instruct=instruct,
            ref_audio_path=audio_path,
            ref_text=ref_text,
            speed=speed,
            language=language if language != "en" else None,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            t_shift=t_shift,
            duration=duration,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            layer_penalty_factor=layer_penalty_factor,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_threshold=audio_chunk_threshold,
        )

        try:
            result = await inference_svc.synthesize(syn_req)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("WebSocket synthesis failed")
            await _send_error(ws, str(e), ctx_id)
            return

        logger.info(
            "[ws] ctx=%s sentence[%d] synthesis=%.3fs",
            ctx_id, i, time.monotonic() - t_sent,
        )

        for tensor in result.tensors:
            if isinstance(tensor, np.ndarray):
                flat = tensor.squeeze().astype(np.float32)
            else:
                flat = tensor.squeeze(0).cpu().float().numpy()
            pcm = (flat * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_buffer.extend(pcm.tobytes())

    # Wait for preceding tasks to preserve chunk ordering.
    for t in preceding:
        try:
            await t
        except Exception:
            pass

    if ctx_id in cancelled_contexts:
        return

    chunk_idx = 0
    chunk_idx = await _send_buffered_chunks(
        ws, pcm_buffer, ctx_id, cancelled_contexts, chunk_idx, flush=True,
    )
    logger.info(
        "[ws] ctx=%s ttfc=%.3fs total=%.3fs chunks=%d",
        ctx_id, time.monotonic() - t0, time.monotonic() - t0, chunk_idx,
    )


@router.get("/api/docs/websocket", tags=["Docs"])
async def websocket_docs():
    """WebSocket TTS API documentation."""
    return _build_ws_docs()


def _build_ws_docs() -> dict:
    p = {
        "transcript": {
            "type": "string", "required": True,
            "description": "Text to synthesize",
        },
        "context_id": {
            "type": "string", "default": "",
            "description": "ID for concurrent streams",
        },
        "continue": {
            "type": "bool", "default": True,
            "description": "If false, sends done after synthesis",
        },
        "language": {
            "type": "string", "default": "en",
            "description": "Language code (en, ms, etc.)",
        },
        "mode": {
            "type": "string", "default": "auto",
            "enum": ["auto", "design", "clone"],
            "description": "Synthesis mode",
        },
        "instruct": {
            "type": "string|null",
            "description": "Instruction for mode=design",
        },
        "voice": {
            "type": "object", "default": {},
            "description": "Voice config with 'id' field",
        },
        "voice.id": {
            "type": "string", "default": "anwar",
            "description": "Voice name from /voices",
        },
        "speed": {
            "type": "float", "default": 1.0,
            "description": "Speech speed multiplier",
        },
        "num_step": {
            "type": "int|null",
            "description": "Diffusion steps (null=server default)",
        },
        "guidance_scale": {
            "type": "float|null",
            "description": "Classifier-free guidance scale",
        },
        "denoise": {
            "type": "bool|null",
            "description": "Apply denoising",
        },
        "t_shift": {
            "type": "float|null",
            "description": "Time shift for diffusion sampling",
        },
        "duration": {
            "type": "float|null",
            "description": "Fixed output duration in seconds",
        },
        "position_temperature": {
            "type": "float|null",
            "description": "Position temperature",
        },
        "class_temperature": {
            "type": "float|null",
            "description": "Class temperature",
        },
        "layer_penalty_factor": {
            "type": "float|null",
            "description": "Layer penalty factor",
        },
        "preprocess_prompt": {
            "type": "bool|null",
            "description": "Preprocess text prompt",
        },
        "postprocess_output": {
            "type": "bool|null",
            "description": "Postprocess audio output",
        },
        "audio_chunk_duration": {
            "type": "float|null",
            "description": "Audio chunk duration in seconds",
        },
        "audio_chunk_threshold": {
            "type": "float|null",
            "description": "Audio chunk threshold",
        },
    }
    return {
        "endpoint": "ws://{host}:{port}/tts/websocket",
        "protocol": "JSON over WebSocket",
        "messages": {
            "generate": {
                "description": "Synthesize speech and stream audio chunks.",
                "params": p,
                "example": {
                    "transcript": "Hello, how are you?",
                    "context_id": "ctx-1",
                    "continue": False,
                    "language": "en",
                    "mode": "auto",
                    "voice": {"id": "anwar"},
                    "speed": 1.0,
                    "num_step": 16,
                },
            },
            "cancel": {
                "description": "Cancel ongoing synthesis for a context.",
                "params": {
                    "cancel": {"type": "bool", "required": True},
                    "context_id": {"type": "string", "required": True},
                },
                "example": {"cancel": True, "context_id": "ctx-1"},
            },
        },
        "response_messages": {
            "audio_chunk": {
                "type": "audio_chunk",
                "data": "base64 int16 PCM 24kHz mono, 1s chunks",
                "done": False,
                "status_code": 206,
                "context_id": "ctx-1",
            },
            "done": {
                "type": "done",
                "done": True,
                "status_code": 200,
                "context_id": "ctx-1",
            },
            "error": {
                "type": "error",
                "error": "error message",
                "done": True,
                "status_code": 500,
                "context_id": "ctx-1",
            },
        },
        "audio_format": "int16 PCM, 24kHz, mono, base64, 1s chunks.",
    }


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
