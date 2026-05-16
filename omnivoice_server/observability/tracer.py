"""
Central Langfuse tracer configuration and utilities.

Supports graceful degradation when Langfuse credentials are not configured.
The Langfuse SDK reads LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from env
automatically — no need to pass them through Settings.
Supports background flushing to avoid blocking user requests.
"""

from __future__ import annotations

import base64
import contextvars
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langfuse import Langfuse

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

logger = logging.getLogger(__name__)

# Singleton client
_langfuse_client: Langfuse | None = None
_langfuse_enabled: bool | None = None
_flush_executor: ThreadPoolExecutor | None = None


def get_langfuse_client(host: str | None = None) -> Langfuse | None:
    """Get or create Langfuse client singleton. Returns None if not configured.

    The Langfuse SDK reads LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from
    the environment automatically. Optionally pass host to override LANGFUSE_HOST.
    """
    global _langfuse_client, _langfuse_enabled, _flush_executor

    # Ensure .env is loaded so os.getenv() can see the vars
    if _langfuse_enabled is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

    if _langfuse_enabled is not None:
        return _langfuse_client if _langfuse_enabled else None

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        _langfuse_enabled = False
        logger.info("Langfuse not configured (missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY)")
        return None

    try:
        kwargs: dict = {}
        if host:
            kwargs["host"] = host
        _langfuse_client = Langfuse(**kwargs)
        _langfuse_enabled = True
        _flush_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="langfuse_flush")
        logger.info("Langfuse client initialized with background flushing")
        return _langfuse_client
    except Exception as e:
        _langfuse_enabled = False
        logger.warning("Failed to initialize Langfuse: %s", e)
        return None


def is_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    if _langfuse_enabled is None:
        get_langfuse_client()
    return _langfuse_enabled or False


def get_observe():
    """Return a decorator factory that creates a Langfuse trace around handlers.

    Defers client lookup to call-time so it works even when applied at
    import-time (before lifespan initializes the client).
    """

    def _make_decorator(func=None, *, name=None, capture_output=True, **_kw):
        import functools

        def decorator(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                client = get_langfuse_client()
                if client is None:
                    return await fn(*args, **kwargs)
                trace_name = name or fn.__name__
                span = client.start_observation(name=trace_name)
                _current_trace.set(span)
                try:
                    result = await fn(*args, **kwargs)
                    span.end()
                    return result
                except Exception as exc:
                    span.update(status_message=str(exc))
                    span.end()
                    raise
                finally:
                    _current_trace.set(None)

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                client = get_langfuse_client()
                if client is None:
                    return fn(*args, **kwargs)
                trace_name = name or fn.__name__
                span = client.start_observation(name=trace_name)
                _current_trace.set(span)
                try:
                    return fn(*args, **kwargs)
                finally:
                    span.end()
                    _current_trace.set(None)

            import asyncio
            if asyncio.iscoroutinefunction(fn):
                return async_wrapper
            return sync_wrapper

        if func is not None:
            return decorator(func)
        return decorator

    return _make_decorator


_current_trace = contextvars.ContextVar("omnivoice_trace", default=None)


def get_current_trace_id() -> str | None:
    """Get the current request's trace ID (set by get_observe wrapper)."""
    trace = _current_trace.get()
    return trace.trace_id if trace else None


def flush_in_background() -> None:
    """Flush Langfuse traces in a background thread (non-blocking)."""
    client = get_langfuse_client()
    if client is None or _flush_executor is None:
        return

    if getattr(_flush_executor, "_shutdown", False):
        return

    def do_flush() -> None:
        import threading

        done = threading.Event()
        error: list[Exception | None] = [None]

        def worker() -> None:
            try:
                client.flush()
            except Exception as e:
                error[0] = e
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        if not done.wait(timeout=30):
            logger.warning("Langfuse flush timeout after 30s")
            return

        if error[0] is not None:
            logger.warning("Background flush failed: %s", error[0])

    try:
        _flush_executor.submit(do_flush)
    except RuntimeError:
        logger.debug("Flush executor shut down, skipping trace flush")


def flush_blocking() -> None:
    """Blocking flush. Use sparingly (e.g., on shutdown)."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
        logger.debug("Langfuse flush completed")
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)


def join_background_flushes() -> None:
    """Wait for pending background flushes to complete. Use on shutdown."""
    global _flush_executor
    if _flush_executor is not None:
        try:
            _flush_executor.shutdown(wait=True)
        except Exception as e:
            logger.warning("Failed to shutdown flush executor: %s", e)
        _flush_executor = None


def update_current_trace(*, metadata: dict | None = None, output: dict | None = None) -> None:
    """Update the current request trace with metadata and/or output."""
    trace = _current_trace.get()
    if trace is None:
        return
    try:
        kwargs = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if output is not None:
            kwargs["output"] = output
        if kwargs:
            trace.update(**kwargs)
    except Exception as e:
        logger.warning("Could not update current trace: %s", e)


def create_audio_data_uri_from_bytes(wav_bytes: bytes) -> str | None:
    """Convert WAV bytes to a base64 data URI for Langfuse (playable in dashboard).

    Returns None if the audio exceeds MAX_AUDIO_SIZE_BYTES.
    """
    if len(wav_bytes) > MAX_AUDIO_SIZE_BYTES:
        logger.warning(
            "Audio too large for Langfuse trace: %.1fMB > %.0fMB",
            len(wav_bytes) / (1024 * 1024),
            MAX_AUDIO_SIZE_BYTES / (1024 * 1024),
        )
        return None
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{audio_b64}"


# ── ChatterBox-compatible structured output builders ─────────────────────────


def _compute_signal_metrics(tensors: list) -> dict:
    """Compute signal-level quality metrics from output tensors (no ML models needed)."""
    import numpy as np

    # Concatenate all tensors into one flat float32 array
    samples = []
    for t in tensors:
        if hasattr(t, "cpu"):
            flat = t.squeeze().cpu().float().numpy()
        else:
            flat = t.squeeze().astype(np.float32)
        samples.append(flat)
    audio = np.concatenate(samples) if samples else np.array([], dtype=np.float32)

    if len(audio) == 0:
        return {
            "clipping_percent": 0.0, "dynamic_range_db": 0.0,
            "rms_db": -60.0, "crest_factor_db": 0.0,
        }

    # Clipping: samples within 1% of full scale
    abs_audio = np.abs(audio)
    clipping_percent = float(np.mean(abs_audio > 0.99) * 100)

    # RMS in dB
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_db = float(20 * np.log10(rms)) if rms > 1e-10 else -60.0

    # Peak in dB
    peak = float(np.max(abs_audio))
    peak_db = float(20 * np.log10(peak)) if peak > 1e-10 else -60.0

    # Dynamic range
    silence_floor = float(np.percentile(abs_audio, 5))
    silence_db = float(20 * np.log10(silence_floor)) if silence_floor > 1e-10 else -60.0
    dynamic_range_db = round(peak_db - silence_db, 2)

    # Crest factor
    crest_factor_db = round(peak_db - rms_db, 2) if rms_db > -60.0 else 0.0

    return {
        "clipping_percent": round(clipping_percent, 3),
        "dynamic_range_db": dynamic_range_db,
        "rms_db": round(rms_db, 2),
        "crest_factor_db": crest_factor_db,
    }


def _compute_quality_assessment(
    signal: dict,
    duration_ratio: float,
    is_outlier: bool,
) -> dict:
    """Determine quality assessment flags from signal + duration metrics."""
    issues: dict[str, str | bool] = {}
    has_problems = False
    has_issues = False

    # Clipping check
    if signal["clipping_percent"] > 5.0:
        issues["clipping"] = "severe"
        has_problems = True
    elif signal["clipping_percent"] > 1.0:
        issues["clipping"] = "moderate"
        has_issues = True

    # Dynamic range check
    if signal["dynamic_range_db"] < 10:
        issues["dynamic_range"] = "too_narrow"
        has_issues = True

    # Duration outlier check
    if is_outlier:
        issues["duration_outlier"] = True
        has_issues = True

    # RMS level check
    if signal["rms_db"] < -30:
        issues["low_volume"] = True
        has_issues = True

    is_excellent = not has_issues and not has_problems
    is_good = has_issues and not has_problems

    return {
        "is_excellent": is_excellent,
        "is_good": is_good,
        "has_issues": has_issues,
        "has_problems": has_problems,
        "issues": issues,
    }


def build_synthesis_input(
    text: str,
    voice: str,
    mode: str,
    speed: float = 1.0,
    num_step: int | None = None,
    guidance_scale: float | None = None,
    denoise: bool | None = None,
    t_shift: float | None = None,
    language: str | None = None,
) -> dict:
    """Build ChatterBox-style structured input dict for a synthesis span."""
    return {
        "text": text[:2000],
        "speaker": voice,
        "mode": mode,
        "model_parameters": {
            "speed": speed,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
            "t_shift": t_shift,
        },
        "language": language,
    }


def build_synthesis_output(
    tensors: list,
    wav_bytes: bytes,
    latency_s: float,
    text: str,
    voice: str,
    mode: str,
    speed: float,
    chunks_succeeded: int = 1,
    chunks_total: int = 1,
    ttfc_ms: float | None = None,
    device: str = "cuda",
    *,
    extra: dict | None = None,
) -> dict:
    """Build ChatterBox-style structured output dict for a synthesis span.

    Matches the field structure from ChatterBox's quality_tracer.py:
      performance, quality (signal + duration + assessment), resources, health.
    """
    import torch

    # Duration
    duration_s = sum(
        (t.shape[-1] if hasattr(t, "shape") else len(t)) for t in tensors
    ) / 24_000
    duration_ms = duration_s * 1000

    # Performance
    rtf = latency_s / duration_s if duration_s > 0 else float("inf")
    throughput = chunks_succeeded / latency_s if latency_s > 0 else 0

    # Duration outlier detection (following ChatterBox pattern)
    text_length = len(text)
    expected_duration_s = text_length * 0.08
    duration_ratio = duration_s / expected_duration_s if expected_duration_s > 0 else 1.0
    is_outlier = duration_ratio < 0.3 or duration_ratio > 3.0

    # Signal quality metrics
    signal = _compute_signal_metrics(tensors)
    assessment = _compute_quality_assessment(signal, duration_ratio, is_outlier)

    # GPU memory
    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    gpu_available = False
    if device == "cuda" and torch.cuda.is_available():
        gpu_available = True
        gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

    # Health
    success_rate = chunks_succeeded / chunks_total if chunks_total > 0 else 1.0

    output: dict = {
        "audio": create_audio_data_uri_from_bytes(wav_bytes),
        "file_size_bytes": len(wav_bytes),
        "normalized_text": text[:2000],
        "status": "completed",
        # Performance metrics
        "performance": {
            "rtf": round(rtf, 3),
            "generation_time_ms": round(latency_s * 1000, 2),
            "audio_duration_ms": round(duration_ms, 2),
            "throughput_chunks_per_sec": round(throughput, 2),
            **(
                {"time_to_first_byte_ms": round(ttfc_ms, 2)}
                if ttfc_ms is not None else {}
            ),
        },
        # Quality metrics
        "quality": {
            "duration_ratio": round(duration_ratio, 2),
            "is_outlier": is_outlier,
            "expected_duration_ms": round(expected_duration_s * 1000, 2),
            "actual_duration_ms": round(duration_ms, 2),
            "signal": signal,
            "assessment": assessment,
        },
        # Resource usage
        "resources": {
            "gpu_memory_allocated_mb": round(gpu_allocated_mb, 2),
            "gpu_memory_reserved_mb": round(gpu_reserved_mb, 2),
            "gpu_available": gpu_available,
            "device": device,
        },
        # Health
        "health": {
            "success_rate": round(success_rate, 3),
            "chunks_succeeded": chunks_succeeded,
            "chunks_failed": chunks_total - chunks_succeeded,
            "status": "success",
        },
    }

    if extra:
        output.update(extra)

    return output


__all__ = [
    "get_langfuse_client",
    "get_current_trace_id",
    "get_observe",
    "is_enabled",
    "update_current_trace",
    "create_audio_data_uri_from_bytes",
    "build_synthesis_input",
    "build_synthesis_output",
    "flush_in_background",
    "flush_blocking",
    "join_background_flushes",
]
