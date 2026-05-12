"""
Auto-benchmark: find optimal batch size and timeout for the current GPU.

First run on new hardware: benchmarks all batch sizes and saves a profile.
Subsequent runs: loads cached profile (keyed by GPU + num_step).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "gpu_benchmark_profile.json"
BATCH_SIZES = [1, 2, 4, 8, 32, 64, 128]
TEXT = "Benchmark test sentence for batch throughput measurement."


# ── Helpers ──────────────────────────────────────────────────────────


def _gpu_fingerprint() -> dict:
    """Hardware fingerprint for cache invalidation."""
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": round(props.total_memory / 1024 / 1024),
    }


def _profile_path(cache_dir: str | None) -> Path | None:
    """Where to store/load the benchmark profile."""
    if cache_dir:
        return Path(cache_dir) / PROFILE_FILENAME
    return None


def _get_vram_mb() -> int:
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024 / 1024)


def _run_batch(model: OmniVoice, batch_size: int, num_step: int) -> tuple[float, int]:
    texts = [TEXT] * batch_size
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(text=texts, num_step=num_step)
    elapsed = time.perf_counter() - t0
    nans = sum(1 for t in output if isinstance(t, torch.Tensor) and torch.isnan(t).any().item())
    ok = len(output) - nans
    return elapsed, ok


def _compute_optimal_timeout(single_latency_s: float) -> int:
    timeout_ms = round(single_latency_s * 0.5 * 1000)
    return max(5, min(100, timeout_ms))


def _pick_optimal(results: list[dict]) -> dict:
    if not results:
        return {"batch_size": 4, "throughput_req_s": 0.0}

    best = max(results, key=lambda r: r["throughput_req_s"])
    best_tp = best["throughput_req_s"]

    # Among results within 3% of peak, pick smallest batch
    near_peak = [r for r in results if r["throughput_req_s"] >= best_tp * 0.97]
    chosen = min(near_peak, key=lambda r: r["batch_size"])

    if chosen["batch_size"] != best["batch_size"]:
        logger.info(
            "Peak at batch=%d (%.1f req/s), batch=%d gives 97%%+ (%.1f req/s)",
            best["batch_size"], best_tp,
            chosen["batch_size"], chosen["throughput_req_s"],
        )
    return chosen


# ── Cache ────────────────────────────────────────────────────────────


def _load_cached_profile(
    cache_dir: str | None,
    num_step: int,
) -> dict | None:
    """Load cached benchmark profile if it matches current hardware."""
    ppath = _profile_path(cache_dir)
    if ppath is None or not ppath.exists():
        return None

    try:
        with open(ppath) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load benchmark profile: %s", e)
        return None

    # Validate fingerprint — re-run if hardware changed
    current_fp = _gpu_fingerprint()
    cached_fp = cached.get("fingerprint", {})
    if cached_fp != current_fp:
        logger.info(
            "Hardware changed (was %s, now %s) — re-benchmarking",
            cached_fp.get("gpu_name", "?"),
            current_fp["gpu_name"],
        )
        return None

    # Validate num_step matches
    if cached.get("num_step") != num_step:
        logger.info(
            "num_step changed (%d -> %d) — re-benchmarking",
            cached.get("num_step"), num_step,
        )
        return None

    logger.info(
        "Loaded cached benchmark profile: batch=%d, timeout=%dms, %.1f req/s",
        cached["optimal_batch_size"],
        cached["optimal_batch_timeout_ms"],
        cached["optimal_throughput_req_s"],
    )
    return cached


def _save_profile(profile: dict, cache_dir: str | None) -> None:
    """Save benchmark profile to disk."""
    ppath = _profile_path(cache_dir)
    if ppath is None:
        return

    try:
        os.makedirs(ppath.parent, exist_ok=True)
        tmp = ppath.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(profile, f, indent=2)
        tmp.replace(ppath)
        logger.info("Benchmark profile saved to %s", ppath)
    except OSError as e:
        logger.warning("Failed to save benchmark profile: %s", e)


# ── Main ─────────────────────────────────────────────────────────────


def find_optimal_batch_size(
    model: OmniVoice,
    num_step: int = 16,
    cache_dir: str | None = None,
    bench_rounds: int = 3,
) -> dict:
    """
    Find optimal batch config for this GPU.

    Checks cache first. If no cached profile (or hardware changed),
    runs the full benchmark and saves results.
    """
    # Try cache first
    cached = _load_cached_profile(cache_dir, num_step)
    if cached is not None:
        return cached

    logger.info("Starting GPU auto-benchmark (num_step=%d)...", num_step)

    # Warmup
    warmup_step = min(num_step, 4)
    for bs in [1, 2, 4, 8]:
        try:
            _run_batch(model, bs, warmup_step)
        except Exception as e:
            logger.warning("Warmup batch=%d failed: %s", bs, e)

    # Benchmark
    results = []
    for bs in BATCH_SIZES:
        try:
            timings = []
            total_ok = 0

            for _ in range(bench_rounds):
                elapsed, ok = _run_batch(model, bs, num_step)
                if ok == bs:
                    timings.append(elapsed)
                    total_ok += ok
                else:
                    logger.warning("Batch=%d: %d/%d outputs had NaN", bs, bs - ok, bs)

            if not timings:
                logger.warning("Batch=%d: all attempts had errors, stopping", bs)
                break

            mean_latency = statistics.mean(timings)
            total_wall = sum(timings)
            throughput = total_ok / total_wall
            vram_after = _get_vram_mb()

            entry = {
                "batch_size": bs,
                "latency_s": round(mean_latency, 3),
                "throughput_req_s": round(throughput, 2),
                "vram_mb": vram_after,
            }
            results.append(entry)
            logger.info(
                "  batch=%2d: %.3fs, %.1f req/s, %d MB VRAM",
                bs, mean_latency, throughput, vram_after,
            )

            # Stop once latency exceeds 1 second
            if mean_latency > 1.0:
                logger.info("Latency >1s at batch=%d, stopping search", bs)
                gc.collect()
                torch.cuda.empty_cache()
                break

        except Exception as e:
            logger.warning("Batch=%d failed: %s — stopping search", bs, e)
            gc.collect()
            torch.cuda.empty_cache()
            break

    if not results:
        logger.warning("Benchmark failed, defaulting to batch_size=4")
        return {
            "optimal_batch_size": 4,
            "optimal_batch_timeout_ms": 50,
            "optimal_throughput_req_s": 0.0,
            "results": [],
        }

    optimal = _pick_optimal(results)
    single_latency = results[0]["latency_s"]
    timeout_ms = _compute_optimal_timeout(single_latency)
    vram_end = _get_vram_mb()

    summary = {
        "fingerprint": _gpu_fingerprint(),
        "num_step": num_step,
        "optimal_batch_size": optimal["batch_size"],
        "optimal_batch_timeout_ms": timeout_ms,
        "optimal_throughput_req_s": optimal["throughput_req_s"],
        "single_request_latency_ms": round(single_latency * 1000, 1),
        "vram_used_mb": vram_end,
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "results": results,
    }

    logger.info(
        "Benchmark complete: batch=%d, timeout=%dms, %.1f req/s "
        "(single=%.1fms), VRAM %d/%d MB",
        summary["optimal_batch_size"],
        summary["optimal_batch_timeout_ms"],
        summary["optimal_throughput_req_s"],
        summary["single_request_latency_ms"],
        summary["vram_used_mb"],
        summary["vram_total_mb"],
    )

    # Save for next startup
    _save_profile(summary, cache_dir)

    return summary
