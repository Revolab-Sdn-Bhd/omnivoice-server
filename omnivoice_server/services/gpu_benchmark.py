"""
Auto-benchmark: find optimal batch size for the current GPU at startup.

Tests increasing batch sizes with num_step=4 (fast), finds where
throughput plateaus, returns the optimal batch_max_size.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice

logger = logging.getLogger(__name__)

# Batch sizes to test (skip 16 — known torch.compile recompilation spike)
BATCH_SIZES = [1, 2, 4, 8, 32, 64, 128]
WARMUP_STEP = 4
BENCH_STEP = 4
PLATEAU_THRESHOLD = 0.10  # 10% throughput gain threshold for plateau
TEXT = "Benchmark test sentence for batch throughput measurement."


def _get_vram_mb() -> int:
    """Get current GPU VRAM usage in MB."""
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024 / 1024)


def _run_batch(model: OmniVoice, batch_size: int, num_step: int) -> tuple[float, int]:
    """Run one batched inference. Returns (elapsed_seconds, num_ok_outputs)."""
    texts = [TEXT] * batch_size
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(text=texts, num_step=num_step)
    elapsed = time.perf_counter() - t0
    nans = sum(1 for t in output if isinstance(t, torch.Tensor) and torch.isnan(t).any().item())
    ok = len(output) - nans
    return elapsed, ok


def find_optimal_batch_size(model: OmniVoice) -> dict:
    """
    Benchmark increasing batch sizes and find the optimal one.

    Returns dict with:
        optimal_batch_size: int
        throughput_req_s: float
        results: list of {batch_size, latency_s, throughput_req_s, vram_mb}
    """
    logger.info("Starting GPU auto-benchmark to find optimal batch size...")

    # Warmup for each batch size (trigger compilation)
    for bs in [1, 2, 4, 8]:
        try:
            _run_batch(model, bs, WARMUP_STEP)
        except Exception as e:
            logger.warning("Warmup batch=%d failed: %s", bs, e)

    # Benchmark each batch size
    results = []
    for bs in BATCH_SIZES:
        try:
            timings = []
            total_ok = 0

            for _ in range(3):
                elapsed, ok = _run_batch(model, bs, BENCH_STEP)
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

        except Exception as e:
            logger.warning("Batch=%d failed: %s — stopping search", bs, e)
            gc.collect()
            torch.cuda.empty_cache()
            break

    if not results:
        logger.warning("Benchmark failed, defaulting to batch_size=4")
        return {"optimal_batch_size": 4, "throughput_req_s": 0.0, "results": []}

    # Find plateau: first batch size where next gain < PLATEAU_THRESHOLD
    optimal = results[-1]  # default to largest tested
    for i in range(len(results) - 1):
        curr_tp = results[i]["throughput_req_s"]
        next_tp = results[i + 1]["throughput_req_s"]
        gain = (next_tp - curr_tp) / curr_tp if curr_tp > 0 else 0

        if gain < PLATEAU_THRESHOLD:
            optimal = results[i]
            logger.info(
                "Throughput plateau at batch=%d (%.1f req/s), "
                "next gain only %.1f%%",
                optimal["batch_size"], curr_tp, gain * 100,
            )
            break

    vram_end = _get_vram_mb()
    summary = {
        "optimal_batch_size": optimal["batch_size"],
        "optimal_throughput_req_s": optimal["throughput_req_s"],
        "vram_used_mb": vram_end,
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "results": results,
    }

    logger.info(
        "Benchmark complete: optimal batch=%d, %.1f req/s, VRAM %d/%d MB",
        summary["optimal_batch_size"],
        summary["optimal_throughput_req_s"],
        summary["vram_used_mb"],
        summary["vram_total_mb"],
    )
    return summary
