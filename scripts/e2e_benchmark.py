#!/usr/bin/env python3
"""
E2E Benchmark — Real inference test across all TTS endpoints.

Runs against a live server with the actual model loaded.
Tracks performance metrics and detects regressions over time.

Usage:
    # Start server first:
    make run
    # Then run benchmarks:
    uv run python scripts/e2e_benchmark.py

Results are appended to tests/benchmark_results.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

import httpx

BASE_URL = os.environ.get("SERVER_URL", "http://localhost:8880")
RESULTS_FILE = Path(__file__).resolve().parent.parent / "tests" / "benchmark_results.jsonl"
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10
REGRESSION_THRESHOLD = 0.30  # 30% — model has ~35% natural duration variance

TEST_CASES = [
    {"name": "short_en", "text": "Hello, how are you doing today?"},
    {
        "name": "medium_ms",
        "text": "Okay, terima kasih sebab call Maybank Customer Service. Have a nice day!",
    },
    {
        "name": "code_mix",
        "text": "So easy also dunno meh? Kalau nak beli rumah, better check your DSR dulu.",
    },
]


def get_gpu_memory() -> dict | None:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            used, total = r.stdout.strip().split(",")
            return {"used_mb": float(used.strip()), "total_mb": float(total.strip())}
    except Exception:
        pass
    return None


def get_ram_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def parse_wav_duration(wav_bytes: bytes) -> float:
    if wav_bytes[:4] != b"RIFF":
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    data_size = struct.unpack_from("<I", wav_bytes, 40)[0]
    return (data_size // 2) / sample_rate


def check_server(url: str) -> dict | None:
    try:
        r = httpx.get(f"{url}/health", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_voice(url: str) -> str | None:
    try:
        r = httpx.get(f"{url}/api/voices", timeout=5)
        voices = r.json().get("voices", [])
        if voices:
            return voices[0]
    except Exception:
        pass
    try:
        r = httpx.get(f"{url}/v1/audio/voices", timeout=5)
        data = r.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None


def benchmark_generate(
    text: str, voice: str, url: str, seed: int | None = None
) -> dict:
    payload = {"text": text, "voice_ref_path": voice}
    if seed is not None:
        payload["seed"] = seed
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/generate", json=payload, timeout=120)
    wall = time.perf_counter() - t0

    assert r.status_code == 200, f"generate failed: {r.status_code} {r.text[:200]}"

    latency = float(r.headers.get("X-Latency-S", wall))
    duration = float(r.headers.get("X-Audio-Duration-S", "0"))
    rtf = float(r.headers.get("X-RTF", "0"))
    audio = r.content
    audio_dur = parse_wav_duration(audio)

    return {
        "wall_time_s": round(wall, 4),
        "latency_s": round(latency, 4),
        "audio_duration_s": round(audio_dur or duration, 4),
        "rtf": round(rtf or (wall / audio_dur if audio_dur > 0 else 0), 4),
        "audio_size_bytes": len(audio),
        "audio_hash": hashlib.sha256(audio).hexdigest()[:16],
        "status": r.status_code,
    }


def benchmark_speech(
    text: str, voice: str, url: str, seed: int | None = None
) -> dict:
    payload = {"input": text, "voice": voice}
    if seed is not None:
        payload["seed"] = seed
    t0 = time.perf_counter()
    r = httpx.post(f"{url}/v1/audio/speech", json=payload, timeout=120)
    wall = time.perf_counter() - t0

    assert r.status_code == 200, f"speech failed: {r.status_code} {r.text[:200]}"

    audio = r.content
    audio_dur = parse_wav_duration(audio)
    rtf = wall / audio_dur if audio_dur > 0 else 0

    return {
        "wall_time_s": round(wall, 4),
        "latency_s": round(wall, 4),
        "audio_duration_s": round(audio_dur, 4),
        "rtf": round(rtf, 4),
        "audio_size_bytes": len(audio),
        "audio_hash": hashlib.sha256(audio).hexdigest()[:16],
        "status": r.status_code,
    }


def benchmark_websocket(
    text: str, voice: str, url: str, seed: int | None = None
) -> dict:
    import asyncio
    import base64

    try:
        import websockets
    except ImportError:
        return {"error": "websockets not installed, skipped"}

    ws_url = url.replace("http", "ws") + "/tts/websocket"

    async def _run():
        msg = {"transcript": text, "voice": {"id": voice}, "continue": False}
        if seed is not None:
            msg["seed"] = seed
        t0 = time.perf_counter()
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(msg))
            chunks = []
            while True:
                resp = await ws.recv()
                data = json.loads(resp)
                if data.get("done"):
                    break
                if data.get("data"):
                    chunks.append(base64.b64decode(data["data"]))
        wall = time.perf_counter() - t0

        total = b"".join(chunks)
        dur = (len(total) // 2) / 24000
        rtf = wall / dur if dur > 0 else 0
        return {
            "wall_time_s": round(wall, 4),
            "latency_s": round(wall, 4),
            "audio_duration_s": round(dur, 4),
            "rtf": round(rtf, 4),
            "audio_size_bytes": len(total),
            "audio_hash": hashlib.sha256(total).hexdigest()[:16],
            "status": 200,
        }

    return asyncio.run(_run())


ENDPOINTS = [
    ("generate", benchmark_generate),
    ("v1_audio_speech", benchmark_speech),
    ("websocket", benchmark_websocket),
]


def run_benchmark(
    test_name: str,
    text: str,
    voice: str,
    endpoint_name: str,
    benchmark_fn,
    seed: int | None = None,
) -> list[dict]:
    # Warmup
    for i in range(WARMUP_RUNS):
        try:
            benchmark_fn(text, voice, BASE_URL, seed=seed)
        except Exception as e:
            print(f"    warmup {i+1} failed: {e}")
            return []

    # Benchmarked runs
    results = []
    for i in range(BENCHMARK_RUNS):
        try:
            r = benchmark_fn(text, voice, BASE_URL, seed=seed)
            r["test_name"] = test_name
            r["endpoint"] = endpoint_name
            r["voice"] = voice
            r["run"] = i + 1
            results.append(r)
        except Exception as e:
            print(f"    run {i+1} failed: {e}")

    return results


def print_summary(test_name: str, endpoint_name: str, results: list[dict]):
    if not results:
        print(f"  {endpoint_name}: NO RESULTS")
        return

    walls = [r["wall_time_s"] for r in results]
    rtfs = [r["rtf"] for r in results]
    durs = [r["audio_duration_s"] for r in results]
    sizes = [r["audio_size_bytes"] for r in results]

    p50 = median(walls)
    p95 = sorted(walls)[min(int(len(walls) * 0.95), len(walls) - 1)]
    sd = stdev(walls) if len(walls) > 1 else 0

    print(f"  {endpoint_name}:")
    print(f"    latency: avg={mean(walls):.3f}s  p50={p50:.3f}s  p95={p95:.3f}s  sd={sd:.3f}s")
    print(f"    RTF:     avg={mean(rtfs):.3f}  p50={median(rtfs):.3f}")
    print(f"    audio:   {mean(durs):.2f}s  ({mean(sizes)/1024:.0f} KB)")


def check_cross_endpoint_consistency(all_results: list[dict]):
    """Check that all endpoints produce similar audio duration for same input."""
    print("\n--- Cross-endpoint consistency ---")
    for tc in TEST_CASES:
        durations = {}
        for r in all_results:
            if r["test_name"] == tc["name"] and "error" not in r:
                ep = r["endpoint"]
                if ep not in durations:
                    durations[ep] = []
                durations[ep].append(r["audio_duration_s"])

        if len(durations) < 2:
            continue

        avg_by_ep = {ep: mean(ds) for ep, ds in durations.items()}
        eps = list(avg_by_ep.keys())
        base_ep = eps[0]
        base_dur = avg_by_ep[base_ep]

        for ep in eps[1:]:
            diff_pct = abs(avg_by_ep[ep] - base_dur) / base_dur * 100 if base_dur > 0 else 0
            status = "OK" if diff_pct < 40 else "MISMATCH"
            print(
                f"  {tc['name']}: {base_ep}={base_dur:.2f}s vs "
                f"{ep}={avg_by_ep[ep]:.2f}s  ({diff_pct:.1f}% diff)  [{status}]"
            )


def check_seed_consistency(voice: str, seed: int = 42):
    """Run each endpoint once with same seed, compare audio hashes."""
    print(f"\n--- Seed consistency (seed={seed}) ---")
    for tc in TEST_CASES:
        hashes = {}
        durations = {}
        for ep_name, bench_fn in ENDPOINTS:
            try:
                r = bench_fn(tc["text"], voice, BASE_URL, seed=seed)
                hashes[ep_name] = r.get("audio_hash", "?")
                durations[ep_name] = r.get("audio_duration_s", 0)
            except Exception as e:
                print(f"  {tc['name']}/{ep_name}: FAILED ({e})")
                continue

        if len(hashes) < 2:
            continue

        unique_hashes = set(hashes.values())
        eps = list(durations.keys())
        base_ep = eps[0]
        base_dur = durations[base_ep]

        print(f"  {tc['name']}:")
        for ep in eps:
            diff = abs(durations[ep] - base_dur) / base_dur * 100 if base_dur > 0 else 0
            print(f"    {ep}: dur={durations[ep]:.3f}s hash={hashes[ep]} ({diff:.1f}% diff)")

        if len(unique_hashes) == 1:
            print("    IDENTICAL audio across all endpoints")
        else:
            print(f"    DIFFERENT audio ({len(unique_hashes)} distinct hashes)")


def load_previous() -> dict | None:
    if not RESULTS_FILE.exists():
        return None
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) < 2:
        return None
    try:
        return json.loads(lines[-2])
    except (json.JSONDecodeError, IndexError):
        return None


def detect_regressions(current: dict, previous: dict) -> list[str]:
    regressions = []
    prev_by_key = {}
    for r in previous.get("results", []):
        if "error" in r:
            continue
        key = f"{r['test_name']}/{r['endpoint']}"
        prev_by_key.setdefault(key, []).append(r)

    for r in current.get("results", []):
        if "error" in r:
            continue
        key = f"{r['test_name']}/{r['endpoint']}"
        prev = prev_by_key.get(key)
        if not prev:
            continue

        prev_avg_wall = mean(p["wall_time_s"] for p in prev)
        prev_avg_rtf = mean(p["rtf"] for p in prev)

        wall_change = (r["wall_time_s"] - prev_avg_wall) / prev_avg_wall
        rtf_change = (r["rtf"] - prev_avg_rtf) / prev_avg_rtf if prev_avg_rtf > 0 else 0

        if wall_change > REGRESSION_THRESHOLD:
            regressions.append(
                f"  REGRESSION: {key} wall_time "
                f"{prev_avg_wall:.3f}s -> {r['wall_time_s']:.3f}s (+{wall_change*100:.1f}%)"
            )
        if rtf_change > REGRESSION_THRESHOLD:
            regressions.append(
                f"  REGRESSION: {key} RTF "
                f"{prev_avg_rtf:.3f} -> {r['rtf']:.3f} (+{rtf_change*100:.1f}%)"
            )

    return regressions


def main():
    print("=" * 60)
    print("OmniVoice E2E Benchmark")
    print("=" * 60)

    health = check_server(BASE_URL)
    if not health:
        print(f"\nServer not running at {BASE_URL}")
        print("Start with: make run")
        sys.exit(1)

    voice = get_voice(BASE_URL)
    if not voice:
        print("No voices available on server")
        sys.exit(1)

    print(f"\nServer:   {BASE_URL}")
    print(f"Model:    {health.get('model_id', 'unknown')}")
    print(f"Device:   {health.get('device', 'unknown')}")
    print(f"Voice:    {voice}")
    print(f"Warmup:   {WARMUP_RUNS}")
    print(f"Runs:     {BENCHMARK_RUNS}")

    gpu_before = get_gpu_memory()
    ram_before = get_ram_mb()
    if gpu_before:
        print(f"GPU:      {gpu_before['used_mb']:.0f}/{gpu_before['total_mb']:.0f} MB")
    print(f"RAM:      {ram_before:.0f} MB")

    all_results = []

    # Deterministic seed for reproducible benchmark runs
    benchmark_seed = 42

    for tc in TEST_CASES:
        print(f"\n--- {tc['name']} ---")
        print(f"Text: {tc['text'][:60]}...")

        for ep_name, bench_fn in ENDPOINTS:
            results = run_benchmark(
                tc["name"], tc["text"], voice, ep_name, bench_fn,
                seed=benchmark_seed,
            )
            print_summary(tc["name"], ep_name, results)
            all_results.extend(results)

    gpu_after = get_gpu_memory()
    ram_after = get_ram_mb()
    print("\n--- Resources after ---")
    if gpu_after:
        delta = gpu_after["used_mb"] - gpu_before["used_mb"] if gpu_before else 0
        print(
            f"GPU: {gpu_after['used_mb']:.0f}/{gpu_after['total_mb']:.0f} MB "
            f"(delta: {delta:+.0f} MB)"
        )
    print(f"RAM: {ram_after:.0f} MB (delta: {ram_after - ram_before:+.0f} MB)")

    # Cross-endpoint consistency
    check_cross_endpoint_consistency(all_results)
    check_seed_consistency(voice, seed=benchmark_seed)

    # Save
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": health.get("model_id"),
        "device": health.get("device"),
        "voice": voice,
        "gpu_before_mb": gpu_before["used_mb"] if gpu_before else None,
        "gpu_after_mb": gpu_after["used_mb"] if gpu_after else None,
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(ram_after, 1),
        "results": all_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nResults appended to {RESULTS_FILE}")

    # Regression check
    previous = load_previous()
    if previous:
        regressions = detect_regressions(entry, previous)
        if regressions:
            print("\n" + "!" * 60)
            print("REGRESSIONS DETECTED:")
            for r in regressions:
                print(r)
            print("!" * 60)
            sys.exit(1)
        else:
            print("No regressions vs previous run.")
    else:
        print("First run — no previous results to compare.")


if __name__ == "__main__":
    main()
