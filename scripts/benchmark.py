#!/usr/bin/env python3
"""
Benchmark omnivoice-server latency and throughput.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --host http://localhost:8880 --voice anwar --runs 5
"""

import argparse
import json
import statistics
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

TEXTS = [
    ("short",   "Hello, this is a benchmark test."),
    ("medium",  "The quick brown fox jumps over the lazy dog."),
    ("long",    "OmniVoice is a high quality text to speech system built for production use."),
]


def synthesize(host: str, text: str, voice: str, api_key: str | None) -> tuple[float, int]:
    body = json.dumps({"model": "omnivoice", "input": text, "voice": voice}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(f"{host}/v1/audio/speech", data=body, headers=headers)
    t0 = time.perf_counter()
    with urlopen(req) as resp:
        data = resp.read()
    return (time.perf_counter() - t0) * 1000, len(data)


def main():
    parser = argparse.ArgumentParser(description="Benchmark omnivoice-server")
    parser.add_argument("--host", default="http://localhost:8880")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    # Wait for server
    for attempt in range(10):
        try:
            urlopen(f"{args.host}/ready", timeout=2)
            break
        except (URLError, OSError):
            if attempt == 9:
                print("ERROR: server not ready at", args.host)
                raise
            time.sleep(1)

    print(f"Host:  {args.host}")
    print(f"Voice: {args.voice}")
    print(f"Runs:  {args.runs}  (+ {args.warmup} warmup)\n")

    print(f"Warming up ({args.warmup} run)...")
    for _ in range(args.warmup):
        synthesize(args.host, TEXTS[0][1], args.voice, args.api_key)

    print(f"\n{'Label':<10}  {'min':>7}  {'avg':>7}  {'max':>7}  {'KB':>6}  Text")
    print("─" * 78)

    all_avgs = []
    for label, text in TEXTS:
        times = []
        kb = 0
        for _ in range(args.runs):
            ms, size = synthesize(args.host, text, args.voice, args.api_key)
            times.append(ms)
            kb = size // 1024
        avg = statistics.mean(times)
        all_avgs.append(avg)
        print(
            f"{label:<10}  {min(times):>6.0f}ms  {avg:>6.0f}ms  {max(times):>6.0f}ms"
            f"  {kb:>5}KB  \"{text[:40]}\""
        )

    print("─" * 78)
    print(f"{'Overall':<10}  {'':>7}  {statistics.mean(all_avgs):>6.0f}ms  {'':>7}")


if __name__ == "__main__":
    main()
