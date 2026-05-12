#!/usr/bin/env python3
"""Benchmark num_step reduction for OmniVoice TTS.

Tests different num_step values, measures latency, and compares audio
quality against a reference (num_step=32) using spectrogram distance metrics.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_num_step.py
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Config
STEP_VALUES = [32, 16, 8, 6, 4, 2]
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5
SAMPLE_TEXT = "Selamat pagi, bagaimana dengan keadaan anda hari ini?"
OUTPUT_DIR = Path("bench_output")
RESULTS_FILE = OUTPUT_DIR / "bench_num_step_results.json"

# Use the first available voice
VOICES_DIR = Path(
    "/mnt/data/shared/hf/hub/datasets--Revolab--voices/snapshots/"
    "80334bdb4f0fe690ab2680f54dd72bf835d72ce6/dataset"
)


def find_voice() -> tuple[str, str]:
    """Find a voice WAV and its companion text."""
    import glob

    wavs = sorted(glob.glob(str(VOICES_DIR / "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No WAV files in {VOICES_DIR}")

    wav_path = wavs[0]
    stem = Path(wav_path).stem
    txt_path = VOICES_DIR / f"{stem}.txt"
    json_path = VOICES_DIR / f"{stem}.json"

    ref_text = ""
    if txt_path.exists():
        ref_text = txt_path.read_text().strip()
    elif json_path.exists():
        data = json.loads(json_path.read_text())
        ref_text = data.get("transcript", "")

    logger.info("Using voice: %s (ref_text: '%s')", stem, ref_text[:60])
    return wav_path, ref_text


def load_model(device: str = "cuda:0"):
    """Load OmniVoice model."""
    from omnivoice import OmniVoice

    logger.info("Loading model...")
    t0 = time.monotonic()
    revision = os.environ.get("MODEL_REVISION", "")
    kwargs = dict(device_map=device, dtype=torch.float16)
    if revision:
        kwargs["revision"] = revision
    model = OmniVoice.from_pretrained("Revolab/omnivoice", **kwargs)
    elapsed = time.monotonic() - t0
    logger.info("Model loaded in %.1fs", elapsed)
    return model


def generate_audio(
    model, text: str, ref_audio_path: str, ref_text: str, num_step: int,
) -> list[torch.Tensor]:
    """Generate audio with specific num_step."""
    return model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        num_step=num_step,
    )


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert audio tensor to numpy float32 array."""
    return tensor.squeeze(0).cpu().float().numpy()


def compute_mfcc(audio: np.ndarray, sr: int = 24000, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCC features."""
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def compute_mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Compute mel spectrogram."""
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def spectrogram_distance(mel_a: np.ndarray, mel_b: np.ndarray) -> float:
    """MSE between mel spectrograms, handling different lengths."""
    min_len = min(mel_a.shape[1], mel_b.shape[1])
    a = mel_a[:, :min_len]
    b = mel_b[:, :min_len]
    return float(np.mean((a - b) ** 2))


def mfcc_cosine_distance(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """Cosine distance between mean MFCC vectors."""
    min_len = min(mfcc_a.shape[1], mfcc_b.shape[1])
    a = mfcc_a[:, :min_len].mean(axis=1)
    b = mfcc_b[:, :min_len].mean(axis=1)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 1.0
    return float(1.0 - dot / norm)


def duration_diff(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 24000) -> float:
    """Absolute duration difference in seconds."""
    return abs(len(audio_a) - len(audio_b)) / sr


def run_benchmark():
    """Main benchmark loop."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    wav_path, ref_text = find_voice()

    # Generate reference audio at num_step=32
    logger.info("Generating reference audio (num_step=32)...")
    ref_tensors = generate_audio(model, SAMPLE_TEXT, wav_path, ref_text, 32)
    ref_audio = tensor_to_np(ref_tensors[0])

    ref_mel = compute_mel_spectrogram(ref_audio)
    ref_mfcc = compute_mfcc(ref_audio)

    # Save reference
    import soundfile as sf

    sf.write(str(OUTPUT_DIR / "reference_s32.wav"), ref_audio, 24000)
    logger.info("Reference saved. Duration: %.2fs", len(ref_audio) / 24000)

    results = {
        "timestamp": datetime.now().isoformat(),
        "text": SAMPLE_TEXT,
        "voice": Path(wav_path).stem,
        "reference_step": 32,
        "reference_duration_s": len(ref_audio) / 24000,
        "benchmarks": [],
    }

    for num_step in STEP_VALUES:
        logger.info("=== Benchmarking num_step=%d ===", num_step)

        # Warmup
        for i in range(WARMUP_RUNS):
            _ = generate_audio(model, SAMPLE_TEXT, wav_path, ref_text, num_step)
            logger.info("  Warmup %d/%d done", i + 1, WARMUP_RUNS)

        # Timed runs
        latencies = []
        audios = []
        for i in range(BENCHMARK_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tensors = generate_audio(model, SAMPLE_TEXT, wav_path, ref_text, num_step)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            audios.append(tensor_to_np(tensors[0]))
            logger.info("  Run %d/%d: %.3fs", i + 1, BENCHMARK_RUNS, elapsed)

        # Use the last generated audio for quality comparison
        audio = audios[-1]
        mel = compute_mel_spectrogram(audio)
        mfcc = compute_mfcc(audio)

        mel_mse = spectrogram_distance(ref_mel, mel)
        mfcc_dist = mfcc_cosine_distance(ref_mfcc, mfcc)
        dur_diff = duration_diff(ref_audio, audio)

        # Save audio
        sf.write(str(OUTPUT_DIR / f"sample_s{num_step}.wav"), audio, 24000)

        entry = {
            "num_step": num_step,
            "latency_s": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
            },
            "audio_duration_s": len(audio) / 24000,
            "quality": {
                "mel_mse_vs_ref": mel_mse,
                "mfcc_cosine_dist_vs_ref": mfcc_dist,
                "duration_diff_s": dur_diff,
            },
        }
        results["benchmarks"].append(entry)

        ref_mean = results["benchmarks"][0]["latency_s"]["mean"]
        cur_mean = entry["latency_s"]["mean"]
        speedup = ref_mean / cur_mean if cur_mean > 0 else 0
        logger.info(
            "  num_step=%d: mean=%.3fs, mel_mse=%.4f, mfcc_dist=%.4f, speedup=%.2fx",
            num_step,
            entry["latency_s"]["mean"],
            mel_mse,
            mfcc_dist,
            speedup,
        )

    # Save results
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=to_serializable))
    logger.info("Results saved to %s", RESULTS_FILE)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    ref_latency = results["benchmarks"][0]["latency_s"]["mean"]
    hdr = (
        f"{'num_step':>8} | {'mean(s)':>8} | {'speedup':>8} | "
        f"{'mel_mse':>8} | {'mfcc_d':>8} | {'dur_diff':>8}"
    )
    print(hdr)
    print("-" * 80)
    for b in results["benchmarks"]:
        speedup = ref_latency / b["latency_s"]["mean"]
        q = b["quality"]
        print(
            f"{b['num_step']:>8} | {b['latency_s']['mean']:>8.3f} | "
            f"{speedup:>7.2f}x | {q['mel_mse_vs_ref']:>8.4f} | "
            f"{q['mfcc_cosine_dist_vs_ref']:>8.4f} | "
            f"{q['duration_diff_s']:>8.3f}"
        )
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_benchmark()
