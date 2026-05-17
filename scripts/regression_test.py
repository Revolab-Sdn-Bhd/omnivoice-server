"""
Regression test: generate all quotes with a specific model revision,
transcribe with WhisperX, compute CER, and record results.

Usage:
    uv run python scripts/regression_test.py \
      --revision abc12345 [--voice anwar] [--runs 5]

Results saved to regression_results/<revision>/.
If all 5 runs are recorded, revision is skipped.
Delete a run's JSONL to re-run it.
"""

import argparse
import gc
import io
import json
import logging
import time
from pathlib import Path

import requests
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("regression_results")
RUNS = 5
WHISPERX_MODEL = "large-v3"
WHISPERX_LANGUAGE = "en"
WHISPERX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_quotes(base_url: str) -> list[str]:
    r = requests.get(f"{base_url}/api/quotes", timeout=10)
    r.raise_for_status()
    return r.json()["quotes"]


def get_health(base_url: str) -> dict:
    r = requests.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def switch_revision(base_url: str, revision: str) -> None:
    health = get_health(base_url)
    current = health.get("model_revision", "")
    if current == revision[:8] or revision.startswith(current):
        logger.info("Already on revision %s", revision[:8])
        return

    logger.info("Switching to revision %s...", revision[:8])
    r = requests.post(
        f"{base_url}/api/model/switch-revision",
        json={"revision": revision},
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    logger.info("Switch result: %s -> %s", current, data.get("revision"))

    for _ in range(120):
        try:
            h = get_health(base_url)
            if h["status"] == "ok":
                logger.info("Model ready: %s", h.get("model_revision"))
                return
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("Model failed to become ready after switching")


def generate_speech(base_url: str, text: str, voice: str) -> bytes:
    r = requests.post(
        f"{base_url}/v1/audio/speech",
        json={"model": "revovoice", "input": text, "voice": voice},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def load_whisperx_model():
    import whisperx

    logger.info(
        "Loading WhisperX (%s, %s)...", WHISPERX_MODEL, WHISPERX_DEVICE
    )
    compute = "float16" if WHISPERX_DEVICE == "cuda" else "int8"
    model = whisperx.load_model(
        WHISPERX_MODEL, WHISPERX_DEVICE,
        language=WHISPERX_LANGUAGE, compute_type=compute,
    )
    return model


def transcribe(whx_model, wav_bytes: bytes) -> str:
    import numpy as np
    import torchaudio

    try:
        buf = io.BytesIO(wav_bytes)
        waveform, sr = torchaudio.load(buf)
    except Exception:
        return ""

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    if len(audio_np) < 1600:  # < 0.1s
        return ""

    try:
        result = whx_model.transcribe(
            audio_np, batch_size=16, language=WHISPERX_LANGUAGE
        )
        segments = result.get("segments", [])
        return " ".join(s["text"].strip() for s in segments)
    except (IndexError, ValueError):
        return ""


def normalize_text_for_comparison(text: str) -> str:
    """Normalize reference text using revo-norm for fair CER comparison."""
    try:
        from omnivoice_server.utils.text import normalize_for_tts
        return normalize_for_tts(text)
    except Exception:
        return text


def compute_cer(reference: str, hypothesis: str) -> float:
    from jiwer import cer
    return cer(reference, hypothesis)


def run_test(
    base_url: str,
    revision: str,
    voice: str,
    quotes: list[str],
    run_idx: int,
    result_file: Path,
    audio_dir: Path,
) -> None:
    whx_model = load_whisperx_model()

    # Resume: skip already-processed quotes
    done_indices = set()
    if result_file.exists():
        for line in result_file.read_text().strip().split("\n"):
            if line.strip():
                done_indices.add(json.loads(line)["quote_idx"])

    for i, quote in enumerate(quotes):
        if i in done_indices:
            continue
        logger.info(
            "[run %d] Quote %d/%d: %s",
            run_idx, i + 1, len(quotes), quote[:60],
        )

        t0 = time.monotonic()
        wav_bytes = generate_speech(base_url, quote, voice)
        gen_time = time.monotonic() - t0

        audio_path = audio_dir / f"run{run_idx}_q{i:03d}.wav"
        audio_path.write_bytes(wav_bytes)

        t0 = time.monotonic()
        hypothesis = transcribe(whx_model, wav_bytes)
        trans_time = time.monotonic() - t0

        normalized_ref = normalize_text_for_comparison(quote)
        normalized_hyp = normalize_text_for_comparison(hypothesis)
        cer_raw = compute_cer(quote, hypothesis)
        cer_norm = compute_cer(normalized_ref, normalized_hyp)

        result = {
            "run": run_idx,
            "quote_idx": i,
            "reference": quote,
            "reference_normalized": normalized_ref,
            "hypothesis": hypothesis,
            "hypothesis_normalized": normalized_hyp,
            "cer": round(cer_raw, 4),
            "cer_normalized": round(cer_norm, 4),
            "gen_time_s": round(gen_time, 2),
            "trans_time_s": round(trans_time, 2),
        }

        with open(result_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

        logger.info(
            "  CER=%.4f gen=%.1fs trans=%.1fs",
            cer_raw, gen_time, trans_time,
        )

    del whx_model
    torch.cuda.empty_cache()
    gc.collect()


def compute_summary(result_file: Path) -> dict:
    lines = result_file.read_text().strip().split("\n")
    results = [json.loads(line) for line in lines if line.strip()]
    if not results:
        return {}

    cers = [r["cer"] for r in results]
    cers_norm = [r.get("cer_normalized", r["cer"]) for r in results]
    return {
        "num_quotes": len(results),
        "mean_cer": round(sum(cers) / len(cers), 4),
        "max_cer": round(max(cers), 4),
        "min_cer": round(min(cers), 4),
        "mean_cer_normalized": round(sum(cers_norm) / len(cers_norm), 4),
        "max_cer_normalized": round(max(cers_norm), 4),
        "min_cer_normalized": round(min(cers_norm), 4),
        "total_gen_time_s": round(sum(r["gen_time_s"] for r in results), 2),
        "total_trans_time_s": round(
            sum(r["trans_time_s"] for r in results), 2
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regression test for model revisions"
    )
    parser.add_argument(
        "--revision", required=True,
        help="Model revision hash (full or short)",
    )
    parser.add_argument(
        "--voice", default="anwar", help="Voice (default: anwar)",
    )
    parser.add_argument(
        "--runs", type=int, default=RUNS,
        help=f"Number of runs (default: {RUNS})",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--skip-switch", action="store_true",
        help="Skip revision switch",
    )
    args = parser.parse_args()

    revision = args.revision
    base_url = args.base_url.rstrip("/")

    rev_dir = RESULTS_DIR / revision[:8]
    rev_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = rev_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    logger.info("Fetching quotes from %s", base_url)
    quotes = get_quotes(base_url)
    logger.info("Got %d quotes", len(quotes))

    if not args.skip_switch:
        switch_revision(base_url, revision)

    health = get_health(base_url)
    active_rev = health.get("model_revision", "")
    logger.info("Active revision: %s", active_rev)

    for run_idx in range(1, args.runs + 1):
        result_file = rev_dir / f"run{run_idx}.jsonl"

        if result_file.exists():
            existing = result_file.read_text().strip().split("\n")
            if len(existing) == len(quotes):
                logger.info(
                    "Run %d complete (%d results), skipping",
                    run_idx, len(existing),
                )
                continue

        logger.info("=== Run %d/%d ===", run_idx, args.runs)
        run_test(
            base_url, revision, args.voice,
            quotes, run_idx, result_file, audio_dir,
        )

    logger.info("=== Summary for revision %s ===", revision[:8])
    all_summaries = []
    for run_idx in range(1, args.runs + 1):
        result_file = rev_dir / f"run{run_idx}.jsonl"
        if result_file.exists():
            summary = compute_summary(result_file)
            summary["run"] = run_idx
            all_summaries.append(summary)
            logger.info(
                "Run %d: mean_cer=%.4f (norm=%.4f) max_cer=%.4f",
                run_idx, summary["mean_cer"],
                summary.get("mean_cer_normalized", 0), summary["max_cer"],
            )

    overall_cer = 0.0
    overall_cer_norm = 0.0
    if all_summaries:
        overall_cer = sum(s["mean_cer"] for s in all_summaries) / len(all_summaries)
        overall_cer_norm = sum(
            s.get("mean_cer_normalized", s["mean_cer"]) for s in all_summaries
        ) / len(all_summaries)
        logger.info(
            "Overall mean CER across %d runs: %.4f (normalized: %.4f)",
            len(all_summaries), overall_cer, overall_cer_norm,
        )

    summary_file = rev_dir / "summary.json"
    summary_data = {
        "revision": revision,
        "active_revision": active_rev,
        "voice": args.voice,
        "num_quotes": len(quotes),
        "runs": all_summaries,
        "overall_mean_cer": round(overall_cer, 4) if all_summaries else None,
        "overall_mean_cer_normalized": round(overall_cer_norm, 4) if all_summaries else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_file.write_text(json.dumps(summary_data, indent=2))
    logger.info("Summary saved to %s", summary_file)


if __name__ == "__main__":
    main()
