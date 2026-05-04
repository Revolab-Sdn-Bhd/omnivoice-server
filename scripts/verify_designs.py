#!/usr/bin/env python3
"""
Verify voice design audio files using faster-whisper ASR.

Checks each WAV against its companion .txt for corruption.
Optionally regenerates failed files.

Usage:
    uv run python scripts/verify_designs.py
    uv run python scripts/verify_designs.py --regenerate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DESIGN_DIR = ROOT / "designed_voice"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def similarity(a: str, b: str) -> float:
    """Simple word overlap ratio."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def verify_all(design_dir: Path, model_size: str, threshold: float):
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model '%s' ...", model_size)
    asr = WhisperModel(model_size, device="cuda", compute_type="float16")

    wav_files = sorted(design_dir.rglob("*.wav"))
    logger.info("Found %d WAV files", len(wav_files))

    good = []
    bad = []

    for i, wav in enumerate(wav_files):
        txt_path = wav.with_suffix(".txt")
        expected = txt_path.read_text().strip() if txt_path.exists() else ""

        try:
            segments, info = asr.transcribe(str(wav), beam_size=1)
            transcribed = " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(wav_files), wav.relative_to(design_dir), e)
            bad.append((wav, 0.0, "", expected))
            continue

        sim = similarity(normalize(transcribed), normalize(expected))
        rel = wav.relative_to(design_dir)
        status = "OK" if sim >= threshold else "BAD"

        if sim >= threshold:
            good.append(wav)
            logger.info("[%d/%d] %s %s (%.0f%%)", i + 1, len(wav_files), status, rel, sim * 100)
        else:
            bad.append((wav, sim, transcribed, expected))
            logger.warning(
                "[%d/%d] %s %s (%.0f%%) heard: %s",
                i + 1, len(wav_files), status, rel, sim * 100, transcribed[:80],
            )

    logger.info("Result: %d OK, %d BAD (threshold=%.0f%%)", len(good), len(bad), threshold * 100)

    if bad:
        logger.info("=== Failed files ===")
        for wav, sim, transcribed, expected in bad:
            logger.info("  %s (%.0f%%)", wav.relative_to(design_dir), sim * 100)
            logger.info("    expected: %s", expected[:80])
            logger.info("    heard:    %s", transcribed[:80])

    return bad


def regenerate(bad_files, model_id: str, num_step: int):
    import torch
    import torchaudio

    from omnivoice import OmniVoice

    SAMPLE_RATE = 24_000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    omni = OmniVoice.from_pretrained(model_id, device_map=device, dtype=torch.float16, load_asr=False)
    logger.info("OmniVoice loaded. Regenerating %d files ...", len(bad_files))

    for wav, sim, _, expected in bad_files:
        txt_path = wav.with_suffix(".txt")
        text = txt_path.read_text().strip() if txt_path.exists() else expected
        # Reconstruct instruct from folder/file naming
        rel = wav.relative_to(ROOT / "designed_voice")
        parts = rel.parts
        folder = parts[0]
        stem = Path(parts[1]).stem  # e.g. "female_young adult"

        # Parse gender and age from stem
        gender = stem.split("_")[0].title()
        age_raw = stem.split("_", 1)[1] if "_" in stem else "young adult"
        age = age_raw.replace("adult", "Adult").replace("aged", "-aged")

        if folder == "pitch":
            pitch_raw = age_raw
            pitch_map = {
                "very low pitch": "Very Low Pitch",
                "low pitch": "Low Pitch",
                "moderate pitch": "Moderate Pitch",
                "high pitch": "High Pitch",
                "very high pitch": "Very High Pitch",
            }
            pitch = pitch_map.get(pitch_raw.lower(), "Moderate Pitch")
            instruct = f"{gender}, Young Adult, {pitch}, American Accent"
        elif folder == "whisper":
            instruct = f"{gender}, Young Adult, Moderate Pitch, American Accent, Whisper"
        else:
            accent = folder.replace("_", " ").title()
            instruct = f"{gender}, {age}, Moderate Pitch, {accent}"

        logger.info("Regenerating %s ← %s", rel, instruct)
        try:
            audio = omni.generate(text=text, instruct=instruct, num_step=num_step)
            import torch as th
            tensor = th.cat([t.cpu() for t in audio], dim=-1)
            torchaudio.save(
                str(wav), tensor, SAMPLE_RATE,
                format="wav", encoding="PCM_S", bits_per_sample=16,
            )
            logger.info("  OK (%.1fs)", tensor.shape[-1] / SAMPLE_RATE)
        except Exception as e:
            logger.error("  FAILED: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Verify voice design audio with ASR")
    parser.add_argument("--dir", default=str(DESIGN_DIR))
    parser.add_argument("--model", default="base.en", help="Whisper model size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Word overlap threshold (0-1)")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate bad files")
    parser.add_argument("--omnivoice-model", default="k2-fsa/OmniVoice")
    parser.add_argument("--num-step", type=int, default=16)
    args = parser.parse_args()

    bad = verify_all(Path(args.dir), args.model, args.threshold)

    if bad and args.regenerate:
        regenerate(bad, args.omnivoice_model, args.num_step)


if __name__ == "__main__":
    main()
